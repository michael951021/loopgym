"""GRPO training loop and CLI: sample tasks, rollout, grade, one GRPO step per step.

`python -m loopgym.rl.train --config configs/X.toml [--resume | --resume-if-any] [--field value ...]`

Exact resume: the checkpoint stores the step count, the LoRA adapter, the optimizer
state, and both RNG states (task sampling and torch). A killed run resumed with
--resume continues the step counter and the data order exactly. Compute jobs use
--resume-if-any, so the same command line starts the run and continues it after a kill.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import random
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
import safetensors.torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Qwen2ForCausalLM

from loopgym.env import exercism, toy
from loopgym.models import MODELS_DIR, tiny_config, tiny_tokenizer
from loopgym.rl import batch as B
from loopgym.rl import dist as D
from loopgym.rl.rollout import HFBackend

ROOT = Path(__file__).resolve().parents[2]
EVAL_SEED_OFFSET = 1_000_000_003

# fields that must match a checkpoint's config for --resume to be safe
RESUME_CHECK = ("model", "lora_r", "lora_alpha", "lora_targets", "task", "toy_task",
                "split", "max_difficulty", "batch", "group", "max_new_tokens",
                "temperature", "top_p", "seed", "runner", "eval_n",
                "backend", "trainer_device", "micro_batch")


@dataclass
class TrainCfg:
    model: str = "tiny"  # "tiny" | "0.5b" | "1.5b" | path to a HF model dir
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    task: str = "toy"  # exercism | toy | loopgym
    toy_task: str = "banana"  # banana | coinflip
    split: str = "train"
    max_difficulty: int = 0  # exercism only; 0 = no cap (data runs 1-9)
    batch: int = 4  # tasks per step
    group: int = 8  # completions per task
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    lr: float = 2e-5
    beta: float = 0.0  # KL coefficient
    clip: float = 0.2
    agg: str = "token-mean"
    scale: str = "std"  # "std" | "none" (Dr. GRPO)
    max_grad_norm: float = 1.0
    chunk: int = 8192
    micro_batch: int = 8  # rows per rollout/logp chunk; sized to fit the 24 GB job cap
    logprob_impl: str = "torch"
    runner: str = "local"  # local | http (lgbox)
    backend: str = "hf"  # hf | vllm (rollout backend; vllm is host-only)
    trainer_device: str = ""  # "" = CPU; "cuda:1" on the GPU host
    ds_config: str = ""  # deepspeed only: engine-config base json ("" = configs/ds_zero3.json)
    seed: int = 1337
    steps: int = 100
    eval_every: int = 0  # 0 disables eval rollouts
    eval_n: int = 0  # 0 = eval the step's tasks; >0 = draw that many val tasks
    save_every: int = 10
    timeout: int = 30  # per-grade timeout, seconds
    out: str = "runs/unnamed"


_FIELD_DEFAULTS = {f.name: f.default for f in dataclasses.fields(TrainCfg)
                   if f.default is not dataclasses.MISSING}


def load_cfg(path: str, overrides: dict) -> TrainCfg:
    doc = tomllib.loads(Path(path).read_text())
    doc.update(overrides)
    fields = {f.name: f for f in dataclasses.fields(TrainCfg)}
    unknown = set(doc) - set(fields)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    for k, v in doc.items():
        if k == "lora_targets" and isinstance(v, list):
            doc[k] = tuple(v)
        elif isinstance(v, str) and k in _INT_FIELDS:
            doc[k] = int(v)
        elif isinstance(v, str) and k in _FLOAT_FIELDS:
            doc[k] = float(v)
        elif isinstance(v, str) and k in _BOOL_FIELDS:
            doc[k] = v.lower() == "true"
    return TrainCfg(**doc)


_INT_FIELDS = {f.name for f in dataclasses.fields(TrainCfg) if f.type is int or f.type == "int"}
_FLOAT_FIELDS = {f.name for f in dataclasses.fields(TrainCfg) if f.type is float or f.type == "float"}
_BOOL_FIELDS = {f.name for f in dataclasses.fields(TrainCfg) if f.type is bool or f.type == "bool"}


def model_dir(cfg: TrainCfg) -> Path:
    for cand in (Path(cfg.model), MODELS_DIR / cfg.model,
                 MODELS_DIR / f"qwen2.5-coder-{cfg.model}-instruct"):
        if cand.is_dir():
            return cand
    raise SystemExit(f"model dir not found for {cfg.model!r}")


def build_base(cfg: TrainCfg) -> Qwen2ForCausalLM:
    """Fresh base model, deterministically initialized from cfg.seed."""
    torch.manual_seed(cfg.seed)
    if cfg.model == "tiny":
        return Qwen2ForCausalLM(tiny_config())
    return Qwen2ForCausalLM.from_pretrained(str(model_dir(cfg)), torch_dtype=torch.float32)


def load_tokenizer(cfg: TrainCfg):
    if cfg.model == "tiny":
        return tiny_tokenizer()
    return AutoTokenizer.from_pretrained(str(model_dir(cfg)))


class TaskSampler:
    """Draws the tasks of each step from a fixed order controlled by one random.Random."""

    def __init__(self, names, seed: int):
        self.names = list(names)
        if not self.names:
            raise SystemExit("task source has no tasks")
        self.rng = random.Random(seed)

    def draw(self, n: int) -> list:
        if n >= len(self.names):
            return [self.rng.choice(self.names) for _ in range(n)]
        return self.rng.sample(self.names, n)

    def state(self):
        return self.rng.getstate()

    def load_state(self, st):
        self.rng.setstate(st)


class Run:
    def __init__(self, cfg: TrainCfg, dist: str = ""):
        self.cfg = cfg
        self.dist = dist
        self.engine = None
        self.world, self.rank = 1, 0
        self.tokenizer = load_tokenizer(cfg)
        self.val_names = None
        if cfg.eval_n > 0 and cfg.task != "exercism":
            raise SystemExit("eval_n requires task = \"exercism\"")
        if cfg.backend not in ("hf", "vllm"):
            raise SystemExit(f"unknown backend {cfg.backend!r} (hf | vllm)")
        if cfg.task == "toy":
            self.toy_task = toy.load_toy(cfg.toy_task)
            self.sampler = None
            self.exdir = None
        elif cfg.task == "exercism":
            self.toy_task = None
            self.exdir = Path(os.environ.get(
                "LOOPGYM_EXERCISES",
                ROOT / "data" / "raw" / "exercism-python" / "exercises" / "practice"))
            splits_doc = json.loads((ROOT / "results" / "splits.json").read_text())
            by_diff = splits_doc[cfg.split]
            names = list(by_diff)
            if cfg.max_difficulty > 0:
                names = [n for n in names if by_diff[n] <= cfg.max_difficulty]
            self.val_names = list(splits_doc.get("val", []))
            self.sampler = TaskSampler(names, cfg.seed)
        elif cfg.task == "loopgym":
            raise NotImplementedError("loopgym task source is not mined yet")
        else:
            raise SystemExit(f"unknown task source {cfg.task!r}")
        self.policy = get_peft_model(
            build_base(cfg),
            LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                       target_modules=list(cfg.lora_targets)))
        if cfg.trainer_device:
            self.policy = self.policy.to(cfg.trainer_device)
        if dist in ("fsdp", "deepspeed") and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            D.init_dist()
            self.rank, self.world = D.rank(), D.world_size()
            if dist == "fsdp":
                D.wrap_fsdp(self.policy)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad], lr=cfg.lr)
        if dist == "deepspeed" and self.world > 1:
            from loopgym.rl import ds as _ds
            base = json.loads(Path(cfg.ds_config).read_text()) if cfg.ds_config else None
            self.engine, _, _ = _ds.build_engine(self.policy, self._ds_doc(), base=base)
        self.backend = self._make_backend()
        self.step_cfg = B.StepCfg(logprob_impl=cfg.logprob_impl, clip_eps=cfg.clip,
                                  kl_coef=cfg.beta, agg=cfg.agg,
                                  max_grad_norm=cfg.max_grad_norm, chunk=cfg.chunk,
                                  micro_batch=cfg.micro_batch)
        self.step = 0  # completed steps

    def _make_backend(self):
        if self.cfg.backend == "hf":
            return HFBackend(self.policy, self.tokenizer, micro_batch=self.cfg.micro_batch)
        from loopgym.rl import rollout_vllm as rv
        model_path = "tiny" if self.cfg.model == "tiny" else str(model_dir(self.cfg))
        args = rv.engine_args(model_path, lora_rank=self.cfg.lora_r)
        return rv.VLLMBackend(rv.build_llm(args), self.tokenizer)

    def _ds_doc(self) -> dict:
        """The train-config doc the DeepSpeed engine reads (lr, clip, per-rank batch)."""
        cfg = self.cfg
        return {"lr": cfg.lr, "max_grad_norm": cfg.max_grad_norm,
                "batch": cfg.batch, "group": cfg.group}

    def _seeds(self, s: int, count: int, eval_offset: int = 0) -> list[int]:
        """Per-prompt rollout seeds: prompt i (global task index) gets its own seed so
        its completions match across topologies (single process vs rank slice)."""
        base = self.cfg.seed + eval_offset + s
        g0 = self.rank * self.cfg.batch
        return [base * 1000 + (g0 + i) for i in range(count)]

    def sync_vllm(self, s: int, adapter_dir: Path) -> None:
        """Save the current LoRA adapter and point the vLLM engine at it."""
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(str(adapter_dir))
        self.backend.sync_lora(adapter_dir, f"step_{s}")

    # -- data -------------------------------------------------------------
    def tasks(self) -> list:
        if self.cfg.task == "toy":
            return [self.toy_task] * self.cfg.batch
        # all ranks draw the same global order (same sampler seed, same draws) and
        # take disjoint slices, so the step's data is the single-process data
        names = self.sampler.draw(self.cfg.batch * self.world)
        if self.world > 1:
            names = names[self.rank * self.cfg.batch:(self.rank + 1) * self.cfg.batch]
        exdir = self.exdir
        return [exercism.load_exercise(exdir / n) for n in names]

    # -- grading ----------------------------------------------------------
    def grade_all(self, tasks, groups, s: int) -> list[list[float]]:
        rows = []
        for b, task in enumerate(tasks):
            for g, c in enumerate(groups[b]):
                rows.append((b, g, task, c))

        def one(row):
            b, g, task, c = row
            if self.cfg.task == "toy":
                gi = (self.rank * self.cfg.batch + b) * self.cfg.group + g
                r, _ = toy.grade(task, task.parse(c.text), step=s, index=gi)
            else:
                r, _ = exercism.grade(task, task.parse(c.text), runner=self.cfg.runner,
                                      timeout=self.cfg.timeout)
            return r

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, max(1, len(rows)))) as pool:
            rewards = list(pool.map(one, rows))
        return [rewards[b * self.cfg.group:(b + 1) * self.cfg.group] for b in range(len(tasks))]

    # -- one step ---------------------------------------------------------
    def run_step(self, s: int, tasks, prompts) -> dict:
        t0 = time.time()
        cfg = self.cfg
        groups = self.backend.generate(
            prompts, cfg.group, cfg.max_new_tokens, cfg.temperature, cfg.top_p,
            seed=cfg.seed + s, seeds=self._seeds(s, len(prompts)))
        roll_s = max(time.time() - t0, 1e-9)
        rewards = self.grade_all(tasks, groups, s)
        batch_ = B.build_batch(groups, rewards, self.tokenizer, scale=cfg.scale)
        n_local = float(batch_.target_mask.sum())
        n_global = D.global_token_count(n_local)
        # the per-rank token-mean backward is AVG-reduced across ranks; the world
        # factor rescales it to the single-process global token-mean step
        loss_factor = self.world * n_local / n_global
        stats = B.grpo_step(self.policy, None, batch_, self.optimizer, self.step_cfg,
                            loss_factor=loss_factor, engine=self.engine)
        loss = float(D.global_mean_loss(torch.tensor(stats["loss"]) * n_local, n_global))
        if self.cfg.backend == "vllm":
            self.sync_vllm(s, Path(cfg.out) / "vllm_adapter")
        r = torch.tensor(rewards, dtype=torch.float32).reshape(-1)
        same = sum(1 for b in range(len(tasks)) if len({x for x in rewards[b]}) == 1)
        if self.world > 1:
            # report the global (all-rank) reward stats: rank 0's local slice alone
            # would not match the single-process run the step is equivalent to
            n_g = D.global_token_count(float(len(r)))
            sum_g = D.global_token_count(float(r.sum()))
            sumsq_g = D.global_token_count(float((r * r).sum()))
            mean = sum_g / n_g
            var = (sumsq_g - n_g * mean * mean) / (n_g - 1) if n_g > 1 else 0.0
            mean, std = float(mean), float(math.sqrt(max(var, 0.0)))
        else:
            mean = float(r.mean())
            std = float(r.std(unbiased=True)) if len(r) > 1 else 0.0
        rec = {
            "step": s,
            "reward_mean": mean,
            "reward_std": std,
            "frac_groups_all_same": same / len(tasks),
            "loss": loss,
            "kl": stats["kl"],
            "clip_frac": stats["clip_frac"],
            "grad_norm": stats["grad_norm"],
            "gen_tok_s": sum(len(c.completion_ids) for grp in groups for c in grp) / roll_s,
            "step_s": time.time() - t0,
        }
        return rec

    def eval_tasks(self, s: int):
        """Draw this step's val eval tasks. Stateless per step (fresh sampler) so a
        resumed run re-derives the same tasks without checkpointed RNG state."""
        sampler = TaskSampler(self.val_names, self.cfg.seed + EVAL_SEED_OFFSET + s)
        return [exercism.load_exercise(self.exdir / n) for n in sampler.draw(self.cfg.eval_n)]

    def eval_step(self, s: int, fallback_tasks) -> dict | None:
        cfg = self.cfg
        if self.world > 1 and cfg.eval_n <= 0:
            return None  # the per-rank train slice is not an aggregate; all ranks skip
        tasks = self.eval_tasks(s) if cfg.eval_n > 0 else fallback_tasks
        prompts = [t.prompt() for t in tasks]
        groups = self.backend.generate(
            prompts, cfg.group, cfg.max_new_tokens, cfg.temperature, cfg.top_p,
            seed=cfg.seed + EVAL_SEED_OFFSET + s)

        def one(i):
            b, g = divmod(i, cfg.group)
            task = tasks[b]
            if cfg.task == "toy":
                r, _ = toy.grade(task, task.parse(groups[b][g].text),
                                 step=s, index=EVAL_SEED_OFFSET + i)
            else:
                r, _ = exercism.grade(task, task.parse(groups[b][g].text),
                                      runner=cfg.runner, timeout=cfg.timeout)
            return r

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tasks) * cfg.group)) as pool:
            rewards = list(pool.map(one, range(len(tasks) * cfg.group)))
        r = torch.tensor(rewards, dtype=torch.float32)
        return {"step": s, "kind": "eval",
                "tasks": [t.name for t in tasks],
                "reward_mean": float(r.mean()),
                "reward_std": float(r.std(unbiased=True)) if len(r) > 1 else 0.0}

    # -- checkpoint -------------------------------------------------------
    def _dist_adapter_tensors(self) -> dict:
        """Collective (all ranks): the full LoRA tensors on CPU, one-process format."""
        if self.dist == "deepspeed":
            import deepspeed  # pinned source aliases `zero` as an attribute, not a submodule
            lora = {n: p for n, p in self.policy.named_parameters() if "lora_" in n}
            with deepspeed.zero.GatheredParameters(list(lora.values())):
                full = {k: v.detach().cpu().clone() for k, v in lora.items()}
            from peft.utils.save_and_load import get_peft_model_state_dict
            out = get_peft_model_state_dict(self.policy, state_dict=full, adapter_name="default")
            return {k: v for k, v in out.items()}
        full = D.full_state_dict(self.policy)  # FSDP2: gather the sharded params
        from peft.utils.save_and_load import get_peft_model_state_dict
        # name the keys exactly as save_pretrained does (drops the "default" adapter
        # level) so the checkpoint is byte-identical to the single-process format
        out = get_peft_model_state_dict(self.policy, state_dict=full, adapter_name="default")
        return {k: v.detach().cpu() for k, v in out.items()}

    @staticmethod
    def _full_tensors(x):
        """Recursively materialize DTensors (FSDP2 optimizer state) as plain tensors."""
        from torch.distributed.tensor import DTensor
        if isinstance(x, DTensor):
            return x.full_tensor().detach().cpu()
        if isinstance(x, dict):
            return {k: Run._full_tensors(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(Run._full_tensors(v) for v in x)
        return x

    def _opt_state(self) -> dict:
        if self.dist == "deepspeed" and self.engine is not None:
            return self.engine.optimizer.state_dict()
        return self._full_tensors(self.optimizer.state_dict())

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        if self.world > 1:
            # both gathers are collectives: every rank joins, only rank 0 persists.
            tensors = self._dist_adapter_tensors()
            opt_state = self._opt_state()
            if D.is_rank0():
                self.policy.peft_config["default"].save_pretrained(str(d))
                safetensors.torch.save_file(tensors, str(d / "adapter_model.safetensors"))
                torch.save({"optimizer": opt_state,
                            "torch_rng": torch.random.get_rng_state()}, d / "opt_rng.pt")
                state = {"step": self.step,
                         "config": dataclasses.asdict(self.cfg),
                         "task_rng": self.sampler.state() if self.sampler else None,
                         "dist": self.dist,
                         "world": self.world}
                (d / "state.json").write_text(json.dumps(state, indent=1) + "\n")
            return
        self.policy.save_pretrained(str(d))
        torch.save({"optimizer": self.optimizer.state_dict(),
                    "torch_rng": torch.random.get_rng_state()}, d / "opt_rng.pt")
        state = {"step": self.step,
                 "config": dataclasses.asdict(self.cfg),
                 "task_rng": self.sampler.state() if self.sampler else None}
        (d / "state.json").write_text(json.dumps(state, indent=1) + "\n")

    def restore(self, d: Path):
        state = json.loads((d / "state.json").read_text())
        saved = state["config"]
        cur = dataclasses.asdict(self.cfg)
        for k in RESUME_CHECK:
            a, b = saved.get(k, _FIELD_DEFAULTS.get(k)), cur[k]
            if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
                a, b = list(a), list(b)  # lora_targets: tuple in-memory, list in JSON
            if a != b:
                raise SystemExit(f"--resume config mismatch on {k!r}: "
                                 f"checkpoint {a!r} != run {b!r}")
        if state.get("world", 1) != self.world or (
                self.world > 1 and state.get("dist", "") != self.dist):
            raise SystemExit(
                f"--resume topology mismatch: checkpoint world={state.get('world', 1)} "
                f"dist={state.get('dist', '')!r} != run world={self.world} dist={self.dist!r}")
        policy = PeftModel.from_pretrained(build_base(self.cfg), str(d))
        for n, p in policy.named_parameters():  # from_pretrained leaves all frozen
            p.requires_grad = "lora_" in n
        if self.dist == "fsdp" and self.world > 1:
            D.wrap_fsdp(policy)
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=self.cfg.lr)
        blob = torch.load(d / "opt_rng.pt", weights_only=True)
        if self.dist == "deepspeed" and self.world > 1:
            from loopgym.rl import ds as _ds
            base = json.loads(Path(self.cfg.ds_config).read_text()) if self.cfg.ds_config else None
            self.engine, _, _ = _ds.build_engine(policy, self._ds_doc(), base=base)
            self.engine.optimizer.load_state_dict(blob["optimizer"])
        else:
            optimizer.load_state_dict(blob["optimizer"])
        torch.random.set_rng_state(blob["torch_rng"])
        if self.sampler is not None:
            self.sampler.load_state(state["task_rng"])
        self.step = state["step"]
        self.policy = policy
        self.optimizer = optimizer
        if self.cfg.backend == "vllm":
            self.sync_vllm(self.step, d)  # re-point the live engine at the ckpt adapter
        else:
            self.backend = HFBackend(self.policy, self.tokenizer,
                                     micro_batch=self.cfg.micro_batch)


def metrics_path(cfg: TrainCfg) -> Path:
    return Path(cfg.out) / "metrics.jsonl"


def ckpt_dir(cfg: TrainCfg) -> Path:
    # out is "<bucket>/<name>" (e.g. runs/smoke); the checkpoint sits in a sibling
    # "ckpt/<name>" of the bucket, so the real run uses ROOT/ckpt and a test's
    # "<tmp>/runs/<name>" stays under "<tmp>".
    out = Path(cfg.out)
    return out.parent.parent / "ckpt" / out.name


def trim_metrics(mpath: Path, step: int) -> None:
    """Drop metrics rows past the checkpoint's step: a run killed between saves logged
    steps that the resumed run will redo, and must not appear twice."""
    if not mpath.is_file():
        return
    lines = mpath.read_text().splitlines(keepends=True)
    keep = [ln for ln in lines if ln.strip() and json.loads(ln)["step"] <= step]
    if len(keep) != len(lines):
        mpath.write_text("".join(keep))
        print(f"dropped {len(lines) - len(keep)} metrics rows past checkpoint step {step}")


def peak_mem_gb() -> float:
    """Peak memory the process used: CUDA max on GPU, else ru_maxrss (Linux, KB)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run(cfg: TrainCfg, resume: bool | str, dist: str = "") -> None:
    """resume: False = fresh run, True = must continue a checkpoint, "auto" = continue
    one if it exists, else start fresh (what a retried compute job needs).

    Under torchrun (WORLD_SIZE>1) every rank runs the same loop; the per-rank
    collectives (run_step, eval_step, save) stay in lockstep, only rank 0 writes
    metrics.jsonl and prints step lines, and every rank prints a PEAKMEM line."""
    out, ck = Path(cfg.out), ckpt_dir(cfg)
    mpath = metrics_path(cfg)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    have_ckpt = (ck / "state.json").is_file()
    if resume == "auto":
        resume = have_ckpt
        if not have_ckpt and mpath.is_file() and rank0:  # died before its first save
            mpath.rename(mpath.with_name(mpath.name + ".discarded"))
            print(f"no checkpoint at {ck}; moved {mpath} aside and starting fresh")
    if resume and not have_ckpt:
        raise SystemExit(f"nothing to resume: no checkpoint at {ck}")
    if not resume and (have_ckpt or mpath.is_file()):
        raise SystemExit(f"{ck} or {mpath} already exists; use --resume or a new out")
    run_ = Run(cfg, dist)
    if resume:
        run_.restore(ck)
        if rank0:
            trim_metrics(mpath, run_.step)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    start = run_.step + 1
    if start > cfg.steps:
        if not mpath.is_file():
            # orphaned checkpoint (metrics missing): a retried job must not report
            # success with no artifacts
            raise SystemExit(f"checkpoint at {ck} is past step {cfg.steps} but "
                             f"{mpath} is missing; delete the checkpoint to rerun")
        if rank0:
            print(f"[{out.name}] already complete at step {run_.step}; nothing to do")
        return
    for s in range(start, cfg.steps + 1):
        tasks = run_.tasks()
        prompts = [t.prompt() for t in tasks]
        rec = run_.run_step(s, tasks, prompts)
        run_.step = s
        if rank0:
            with mpath.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{out.name}] step {s}/{cfg.steps} reward={rec['reward_mean']:.3f} "
                  f"loss={rec['loss']:.4f} kl={rec['kl']:.5f} step_s={rec['step_s']:.1f}")
        if cfg.eval_every > 0 and s % cfg.eval_every == 0:
            erec = run_.eval_step(s, tasks)
            if erec is not None and rank0:
                with mpath.open("a") as f:
                    f.write(json.dumps(erec) + "\n")
                print(f"[{out.name}] eval step {s} reward={erec['reward_mean']:.3f}")
        if s % cfg.save_every == 0 or s == cfg.steps:
            run_.save(ck)
    print(f"PEAKMEM rank={run_.rank} gb={peak_mem_gb():.3f}", flush=True)
    if rank0:
        print(f"[{out.name}] done at step {run_.step}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loopgym.rl.train")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-if-any", action="store_true",
                    help="resume if a checkpoint exists, else start fresh (for retried jobs)")
    ap.add_argument("--dist", choices=("fsdp", "deepspeed"), default="fsdp")
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args(argv)
    if len(extra) % 2:
        ap.error("overrides must be --key value pairs")
    overrides = {}
    for i in range(0, len(extra), 2):
        key = extra[i].lstrip("-").replace("-", "_")
        if not key or key in ("config", "resume", "resume_if_any", "dist", "dry_run"):
            ap.error(f"unexpected argument {extra[i]!r}")
        overrides[key] = extra[i + 1]
    if args.dry_run:
        if args.dist != "deepspeed":
            ap.error("--dry-run requires --dist deepspeed")
        from loopgym.rl import ds as _ds
        print(json.dumps(_ds.engine_cfg(_ds.load_train_doc(args.config)), indent=2, sort_keys=True))
        return 0
    cfg = load_cfg(args.config, overrides)
    if args.resume and args.resume_if_any:
        ap.error("--resume and --resume-if-any are exclusive")
    run(cfg, resume="auto" if args.resume_if_any else args.resume, dist=args.dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
