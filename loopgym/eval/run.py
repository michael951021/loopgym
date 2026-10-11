"""Eval CLI: one evaluator for every model and checkpoint.

Samples completions with the HF backend, parses and grades them locally, and
summarizes with pass@k over a task-clustered bootstrap CI. Samples are appended
to a sidecar JSONL next to --out, so an interrupted run resumes per task: each
sample is seeded by (task index, sample index) and re-runs skip finished tasks.
`--reference` scores the reference solutions as the "model" (known answer:
pass@1 = 1.0; the loopgym v0 reference is the good commit's unified diff, which
grades 1.0 by construction).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..env import grade
from ..env.exercism import load_exercise
from ..env.splits import ROOT
from ..rl.rollout import HFBackend
from .passk import summarize

MODELS_DIR = ROOT / "models"


def resolve_model_dir(name: str) -> Path:
    for cand in (Path(name), MODELS_DIR / name,
                 MODELS_DIR / f"qwen2.5-coder-{name}-instruct"):
        if cand.is_dir():
            return cand
    raise SystemExit(f"model dir not found for {name!r}")


def load_model(model: str, adapter: str | None):
    d = resolve_model_dir(model)
    tokenizer = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=torch.float32)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m, tokenizer


def load_tasks(source: str, split: str, limit: int | None) -> list:
    if source == "loopgym":
        from ..env import loopgym as _lg
        return _lg.load_v0(split, limit)
    if source != "exercism":
        raise NotImplementedError(f"task source {source!r} is not mined yet "
                                  "(see loopgym/data/verify.py)")
    doc = json.loads((ROOT / "results" / "splits.json").read_text())
    if split not in doc:
        raise SystemExit(f"unknown split {split!r}; have {sorted(doc)}")
    names = sorted(doc[split])
    if limit is not None:
        names = names[:limit]
    exdir = Path(os.environ.get(
        "LOOPGYM_EXERCISES",
        ROOT / "data" / "raw" / "exercism-python" / "exercises" / "practice"))
    return [load_exercise(exdir / n) for n in names]


def sidecar_path(out: Path) -> Path:
    return out.with_name(out.stem + ".samples.jsonl")


def load_done(side: Path) -> dict:
    """{task name: [sample records]} from the sidecar, in file order."""
    done: dict = {}
    if side.exists():
        for line in side.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.setdefault(r["task"], []).append(r)
    return done


def eval_reference(tasks, n: int, side: Path, done: dict, timeout: int = 30) -> None:
    with side.open("a") as f:
        for t in tasks:
            have = len(done.get(t.name, []))
            if have >= n:
                continue
            reward, info = grade(t, t.reference(), "local", timeout=timeout)
            for i in range(n - have):
                r = {"task": t.name, "sample": have + i, "seed": None,
                     "reward": reward, "timed_out": info["timed_out"]}
                f.write(json.dumps(r) + "\n")
                done.setdefault(t.name, []).append(r)


def eval_model(backend: HFBackend, tasks, n: int, args, side: Path,
               done: dict, timeout: int = 30) -> None:
    with side.open("a") as f:
        for idx, t in enumerate(tasks):
            have = len(done.get(t.name, []))
            if have >= n:
                continue
            for i in range(have, n):
                # one sample per generate call: seed then draw, so sample i is
                # reproducible regardless of how many samples a run requested
                comp = backend.generate([t.prompt()], 1, args.max_new_tokens,
                                        args.temperature, args.top_p,
                                        args.seed + idx * 100 + i)[0][0]
                files = t.parse(comp.text)
                reward, info = grade(t, files, "local", timeout=timeout)
                r = {"task": t.name, "sample": i, "seed": args.seed + idx * 100 + i,
                     "reward": reward, "timed_out": info["timed_out"]}
                f.write(json.dumps(r) + "\n")
                done.setdefault(t.name, []).append(r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="model dir (required unless --reference)")
    ap.add_argument("--adapter", help="LoRA adapter checkpoint dir")
    ap.add_argument("--source", choices=["exercism", "loopgym"], default="exercism")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n", type=int, default=4, help="samples per task")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="only the first N tasks of the split")
    ap.add_argument("--reference", action="store_true",
                    help="score reference solutions as the model")
    ap.add_argument("--out", help="summary JSON path (default results/eval/)")
    args = ap.parse_args(argv)
    if args.reference == bool(args.model):
        raise SystemExit("give exactly one of --reference / --model")
    if args.reference and args.adapter:
        raise SystemExit("--adapter needs --model")

    tasks = load_tasks(args.source, args.split, args.limit)
    if not tasks:
        raise SystemExit(f"no tasks for {args.source} {args.split}")
    tag = "reference" if args.reference else args.model.rsplit("/", 1)[-1]
    out = Path(args.out) if args.out else ROOT / "results" / "eval" / \
        f"{tag}_{args.source}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    side = sidecar_path(out)
    done = load_done(side)

    timeout = 120 if args.source == "loopgym" else 30  # verify.py's check budget
    if args.reference:
        eval_reference(tasks, args.n, side, done, timeout)
    else:
        model, tokenizer = load_model(args.model, args.adapter)
        eval_model(HFBackend(model, tokenizer), tasks, args.n, args, side, done,
                   timeout)

    records, per_task = [], {}
    for t in tasks:
        rs = sorted(done.get(t.name, []), key=lambda r: r["sample"])[: args.n]
        if len(rs) < args.n:
            raise SystemExit(f"task {t.name} incomplete: {len(rs)}/{args.n} samples")
        c = sum(r["reward"] >= 1.0 for r in rs)
        records.append({"task": t.name, "n": args.n, "c": c})
        per_task[t.name] = {"n": args.n, "correct": c}
    s = summarize(records, k=1)
    for name, pt in per_task.items():
        pt["pass_at_1"] = s["per_task"][name]
    doc = {
        "model": args.model, "adapter": args.adapter, "source": args.source,
        "split": args.split, "n": args.n, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "k": 1, "pass_at_k": s["pass_at_k"], "ci": list(s["ci"]),
        "n_tasks": len(tasks), "per_task": per_task,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"pass@{s['k']} {s['pass_at_k']:.3f} "
          f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] "
          f"over {len(tasks)} tasks (n={args.n} each) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
