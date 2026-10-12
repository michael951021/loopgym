"""GRPO training loop + CLI: 3-step smoke via the real CLI, and exact resume.

The resume test is the independent reference: a run stopped after step 2 and resumed
with --resume must reproduce an uninterrupted 3-step run bit-for-bit on the
deterministic metric fields. Timing fields (gen_tok_s, step_s) are wall-clock and
excluded.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DET = ["step", "reward_mean", "reward_std", "frac_groups_all_same",
       "loss", "kl", "clip_frac", "grad_norm"]

# every run overrides --out and --steps on the CLI, so both override paths are
# exercised each time (a break in either would misroute output or the step count).
BASE_TOML = """
model = "tiny"
lora_r = 4
lora_alpha = 8
lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
task = "toy"
toy_task = "banana"
batch = 2
group = 4
max_new_tokens = 16
temperature = 1.0
lr = 0.01
beta = 0.0
clip = 0.2
steps = 99
eval_every = 1
save_every = 1
seed = 42
out = "runs/placeholder"
"""


def write_cfg(tmp: Path) -> Path:
    p = tmp / "cfg.toml"
    p.write_text(BASE_TOML)
    return p


def run_cli(cfg: Path, *extra, timeout=300) -> tuple[subprocess.CompletedProcess, float]:
    cmd = [sys.executable, "-m", "loopgym.rl.train", "--config", str(cfg), *extra]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return p, time.time() - t0


def rows(out: str) -> list[dict]:
    p = Path(out) / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def train_rows(out: str) -> list[dict]:
    return [r for r in rows(out) if r.get("kind") != "eval"]


def eval_rows(out: str) -> list[dict]:
    return [r for r in rows(out) if r.get("kind") == "eval"]


def test_smoke_cli(tmp_path):
    cfg = write_cfg(tmp_path)
    out = str(tmp_path / "runs" / "smoke")
    p, elapsed = run_cli(cfg, "--out", out, "--steps", "3")
    assert p.returncode == 0, p.stderr
    assert elapsed < 180, f"smoke took {elapsed:.1f} s (must be < 3 min)"

    tr = train_rows(out)
    assert [r["step"] for r in tr] == [1, 2, 3], tr
    for r in tr:
        for f in DET + ["gen_tok_s", "step_s"]:
            assert f in r and (isinstance(r[f], (int, float))) and r[f] == r[f], (f, r)
        assert 0.0 <= r["reward_mean"] <= 1.0
    assert len(eval_rows(out)) == 3  # eval_every = 1

    ck = tmp_path / "ckpt" / "smoke"
    state = json.loads((ck / "state.json").read_text())
    assert state["step"] == 3
    assert (ck / "adapter_model.safetensors").is_file()
    assert (ck / "opt_rng.pt").is_file()


def test_resume_matches_uninterrupted(tmp_path):
    cfg = write_cfg(tmp_path)
    full = str(tmp_path / "runs" / "full")
    part = str(tmp_path / "runs" / "part")

    # uninterrupted 3 steps
    p, _ = run_cli(cfg, "--out", full, "--steps", "3")
    assert p.returncode == 0, p.stderr

    # stop after 2, then resume to 3 (exercises --steps override + --resume)
    p, _ = run_cli(cfg, "--out", part, "--steps", "2")
    assert p.returncode == 0, p.stderr
    p, _ = run_cli(cfg, "--out", part, "--steps", "3", "--resume")
    assert p.returncode == 0, p.stderr

    a, b = train_rows(full), train_rows(part)
    assert [r["step"] for r in a] == [1, 2, 3], a
    assert [r["step"] for r in b] == [1, 2, 3], f"resume miscounted steps: {b}"
    for ra, rb in zip(a, b):
        for f in DET:
            assert ra[f] == rb[f], (
                f"resume diverged at step {ra['step']} {f}: "
                f"uninterrupted={ra[f]!r} resumed={rb[f]!r}")


def test_fresh_into_existing_fails(tmp_path):
    cfg = write_cfg(tmp_path)
    out = str(tmp_path / "runs" / "c")
    p, _ = run_cli(cfg, "--out", out, "--steps", "1")
    assert p.returncode == 0, p.stderr
    p, _ = run_cli(cfg, "--out", out, "--steps", "1")
    assert p.returncode != 0
    assert "exists" in p.stderr and "resume" in p.stderr, p.stderr


def test_resume_with_no_checkpoint_fails(tmp_path):
    cfg = write_cfg(tmp_path)
    out = str(tmp_path / "runs" / "d")
    p, _ = run_cli(cfg, "--out", out, "--steps", "1", "--resume")
    assert p.returncode != 0
    assert "nothing to resume" in p.stderr, p.stderr


def test_resume_if_any_starts_then_continues(tmp_path):
    """The compute-job command line: one command must both start the run and continue it."""
    cfg = write_cfg(tmp_path)
    out = tmp_path / "runs" / "e"
    out.mkdir(parents=True)
    # a run that died before its first checkpoint: its rows cannot be continued
    (out / "metrics.jsonl").write_text(json.dumps({"step": 1, "reward_mean": -1.0}) + "\n")
    p, _ = run_cli(cfg, "--out", str(out), "--steps", "1", "--resume-if-any")
    assert p.returncode == 0, p.stderr
    assert (out / "metrics.jsonl.discarded").is_file()
    assert [r["step"] for r in train_rows(str(out))] == [1]
    assert train_rows(str(out))[0]["reward_mean"] != -1.0
    # killed between saves: a row logged past the checkpoint is redone, not duplicated
    with (out / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"step": 2, "reward_mean": -1.0}) + "\n")
    p, _ = run_cli(cfg, "--out", str(out), "--steps", "2", "--resume-if-any")
    assert p.returncode == 0, p.stderr
    rows_ = train_rows(str(out))
    assert [r["step"] for r in rows_] == [1, 2], rows_
    assert rows_[1]["reward_mean"] != -1.0


def test_orphaned_checkpoint_fails_resume_if_any(tmp_path):
    """Stale ckpt with metrics lost (the toybanana2 incident): a retried job must not
    report success with no artifacts."""
    cfg = write_cfg(tmp_path)
    out = str(tmp_path / "runs" / "o")
    p, _ = run_cli(cfg, "--out", out, "--steps", "1")
    assert p.returncode == 0, p.stderr
    (Path(out) / "metrics.jsonl").unlink()
    p, _ = run_cli(cfg, "--out", out, "--steps", "1", "--resume-if-any")
    assert p.returncode != 0, "orphaned checkpoint must not exit 0"
    assert "missing" in p.stderr, p.stderr


def test_complete_run_resume_if_any_noops(tmp_path):
    cfg = write_cfg(tmp_path)
    out = str(tmp_path / "runs" / "n")
    p, _ = run_cli(cfg, "--out", out, "--steps", "1")
    assert p.returncode == 0, p.stderr
    p, _ = run_cli(cfg, "--out", out, "--steps", "1", "--resume-if-any")
    assert p.returncode == 0, p.stderr
    assert "nothing to do" in p.stdout, p.stdout


def test_max_difficulty_filters_train_sampler(tmp_path):
    from loopgym.rl.train import ROOT, Run, TrainCfg
    splits = json.loads((ROOT / "results" / "splits.json").read_text())
    base = dict(model="tiny", task="exercism", split="train", batch=1, group=1,
                max_new_tokens=4, out=str(tmp_path / "runs" / "x"))
    r = Run(TrainCfg(max_difficulty=1, **base))
    want = {n for n, d in splits["train"].items() if d == 1}
    assert want and set(r.sampler.names) == want
    for cap in (0, 9):  # 0 = no cap; 9 = above every difficulty in the data
        full = Run(TrainCfg(max_difficulty=cap, **base))
        assert set(full.sampler.names) == set(splits["train"])


def test_eval_tasks_deterministic_and_from_val(tmp_path):
    from loopgym.rl.train import ROOT, Run, TrainCfg
    splits = json.loads((ROOT / "results" / "splits.json").read_text())
    r = Run(TrainCfg(model="tiny", task="exercism", split="train", max_difficulty=1,
                     batch=1, group=1, max_new_tokens=4, eval_n=2, seed=42,
                     out=str(tmp_path / "runs" / "y")))
    a = [t.name for t in r.eval_tasks(20)]
    b = [t.name for t in r.eval_tasks(20)]  # stateless per step: no drift on resume
    assert a == b and len(a) == 2
    assert set(a) <= set(splits["val"])


def test_exercism_eval_step_uses_val_tasks(tmp_path):
    """tiny model's 512-token context cannot take real exercism prompts, so the
    generator is stubbed; the val draw and the local grading are the real code."""
    from loopgym.rl.train import ROOT, Run, TrainCfg
    splits = json.loads((ROOT / "results" / "splits.json").read_text())
    r = Run(TrainCfg(model="tiny", task="exercism", split="train", max_difficulty=1,
                     batch=1, group=1, max_new_tokens=4, eval_n=2, seed=42,
                     out=str(tmp_path / "runs" / "z")))

    class _C:
        text = "```python\nx = 1\n```"

    r.backend.generate = (lambda prompts, g, mnt, temp, top_p, seed:
                          [[_C()] for _ in prompts])
    ev = r.eval_step(20, None)
    assert ev["kind"] == "eval" and ev["step"] == 20
    assert set(ev["tasks"]) <= set(splits["val"]), ev
    assert len(ev["tasks"]) == 2
    assert ev["reward_mean"] == 0.0  # junk solution, real local grading


def test_resume_roundtrips_exercism_sampler(tmp_path):
    """The multi-hour exercism run resumes through state.json with a LIVE TaskSampler
    (task='exercism'), but test_resume_matches_uninterrupted uses task='toy' where
    sampler is None, so it never covers restore()'s load_state branch — which is where
    grpo05 crashed. Drive the real save()/restore() on an exercism Run and check the
    sampler position survives the JSON round-trip (tiny model, no grading, no steps)."""
    from loopgym.rl.train import ROOT, Run, TrainCfg, ckpt_dir
    json.loads((ROOT / "results" / "splits.json").read_text())  # splits.json must exist
    base = dict(model="tiny", task="exercism", split="train", max_difficulty=1,
                batch=1, group=1, max_new_tokens=4, seed=42,
                out=str(tmp_path / "runs" / "resume_ex"))
    r = Run(TrainCfg(**base))
    assert r.sampler is not None, "exercism run must have a live sampler"
    r.sampler.draw(3)          # advance past the fresh position
    r.step = 2
    r.save(ckpt_dir(TrainCfg(**base)))   # writes state.json with the sampler's position
    nxt = r.sampler.draw(3)    # what the uninterrupted run would draw next

    r2 = Run(TrainCfg(**base))           # fresh sampler at position 0
    r2.restore(ckpt_dir(TrainCfg(**base)))   # the line that crashed grpo05
    assert r2.step == 2
    assert r2.sampler.draw(3) == nxt, "restored sampler did not continue from the checkpoint"


def test_task_sampler_continuity_after_state_roundtrip():
    from loopgym.rl.train import TaskSampler
    names = ["a", "b", "c", "d", "e", "f"]
    s1 = TaskSampler(names, 7)
    s1.draw(2)                # advance a bit
    st = s1.state()           # capture position
    second = s1.draw(3)       # s1's next 3 draws from st
    s2 = TaskSampler(names, 999)  # different seed; restored state must override it
    s2.load_state(st)
    assert s2.draw(3) == second, "state did not restore the sampler's position"


def test_task_sampler_continuity_after_json_roundtrip():
    # Checkpoints store the state in state.json, which turns getstate()'s
    # tuples into lists; load_state must restore from that just as well.
    import json
    from loopgym.rl.train import TaskSampler
    names = ["a", "b", "c", "d", "e", "f"]
    s1 = TaskSampler(names, 7)
    s1.draw(2)
    st = json.loads(json.dumps(s1.state()))
    assert not isinstance(st, tuple), "JSON must have degraded the tuple to a list"
    second = s1.draw(3)
    s2 = TaskSampler(names, 999)
    s2.load_state(st)
    assert s2.draw(3) == second, "state did not restore after a JSON round-trip"
