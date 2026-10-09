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
