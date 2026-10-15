"""scripts/gpu_run.sh: the dry-run contract (sandbox-safe; rule 2 = no GPU here).

The dry run must print the plan, check every required file, and exit 0 when all
are present; unknown flags exit 2; --stop-agents must appear in the plan as the
agent-stop step (the real window needs it: the model server holds ~21 GB/GPU)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_run.sh"

# gitignored host assets from REQUIRED_FILES in scripts/gpu_run.sh (keep in sync)
HOST_ASSETS = [
    "models/qwen2.5-coder-1.5b-instruct",
    "models/qwen2.5-coder-0.5b-instruct",
    "data/raw/exercism-python/exercises/practice",
    "data/loopgym/v0/tasks.jsonl",
    "data/loopgym/v0/checks",
    "data/loopgym/candidates.jsonl",
    "data/raw/repos/draftlab.git",
    "data/raw/repos/harness.git",
]
NEED_HOST = pytest.mark.skipif(
    any(not (ROOT / a).exists() for a in HOST_ASSETS),
    reason="host model/data assets absent (fresh clone)")


def run_script(*flags):
    return subprocess.run(["bash", str(SCRIPT), *flags], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


@NEED_HOST
def test_dry_run_ok():
    p = run_script("--dry-run")
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "gpu_run.sh plan" in out
    for step in ("preflight", ".venv-gpu", "known", "vllm-vs-hf", "GRPO 1.5B",
                 "eval", "results/gpu"):
        assert step in out, f"plan missing {step!r}"
    assert "dry-run OK" in out
    assert "MISSING" not in out
    # the gate values from CLAUDE.md / the config must be visible in the plan
    assert "configs/grpo15_gpu.toml" in out


def test_dry_run_without_stop_agents_does_not_stop():
    p = run_script("--dry-run")
    assert "./agent stop all --now" not in p.stdout


@NEED_HOST
def test_stop_agents_flag_shows_stop_step():
    p = run_script("--dry-run", "--stop-agents")
    assert p.returncode == 0, p.stderr
    assert "./agent stop all --now" in p.stdout


def test_unknown_flag_rejected():
    p = run_script("--bogus")
    assert p.returncode == 2
    assert "unknown flag" in p.stderr


def test_help_exits_zero():
    assert run_script("--help").returncode == 0


def test_shellcheck_clean():
    p = subprocess.run([str(ROOT / ".tools" / "bin" / "shellcheck"), str(SCRIPT)],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr


def test_check_vllm_hf_dry_run():
    p = subprocess.run([sys.executable, "-m", "loopgym.rl.check_vllm_hf",
                        "--dry-run", "--model", "m"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "check_vllm_hf plan" in out
    assert "float32" in out and "bfloat16" in out
    assert "0.02" in out
