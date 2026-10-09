"""tools/toy_check.py: verdict logic on synthetic runs, config field checks."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "toy_check.py"


def write_run(base: Path, name: str, rewards: list[float], eval_noise: bool = False) -> None:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for s, r in enumerate(rewards):
        lines.append(json.dumps({"step": s, "reward_mean": r, "kind": None}))
        if eval_noise:  # eval lines must be ignored by the checker
            lines.append(json.dumps({"step": s, "reward_mean": 0.99, "kind": "eval"}))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def run_tool(d: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    return subprocess.run([sys.executable, str(TOOL), "--dir", str(d)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_passing_pair_exits_zero(tmp_path):
    # banana: 0.1 early, rising to 0.9; null: flat 0.5 with deterministic jitter
    banana = [0.1 + 0.8 * s / 59 for s in range(60)]
    null = [0.5 + 0.01 * (s % 3 - 1) for s in range(60)]
    write_run(tmp_path, "toy_banana", banana, eval_noise=True)
    write_run(tmp_path, "toy_null", null)
    p = run_tool(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PASS" in p.stdout


def test_flat_banana_fails(tmp_path):
    write_run(tmp_path, "toy_banana", [0.5] * 60)
    write_run(tmp_path, "toy_null", [0.5] * 60)
    p = run_tool(tmp_path)
    assert p.returncode == 1
    assert "FAIL" in p.stdout


def test_trending_null_fails(tmp_path):
    banana = [0.1 + 0.8 * s / 59 for s in range(60)]
    null = [0.2 + 0.6 * s / 59 for s in range(60)]  # clear trend: CI excludes 0
    write_run(tmp_path, "toy_banana", banana)
    write_run(tmp_path, "toy_null", null)
    p = run_tool(tmp_path)
    assert p.returncode == 1


def test_short_run_cannot_pass(tmp_path):
    # 2-step probe: no steps 55-59 -> banana window n/a -> exit 1
    write_run(tmp_path, "toy_banana", [0.1, 0.15])
    write_run(tmp_path, "toy_null", [0.5, 0.5])
    p = run_tool(tmp_path)
    assert p.returncode == 1
    assert "n/a" in p.stdout


def test_missing_file_fails(tmp_path):
    write_run(tmp_path, "toy_banana", [0.1 + 0.8 * s / 59 for s in range(60)])
    p = run_tool(tmp_path)
    assert p.returncode == 1
    assert "missing" in p.stdout


def _load_cfg(name: str):
    sys.path.insert(0, str(ROOT))
    try:
        from loopgym.rl.train import load_cfg
        return load_cfg(str(ROOT / "configs" / name), {})
    finally:
        sys.path.remove(str(ROOT))


def test_toy_configs_fields():
    for name, toy_task in (("toy_banana.toml", "banana"), ("toy_null.toml", "coinflip")):
        cfg = _load_cfg(name)
        assert cfg.model == "0.5b"
        assert cfg.lora_r == 16
        assert cfg.lora_targets == ("q_proj", "k_proj", "v_proj", "o_proj")
        assert cfg.task == "toy" and cfg.toy_task == toy_task
        assert cfg.batch == 4 and cfg.group == 8
        assert cfg.max_new_tokens == 24
        assert math.isclose(cfg.lr, 5e-5)
        assert cfg.beta == 0.0
        assert cfg.steps == 60 and cfg.save_every == 10
        assert cfg.out == ("runs/toy_banana" if toy_task == "banana" else "runs/toy_null")
