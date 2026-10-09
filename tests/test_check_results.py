"""tools/check_results.py: range/known-answer verdicts on synthetic results."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_results.py"


def write_eval(results: Path, fname: str, p: float, ci=(0.0, 0.5)) -> None:
    d = results / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps(
        {"pass_at_k": p, "ci": list(ci), "n_tasks": 9}))


def run_tool(results: Path, only: str | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--results-dir", str(results)]
    if only:
        cmd += ["--only", only]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=120)


def test_in_range_exits_zero(tmp_path):
    write_eval(tmp_path, "base05_val.json", 0.20, (0.10, 0.30))
    write_eval(tmp_path, "base15_val.json", 0.40, (0.30, 0.50))
    write_eval(tmp_path, "reference_exercism_val.json", 1.0)
    p = run_tool(tmp_path, "baselines")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PASS" in p.stdout and "FAIL" not in p.stdout


def test_out_of_range_fails(tmp_path):
    write_eval(tmp_path, "base05_val.json", 0.0833, (0.0, 0.167))
    write_eval(tmp_path, "base15_val.json", 0.3611, (0.167, 0.583))
    write_eval(tmp_path, "reference_exercism_val.json", 1.0)
    p = run_tool(tmp_path, "baselines")
    assert p.returncode == 1
    assert "FAIL" in p.stdout and "0.5B val pass@1" in p.stdout


def test_missing_baseline_fails(tmp_path):
    write_eval(tmp_path, "base15_val.json", 0.40)
    p = run_tool(tmp_path, "baselines")
    assert p.returncode == 1
    assert "missing" in p.stdout


def test_size_order_fails(tmp_path):
    # both in range but 1.5B below 0.5B: known answer broken
    write_eval(tmp_path, "base05_val.json", 0.30)
    write_eval(tmp_path, "base15_val.json", 0.25)
    write_eval(tmp_path, "reference_exercism_val.json", 1.0)
    p = run_tool(tmp_path, "baselines")
    assert p.returncode == 1
    assert "1.5B > 0.5B" in p.stdout
    assert "FAIL" in p.stdout


def test_reference_not_one_fails(tmp_path):
    write_eval(tmp_path, "base05_val.json", 0.20)
    write_eval(tmp_path, "base15_val.json", 0.40)
    write_eval(tmp_path, "reference_exercism_val.json", 0.9)
    p = run_tool(tmp_path, "baselines")
    assert p.returncode == 1
    assert "reference" in p.stdout


def test_toy_missing_is_pending_not_failure(tmp_path):
    p = run_tool(tmp_path, "toy")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PENDING" in p.stdout


def test_benches(tmp_path):
    (tmp_path / "bench_logprob.json").write_text(json.dumps({
        "T": [2048], "cells": [
            {"impl": "naive", "T": 2048, "peak_extra_bytes": 1_000_000_000},
            {"impl": "torch", "T": 2048, "peak_extra_bytes": 100_000_000},
        ]}))
    (tmp_path / "lgbox_bench.json").write_text(json.dumps({"cells": [
        {"runner": "local", "concurrency": 1, "lat_mean_s": 0.20},
        {"runner": "http", "concurrency": 1, "lat_mean_s": 0.21},
    ]}))
    p = run_tool(tmp_path, "benches")
    assert p.returncode == 0, p.stdout + p.stderr
    # overhead above 50 ms -> FAIL
    (tmp_path / "lgbox_bench.json").write_text(json.dumps({"cells": [
        {"runner": "local", "concurrency": 1, "lat_mean_s": 0.20},
        {"runner": "http", "concurrency": 1, "lat_mean_s": 0.26},
    ]}))
    p = run_tool(tmp_path, "benches")
    assert p.returncode == 1
    assert "lgbox overhead" in p.stdout
