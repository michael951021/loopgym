"""Hidden check, draftlab-5bf9b0d-ca4403c.

results/alpha/tiny.json: hold-out alpha over 14 runs, run-level CIs,
tpc/speedup for k in 1..8 at the measured cost ratio c.
Fails at the bad commit (val split, 3 runs). Stdlib only, instant.
"""
import json
import sys
from pathlib import Path


def root() -> Path:
    for p in sys.path:
        d = Path(p or ".")
        if (d / "results" / "alpha" / "tiny.json").is_file():
            return d
    raise AssertionError("worktree with results/alpha/tiny.json not on sys.path")


def data() -> dict:
    return json.loads((root() / "results" / "alpha" / "tiny.json").read_text())


def test_heldout_14_runs():
    j = data()
    o = j["overall"]
    assert j["split"] == "heldout" and j["n_runs"] == 14
    assert o["n_runs"] == 14
    for v in j["by_kind"].values():
        assert 1 <= v["n_runs"] <= 14  # sparse kinds may lack tokens in some runs


def test_run_level_ci_and_k_grid():
    j = data()
    assert "runs" in j["bootstrap"]["unit"].lower()  # CI is over runs, not blocks
    assert j["k_values"] == list(range(1, 9))
    assert j["c"] > 0  # measured draft/target cost ratio
    for v in [j["overall"]] + list(j["by_kind"].values()):
        for lo, hi in (v["alpha_ci95"], v["greedy_ci95"]):
            assert 0.0 <= lo <= hi <= 1.0


def test_known_answer_tpc_and_speedup():
    j = data()
    o, a, c = j["overall"], j["overall"]["alpha"], j["c"]
    for k in (1, 4, 8):  # Leviathan: tpc(k) = 1 + sum_{i<=k} alpha^i
        tpc = 1 + sum(a**i for i in range(1, k + 1))
        assert abs(o["tokens_per_call"][str(k)] - tpc) < 0.02
        assert abs(o["speedup"][str(k)] - tpc / (1 + k * c)) < 0.02


def test_internal_consistency():
    j = data()
    assert sum(v["n_tokens"] for v in j["by_kind"].values()) == j["overall"]["n_tokens"]
    assert j["max_tokens"] <= 256
