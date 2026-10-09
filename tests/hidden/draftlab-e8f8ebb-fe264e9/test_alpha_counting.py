"""Hidden check, draftlab-e8f8ebb-fe264e9.

results/alpha/{target,ngram}.json were regenerated after the counting fix
(n_pos += valid_t.sum()): the target-vs-target run must give alpha(T,T)=1.0000
at the full 400-block default budget, and the two artifacts must agree with
each other (same run config, same valid position count). Stale pre-fix jsons
(bad commit) break all of this. Stdlib only, instant.
"""
import json
import sys
from pathlib import Path


def root() -> Path:
    for p in sys.path:
        d = Path(p or ".")
        if (d / "results" / "alpha" / "target.json").is_file():
            return d
    raise AssertionError("worktree with results/alpha/target.json not on sys.path")


def load(name: str) -> dict:
    return json.loads((root() / "results" / "alpha" / name).read_text())


def test_alpha_T_T_is_one():
    t = load("target.json")
    assert t["drafter"] == "target"
    for v in [t["overall"]] + list(t["by_kind"].values()):
        assert abs(v["alpha"]["mean"] - 1.0) < 1e-4
        assert abs(v["greedy_match"]["mean"] - 1.0) < 1e-4


def test_full_default_budget():
    t = load("target.json")
    assert t["n_blocks"] == 400 and t["n_runs"] == 3 and t["split"] == "val"
    assert t["n_positions"] == t["overall"]["n_positions"]
    assert sum(v["n_positions"] for v in t["by_kind"].values()) == t["n_positions"]


def test_artifacts_regenerated_together():
    t, n = load("target.json"), load("ngram.json")
    assert (t["n_blocks"], t["split"], t["n_runs"]) == (n["n_blocks"], n["split"], n["n_runs"])
    assert t["n_positions"] == n["n_positions"]
    assert sum(v["n_positions"] for v in n["by_kind"].values()) == n["n_positions"]


def test_ngram_known_range_and_leviathan():
    n = load("ngram.json")
    o, a, c = n["overall"], n["overall"]["alpha"]["mean"], n["cost_ratio_c"]
    assert 0.05 <= a <= 0.5  # range written in NOTES.md before the run
    assert 0.0 < c < 0.1
    for k in (1, 4, 8):  # Leviathan: tpc(k) = 1 + sum_{i<=k} a^i
        tpc = 1 + sum(a**i for i in range(1, k + 1))
        assert abs(o["by_k"][str(k)]["exp_tokens_per_call"] - tpc) < 0.02
        assert abs(o["by_k"][str(k)]["speedup"] - tpc / (1 + k * c)) < 0.02
