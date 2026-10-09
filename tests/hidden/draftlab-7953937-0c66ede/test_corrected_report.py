"""Hidden check, draftlab-7953937-0c66ede.

Corrected REPORT: numbers recomputed from results/ (held-out alpha + run-level
95% CIs), no significance claim where CIs overlap, Corrections section.
Fails at the bad commit (stale val-based alphas, disjoint-CI claim, no
Corrections); passes at the fix. Stdlib only, instant.
"""
import json
import re
import sys
from pathlib import Path


def root() -> Path:
    for p in sys.path:
        d = Path(p or ".")
        if (d / "REPORT.md").exists() and (d / "results" / "alpha").is_dir():
            return d
    raise AssertionError("worktree with REPORT.md + results/alpha not on sys.path")


def overall(d: str) -> dict:
    return json.loads((root() / "results" / "alpha" / f"{d}.json").read_text())["overall"]


DRAFTERS = ("ngram", "lora", "dpo", "tiny")


def test_alpha_runs_are_heldout_14():
    for d in DRAFTERS:
        j = json.loads((root() / "results" / "alpha" / f"{d}.json").read_text())
        assert j["split"] == "heldout" and j["n_runs"] == 14


def test_report_states_recomputed_alpha_and_run_ci():
    rep = (root() / "REPORT.md").read_text()
    for d in DRAFTERS:
        o = overall(d)
        assert f"{o['alpha']:.4f}" in rep
        assert f"{o['alpha_ci95'][0]:.4f}" in rep
        assert f"{o['alpha_ci95'][1]:.4f}" in rep


def test_known_answer_tpc_formula():
    for d in DRAFTERS:  # Leviathan: tpc(k) = 1 + sum_{i<=k} alpha^i
        a, tpc = overall(d)["alpha"], overall(d)["tokens_per_call"]
        for k in (1, 8):
            assert abs(tpc[str(k)] - (1 + sum(a**i for i in range(1, k + 1)))) < 0.02


def test_no_significance_claim_where_cis_overlap():
    lo, hi = overall("lora")["alpha_ci95"]
    dlo, dhi = overall("dpo")["alpha_ci95"]
    assert max(lo, dlo) < min(hi, dhi)  # results: CIs overlap
    rep = (root() / "REPORT.md").read_text()
    assert re.search(r"overlap|indistinguishable", rep, re.I)


def test_corrections_section():
    assert re.search(r"corrections", (root() / "REPORT.md").read_text(), re.I)
