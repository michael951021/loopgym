"""Hidden check, draftlab-bea1633-9b213ef: REPORT.md states the heldout 14-run
alphas with run-level 95% CIs recomputed from results/alpha/*.json, has a
Corrections section, no stale pre-fix numbers, and no DPO-vs-LoRA significance
claim where the CIs overlap. Stdlib only, instant.
"""
import json
import re
import sys
from pathlib import Path

DRAFTERS = ("ngram", "lora", "dpo", "tiny")


def root() -> Path:
    for p in sys.path:
        d = Path(p or ".")
        if (d / "REPORT.md").exists() and (d / "results" / "alpha").is_dir():
            return d
    raise AssertionError("worktree with REPORT.md + results/alpha not on sys.path")


def overall(d: str) -> dict:
    return json.loads((root() / "results" / "alpha" / f"{d}.json").read_text())["overall"]


def test_alphas_are_heldout_14_runs():
    for d in DRAFTERS:
        j = json.loads((root() / "results" / "alpha" / f"{d}.json").read_text())
        assert j["split"] == "heldout" and j["n_runs"] == 14


def test_report_states_recomputed_alpha_and_ci():
    rep = (root() / "REPORT.md").read_text()
    for d in DRAFTERS:
        a, (lo, hi) = overall(d)["alpha"], overall(d)["alpha_ci95"]
        assert f"{a:.4f}" in rep, f"{d} alpha"
        assert f"{lo:.4f}" in rep and f"{hi:.4f}" in rep, f"{d} run-level CI"


def test_run_level_ci_not_block_narrow():
    # pre-fix block-level ngram CI width was 7e-4; run-level is >= 10x wider
    lo, hi = overall("ngram")["alpha_ci95"]
    assert (hi - lo) >= 10 * (0.3343 - 0.3336)


def test_tpc_known_answer():
    for d in DRAFTERS:  # Leviathan: tpc(k) = 1 + sum_{i<=k} alpha^i
        a, tpc = overall(d)["alpha"], overall(d)["tokens_per_call"]
        for k in (1, 8):
            assert abs(tpc[str(k)] - (1 + sum(a ** i for i in range(1, k + 1)))) < 0.02


def test_no_significance_claim_where_cis_overlap():
    (lo, hi), (dlo, dhi) = overall("lora")["alpha_ci95"], overall("dpo")["alpha_ci95"]
    assert max(lo, dlo) < min(hi, dhi)  # results: CIs overlap
    assert re.search(r"overlap|indistinguishable", (root() / "REPORT.md").read_text(), re.I)


def test_corrections_section_stale_numbers_gone():
    rep = (root() / "REPORT.md").read_text()
    assert re.search(r"^#+ .*[Cc]orrection", rep, re.M)
    assert not any(s in rep for s in ("0.3341", "0.7306", "0.7319"))
