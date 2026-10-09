"""Hidden check, draftlab-44cfa02-9b213ef: corrected REPORT — held-out 14-run
alphas with run-level 95% CIs stated in REPORT.md, DPO-vs-LoRA as CI overlap,
Corrections section, stale pre-audit numbers gone. Stdlib only, instant.
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


def test_run_level_ci_not_block_narrow():
    # pre-fix ngram CI [0.3336, 0.3343] (width 7e-4); run-level >= 10x wider
    lo, hi = overall("ngram")["alpha_ci95"]
    assert (hi - lo) >= 10 * (0.3343 - 0.3336)


def test_report_states_recomputed_alpha_and_ci():
    rep = (root() / "REPORT.md").read_text()
    for a, ci in ((o["alpha"], o["alpha_ci95"]) for o in map(overall, DRAFTERS)):
        assert f"{a:.4f}" in rep and f"{ci[0]:.4f}" in rep and f"{ci[1]:.4f}" in rep


def test_known_answer_tpc_formula():
    for d in DRAFTERS:  # Leviathan: tpc(k) = 1 + sum_{i<=k} alpha^i
        a, tpc = overall(d)["alpha"], overall(d)["tokens_per_call"]
        for k in (1, 8):
            assert abs(tpc[str(k)] - (1 + sum(a ** i for i in range(1, k + 1)))) < 0.02


def test_no_significance_claim_where_cis_overlap():
    lo, hi = overall("lora")["alpha_ci95"]
    dlo, dhi = overall("dpo")["alpha_ci95"]
    assert max(lo, dlo) < min(hi, dhi)  # results: CIs overlap
    assert re.search(r"overlap|indistinguishable", (root() / "REPORT.md").read_text(), re.I)


def test_corrections_section_and_stale_numbers_gone():
    rep = (root() / "REPORT.md").read_text()
    assert re.search(r"^#+ .*[Cc]orrection", rep, re.M)
    assert not any(s in rep for s in ("0.3341", "0.3295", "0.7306", "0.7319"))
