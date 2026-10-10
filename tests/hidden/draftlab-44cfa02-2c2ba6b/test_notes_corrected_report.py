"""Hidden check, draftlab-44cfa02-2c2ba6b.

NOTES.md aligned with the corrected REPORT: REPORT carries the Corrections
section; the heldout lora/dpo alpha CIs recomputed from results/ overlap
(NOTES drops the stale "disjoint" claim); NOTES' spec_vs_target line agrees
with results/judge.json; a done 42-cell sweep grid is kept only if marked
superseded. Stdlib only.
"""
import json
import re
import sys
from pathlib import Path


def _wt():
    for p in sys.path:
        d = Path(p or ".")
        if (d / "NOTES.md").exists() and (d / "results" / "judge.json").exists():
            return d
    raise AssertionError("draftlab worktree (NOTES.md + results/judge.json) not on sys.path")


WT = _wt()
NOTES = (WT / "NOTES.md").read_text()


def test_report_is_corrected():
    rep = (WT / "REPORT.md").read_text()
    assert re.search(r"^##\s+Corrections", rep, re.M), "REPORT.md lacks Corrections section"


def _alpha(name):
    d = json.loads((WT / "results" / "alpha" / f"{name}.json").read_text())
    return d["split"], d["overall"]["alpha_ci95"]


def test_heldout_alpha_cis_overlap_and_notes_aligned():
    lora_split, lo = _alpha("lora")
    dpo_split, hi = _alpha("dpo")
    assert lora_split == dpo_split == "heldout", "alpha results not the heldout re-measure"
    assert lo[0] < hi[1] and hi[0] < lo[1], "heldout lora/dpo CIs overlap"
    line = next((ln for ln in NOTES.splitlines() if "lora" in ln and "dpo" in ln), "")
    assert "disjoint" not in line.lower(), "NOTES still claims lora/dpo CIs disjoint"


def test_judge_line_agrees_with_judge_json():
    svt = json.loads((WT / "results" / "judge.json").read_text())["spec_vs_target"]
    line = next((ln for ln in NOTES.splitlines() if "spec_vs_target" in ln), "")
    m = re.search(r"(\d?\.\d{2,4})", line)
    assert m, "no spec_vs_target win rate in NOTES"
    assert abs(float(m.group(1)) - svt["win_rate"]) < 0.001, "win rate != judge.json"
    m = re.search(r"n=(\d+)|(\d+)\s+verdicts", line)
    assert m, "no verdict count on spec_vs_target line"
    assert int(m.group(1) or m.group(2)) == svt["n_verdicts"], "count != judge.json"


def test_done_sweep_grid_marked_superseded():
    if re.search(r"42[\s-]*cells?\b[^.\n]*\bdone\b|\bdone\b[^.\n]*42[\s-]*cells?", NOTES, re.I):
        assert re.search(r"superseded", NOTES, re.I), "42-cell grid not marked superseded"
