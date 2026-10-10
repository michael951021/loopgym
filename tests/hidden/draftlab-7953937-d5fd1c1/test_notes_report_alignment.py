"""Hidden check, draftlab-7953937-d5fd1c1.

NOTES.md aligned with the corrected REPORT: no dead attempt-1 report line,
judge numbers match results/judge.json (identical at both commits), and a
done 42-cell sweep grid is kept only if marked superseded. Stdlib only.
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
SVT = json.loads((WT / "results" / "judge.json").read_text())["spec_vs_target"]


def judge_line():
    for ln in (WT / "NOTES.md").read_text().splitlines():
        if "spec_vs_target" in ln:
            return ln
    raise AssertionError("no spec_vs_target line in NOTES.md")


def test_no_stale_report_state():
    n = (WT / "NOTES.md").read_text()
    assert not re.search(r"attempt 1 died", n, re.I), "stale attempt-1 line kept"
    assert re.search(r"check_report", n) and re.search(r"exit 0", n)


def test_judge_win_rate_matches_judge_json():
    m = re.search(r"(\d?\.\d{2,4})", judge_line())
    assert m, "no win rate on spec_vs_target line"
    assert abs(float(m.group(1)) - SVT["win_rate"]) < 0.001, \
        f"NOTES cites {m.group(1)} != judge.json {SVT['win_rate']:.4f}"


def test_judge_verdict_count_matches_judge_json():
    m = re.search(r"n=(\d+)|(\d+)\s+verdicts", judge_line())
    assert m, "no verdict count on spec_vs_target line"
    assert int(m.group(1) or m.group(2)) == SVT["n_verdicts"], \
        f"count != judge.json n_verdicts={SVT['n_verdicts']}"


def test_done_sweep_grid_marked_superseded():
    n = (WT / "NOTES.md").read_text()
    if re.search(r"42[\s-]*cells?\b[^.\n]*\bdone\b|\bdone\b[^.\n]*42[\s-]*cells?", n, re.I):
        assert re.search(r"superseded", n, re.I), "42-cell grid not marked superseded"
