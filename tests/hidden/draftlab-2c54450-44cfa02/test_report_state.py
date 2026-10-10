"""Hidden check draftlab-2c54450-44cfa02: report work is complete in this worktree —
REPORT.md + README.md exist with the ticket's required sections,
tools/check_report.py recomputes headline numbers from results/ and exits 0,
tools/ is ruff-clean, and NOTES.md records the report state (checker exit 0,
temp-filter + ruff fixes). CPU, seconds."""
import re
import subprocess
import sys
from pathlib import Path

WT = next((Path(p) for p in sys.path if p and (Path(p) / "NOTES.md").exists()), Path.cwd())


def sh(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=90)


def test_report_and_readme_have_required_sections():
    assert (WT / "REPORT.md").exists(), "REPORT.md missing"
    rep = (WT / "REPORT.md").read_text()
    for sec in ("question", "corpus", "method", "results", "judge", "failed", "gpu"):
        assert re.search(sec, rep, re.I), f"REPORT.md missing section: {sec}"
    assert (WT / "README.md").exists(), "README.md missing"
    readme = (WT / "README.md").read_text()
    for sec in ("mermaid", "quickstart", "skills?"):
        assert re.search(sec, readme, re.I), f"README.md missing: {sec}"


def test_checker_recomputes_numbers_and_exit0():
    p = sh([sys.executable, "tools/check_report.py"], WT)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]


def test_tools_ruff_clean():
    p = sh([sys.executable, "-m", "ruff", "check", "tools"], WT)
    assert p.returncode == 0, (p.stdout + p.stderr)[-800:]


def test_notes_record_report_state():
    notes = (WT / "NOTES.md").read_text()
    assert re.search(r"check_report", notes), "NOTES.md: checker not recorded"
    assert re.search(r"exit 0", notes), "NOTES.md: checker exit 0 not recorded"
    assert re.search(r"REPORT\.md", notes) and re.search(r"README\.md", notes), \
        "NOTES.md: REPORT.md/README.md not recorded"
    assert re.search(r"temp", notes, re.I), "NOTES.md: temp-filter fix not recorded"
    assert re.search(r"ruff", notes, re.I), "NOTES.md: ruff fix not recorded"
