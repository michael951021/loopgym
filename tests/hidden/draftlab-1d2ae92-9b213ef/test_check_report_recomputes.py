"""Hidden check, draftlab-1d2ae92-9b213ef: the worktree's tools/check_report.py
must exit 0 on the corrected REPORT (numbers recomputed from results/, run-level
CIs, Corrections section) and reject a tampered one. data/corpus is gitignored,
so a synthetic corpus parquet is supplied for the no-quote check. CPU, ~5 s.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl


def build_root(tmp: Path) -> Path:
    wt = next((Path(p) for p in sys.path if p and (Path(p) / "tools" / "check_report.py").exists()), None)
    assert wt, "worktree with tools/check_report.py not on sys.path"
    root = tmp / "root"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(wt / "tools" / "check_report.py", root / "tools" / "check_report.py")
    for f in ("REPORT.md", "README.md"):
        if (wt / f).exists():
            shutil.copy2(wt / f, root / f)
    shutil.copytree(wt / "results", root / "results")
    cor = root / "data" / "corpus"
    cor.mkdir(parents=True)
    for s in ("heldout", "val", "test"):
        pl.DataFrame({"text": [f"synthetic corpus line {s} {i} zqjxwv padding"
                               for i in range(120)]}).write_parquet(cor / f"{s}.parquet")
    return root


def run_checker(root: Path):
    return subprocess.run([sys.executable, str(root / "tools" / "check_report.py")],
                          capture_output=True, text=True, cwd=root, timeout=100)


def test_checker_passes_corrected_report(tmp_path):
    p = run_checker(build_root(tmp_path))
    assert p.returncode == 0, p.stdout[-1500:]


def test_checker_catches_stale_number(tmp_path):
    root = build_root(tmp_path)
    t0 = (root / "REPORT.md").read_text()
    o = json.loads((root / "results" / "alpha" / "ngram.json").read_text())["overall"]
    assert f"{o['alpha']:.4f}" in t0, "headline alpha not in REPORT.md"
    (root / "REPORT.md").write_text(t0.replace(f"{o['alpha']:.4f}", "0.0000"))
    p = run_checker(root)
    assert p.returncode != 0, "checker accepted a stale headline alpha"


def test_checker_catches_missing_corrections(tmp_path):
    root = build_root(tmp_path)
    t0 = (root / "REPORT.md").read_text()
    m = re.search(r"^#+ .*[Cc]orrection.*\n", t0, re.M)
    assert m, "no Corrections section heading"
    (root / "REPORT.md").write_text(t0[: m.start()] + t0[m.end():])
    assert run_checker(root).returncode != 0, "checker accepted missing Corrections"
