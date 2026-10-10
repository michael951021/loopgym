"""Hidden check draftlab-f0371d7-2c54450: sweep batch 3 state is real and
recorded in NOTES.md. Behaviour the fix restored:
(a) results/sweep.parquet holds the full 42-cell grid (21 T=0 + 21 T=0.7,
    393 prompts in every cell) -- the ground truth the notes' claim refers to;
(b) tools/check_sweep_batch3.py exits 0 printing "BATCH 3 CHECK: PASS"
    (recalibrated model-drafter gates hold);
(c) NOTES.md's Ray-sweep section records that state -- all 3 batches done,
    42 cells, T=0.7, gate recalibration, checker exit 0 -- consistent with
    (a), not a bare claim. CPU, seconds."""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

WT = next((Path(p) for p in sys.path if p and (Path(p) / "NOTES.md").exists()), Path.cwd())


def test_parquet_grid_42_cells_393_each():
    df = pd.read_parquet(WT / "results" / "sweep.parquet")
    g = df.groupby(["drafter", "k", "temp"])
    assert g.ngroups == 42, f"grid has {g.ngroups} cells, expected 42"
    sizes = g.size()
    assert (sizes == 393).all(), f"incomplete cells: {sizes[sizes != 393].to_dict()}"


def test_batch3_checker_pass():
    p = subprocess.run([sys.executable, "tools/check_sweep_batch3.py"], cwd=WT,
                       capture_output=True, text=True, timeout=100)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    assert "BATCH 3 CHECK: PASS" in p.stdout, (p.stdout + p.stderr)[-800:]


def test_notes_sweep_section_records_batch3_state():
    notes = (WT / "NOTES.md").read_text()
    m = re.search(r"## Ray sweep.*?(?=\n## |\Z)", notes, re.S)
    assert m, "NOTES.md: no Ray sweep section"
    sec = m.group(0)
    pats = [(r"all\s*3\s*batches\s*done.*?42\s*cells", "all-batches-done / 42 cells"),
            (r"T=0\.7", "T=0.7 batch"), (r"recalibrat", "gate recalibration"),
            (r"check_sweep_batch", "batch checker name"), (r"exit 0", "checker exit 0")]
    for pat, why in pats:
        assert re.search(pat, sec, re.I), f"NOTES.md sweep section: {why} not recorded"
