"""Hidden check draftlab-35a5d46-2c54450: the Ray-sweep ticket's end state.
At the bad commit only the ticket and sweep.py exist; the fix restored the
measured result: (a) results/sweep.parquet holds the full 42-cell grid
(5 drafters x k 1/2/4/8 + none, T in {0, 0.7}, 393 prompts per cell);
(b) tools/check_sweep_batch3.py (completeness + recalibrated model-drafter
gates) exits 0 with "BATCH 3 CHECK: PASS" and writes the 42-cell table;
(c) NOTES.md's Ray-sweep section records that completed state (claim backed
by (a)/(b), not a bare number). CPU, seconds, no models."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

WT = next((Path(p) for p in sys.path if p and (Path(p) / "NOTES.md").exists()), Path.cwd())
DRAFTERS = ["ngram", "tiny", "base", "lora", "lora_dpo"]
KS = [1, 2, 4, 8]
TABLE = WT / "results" / "sweep_batch3_table.json"


def test_parquet_grid_42_cells_393_each():
    p = WT / "results" / "sweep.parquet"
    assert p.exists(), f"{p} missing: sweep grid never produced"
    df = pd.read_parquet(p)
    ng = df.groupby(["drafter", "k", "temp"])
    assert ng.ngroups == 42, f"grid has {ng.ngroups} cells, expected 42"
    g = ng.size()
    assert (g == 393).all(), f"incomplete cells: {g[g != 393].to_dict()}"


def test_batch3_checker_pass_and_table():
    p = subprocess.run([sys.executable, "tools/check_sweep_batch3.py"], cwd=WT,
                       capture_output=True, text=True, timeout=100)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    assert "BATCH 3 CHECK: PASS" in p.stdout, (p.stdout + p.stderr)[-800:]
    assert "MISSED" not in p.stdout, "a pre-registered range was missed"
    cells = json.loads(TABLE.read_text())["cells"]
    assert sum(len(v) for v in cells.values()) == 42, "table must cover all 42 cells"


def test_notes_record_batch3_completion():
    notes = (WT / "NOTES.md").read_text()
    m = re.search(r"## Ray sweep.*?(?=\n## |\Z)", notes, re.S)
    assert m, "NOTES.md: no Ray sweep section"
    sec = m.group(0)
    for pat, why in [(r"all\s*3\s*batches\s*done", "completion claim"),
                     (r"42\s*cells", "grid size"),
                     (r"recalibrat", "gate recalibration"),
                     (r"exit 0", "checker result")]:
        assert re.search(pat, sec, re.I), f"NOTES.md sweep section: {why} not recorded"
