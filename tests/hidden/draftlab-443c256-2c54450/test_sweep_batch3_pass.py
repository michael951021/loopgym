"""Hidden check draftlab-443c256-2c54450: sweep batch3 grid complete and its
acceptance check PASSes. Behaviour the fix restored:
(a) results/sweep.parquet holds all 42 cells (21 T=0.7 cells x 393 prompts
    added to the 21 T=0 cells of batches 1/2);
(b) tools/check_sweep_batch3.py runs to completion, exits 0 and prints
    "BATCH 3 CHECK: PASS" (grid complete + model-drafter gates satisfied);
(c) that run (re)writes results/sweep_batch3_table.json covering all 42 cells.
CPU, seconds."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

WT = next((Path(p) for p in sys.path
           if p and (Path(p) / "tools" / "check_sweep_batch3.py").exists()),
          Path.cwd())
DRAFTERS = ["ngram", "tiny", "base", "lora", "lora_dpo"]
KS = [1, 2, 4, 8]
TABLE = WT / "results" / "sweep_batch3_table.json"


def test_parquet_grid_complete_42_cells():
    df = pd.read_parquet(WT / "results" / "sweep.parquet")
    assert df.groupby(["drafter", "k", "temp"]).ngroups == 42
    for d, ks in [(d, KS) for d in DRAFTERS] + [("none", [0])]:
        for k in ks:
            s = df[(df["drafter"] == d) & (df["k"] == k) & (df["temp"] == 0.7)]
            assert len(s) == 393, f"{d} k={k} T=0.7: {len(s)}/393 prompts"


def test_checker_passes_exit0():
    if TABLE.exists():
        TABLE.unlink()
    p = subprocess.run([sys.executable, "tools/check_sweep_batch3.py"], cwd=WT,
                       capture_output=True, text=True, timeout=100)
    assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    assert "BATCH 3 CHECK: PASS" in p.stdout, (p.stdout + p.stderr)[-800:]
    assert "FAIL" not in p.stdout and "MISSED" not in p.stdout


def test_table_written_all_42_cells():
    assert TABLE.exists(), "checker run did not write sweep_batch3_table.json"
    cells = json.loads(TABLE.read_text())["cells"]
    n = sum(len(v) for v in cells.values())
    assert n == 42, f"table covers {n} cells, expected 42"
    for key, tv in cells.items():
        assert set(tv) == {"T=0.0", "T=0.7"}, f"{key}: {sorted(tv)}"
