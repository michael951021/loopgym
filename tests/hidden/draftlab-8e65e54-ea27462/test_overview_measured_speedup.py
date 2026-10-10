"""Hidden check, draftlab-8e65e54-ea27462: the Overview measured-best-speedup KPI
must exclude the none baseline arm (the 1.0x reference): tile = max speedup_vs_none
over done non-none cells. Runs the KPI loop from Overview.tsx in node. No network."""
import json
import subprocess
import sys
from pathlib import Path

def root() -> Path:
    for p in sys.path:
        d = Path(p or ".")
        if (d / "web" / "src" / "pages" / "Overview.tsx").exists():
            return d
    raise AssertionError("draftlab worktree root not on sys.path")

def kpi_block() -> str:
    src = (root() / "web" / "src" / "pages" / "Overview.tsx").read_text()
    i = src.index("let measSp = 0;")
    j = src.index("{", src.index("for (", i))
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}" and depth:
            depth -= 1
            if not depth:
                return src[i:k + 1]
    raise AssertionError("KPI loop not balanced")

def kpi(cells: dict) -> list:
    js = "const sweep = %s;\nfunction kpi(sweep) {\n%s\nreturn [measSp, measLabel];}\nconsole.log(JSON.stringify(kpi(sweep)));" % (json.dumps(cells), kpi_block())
    p = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)

def expect(cells: dict, val: float, label: str):
    assert kpi({"cells": cells}) == [val, label]

def test_none_arm_is_reference_not_result():
    expect({"none k=0 T=0.0": {"status": "done", "speedup_vs_none": 1.0},
            "dpo k=8 T=0.0": {"status": "done", "speedup_vs_none": 0.83},
            "base k=1 T=0.7": {"status": "pending", "speedup_vs_none": None}}, 0.83, "dpo k=8 T=0.0")

def test_none_done_but_all_arms_pending():
    expect({"none k=0 T=0.0": {"status": "done", "speedup_vs_none": 1.0},
            "dpo k=8 T=0.0": {"status": "pending", "speedup_vs_none": None}}, 0, "–")

def test_max_over_done_arms_when_no_none():
    expect({"a k=1 T=0.0": {"status": "done", "speedup_vs_none": 1.5},
            "b k=8 T=0.7": {"status": "done", "speedup_vs_none": 0.9}}, 1.5, "a k=1 T=0.0")

def test_real_sweep_json_matches_non_none_max():
    cells = json.loads((root() / "web" / "public" / "data" / "sweep.json").read_text())["cells"]
    val, label = 0, "–"
    for n, c in cells.items():
        if n.startswith("none") or c.get("status") != "done" or c.get("speedup_vs_none") is None:
            continue
        if c["speedup_vs_none"] > val:
            val, label = c["speedup_vs_none"], n
    expect(cells, val, label)
