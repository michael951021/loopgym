"""Hidden check, draftlab-7074368-07efa48: compare_sweep_alpha.py scores the
engine's measured T=0 accept/tpc vs the alpha metric's valid-position CIs on
the same text (exit 0 inside, 1 outside); NOTES.md holds the 393-prompt val
set, per-kind counts summing to 393. Synthetic fixtures; no model, no net."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def root() -> Path:
    for p in sys.path:
        d = Path(p or ".")
        if (d / "tools" / "compare_sweep_alpha.py").exists() and (d / "NOTES.md").exists():
            return d
    raise AssertionError("draftlab worktree root not on sys.path")


def drafters() -> list:
    src = (root() / "tools" / "compare_sweep_alpha.py").read_text()
    m = re.search(r"DRAFTERS\s*=\s*\[(.*?)\]", src)
    return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]


def build(tmp: Path, n_acc: int, n_emit: int) -> Path:
    (tmp / "tools").mkdir(parents=True)
    (tmp / "results").mkdir()
    t = root() / "tools" / "compare_sweep_alpha.py"
    (tmp / "tools" / "compare_sweep_alpha.py").write_bytes(t.read_bytes())
    rows = []
    for d in drafters():
        rows.append(dict(drafter=d, k=1, temp=0.0, kind="text", n_acc=n_acc,
                         n_draft=1000, n_emit=n_emit, n_steps=1000))
        rows.append(dict(drafter=d, k=8, temp=0.7, kind="text", n_acc=990,
                         n_draft=1000, n_emit=8000, n_steps=1000))
    pd.DataFrame(rows).to_parquet(tmp / "results" / "sweep.parquet")
    overall = dict(greedy=0.90, greedy_ci95=[0.85, 0.95], greedy_valid=0.55,
                   greedy_valid_ci95=[0.50, 0.60],
                   tokens_per_call_ci95={"1": [1.9, 2.1], "8": [7.5, 8.5]},
                   tpc_valid_ci95={"1": [1.05, 1.15], "8": [7.5, 8.5]})
    doc = {"overall": overall, "by_kind": {"text": dict(overall)}}
    for d in drafters():
        (tmp / "results" / f"alpha_sweep_{d}.json").write_text(json.dumps(doc))
    return tmp


def run_tool(tmp: Path) -> int:
    p = subprocess.run([sys.executable, str(tmp / "tools" / "compare_sweep_alpha.py")],
                       capture_output=True, text=True, cwd=tmp, timeout=90)
    return p.returncode


def test_measured_in_valid_ci_exits_0(tmp_path):
    assert run_tool(build(tmp_path, 550, 1100)) == 0


def test_measured_out_of_valid_ci_exits_1(tmp_path):
    assert run_tool(build(tmp_path, 900, 2000)) == 1


def test_notes_record_393_prompt_val_set():
    notes = (root() / "NOTES.md").read_text()
    counts = [int(m) for m in re.findall(r"(\d+)\s*val prompts", notes)]
    assert counts and all(c == 393 for c in counts)
    m = re.search(r"code (\d+) text (\d+) thinking (\d+) tool_result (\d+)"
                  r"\s+Bash (\d+) Edit (\d+) Read (\d+) WebFetch (\d+) Write (\d+)", notes)
    assert m and sum(map(int, m.groups())) == 393
