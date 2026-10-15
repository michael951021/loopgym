"""Eval CLI: reference mode is a known answer (pass@1 1.0 on usable exercises)
and the sidecar makes reruns resume (no duplicate samples)."""
import json
from pathlib import Path

import pytest

from loopgym.eval import run as evalrun

# Data assets are gitignored host files; skip the eval runs that read them.
ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "data" / "raw" / "exercism-python"
LG_DATA = ROOT / "data" / "loopgym"


def _run(out, extra=None):
    argv = ["--reference", "--source", "exercism", "--split", "val",
            "--limit", "2", "--out", str(out)] + (extra or [])
    assert evalrun.main(argv) == 0
    return json.loads(out.read_text()), out.with_name(out.stem + ".samples.jsonl")


@pytest.mark.skipif(not EXERCISES.is_dir(), reason="no data/raw exercises (fresh clone)")
def test_reference_limit2_pass_at_1(tmp_path):
    out = tmp_path / "ref.json"
    doc, side = _run(out)
    assert doc["pass_at_k"] == 1.0
    assert doc["ci"] == [1.0, 1.0]
    assert doc["n_tasks"] == 2
    recs = [json.loads(line) for line in side.read_text().splitlines() if line.strip()]
    assert len(recs) == 8  # 2 tasks x n=4
    assert {r["task"] for r in recs} == {"flatten-array", "food-chain"}
    assert all(r["reward"] == 1.0 for r in recs)


@pytest.mark.skipif(not EXERCISES.is_dir(), reason="no data/raw exercises (fresh clone)")
def test_reference_resume_appends_nothing(tmp_path):
    out = tmp_path / "ref.json"
    _run(out)
    _, side = _run(out)
    lines = side.read_text().splitlines()
    assert len(lines) == 8  # second run skipped both completed tasks


@pytest.mark.skipif(not LG_DATA.is_dir(), reason="no data/loopgym (fresh clone)")
def test_loopgym_reference_limit2_n1_pass_at_1(tmp_path):
    """v0 known answer: the good commit's fix (unified diff) grades 1.0 on real
    tasks, and the bad commit (no patch) grades 0.0."""
    out = tmp_path / "ref_lg.json"
    argv = ["--reference", "--source", "loopgym", "--split", "test",
            "--limit", "2", "--n", "1", "--out", str(out)]
    assert evalrun.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["n_tasks"] == 2
    assert doc["pass_at_k"] == 1.0 and doc["ci"] == [1.0, 1.0]
    side = out.with_name(out.stem + ".samples.jsonl")
    rewards = [json.loads(line)["reward"] for line in
               side.read_text().splitlines() if line.strip()]
    assert rewards == [1.0, 1.0]
    tasks = evalrun.load_tasks("loopgym", "test", 2)
    assert [t.name for t in tasks] == ["draftlab-3c28deb-5bf9b0d",
                                       "draftlab-7074368-07efa48"]
    for t in tasks:  # bad commit: an empty reply applies nothing -> check fails
        reward, info = evalrun.grade(t, t.parse(""), "local", timeout=120)
        assert reward == 0.0, (t.name, info)
    # a diff that cannot apply scores 0 without running; a diff outside the
    # task's files scores 0 without running (anti-hack)
    t, p = tasks[0], tasks[0].files[0]
    bad_patch = (f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n"
                 f"@@ -1,1 +1,1 @@\n-AAAAAAAAAA\n+BBBBBBBBBB\n")
    reward, info = evalrun.grade(t, bad_patch, "local", timeout=120)
    assert reward == 0.0 and "does not apply" in info.get("reason", ""), info
    reward, info = evalrun.grade(t, bad_patch.replace(p, "tests/hidden/x/test_e.py"),
                                 "local", timeout=120)
    assert reward == 0.0 and "outside the task" in info.get("reason", ""), info
