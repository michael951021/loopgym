"""Eval CLI: reference mode is a known answer (pass@1 1.0 on usable exercises)
and the sidecar makes reruns resume (no duplicate samples)."""
import json

from loopgym.eval import run as evalrun


def _run(out, extra=None):
    argv = ["--reference", "--source", "exercism", "--split", "val",
            "--limit", "2", "--out", str(out)] + (extra or [])
    assert evalrun.main(argv) == 0
    return json.loads(out.read_text()), out.with_name(out.stem + ".samples.jsonl")


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


def test_reference_resume_appends_nothing(tmp_path):
    out = tmp_path / "ref.json"
    _run(out)
    _, side = _run(out)
    lines = side.read_text().splitlines()
    assert len(lines) == 8  # second run skipped both completed tasks
