"""Harness-5279ea0-429f2af: reports say what became of each task (done / not finished / interrupted / split / merge fix) and why unfinished ones did not finish."""
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(next(p for p in sys.path if (Path(p) / "ctxreport.py").exists()))
sys.path.insert(0, str(ROOT))
import ctxreport  # noqa: E402
import telemetry  # noqa: E402

schedpy = runpy.run_path(str(ROOT / "sched.py"))
result, task_state = getattr(telemetry, "result", None), getattr(telemetry, "task_state", None)

def g(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout

def test_result_says_what_became_of_the_task():
    lab, cls, why = result({}, {"task_state": "open", "merged": "n/a"})
    assert "not finished" in lab and cls == "bad" and why
    assert "done" in result({}, {"task_state": "done", "merged": "yes"})[0] and "interrupted" in result({}, {"task_state": "interrupted", "merged": "paused"})[0]
    assert "split" in result({}, {"task_state": "split", "merged": "n/a"})[0] and "merge fix" in result({"merge_fix": True}, {"task_state": "merge-fix", "merged": "yes"})[0]

def test_task_state_inferred_from_git():
    p = telemetry.ROOT / "projects" / "zz-check"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    g(p, "init", "-q")
    def commit(text):
        (p / "TODO.md").write_text(text)
        g(p, "add", "TODO.md")
        g(p, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
        return g(p, "rev-parse", "HEAD").strip()
    c_open, c_done, c_split = [commit(t) for t in ("- [ ] T: fix it\n", "- [x] T: fix it\n", "- [ ] T2: smaller\n")]
    st = {"project": "zz-check", "task": "T: fix it", "src": "TODO.md"}
    assert task_state(st, {"head": c_done, "merged": "yes"}) == "done" and task_state(st, {"head": c_open, "merged": "n/a"}) == "open"
    assert task_state(st, {"head": c_open, "merged": "paused"}) == "interrupted" and task_state(st, {"head": c_split, "merged": "n/a"}) == "split"
    assert task_state(st, {"head": c_open, "task_state": "done"}) == "done" and task_state(st, None) == "running"

def test_note_stored_and_given_to_next_attempt(tmp_path):
    note = "Blocked: needs the GPU.\nWould have helped: a GPU slot."
    r = subprocess.run([sys.executable, str(ROOT / "sched.py"), "note", str(tmp_path / "st"), "akey", note], capture_output=True)
    assert r.returncode == 0 and json.loads((tmp_path / "st" / "debriefs.json").read_text())["akey"] == note, r.stderr
    p = schedpy["prompt"](None, {"src": "TODO.md", "line": 1, "task": "T", "cl": "", "parent": "", "next": [], "debrief": note}, 2, False, [])
    assert "Blocked: needs the GPU." in p and "Would have helped: a GPU slot." in p

def test_index_shows_result_and_why():
    (telemetry.ROOT / "logs").mkdir(exist_ok=True)
    (telemetry.ROOT / "logs" / "none.jsonl").touch()
    (telemetry.ROOT / "reports" / "zz-check").mkdir(parents=True, exist_ok=True)
    rep = {"run": "r1", "title": "T: fix it", "project": "zz-check", "trigger": "end", "start": "2026-01-01", "wall_min": 5.0, "turns": 3, "peak_pct": 10.0,
           "think_pct": 5.0, "gen_tps": 10.0, "share_pct": None, "outcome": "success", "compactions": 0, "retries": 0,
           "result": {"label": "not finished", "cls": "bad", "why": "task left unchecked"}, "note": "Blocked: needs the GPU.", "attempt": {"n": 1}}
    (telemetry.ROOT / "reports" / "zz-check" / "20260101-0000-r1-end.json").write_text(json.dumps(rep))
    ctxreport.build_index()
    assert "not finished" in (h := (telemetry.ROOT / "reports" / "index.html").read_text()) and "Blocked: needs the GPU." in h and "task left unchecked" in h
