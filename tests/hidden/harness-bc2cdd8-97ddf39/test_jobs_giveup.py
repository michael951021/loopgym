"""harness bc2cdd8..97ddf39 (lib/jobs.py): a job that has already failed JOB_TRIES times must stay
failed after a starting worker wipes run/NAME/failed.json. The fix re-asserts the mark in tick();
before it, the same line fell into `ready` and was retried forever, so a wiped failed.json was never
re-asserted. Discriminator: wipe failed.json, jobs.tick(), is the line's key back in failed.json?"""
import json
import os
import shutil
import sys
import subprocess
from pathlib import Path

ROOT = Path(next(p for p in sys.path if (Path(p) / "lib" / "jobs.py").exists()))
sys.path.insert(0, str(ROOT / "lib"))
import jobs  # noqa: E402
import sched  # noqa: E402

NAME, JID = "vb3proj", "jb"
LINE = "**Bench** — sweep (id: jb) (after: -) (run: `true`)"

def g(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout

def commit(p, f, text):
    (p / f).write_text(text)
    g(p, "add", "-A")
    g(p, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", f)

def test_giveup_survives_failed_reset():
    os.environ.update({"JOB_TRIES": "2", "JOB_CPUS": "0", "JOB_RESERVE_GB": "16"})
    proj, run = ROOT / "projects" / NAME, ROOT / "run" / NAME
    for d in (proj, run):
        shutil.rmtree(d, ignore_errors=True)   # pristine across the 3 re-runs (shared worktree)
    proj.mkdir(parents=True)
    g(proj, "init", "-q")
    commit(proj, "TODO.md", f"- [ ] {LINE}\n")
    key = sched.key(LINE)
    run.mkdir(parents=True)
    # a job that already failed its tries; failed.json wiped by a starting worker
    (run / "jobs.json").write_text(json.dumps(
        {JID: {"state": "failed", "tries": 2, "key": key, "result": "exit-code", "rc": "1", "log": "x.log"}}))
    (run / "failed.json").write_text(json.dumps({}))
    try:
        jobs.tick()   # good: gave_up re-asserts the mark; bad: line goes to `ready`, mark never restored
        assert key in json.loads((run / "failed.json").read_text()), \
            "twice-failed job's failed.json mark was not re-asserted after a reset"
    finally:
        for d in (proj, run):
            shutil.rmtree(d, ignore_errors=True)
