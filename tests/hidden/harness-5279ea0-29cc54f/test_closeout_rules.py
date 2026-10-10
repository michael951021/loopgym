"""harness 5279ea0..29cc54f: loop.md gains acceptance-check-first + known-answer rules;
the agent reports via a RESULT block and no longer self-checks the box or writes TASKLOG.md
(the harness close-out does), and a [fail] run re-asked answers "not done"."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["PYTHONPATH"])

def fab(tmp, task):
    d = tmp / "proj"
    d.mkdir()
    (d / "TODO.md").write_text("# Tasks\n" + task + "\n")
    for a in (["init"], ["config", "user.name", "t"], ["config", "user.email", "t@t"],
              ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", *a], cwd=d, check=True, capture_output=True)
    return d

def fake(d, prompt, *extra):
    return subprocess.run([sys.executable, str(ROOT / "tests" / "fake_claude.py"), *extra, "-p", prompt],
                          cwd=d, env={**os.environ, "AGENT_WORKER": "w1"},
                          capture_output=True, text=True, timeout=60)

def result_event(r):
    evs = [json.loads(ln) for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    return next(e for e in reversed(evs) if e.get("type") == "result")

def test_agent_reports_result_block_and_lets_harness_close_out(tmp_path):
    task = "- [ ] P: works [file p.txt] [sleep 0]"
    d = fab(tmp_path, task)
    ev = result_event(fake(d, "\n## Your task (TODO.md line 2)\n" + task))
    txt = ev.get("result", "")
    assert re.search(r"RESULT:\s*done", txt) and "APPROACH:" in txt and "BLOCKERS:" in txt
    assert "- [ ] P" in (d / "TODO.md").read_text()      # harness checks the box, not the agent
    assert not (d / "TASKLOG.md").exists()               # harness writes TASKLOG, not the agent
    assert (d / "p.txt").exists()

def test_fail_run_reasked_answers_not_done(tmp_path):
    task = "- [ ] Q: always fails [fail] [sleep 0]"
    d = fab(tmp_path, task)
    assert "result" not in result_event(fake(d, "\n## Your task (TODO.md line 2)\n" + task))
    r = fake(d, "Your run ended without the RESULT block; reply with only that block.", "--resume")
    assert r.returncode == 0
    ev = json.loads(r.stdout.strip().splitlines()[-1])
    assert re.search(r"RESULT:\s*not done", ev["result"]) and "WHY:" in ev["result"]

def test_prompt_says_acceptance_check_first():
    t = (ROOT / "prompts" / "loop.md").read_text()
    assert re.search(r"acceptance", t, re.I) and re.search(r"before\s+implement", t, re.I)

def test_prompt_says_numbers_need_known_answer():
    t = (ROOT / "prompts" / "loop.md").read_text()
    assert re.search(r"known-answer", t, re.I)
    assert re.search(r"5x", t) and re.search(r"plausible\s+range", t, re.I)
