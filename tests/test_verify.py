"""Tests for loopgym/data/verify.py: verdict logic, end-to-end on a fake bare
repo (worktree build + 3x runs + record), and the timeout path. No model needed."""
import json
import os
import subprocess

import pytest

from loopgym.data import verify as V


def _info(reward, tests=1):
    return {"runner": "local", "timed_out": False, "returncode": 0, "tests": tests,
            "failures": 0, "errors": 0, "skipped": 0, "reward": reward}


def test_verdict():
    g3 = [_info(1.0)] * 3
    assert V._verdict([_info(0.0)] * 3, g3) == ("verified", 0, 3)
    assert V._verdict([_info(1.0)] * 3, [_info(0.0)] * 3) == ("inverted", 3, 0)
    assert V._verdict([_info(1.0), _info(0.0), _info(0.0)],
                      [_info(0.0)] * 3) == ("inverted", 1, 0)
    assert V._verdict([_info(0.0), _info(1.0), _info(0.0)], g3) == ("flaky", 1, 3)
    assert V._verdict([_info(0.0)] * 3, [_info(1.0), _info(1.0), _info(0.0)]) == ("flaky", 0, 2)
    # a run that produced no tests is an error, regardless of rewards
    assert V._verdict([_info(0.0, tests=0)] + [_info(0.0)] * 2, g3) == ("error", 0, 0)


def _g(d, *a):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(d), *a], check=True,
                          capture_output=True, text=True, env=env).stdout


@pytest.fixture()
def fake_bare(tmp_path):
    """A bare repo with a bad commit (f()==0) and a good commit (f()==1)."""
    src = tmp_path / "src"
    src.mkdir()
    _g(src, "init", "-q")
    (src / "mod.py").write_text("def f():\n    return 0\n")
    _g(src, "add", "mod.py")
    _g(src, "commit", "-qm", "base")
    bad = _g(src, "rev-parse", "HEAD").strip()
    (src / "mod.py").write_text("def f():\n    return 1\n")
    _g(src, "add", "mod.py")
    _g(src, "commit", "-qm", "fix: correct return")
    good = _g(src, "rev-parse", "HEAD").strip()
    bare = tmp_path / "fake.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(bare)],
                   check=True, capture_output=True)
    return bare, bad, good


def test_verify_end_to_end(fake_bare, tmp_path, monkeypatch):
    bare, bad, good = fake_bare
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_discrim.py").write_text(
        "from mod import f\n\n\ndef test_discrim():\n    assert f() == 1\n")
    monkeypatch.setattr(V, "REPOS", {"fake": bare})
    out = tmp_path / "verified.jsonl"
    monkeypatch.setattr(V, "OUT", out)
    verdict, rec = V.verify({"id": "fake-x", "repo": "fake",
                             "bad": bad, "good": good}, hidden, n_runs=2)
    assert verdict == "verified"
    assert rec["bad_passes"] == 0 and rec["good_passes"] == 2 and rec["n_runs"] == 2
    lines = out.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["verdict"] == "verified"
    assert rec["bad"] == bad[:12] and rec["good"] == good[:12]


def test_verify_inverted(fake_bare, tmp_path, monkeypatch):
    bare, bad, good = fake_bare
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_inv.py").write_text(
        "from mod import f\n\n\ndef test_inv():\n    assert f() == 0\n")
    monkeypatch.setattr(V, "REPOS", {"fake": bare})
    monkeypatch.setattr(V, "OUT", tmp_path / "verified.jsonl")
    verdict, rec = V.verify({"id": "fake-inv", "repo": "fake",
                             "bad": bad, "good": good}, hidden, n_runs=1)
    assert verdict == "inverted"
    assert rec["bad_passes"] == 1 and rec["good_passes"] == 0


def test_run_once_timeout(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "mod.py").write_text("def f():\n    return 1\n")
    sdir = tmp_path / "s"
    sdir.mkdir()
    slow = tmp_path / "hidden"
    slow.mkdir()
    (slow / "test_slow.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(6)\n")
    info = V._run_once(wt, list(slow.glob("test_*.py")), sdir, timeout=2)
    assert info["timed_out"] is True and info["reward"] == 0.0 and info["tests"] == 0
