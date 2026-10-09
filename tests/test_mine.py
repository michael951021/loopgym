"""Tests for loopgym/data/mine.py: keyword rule, hunk parsing, e2e tmp repo, seed recall."""
import json
import os
import subprocess

import pytest

from loopgym.data import mine


def test_keyword_rule():
    for m in ("fix", "fixed", "fixes", "correct", "corrected", "wrong",
              "bug", "bugs", "buggy", "audit", "audited", "FIXED the bug",
              "Audit of the deploy config"):
        assert mine.KW.search(m), m
    for nm in ("fixedup", "refix", "debug", "fixup", "workaround"):
        assert not mine.KW.search(nm), nm


def _g(d, *a):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(d), *a], check=True,
                          capture_output=True, text=True, env=env).stdout


@pytest.fixture()
def tmp_repo(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    _g(d, "init", "-q")
    (d / "f.py").write_text("A1\nA2\n")
    _g(d, "add", "f.py")
    _g(d, "commit", "-qm", "base: add module")
    (d / "TODO.md").write_text("- [ ] **Feature X** — does a thing (id: featx)\n")
    (d / "f.py").write_text("A1\nA2\nBUG1\nBUG2\n")
    _g(d, "add", ".")
    _g(d, "commit", "-qm", "featx: **Feature X** — does a thing")
    (d / "f.py").write_text("A1\nA2\n")
    _g(d, "add", ".")
    _g(d, "commit", "-qm", "fix: bug in Feature X")
    return d


def test_mine_repo_end_to_end(tmp_repo):
    recs, stats, hits = mine.mine_repo(tmp_repo, "tmp")
    assert stats["commits"] == 3 and stats["kw_fixes"] == 1
    assert stats["pairs"] == 1 and stats["kept"] == 1
    (cid, rec) = next(iter(recs.items()))
    bad = _g(tmp_repo, "rev-parse", "HEAD^").strip()
    fix = _g(tmp_repo, "rev-parse", "HEAD").strip()
    assert rec["bad"] == bad and rec["good"] == fix
    assert rec["lines"] == 2 and rec["files"] == ["f.py"] and rec["seed"] is False
    assert rec["fix_msg"] == "fix: bug in Feature X"
    assert rec["ticket"] == "- [ ] **Feature X** — does a thing (id: featx)"
    assert bad in hits and fix[:8] in [c for c, _ in hits[bad]]
    assert recs[cid] is rec


def test_deleted_ranges_hunk_forms(tmp_repo):
    # multi-line deletion: "@@ -3,2 +2,0 @@"
    fix = _g(tmp_repo, "rev-parse", "HEAD").strip()
    assert mine.deleted_ranges(tmp_repo, fix) == [("f.py", 3, 2)]
    # single-line modification: git omits the ",1" count ("@@ -1 +1 @@")
    (tmp_repo / "f.py").write_text("A1x\nA2\n")
    _g(tmp_repo, "add", ".")
    _g(tmp_repo, "commit", "-qm", "refine: tweak first line")
    top = _g(tmp_repo, "rev-parse", "HEAD").strip()
    assert mine.deleted_ranges(tmp_repo, top) == [("f.py", 1, 1)]


def test_ticket_message_fallback(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    _g(d, "init", "-q")
    (d / "g.py").write_text("X\n")
    _g(d, "add", ".")
    _g(d, "commit", "-qm", "base: start")
    (d / "g.py").write_text("X\nBUG\n")
    _g(d, "add", ".")
    _g(d, "commit", "-qm", "featy: add thing\n\nTODO: do a thing")
    (d / "g.py").write_text("X\n")
    _g(d, "add", ".")
    _g(d, "commit", "-qm", "fix: the thing")
    recs, stats, _ = mine.mine_repo(d, "tmp")
    assert stats["kept"] == 1
    assert next(iter(recs.values()))["ticket"] == "TODO: do a thing"


def test_seed_recall_real_repos():
    """The 3 human-confirmed seed defects are all paired by the automatic rules."""
    hits = {}
    for name, repo in mine.REPOS.items():
        _, _, h = mine.mine_repo(repo, name)
        hits[name] = h
    for s in json.loads(mine.SEEDS.read_text()):
        bad = mine.resolve(mine.REPOS[s["project"]], s["bad"])
        assert hits[s["project"]].get(bad), s["id"]
