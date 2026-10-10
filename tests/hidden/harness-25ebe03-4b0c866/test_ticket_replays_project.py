"""Harness-25ebe03-4b0c866: ab ticket replays — a task may live in another project (task project/base),
so its trials start from that project's repo at the task's own base commit, and the task's scratch dirs
are private, writable dirs in the trial's worktree (not the shared ro mounts)."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(next(p for p in sys.path if (Path(p) / "lib" / "ab.py").exists()))
sys.path.insert(0, str(ROOT / "lib"))
import ab  # noqa: E402

def g(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout

def commit(p, f, text):
    (p / f).write_text(text)
    g(p, "add", "-A")
    g(p, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", f)

def mkexp():
    main, other = ROOT / "projects" / "zz-main", ROOT / "projects" / "zz-abfix"
    for p in (main, other):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)
        g(p, "init", "-q")
    commit(main, "main.txt", "m")
    commit(other, "fix.txt", "v1")
    c1 = g(other, "rev-parse", "HEAD").strip()
    commit(other, "fix.txt", "v2")
    (other / "ckpt").mkdir()
    (other / "ckpt" / "old.bin").write_bytes(b"0")
    (other / "data").mkdir()
    d = ROOT / "ab" / "zz-replay"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "spec.toml").write_text(
        'project = "zz-main"\nshared = ["ckpt", "data"]\n'
        '[[task]]\nid = "T"\nprompt = "fix"\nproject = "zz-abfix"\nbase = "' + c1 + '"\nscratch = ["ckpt"]\n'
        '[[variant]]\nname = "A"\n[[variant]]\nname = "B"\n')
    return ab.Exp("zz-replay"), other, c1, g(main, "rev-parse", "HEAD").strip()

def test_trial_starts_from_task_project_base():
    exp, other, c1, pinbase = mkexp()
    wt, _ = ab.setup_tree(exp, "T", "A", 1, {"base": pinbase})
    assert exp.pdir_of("T") == other and g(wt, "rev-parse", "HEAD^").strip() == c1 and (wt / "fix.txt").read_text() == "v1"

def test_scratch_private_writable_shared_excluded():
    exp, other, c1, pinbase = mkexp()
    wt = ROOT / "work" / "ab" / "zz-replay" / "scratch.wt"
    shutil.rmtree(wt, ignore_errors=True)
    b = ab.binds(exp, wt, "T")
    assert (wt / "ckpt").is_dir() and list((wt / "ckpt").iterdir()) == []
    (wt / "ckpt" / "new.bin").write_bytes(b"1")
    assert b[0][1] == str(other / ".git")
    ros = [e[1] for e in b if e[0] == "--ro-bind"]
    assert str(other / "data") in ros and str(other / "ckpt") not in ros
