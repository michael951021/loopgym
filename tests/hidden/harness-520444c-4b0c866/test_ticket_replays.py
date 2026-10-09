"""Hidden check, harness-520444c-4b0c866 (ticket replays): a task may live in another project
(project/base/scratch); reviews report server-side tok/s + queue time (_speeds). Fails at bad."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(next(p for p in sys.path if (Path(p) / "lib" / "ab.py").exists()))
sys.path.insert(0, str(ROOT / "lib"))
import ab  # noqa: E402
import abreview  # noqa: E402

def g(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout

def repo(d, n, tag=None):
    d.mkdir(parents=True)
    nums = " ".join(map(str, range(n)))
    subprocess.run(f"cd {d} && git init -q && for i in {nums}; do echo $i > f.txt && git add . && "
                   f"git -c user.name=t -c user.email=t@t commit -qm c$i; done",
                   shell=True, check=True, capture_output=True)
    if tag:
        g(d, "tag", tag)

def make(tmp_path):
    root = tmp_path / "r"
    repo(root / "projects" / "alpha", 1)
    beta = root / "projects" / "beta"
    repo(beta, 2, tag="beta-tip")
    (beta / "libdata").mkdir()
    (beta / "ckpt").mkdir()
    (beta / ".agent-shared").write_text("libdata\nckpt\n")
    (root / "ab" / "demo").mkdir(parents=True)
    (root / "ab" / "demo" / "spec.toml").write_text(
        'project = "alpha"\n[[task]]\nid = "ta"\nprompt = "p"\n[[task]]\nid = "tb"\nprompt = "p"\n'
        'project = "beta"\nbase = "beta-tip"\nscratch = ["ckpt"]\n[[variant]]\nname = "A"\n'
        '[[variant]]\nname = "B"\n')
    ab.ROOT, ab.AB = root, root / "ab"
    return ab.Exp("demo"), root, beta

def test_task_project_base(tmp_path):
    exp, root, beta = make(tmp_path)
    tip = g(beta, "rev-parse", "beta-tip").strip()
    assert exp.pdir_of("tb") == beta and exp.pdir_of("ta") == root / "projects" / "alpha"
    assert exp.base_of("tb", {"base": "pin"}) == tip   # own base, resolved in its own project
    assert exp.base_of("ta", {"base": "pin"}) == "pin"  # no base -> the pinned one

def test_scratch_writable_not_shared(tmp_path):
    exp, _, beta = make(tmp_path)
    wt = tmp_path / "wt"
    wt.mkdir()
    b = ab.binds(exp, wt, "tb")
    assert (wt / "ckpt").is_dir()  # the trial gets its own writable scratch dir
    assert ["--bind", str(beta / ".git"), str(beta / ".git")] in b
    assert ["--ro-bind", str(beta / "libdata"), "/work/libdata"] in b
    assert not any("ckpt" in " ".join(x) for x in b)  # scratch left out of the shared binds

def test_speeds_known_answer():
    assert abreview._speeds([{"prompt_n": 1000, "prompt_ms": 200, "predicted_n": 500, "predicted_ms": 5000}], 7.0) \
        == {"prompt_tok_s": 5000.0, "gen_tok_s": 100.0, "compute_s": 5.2, "queue_s": 1.8}
    assert abreview._speeds([], 3.0) == {}  # no timings -> nothing to report
