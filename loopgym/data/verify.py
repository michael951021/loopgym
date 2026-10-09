"""Verification gate for mined loopgym candidates.

A candidate is a (bad, good) commit pair in a bare repo. The hidden check
(tests/hidden/<id>/test_*.py) must fail at bad and pass at good. verify() builds
pristine worktrees of both commits, copies the hidden check into a fresh dir, and
runs it 3x per side under the local-runner rules (junit XML, 120 s timeout, pgroup
SIGKILL). Verdict:
  verified  bad 0/3 pass, good 3/3 pass
  inverted  check passes more on bad than on good (wrong direction)
  flaky     mixed / non-deterministic results
  error     a run produced no tests (collection/import/setup failure)
The record is appended to data/loopgym/verified.jsonl.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from loopgym.env.exercism import _xml_counts

ROOT = Path(__file__).resolve().parents[2]
REPOS = {n: ROOT / "data" / "raw" / "repos" / f"{n}.git" for n in ("draftlab", "harness")}
HIDDEN = ROOT / "tests" / "hidden"
CANDIDATES = ROOT / "data" / "loopgym" / "candidates.jsonl"
OUT = ROOT / "data" / "loopgym" / "verified.jsonl"
N_RUNS, TIMEOUT = 3, 120


def git(repo, *args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                          cwd=repo).stdout


def check_files(check_dir):
    return sorted(Path(check_dir).glob("test_*.py"))


def _add_worktree(repo, commit, path):
    git(repo, "worktree", "add", "--detach", str(path), commit)


def _remove_worktree(repo, path):
    try:
        git(repo, "worktree", "remove", "--force", str(path))
    except subprocess.CalledProcessError:
        pass
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _run_once(worktree, files, scratch, timeout):
    """One pristine run: fresh dir holds only the check; PYTHONPATH -> worktree."""
    info = {"runner": "local", "timed_out": False, "returncode": None,
            "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "reward": 0.0}
    s = Path(scratch)
    for f in files:
        shutil.copy2(f, s / f.name)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    env.update({"PYTHONPATH": str(worktree), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8")})
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml=r.xml"],
        cwd=s, env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        info["returncode"] = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        info["timed_out"] = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        info["returncode"] = proc.wait()
    xml = s / "r.xml"
    if xml.exists():
        info.update(_xml_counts(xml.read_text()))
    info["reward"] = 1.0 if (not info["timed_out"] and info["tests"] > 0
                             and info["failures"] == 0 and info["errors"] == 0
                             and info["skipped"] == 0) else 0.0
    return info


def _verdict(bad, good):
    """(verdict, bad_passes, good_passes) from per-run info lists."""
    if any(r["tests"] == 0 for r in bad + good):
        return "error", 0, 0
    b = sum(1 for r in bad if r["reward"] == 1.0)
    g = sum(1 for r in good if r["reward"] == 1.0)
    if b == 0 and g == len(good):
        return "verified", b, g
    if b > g:
        return "inverted", b, g
    return "flaky", b, g


def verify(candidate, check_dir, n_runs=N_RUNS, timeout=TIMEOUT):
    """Run the hidden check n_runs times at bad and good; return (verdict, record)."""
    files = check_files(check_dir)
    if not files:
        raise FileNotFoundError(f"no test_*.py in {check_dir}")
    repo = REPOS[candidate["repo"]]
    verdict, b, g = "error", 0, 0
    bad_runs, good_runs = [], []
    with tempfile.TemporaryDirectory(prefix="loopgym-verify-") as base:
        base = Path(base)
        wt_bad, wt_good = base / "bad", base / "good"
        try:
            git(repo, "worktree", "prune")
            _add_worktree(repo, candidate["bad"], wt_bad)
            _add_worktree(repo, candidate["good"], wt_good)
            for _ in range(n_runs):
                with tempfile.TemporaryDirectory(dir=base, prefix="bad-") as sd:
                    bad_runs.append(_run_once(wt_bad, files, sd, timeout))
            for _ in range(n_runs):
                with tempfile.TemporaryDirectory(dir=base, prefix="good-") as sd:
                    good_runs.append(_run_once(wt_good, files, sd, timeout))
            verdict, b, g = _verdict(bad_runs, good_runs)
        finally:
            _remove_worktree(repo, wt_bad)
            _remove_worktree(repo, wt_good)
    record = {
        "id": candidate["id"], "repo": candidate["repo"],
        "bad": candidate["bad"][:12], "good": candidate["good"][:12],
        "verdict": verdict, "bad_passes": b, "good_passes": g, "n_runs": n_runs,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return verdict, record


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m loopgym.data.verify <candidate-id>", file=sys.stderr)
        return 2
    cid = argv[0]
    lines = CANDIDATES.read_text().splitlines()
    rec = next((json.loads(line) for line in lines
                if line and json.loads(line)["id"] == cid), None)
    if rec is None:
        print(f"candidate {cid!r} not in candidates.jsonl", file=sys.stderr)
        return 2
    verdict, _ = verify(rec, HIDDEN / cid)
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
