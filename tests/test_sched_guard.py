"""sched_guard: the dispatcher's gate for human-only TODO.md lines."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / "tools" / "sched_guard.py"
TODO = ROOT / "TODO.md"

SYN = """\
# plan
## S
> why: x
- [ ] **a plain line** — (id: plain) body (after: -)
  why: x
- [ ] **human only** — (id: hum) body (human-only) (after: -)
  why: x
- [ ] **human gated** — (id: gated) body (human-only: results/rulings/r.json) (after: -)
  why: x
- [x] **done line** — (id: done) body (human-only) (after: -)
  why: x
"""


def run(args, cwd):
    return subprocess.run([sys.executable, str(GUARD), *args], cwd=cwd,
                          capture_output=True, text=True)


def test_dispatchable_exit0(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text(SYN)
    r = run(["--line", "plain"], tmp_path)
    assert r.returncode == 0, r.stderr
    r = run(["--line", "done"], tmp_path)  # checked line is dispatchable (closes)
    assert r.returncode == 0, r.stderr
    r = run(["--line", "gated"], tmp_path)  # release path missing -> skip
    assert r.returncode == 1, r.stdout + r.stderr
    r = run(["--line", "nope"], tmp_path)
    assert r.returncode == 2, r.stdout + r.stderr


def test_release_path_unblocks(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text(SYN)
    (tmp_path / "results" / "rulings").mkdir(parents=True)
    (tmp_path / "results" / "rulings" / "r.json").write_text("{}")
    r = run(["--line", "gated"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    r = run(["--line", "hum"], tmp_path)  # plain (human-only) stays skipped
    assert r.returncode == 1, r.stdout + r.stderr


def test_list(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text(SYN)
    r = run(["--list"], tmp_path)
    assert r.returncode == 1
    assert "hum" in r.stdout and "gated" in r.stdout
    (tmp_path / "results" / "rulings").mkdir(parents=True)
    (tmp_path / "results" / "rulings" / "r.json").write_text("{}")
    r = run(["--list"], tmp_path)  # hum still must be skipped -> exit 1
    assert r.returncode == 1 and "releasable" in r.stdout


def test_list_clean_exit0(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text("- [ ] **plain** — (id: plain) body (after: -)\n  why: x\n")
    assert run(["--list"], tmp_path).returncode == 0


def test_real_todo_kevinruling():
    # Known state of the real plan: kevinruling is open, human-only, and the
    # ruling file is absent (until Kevin commits it), so the gate must skip it.
    r = run(["--line", "kevinruling"], ROOT)
    ruling = ROOT / "results" / "rulings" / "base05_val.ruling.json"
    if ruling.exists():
        assert r.returncode == 0, r.stdout + r.stderr
    else:
        assert r.returncode == 1, r.stdout + r.stderr
    assert run(["--line", "grpocfg"], ROOT).returncode == 0  # ordinary line
