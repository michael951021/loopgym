"""Exercism tasks and the local grader.

Reward = 1.0 iff the pristine hidden tests (tests > 0, no failures/errors/skips in the
pytest junit XML) pass when run in a fresh directory holding only the model's solution
file(s). Timeouts and missing XML score 0.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SYSTEM = ("You are an expert Python programmer. Reply with the complete solution file "
          "in one ```python code block and nothing else.")


@dataclass
class Task:
    name: str
    dir: Path
    solution_file: str
    test_file: str
    test_files: list = field(default=None)  # paasio has two; default [test_file]

    def __post_init__(self):
        self.dir = Path(self.dir)
        if self.test_files is None:
            self.test_files = [self.test_file]

    def stub(self):
        return {self.solution_file: (self.dir / self.solution_file).read_text()}

    def reference(self):
        return {self.solution_file: (self.dir / ".meta" / "example.py").read_text()}

    def prompt(self):
        docs = (self.dir / ".docs" / "instructions.md").read_text()
        extra = self.dir / ".docs" / "instructions.append.md"
        if extra.exists():
            docs = docs + "\n\n" + extra.read_text()
        stub = self.stub()[self.solution_file]
        if not stub.endswith("\n"):
            stub += "\n"
        user = (docs + f"\n\nComplete this file (`{self.solution_file}`):\n"
                f"```python\n{stub}```")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def parse(self, reply):
        return {self.solution_file: _last_python_block(reply, reply)}


def _last_python_block(reply, fallback):
    lines = reply.splitlines()
    opens = [i for i, line in enumerate(lines) if line.strip().startswith("```python")]
    if not opens:
        return fallback
    start = opens[-1] + 1
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("```"):
            return "\n".join(lines[start:j]) + "\n"
    return "\n".join(lines[start:])  # unterminated fence: take the rest


def load_exercise(exdir):
    exdir = Path(exdir)
    cfg = json.loads((exdir / ".meta" / "config.json").read_text())
    files = cfg.get("files", {})
    solution = files["solution"][0]
    tests = files.get("test") or [solution[:-3] + "_test.py"]
    return Task(name=exdir.name, dir=exdir, solution_file=solution,
                test_file=tests[0], test_files=list(tests))


def _xml_counts(xml_text):
    """(tests, failures, errors, skipped) summed over all testsuites in a junit XML."""
    root = ET.fromstring(xml_text)
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for s in suites:
        for k in counts:
            counts[k] += int(s.get(k) or 0)
    return counts


def _reward(info):
    """reward 1.0 iff ran without timeout, tests > 0, no failures/errors/skips."""
    ok = (not info["timed_out"] and info["tests"] > 0 and info["failures"] == 0
          and info["errors"] == 0 and info["skipped"] == 0)
    info["reward"] = 1.0 if ok else 0.0
    return info["reward"], info


def _local(task, files, timeout=30):
    info = {"runner": "local", "timed_out": False, "returncode": None,
            "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    with tempfile.TemporaryDirectory(prefix="loopgym-grade-") as td:
        d = Path(td)
        for name in task.test_files:
            shutil.copy2(task.dir / name, d / name)
        sol = files.get(task.solution_file)
        if sol is not None:
            (d / task.solution_file).write_text(sol)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
        proc = subprocess.Popen(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--junitxml=r.xml"],
            cwd=d, env=env, start_new_session=True,
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
        xml_path = d / "r.xml"
        if xml_path.exists():
            info.update(_xml_counts(xml_path.read_text()))
    return _reward(info)


RUNNERS = {"local": _local}


def runner_names():
    """All usable runners: "http" (lgbox.py) only when LOOPGYM_LGBOX_URL is set."""
    names = set(RUNNERS)
    if not os.environ.get("LOOPGYM_LGBOX_URL"):
        names.discard("http")
    return sorted(names)


def grade(task, files, runner="local", timeout=30):
    if runner not in RUNNERS:
        raise ValueError(f"unknown runner {runner!r}; have {runner_names()}")
    return RUNNERS[runner](task, files, timeout=timeout)
