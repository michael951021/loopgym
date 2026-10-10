"""loopgym tasks and grader: a verified (bad, good) defect pair becomes a Task.

prompt() = the ticket text + each changed file at the bad commit (elided to the changed
lines, <= 6k tokens) + the instruction to reply with the full new content of each file in
```path blocks. parse() reads those blocks back into {path: content}. grade() applies the
reply's files to a worktree of the bad commit and runs the task's hidden check
(tests/hidden/<id>/) pristine, under the local-runner rules. A reply that touches any file
outside the task's changed set scores 0 (it could be editing the hidden test).

Representability: a record becomes a Task only if every file in the bad->good diff is
modified (M) or added (A) and is utf-8 text where it exists. Deletions, renames, and
binary files cannot be expressed as "full new content of a file", so load_tasks() skips
those records. Applying the good-commit content of the full M+A set to the bad tree yields
exactly the good tree, so the good side passes the check by construction.
"""
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loopgym.data import verify as V

BUDGET_TOKENS = 6000
CHARS_PER_TOKEN = 3
PER_FILE_CAP = 4500
CONTEXT = 3

SYSTEM = "You are an expert software engineer. Find and fix the defect described below."
INSTRUCTION = ("Find and fix the defect. Reply with the full new content of each file "
               "you change in ```path\\n...``` blocks.")


def _git(repo, *args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                          cwd=repo).stdout


def _show(repo, ref, path):
    p = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                       capture_output=True)
    return p.stdout if p.returncode == 0 else None


def _is_text(b):
    try:
        b.decode("utf-8")
        return True
    except (UnicodeDecodeError, AttributeError):
        return False


@dataclass
class Task:
    name: str          # task id (== the tests/hidden/<name> dir name)
    repo: str          # "draftlab" | "harness"
    bad: str           # bad commit (12-hex, resolvable)
    good: str          # good commit (12-hex)
    ticket: str        # ticket text that produced the bad commit (may be "")
    files: list        # sorted changed paths (M+A): the only files a reply may touch
    added: set         # A paths (absent at the bad commit)
    content_bad: dict  # {path: str} M files, at the bad commit
    content_good: dict  # {path: str} all files, at the good commit
    priority: list     # changed-file order for the prompt (candidate files first)

    def stub(self):
        """The bad-commit content of the files that exist at the bad commit (M files)."""
        return dict(self.content_bad)

    def reference(self):
        """The good-commit content of every changed file (M+A)."""
        return dict(self.content_good)

    def prompt(self):
        ticket = self.ticket.strip()
        head = (ticket + "\n\n" if ticket else "")
        head += "Changed files (content at this commit; unchanged stretches elided):\n"
        blocks = []
        for p in self.priority:
            body = "[new file: does not exist at this commit; create it if the fix needs it]" \
                if p in self.added else _elide(self, p)
            blocks.append((p, body))
        shown, note = _budgeted(head, blocks)
        user = head + "\n" + "\n".join(shown) + note + "\n\n" + INSTRUCTION
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def parse(self, reply):
        return _parse_files(reply, set(self.files))


def _budgeted(head, blocks):
    """Order blocks to fit BUDGET_TOKENS; return (shown_block_strings, omitted_note).
    The omitted note is budgeted too: a wide diff lists many long paths there."""
    target = (BUDGET_TOKENS - 200) * CHARS_PER_TOKEN  # slack for instruction + separators
    rem = target - len(head) - len(INSTRUCTION)
    shown, others = [], []
    for path, body in blocks:
        body = body[:PER_FILE_CAP] + ("\n...[truncated]" if len(body) > PER_FILE_CAP else "")
        if rem <= 40:
            others.append(path)
            continue
        if len(body) > rem:
            body = body[:rem] + "\n...[truncated]"
        shown.append(f"```{path}\n{body}\n```")
        rem -= len(body) + 2
    while len(head) + sum(len(s) for s in shown) + len(INSTRUCTION) > target and shown:
        others.append(blocks[len(shown)][0])
        shown.pop()
    note = ""
    if others:
        # +25 reserves room for the "(and N more)" suffix; base overcounts separators
        base = len(head) + 1 + sum(len(s) for s in shown) + len(shown) + 2 + len(INSTRUCTION)
        pre = "\n\nOther changed files (not shown for length): "
        kept, line = [others[0]], others[0]
        for p in others[1:]:
            if base + len(pre) + len(line) + 2 + len(p) + 25 > target:
                break
            line += ", " + p
            kept.append(p)
        if len(kept) < len(others):
            line += f" (and {len(others) - len(kept)} more)"
        note = pre + line
    return shown, note


def _changed_ranges(task, path, ctx):
    """Half-open [start,end) line ranges of the bad file to show (changed lines +-ctx)."""
    n = len(task.content_bad[path].splitlines())
    try:
        out = _git(V.REPOS[task.repo], "diff", "-U0", "--no-color",
                   task.bad, task.good, "--", path)
    except subprocess.CalledProcessError:
        return [(0, n)]
    ranges = []
    for line in out.splitlines():
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+", line)
        if not m:
            continue
        start = int(m.group(1)) - 1
        count = int(m.group(2)) if m.group(2) is not None else 1
        s = max(0, start - ctx)
        e = min(n, start + count + ctx)
        if e > s:
            ranges.append((s, e))
    ranges.sort()
    merged = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged or [(0, n)]


def _elide(task, path):
    bad = task.content_bad[path]
    lines = bad.splitlines(keepends=True)
    out, prev = [], 0
    for s, e in _changed_ranges(task, path, CONTEXT):
        if s > prev:
            out.append(f"...[{s - prev} lines omitted]...\n")
        out.extend(lines[s:e])
        prev = e
    if prev < len(lines):
        out.append(f"...[{len(lines) - prev} lines omitted]...\n")
    return "".join(out)


def _opener_info(line, known):
    """The path a ``` fence names, or None. A fence opens a file block iff its info is a
    path: it has a slash (any nested path, incl. a would-be edit of the hidden test), or a
    file extension, or is a bare name from the task's own files (e.g. a file literally
    named "agent"). Other bare words are language tags (python, mermaid, ...) and are not
    file blocks, so ``` fences inside a file's content can't end it early."""
    m = re.match(r"^\s*```(.+?)\s*$", line)
    if not m:
        return None
    info = m.group(1).strip()
    if not info or " " in info or "`" in info:
        return None
    if "/" in info or re.search(r"\.[A-Za-z0-9]{1,8}$", info) or info in known:
        return info
    return None


def _parse_files(reply, known):
    """```path blocks -> {path: content}. A block runs to the NEXT path-fence opener (or
    EOF), so ``` fences inside a file (common in .md) don't end it early; one trailing
    closing fence is stripped. Last block per path wins. known = the task's file set."""
    lines = reply.splitlines()
    openers = [(i, _opener_info(lines[i], known)) for i in range(len(lines))]
    openers = [(i, info) for i, info in openers if info is not None]
    files = {}
    for k, (start, info) in enumerate(openers):
        end = openers[k + 1][0] if k + 1 < len(openers) else len(lines)
        body = lines[start + 1:end]
        while body and body[-1].strip() == "":
            body.pop()
        if body and body[-1].strip() == "```":
            body.pop()
        while body and body[-1].strip() == "":
            body.pop()
        files[info] = "\n".join(body) + "\n"
    return files


def _diff_spec(repo, bad, good):
    """(files, added, content_bad, content_good) or None if the fix is not representable
    (any diff file that is not M/A, or not utf-8 where it exists)."""
    out = _git(repo, "diff", "--name-status", bad, good)
    status = []
    for line in out.splitlines():
        if not line.strip():
            continue
        st, path = line.split("\t")[:2]
        if st not in ("M", "A"):
            return None
        status.append((st, path))
    added = {p for st, p in status if st == "A"}
    modified = [p for st, p in status if st == "M"]
    files = sorted({p for _, p in status})
    content_good = {}
    for p in files:
        b = _show(repo, good, p)
        if b is None or not _is_text(b):
            return None
        content_good[p] = b.decode("utf-8")
    content_bad = {}
    for p in modified:
        b = _show(repo, bad, p)
        if b is None or not _is_text(b):
            return None
        content_bad[p] = b.decode("utf-8")
    return files, added, content_bad, content_good


def _verified_records():
    recs = {}
    for line in V.OUT.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        cur = recs.get(r["id"])
        if cur is None or r["ts"] >= cur["ts"]:
            recs[r["id"]] = r
    return recs


def _candidates():
    cands = {}
    for line in V.CANDIDATES.read_text().splitlines():
        if line:
            r = json.loads(line)
            cands[r["id"]] = r
    return cands


def load_tasks():
    """All representable, verified, checked loopgym tasks, sorted by id."""
    cands = _candidates()
    recs = _verified_records()
    tasks = []
    for tid in sorted(recs):
        r = recs[tid]
        if r["verdict"] != "verified":
            continue
        cand = cands.get(tid)
        if cand is None:
            continue
        check_dir = V.HIDDEN / tid
        if not check_dir.is_dir() or not list(check_dir.glob("test_*.py")):
            continue
        spec = _diff_spec(V.REPOS[r["repo"]], r["bad"], r["good"])
        if spec is None:
            continue
        files, added, content_bad, content_good = spec
        prio = [p for p in cand.get("files", []) if p in files]
        prio += [p for p in files if p not in prio]
        tasks.append(Task(name=tid, repo=r["repo"], bad=r["bad"], good=r["good"],
                          ticket=cand.get("ticket", ""), files=files, added=added,
                          content_bad=content_bad, content_good=content_good,
                          priority=prio))
    return tasks


def grade(task, files, runner="local", timeout=V.TIMEOUT):
    """Apply the reply's files to a worktree of the bad commit; run the hidden check.

    A reply that names any file outside task.files scores 0 without running (anti-hack:
    the model could otherwise rewrite the hidden test)."""
    if runner != "local":
        raise ValueError(f"loopgym tasks grade locally; got runner {runner!r}")
    info = {"runner": "local", "timed_out": False, "returncode": None,
            "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    extra = sorted(set(files) - set(task.files))
    if extra:
        info["reward"] = 0.0
        info["reason"] = "reply modifies files outside the task: " + ", ".join(extra)
        return 0.0, info
    checks = V.check_files(V.HIDDEN / task.name)
    if not checks:
        info["reward"] = 0.0
        info["reason"] = "no hidden check"
        return 0.0, info
    repo = V.REPOS[task.repo]
    with tempfile.TemporaryDirectory(prefix="loopgym-grade-") as base:
        base = Path(base)
        wt = base / "wt"
        V._add_worktree(repo, task.bad, wt)
        try:
            for path, content in files.items():
                dst = wt / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(content)
            scratch = base / "scratch"
            scratch.mkdir()
            info = V._run_once(wt, checks, scratch, timeout=timeout)
        finally:
            V._remove_worktree(repo, wt)
    return info["reward"], info
