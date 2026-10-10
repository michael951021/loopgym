#!/usr/bin/env python3
"""Gate for the host dispatcher: skip human-only TODO.md lines.

Open lines may carry `(human-only)` — never dispatch while open — or
`(human-only: RELPATH)` — skip while open and RELPATH (repo-relative) missing.
The dispatcher calls `sched_guard.py --line <id>` before dispatching a line:
exit 0 = may dispatch, 1 = skip, 2 = no such line id. `--list` reports every
open human-only line; exit 1 iff any must be skipped.
"""
import argparse
import os
import re
import sys

TASK = re.compile(r"^\s*- \[( |x)\]\s+(.*)$")
ID = re.compile(r"\(id:\s*([A-Za-z0-9_]+)\)")
TAG = re.compile(r"\(human-only(?::\s*([^)]+?))?\)")


def task_lines(todo_path):
    """Yield (lineno, id, open, tag) for every task line; tag is None (untagged),
    "" (plain human-only) or the release path."""
    with open(todo_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            m = TASK.match(line)
            if not m:
                continue
            mid = ID.search(m.group(2))
            tag = TAG.search(m.group(2))
            yield i, (mid.group(1) if mid else "?"), m.group(1) == " ", (
                tag.group(1) if tag and tag.group(1) else ("" if tag else None))


def skip_reason(open_, tag, root):
    if not open_ or tag is None:
        return None
    if tag == "" or not os.path.exists(os.path.join(root, tag)):
        return "human-only" if tag == "" else "skip until " + tag + " exists"
    return None  # released: the line can now be closed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="report open human-only lines (default)")
    ap.add_argument("--line", metavar="ID", help="gate a single line id")
    ap.add_argument("--todo", default="TODO.md", help="plan file (default: TODO.md)")
    ap.add_argument("--root", default=".", help="repo root for release paths")
    a = ap.parse_args()

    rows = list(task_lines(a.todo))
    if a.line:
        for lineno, lid, open_, tag in rows:
            if lid != a.line:
                continue
            why = skip_reason(open_, tag, a.root)
            if why:
                print(f"line {lineno} (id: {lid}): {why}", file=sys.stderr)
                return 1
            return 0
        return 2

    skipped = 0
    for lineno, lid, open_, tag in rows:
        if not open_ or tag is None:
            continue
        why = skip_reason(True, tag, a.root)
        if why:
            skipped += 1
            print(f"line {lineno} (id: {lid}): {why}")
        else:
            print(f"line {lineno} (id: {lid}): human-only, releasable ({tag} present)")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
