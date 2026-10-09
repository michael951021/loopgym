#!/usr/bin/env python3
"""Host-only: copy the agent history loopgym mines into data/raw/ (the sandbox cannot see other projects).

  python3 tools/host_export.py            (run on the host from the loopgym project root; idempotent)

Writes data/raw/:
  repos/draftlab.git    bare mirror of projects/speculative-agent-workloads (all branches, incl. ab-ref/* fixes)
  repos/harness.git     bare mirror of the loop harness (public: github.com/michael951021/local-agent-loop)
  ticket_review.json    the 5 replayed tickets with confirmed defects: base/bad/good commits, ticket text, defects
  runs.jsonl            one line per draftlab agent run: title, spec, outcome, attempts, wall_min, turns, start
  tasklog_draftlab.md   draftlab's TASKLOG.md (APPROACH / BLOCKERS per finished task)
  exercism-python/      github.com/exercism/python at EXERCISM_COMMIT (142 practice exercises: training tasks)
Excluded on purpose: projects/apply (personal data), projects/contrib-loop (third-party repos, undisclosed findings).
"""
import json
import subprocess
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LOOP = HERE.parent.parent
RAW = HERE / "data" / "raw"
EXERCISM_COMMIT = "6105f6d9d3ef66ce3d5c8f529d2882af3d76b706"


def mirror(src, dst):
    if dst.exists():
        subprocess.run(["git", "-C", str(dst), "fetch", "-q", "--prune", str(src), "+refs/heads/*:refs/heads/*"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--mirror", str(src), str(dst)], check=True)


def main():
    (RAW / "repos").mkdir(parents=True, exist_ok=True)
    mirror(LOOP / "projects" / "speculative-agent-workloads", RAW / "repos" / "draftlab.git")
    mirror(LOOP, RAW / "repos" / "harness.git")
    spec = tomllib.loads((LOOP / "ab" / "ticket-review" / "spec.toml").read_text())
    tasks = [{k: t.get(k) for k in ("id", "project", "base", "bad", "good", "ticket", "paths", "check", "defects")}
             for t in spec.get("task", []) if t.get("project") == "draftlab"]   # contrib-loop: excluded (see top)
    (RAW / "ticket_review.json").write_text(json.dumps(tasks, indent=1))
    keep = ("title", "spec", "result", "attempt", "wall_min", "turns", "outcome", "compactions", "start", "worker", "merged")
    with open(RAW / "runs.jsonl", "w") as out:
        for f in sorted((LOOP / "reports" / "speculative-agent-workloads").glob("*-end.json")):
            try:
                d = json.loads(f.read_text())
            except ValueError:
                continue
            out.write(json.dumps({"run": f.stem, **{k: d.get(k) for k in keep}}) + "\n")
    tl = LOOP / "projects" / "speculative-agent-workloads" / "TASKLOG.md"
    (RAW / "tasklog_draftlab.md").write_text(tl.read_text())
    ex = RAW / "exercism-python"
    if not ex.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/exercism/python.git", str(ex)], check=True)
    subprocess.run(["git", "-C", str(ex), "checkout", "-q", EXERCISM_COMMIT], check=True)
    print(f"exported {len(tasks)} ticket-review tasks, runs.jsonl, 2 repo mirrors -> {RAW}")


if __name__ == "__main__":
    main()
