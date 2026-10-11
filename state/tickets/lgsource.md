# lgsource — Eval loader for loopgym v0 (TODO 169)

## Issue
`python -m loopgym.eval.run --source loopgym` (the Eval CLI line in REPORT.md is
checked off as `exercism|loopgym`) raises `NotImplementedError` at
loopgym/eval/run.py:49. The baselg2 baseline job and every loopgym eval need the
loader + grader for the frozen v0 dataset.

## Question
Can the eval CLI load the frozen v0 tasks (`data/loopgym/v0/tasks.jsonl`) for a
split, prompt from the bad commit, and grade a model reply with the task's hidden
check the way verify.py does — in a reply format that fits the 512-token budget?
Falsified if the good commit's fix does not grade 1.0 or the bad commit 0.0 on
real tasks.

## Evidence
- env/loopgym.py (lgenv) already builds v0-style Tasks from verified records with
  a full-file ```path format, pinned by tests/test_loopgym_env.py (ref=1, stub=0).
- Budget: median v0 fix = 815 diff lines (results/loopgym_v0.json) vs
  --max-new-tokens 512 — whole-file rewrites cannot fit. **Decision: the reply is
  a unified diff in one ```diff block** (only changed hunks; the prompt's elided
  changed-lines give applying context; reference = `git diff bad..good`, which
  grades 1.0 by construction; empty patch = bad state = 0.0).
- 15/26 v0 tasks have empty ticket text; fix_msg is the answer, never in prompt.
- Checks: tests in tests/test_eval_run.py (ref 1.0 / bad 0.0, 2 real test-split
  tasks); CLI run logged under .agent-evidence/lgsource-*.log.

## Done when
1. `pytest -q tests/test_eval_run.py` passes, incl. `--reference --source
   loopgym --split test --limit 2 --n 1` -> pass@1 = 1.0 (good fix graded 1.0)
   and empty-patch grade = 0.0 (bad commit) on those 2 real tasks.
2. `.venv/bin/python -m loopgym.eval.run --model
   models/qwen2.5-coder-0.5b-instruct --source loopgym --split test --n 1
   --limit 2` runs end to end (exit 0).

## Stop when
Not doing: full baselines (baselg2), train.py wiring, the lgenv content-format
Task or its tests, frozen v0 data. If ref=1.0/bad=0.0 cannot be reproduced after
2 attempts: record in NOTES.md, split the ticket.

## Plan
1. env/loopgym.py: `DiffTask(Task)` (diff prompt + parse -> patch + reference =
   git diff), `load_v0(split, limit)` from v0/tasks.jsonl (ticket/priority from
   candidates.jsonl), `_grade_diff` (frozen checks at data/loopgym/v0/checks/,
   git apply -p1/-p0 on bad worktree, pristine V._run_once, timeout 120); one
   dispatch branch in grade().
2. eval/run.py: loopgym branch -> load_v0; env grade dispatcher; thread timeout
   (loopgym 120 / exercism 30).
3. Tests; e2e model run; NOTES.md; commit; merge master.
