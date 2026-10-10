# Ticket: loopgym task environment (id: lgenv)

TODO.md:99. after: vb1, exercism.

## Issue

loopgym/env/loopgym.py is missing. Verified loopgym records exist (data/loopgym/verified.jsonl)
but there is no Task object, no prompt/parse, and no grader that turns one into a reward.

## Question

What file set, prompt, parse, and grading semantics make the known answer hold:
good-commit files score 1, bad-commit files 0, for **every** loaded task — and a reply that
edits the hidden test scores 0?

## Evidence

- Probed all verified tasks (M-only vs full M+A diff, via V._run_once on worktrees):
  * Full bad→good diff (M+A, text only) applied to the bad tree makes the tree exactly equal
    to the good tree → check passes by construction (it was verified 3/3 at good).
  * M-only fails for draftlab-1d2ae92-44cfa02 (its check needs a file added between the
    commits) → files must include A.
  * 6 of 15 records are not expressible as file-content replies: their fix deletes files
    (D) or touches non-text files (parquet/png/jsonl data): f14caf0-1fc3f9e,
    44cfa02-9b213ef, 1d2ae92-9b213ef, fda7335-ca4403c, 5bf9b0d-ca4403c, ed3ae6c-1e9f5d3.
    load_tasks excludes them (a reply cannot delete or emit binary content).
  * b7c6e4f-1f173d1 first probed 0.0 in 0 s: its hidden check dir was missing on agent/w2
    (w1 committed it to master, 15cce4e). Merged master; re-probe = 1.0 in 30 s.
    → load_tasks must require tests/hidden/<id>/ with ≥1 test_*.py, not just the record.
- Grading must reject reply files outside the task's changed set, or a reply that
  "fixes" the hidden test would score 1 (anti-hack, required by the ticket).
- Prompt budget: 6k tokens ≈ 18k chars (chars/3, conservative). Full file content when it
  fits, else unchanged stretches elided around the changed lines (difflib ±2 context).

## Done when

`.venv/bin/python -m pytest -q tests/test_loopgym_env.py` passes (< 5 min):
for every loaded task, grade(reference) = 1.0 and grade(stub) = 0.0; a reply that
edits the hidden test scores 0.
