# vb1 — Hidden checks, batch 1 (candidates 1–12)

## Issue
The 12 top-scored defect candidates in data/loopgym/candidates.jsonl have no hidden checks, so
none is an RL task yet. The loopgym dataset grows only when each candidate gets a deterministic
check that fails at its bad commit and passes at the fix, confirmed by verify.py.

## Question
Is state/verify/batch1.md correctly instantiated: exactly 12 unchecked steps, one per candidate,
in score order, each step's id/repo/bad/good/files matching its candidates.jsonl row? Refuted by
any step id absent from rows 1–12, wrong order, or a field mismatch.

## Evidence
- data/loopgym/candidates.jsonl: 141 rows; mine.py:181 writes them sorted `(-lines, id)` — file
  order IS the score order (score = defect-diff lines; no separate score field exists).
- templates/verify_step.md: step format + the Rules/On-failure text to carry over.
- data/raw/repos/{draftlab,harness}.git exist; tests/hidden/ has only draftlab-alpha so far.
- Smallest check: field-match script comparing the 12 steps against jsonl rows 1–12; output in
  .agent-evidence/batch1-instantiate.log.

## Done when
state/verify/batch1.md exists (template instantiated, 12 unchecked steps in score order, each
ending `Verdict: (fill in)`); field-match script exits 0; the vb1 line in TODO.md left
unchecked; this ticket written; NOTES.md updated; committed. No hidden check is written this run.

## Stop when
Scope: state/tickets/vb1.md, state/verify/batch1.md, and this run's NOTES.md lines only. No
tests/hidden/*, no verify.py runs (one candidate per run; the steps are for later runs). If a
row 1–12 lacked id/bad/good/files, stop and record it — do not guess.

## Plan
1. Extract rows 1–12 of candidates.jsonl (done: ids draftlab-7953937-0c66ede …
   draftlab-a75ef10-1f173d1, 11 draftlab + 1 harness).
2. Write state/verify/batch1.md: template Rules/On-failure + 12 steps (fix_msg ≤ 15 words).
3. Run the field-match script via capture.py; save log.
4. NOTES.md line for this area; commit; git merge master.
