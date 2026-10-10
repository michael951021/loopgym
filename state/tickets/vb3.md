# vb3 — Hidden checks, batch 3 (candidates 25–40)

## Issue
Batches 1–2 verified 24/24 candidates (100% ≥ 30% gate), so the < 30% skip rule does not fire:
candidates 25–40 have no hidden checks yet, so none is an RL task. This is the last mining pass
before the dataset freeze (id: freeze waits on vb3).

## Question
Is state/verify/batch3.md correctly instantiated (16 unchecked steps, score order, fields matching
jsonl rows 25–40) — and can candidate 25, harness-bc2cdd8-97ddf39, get a `verified` verdict?
Refuted by a step/row field mismatch or a verify run that is not 0/3 bad and 3/3 good.

## Evidence
- Row 25: bad=bc2cdd8, good=97ddf39, file lib/jobs.py. Fix diff (git show good -- lib/jobs.py):
  new `gave_up`/`forget`; tick() re-asserts a twice-failed job's failed.json mark after a starting
  worker wipes it (the commit message's "job give-up survives failed.json resets").
- Discriminator: at good, a job line in state `failed` with tries ≥ JOB_TRIES hits the new tick
  branch → `gave_up` re-writes failed.json. At bad, the same line falls into `ready` and is retried
  forever; a wiped failed.json is never re-asserted. Observable: after wiping failed.json and
  calling jobs.tick(), is the line's key re-present in failed.json?
- lib/jobs.py, lib/sched.py, config.env all present at both commits → module-level `import jobs`
  is safe (no lazy-import error path). JOB_CPUS=0 makes the admission check fail so start()/
  systemd-run is never reached on either side (no systemd in the sandbox).
- Reuse the proven harness-check shape (tests/hidden/harness-25ebe03-4b0c866/): find ROOT from
  sys.path, fabricate a git project under ROOT/projects with a committed job line.
- Field-match output: .agent-evidence/batch3-instantiate.log.

## Done when
state/verify/batch3.md exists (16 unchecked steps, score order, `Verdict: (fill in)`); field-match
exits 0; for candidate 25: tests/hidden/harness-bc2cdd8-97ddf39/test_*.py (≤ 60 lines) exists and
`.venv/bin/python -m loopgym.data.verify harness-bc2cdd8-97ddf39` prints its verdict, recorded on
the step line, step checked. Known answer: the good side must re-assert the mark (assert true);
the bad side must not (assert false) — a hand-built jobs.json/failed.json fixture, no model.
TODO vb3 line left unchecked; NOTES.md updated; committed.

## Stop when
Scope: state/tickets/vb3.md, state/verify/batch3.md, tests/hidden/harness-bc2cdd8-97ddf39/, NOTES
lines for this area. Other candidates are for later runs (one per run). Two attempts at candidate
25 without `verified` ⇒ mark not-checkable with reason, check the step, stop.

## Plan
1. Instantiate batch3.md from templates/verify_step.md (rows 25–40, fix_msg ≤ 15 words).
2. Field-match script vs jsonl rows 25–40 via capture.py.
3. Write candidate-25 check (tick() re-asserts give-up after failed.json reset).
4. Run verify; record verdict; check step. NOTES, commit, merge master.

## Close-out (milestone complete)
- All 16 steps checked `verified (bad 0/3, good 3/3)`; step ids match candidates 25–40 in score order.
- All 16 hidden checks present; latest-ts record verified for each id in verified.jsonl (cross-check
  16/16 OK, this close-out run).
- Yield: 12/12 + 12/12 + 16/16 = 40/40 (100%) verified across batches 1–3.
- Fresh regression on current tree: verify draftlab-cf96577-11b1d16 → verified in 3.5 s, exit 0
  (.agent-evidence/verify-cf96577-11b1d16.log).
- vb3 line checked in TODO.md per batch 1's contract (check the line when every step is checked).
