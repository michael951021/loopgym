# toyok — Check toy known answers

## Issue
The planted-effect (banana) and null-control (coinflip) runs are the gate before any real
GRPO run (CLAUDE.md). If the trainer fails them it has a bug, not a finding. No verdict
has been recorded yet: results/toy.json does not exist.

## Question
Does `tools/toy_check.py` exit 0 on 60-step berry-prompt runs: banana steps 0-4 mean
< 0.2 AND steps 55-59 mean > 0.8, null slope bootstrap CI containing 0?

## Evidence
- Observed 2026-10-09: `tools/toy_check.py` exits 1, both metrics.jsonl missing
  (.agent-evidence/toyok-before.log).
- Root cause of the missing runs (observed): toybanana2/toynull2 (marked [x], exit 0)
  no-op'd. The toyokfix wipe deleted runs/toy_*/metrics.jsonl but left the stale
  step-60 fruit-prompt checkpoints at ckpt/toy_banana + ckpt/toy_null (timestamps
  18:53/19:16, pre-fix). train.py --resume-if-any then hit "already complete at step
  60; nothing to do" and exited 0 with no artifacts.
- Fix landed: train.py exits 1 when a resume is past cfg.steps but metrics.jsonl is
  missing (orphaned checkpoint). Reproduced on the real stale ckpt: exit 1
  (.agent-evidence/toyok-guard-repro.log); tests/test_train.py
  test_orphaned_checkpoint_fails_resume_if_any + test_complete_run_resume_if_any_noops
  pass (.agent-evidence/toyok-guard-tests.log). Stale ckpts deleted.
- Fresh runs re-ticketed as compute jobs toybanana3/toynull3 in TODO.md; toyok now
  (after: toybanana3, toynull3).

## Done when
`.venv/bin/python tools/toy_check.py` exits 0 on the toybanana3/toynull3 metrics;
results/toy.json written (both curves' first/last window means + slope, lo, hi);
NOTES.md 3 lines. Known answers: planted effect must rise < 0.2 -> > 0.8, null slope
CI must contain 0 (thresholds pinned in CLAUDE.md, not tunable). If exit 1 on the
fresh runs: find the trainer bug; do NOT adjust thresholds.

## Stop when
Scope: tools/toy_check.py, results/toy.json, NOTES.md, (trainer code if a bug found).
Blocked this run on the fresh 60-step compute jobs (~hours each; rule 3: not a
session job). Two-attempt limit reached for producing the data here -> stop.

## Plan
1. (done) Ran check: exit 1, missing metrics; found stale-ckpt root cause.
2. (done) train.py orphaned-ckpt guard + 2 tests; deleted stale ckpts; re-ticketed
   fresh runs toybanana3/toynull3; retargeted toyok (after:).
3. (next run, after jobs) Run toy_check.py; on exit 0 write results/toy.json from the
   printed numbers + 3 NOTES lines; on exit 1 debug the trainer, never the thresholds.
