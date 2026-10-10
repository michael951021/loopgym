# toyok — Check toy known answers

## Issue
The planted-effect (banana) and null-control (coinflip) runs are the gate before any real
GRPO run (CLAUDE.md). If the trainer fails them it has a bug, not a finding. No verdict has
been recorded yet in results/ or NOTES.md.

## Question
Does `tools/toy_check.py` exit 0 on the existing 60-step runs: banana steps 0-4 mean < 0.2
AND steps 55-59 mean > 0.8, null slope bootstrap CI containing 0?

## Evidence
- runs/toy_banana/metrics.jsonl, runs/toy_null/metrics.jsonl (60 training lines each; steps 1-60).
- Smallest check: `.venv/bin/python tools/toy_check.py` (exit 0 = pass). Evidence log:
  .agent-evidence/toyok-verify.log.

## Done when
`.venv/bin/python tools/toy_check.py` exits 0; results/toy.json written (both curves'
first/last window means + slope, lo, hi); NOTES.md 3 lines. Known answers: planted effect
must rise, null control must show no trend (both pinned in CLAUDE.md; thresholds are the
spec, not tunable). If exit 1: find the trainer bug (tests/test_batch.py style checks),
fix it, add `- [ ]` re-run lines for the toy jobs; do NOT adjust thresholds.

## Stop when
Only tools/toy_check.py, results/toy.json, NOTES.md, (plus trainer code if a bug is found)
are in scope. Stop/split after two attempts without new evidence.

## Plan
1. Run the check, record output + exit code.
2. If pass: write results/toy.json from the printed numbers; 3 NOTES lines; commit.
3. If fail: inspect failing curve, build tests/test_batch.py-style check on the suspect
   piece, fix, note re-run lines in NOTES.md; add the `- [ ]` re-run ticket lines.
