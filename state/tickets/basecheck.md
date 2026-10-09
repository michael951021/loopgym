# Ticket: Check baselines against ranges (id: basecheck)

## Issue
Baseline eval numbers (base05/base15 exercism val) must be checked against the
CLAUDE.md plausible ranges before any GRPO run: a bad prompt, grader, or
truncation would poison every later before/after comparison.

## Question
Are all committed result numbers inside their CLAUDE.md plausible ranges with
known answers holding? Current observation refuting "all good": base05 val
pass@1 = 0.0833 [0.000, 0.167] vs range 0.10–0.55 — point estimate below the
floor. base15 = 0.3611 [0.167, 0.583] inside 0.25–0.75.

## Evidence
- results/eval/base05_val.json (0.0833), base15_val.json (0.3611),
  reference_exercism_val.json (1.000 known answer, grading+prompt sanity).
- Commit 447e745 probe: 0.000 rewards are genuine (model emits trailing junk),
  not a grader bug.
- Ruling-out probes (all .agent-evidence/basecheck-*.log):
  - probe512: regenerate 9 sample-0 completions, same seeds, T=0.8, 512 tok:
    9/9 rewards match committed sidecar (pipeline deterministic); only 2/9
    capped without EOS (food-chain, vlq).
  - probe1024: those two tasks at 1024 tok, n=4: 0/8 recovered, all end in
    EOS → truncation not the cause.
  - probe-parse: 0-reward failures are model content: isogram check
    `len(dict) == len(set(dict))` is a tautology (always True; direct grade:
    14 tests, 8 failures, reward 0 — exactly the 8 non-isogram cases);
    point-mutations raises ValueError on length mismatch where tests require
    truncating comparison; parse picks the correct last ```python block.
  - Grading cross-check: reference val pass@1 = 1.000; junit counts match
    hand analysis of the model's own code.
- Verdict: 0.0833 is an honest 0.5B measurement at the committed protocol;
  below the pre-written 0.10 floor; CI [0, 0.167] overlaps the range.
  Rule 5: record as out-of-range suspected bug, never widen the range.

## Done when
`.venv/bin/python tools/check_results.py --only baselines` exits 0: every
number in its CLAUDE.md range (base05 0.10–0.55, base15 0.25–0.75),
1.5B > 0.5B (CIs: non-overlapping, or overlap reported). Known answers:
reference val pass@1 == 1.0. If base05 stays out of range after ruling out
prompt/grading/truncation, record as suspected bug per rule 5 — never widen
the range.

## Stop when
Scope: this ticket only checks baselines (toy/bench sections grow with later
experiments; missing files in other sections skip, not fail). Stop/split:
after two attempts without new evidence on the base05 cause, record the
blocker in NOTES.md and stop.

## Plan
1. tools/check_results.py: load results JSONs, assert ranges, 1.5B>0.5B with
   CI-overlap report, print table, --only filter. Run → expect base05 FAIL.
2. Probe base05 cause: regenerate 0.5B sample-0 completions (same seeds,
   max_new_tokens 512), check new-token count / EOS to test truncation; grade
   with 1024 budget on the failing tasks to see which flip.
3. If a pipeline bug: fix + rerun affected baseline (compute job if >10 min).
   If genuine performance: document evidence, keep range, report per rule 5.
4. check exits 0; make check; NOTES.md; commit; merge master.

## Outcome — attempt 2 (2026-10-09)
`--only baselines` still exits 1 (base05 0.0833 < 0.10 floor); checker unit
tests 7/7. Attempt 2 re-verified the number end-to-end:
- Split audit: the val bucket (sha1%100 in [70,85)) holds 10 exercises;
  `ledger` is excluded from splits.json because its stub scores 1.0 (not
  usable) — legitimate. The 9 committed val tasks are exactly the usable
  ones. (parallel-letter-frequency's ref times out at the 30 s audit and
  passes at 60 s — an audit artifact, but it is in the TEST split, so it
  does not affect these val numbers.)
- Independent metric recompute: a hand-written Chen pass@1 over the
  sidecars matches the committed values exactly (0.083333 / 0.361111); the
  bootstrap CI differs only by RNG seed and still overlaps the 0.10 floor.
- Full re-grade (.agent-evidence/basecheck2-regrade36.log): all 36 base05
  samples regenerated from the recorded seeds (T=0.8, top_p=1.0, 512 tok)
  and re-graded from scratch — 0 mismatches, 0 timeouts, new pass@1
  0.0833 exactly. No pipeline bug; 0.0833 is an honest 0.5B measurement
  (failures are model content: tautology, wrong exceptions, trailing junk).
Resolution: n=16 re-run of both baselines added to TODO.md as a (run:)
compute job (resumes the sidecars, reuses samples 0–3, then re-runs the
checker verdict). Until it lands, rule 5 blocks "done"; the range author
must decide (accept with overlapping CI, or revise the floor).
