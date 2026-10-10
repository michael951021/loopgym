# rangerule — Range-author ruling: base05 floor

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] (n=16, T=0.8, 512 tok, 9 tasks,
seed 0) is below the pre-run 0.10 floor (CLAUDE.md). Rule 5 blocks "done"
on basecheck — and everything gated after it (grpocfg, grpo05) — until the
range author rules. Only Kevin may accept the number or revise the range;
agents never widen it (rule 5; tools/check_results.py:15).

## Question
Does `.venv/bin/python tools/check_results.py --only baselines` exit 0 now
that the author ruling is on file? Closed: ruling landed at 64ecd1b (kevin,
decision "accept"); this run's fresh checker exit 0 is the verdict
(rangerule-verdict.log).

## Evidence
- basecheck12-verdict.log: verdict exit 1, single FAIL row base05 0.0764
  < 0.10, all other rows PASS.
- Ten pipeline causes ruled out atk1-10 (.agent-evidence/basecheck*.log):
  split, grading, metric, truncation probe, parse, prompt/sampling byte
  audit, model identity, n=16 grading, handoff verbatim vs check_ruling,
  re-dispatch gap.
- Independent recompute from raw sidecars: 11/144 = 0.076389, exact match
  to committed JSON, 0 timeouts — the number is real, not a pipeline bug.
- Ruling on file: results/rulings/base05_val.ruling.json (commit 64ecd1b;
  author kevin, decision "accept", rationale cites the CI straddling the
  0.10 floor + atk1-10). Verified this run, byte-identical to the
  atk11-verified form.
- This run: checker --only baselines exit 0, all 5 rows PASS, base05 row
  notes "in range by ruling — kevin: …" verbatim
  (.agent-evidence/rangerule-verdict.log). Regression:
  tests/test_check_results.py 11/11.
All facts observed; no guesses outstanding.

## Done when
Ruling on file (yes, 64ecd1b) AND `.venv/bin/python tools/check_results.py
--only baselines` exits 0 (yes, this run, rangerule-verdict.log) AND the
verdict is recorded in NOTES.md. basecheck unblocked. Known answer: the
checker applies the ruling verbatim (check_ruling,
tools/check_results.py:61; tests/test_check_results.py 11/11 this run);
malformed rulings fail closed.

## Stop when
Agents never write or widen the range. If the ruling is absent at the end
of the run, stop with RESULT: not done — the one remaining action is
Kevin's; do not re-run ruled-out cause checks (that is the loop atk1-12
escaped). atk2 split the TODO line: kevinruling (the human file) + this
agent residue, now gated (after: kevinruling), so agents stop being
dispatched to the human-only part.

## Plan (executed)
1. Verify ruling on file (64ecd1b) and read it — done: accept.
2. Run the checker as this run's evidence — done: exit 0, 5/5 PASS,
   rangerule-verdict.log.
3. Update NOTES.md verdict line; commit ticket + NOTES.
