# rangerule — Range-author ruling: base05 floor

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] (n=16, T=0.8, 512 tok, 9 tasks,
seed 0) is below the pre-run 0.10 floor (CLAUDE.md). Rule 5 blocks "done"
on basecheck — and everything gated after it (grpocfg, grpo05) — until the
range author rules. Only Kevin may accept the number or revise the range;
agents never widen it (rule 5; tools/check_results.py:15).

## Question
Will the range author accept 0.0764 (task-clustered CI straddles the 0.10
floor) or revise the pre-run range? The question is open iff
results/rulings/base05_val.ruling.json is absent; a committed author ruling
is the observation that closes it.

## Evidence
- basecheck12-verdict.log: verdict exit 1, single FAIL row base05 0.0764
  < 0.10, all other rows PASS.
- Ten pipeline causes ruled out atk1-10 (.agent-evidence/basecheck*.log):
  split, grading, metric, truncation probe, parse, prompt/sampling byte
  audit, model identity, n=16 grading, handoff verbatim vs check_ruling,
  re-dispatch gap.
- Independent recompute from raw sidecars: 11/144 = 0.076389, exact match
  to committed JSON, 0 timeouts — the number is real, not a pipeline bug.
- results/rulings/ is .gitkeep-only: no ruling on file (verified this run;
  master == HEAD at 0c95db8).
- Unblock channel E2E-proven (atk8): an accept ruling flips the checker
  exit to 0 on this exact input.
All facts observed; no guesses outstanding.

## Done when
results/rulings/base05_val.ruling.json exists (author kevin; decision
"accept", or "revise" + new_range [lo,hi] with 0.0764 inside) AND
`.venv/bin/python tools/check_results.py --only baselines` exits 0.
Known answer: the checker applies the ruling verbatim (check_ruling,
tools/check_results.py:61; tests/test_check_results.py); malformed rulings
fail closed. I run the checker first this run: expect exit 1, single FAIL
row, note citing the ruling path. Ruling forms (verbatim-verified atk11; exact single-line paste in
state/tickets/kevinruling.md Plan step 1 — do not re-wrap the JSON):
- accept: {"author": "kevin", "decision": "accept", "rationale": "task-clustered CI [0.014, 0.153] straddles the 0.10 floor; ten pipeline causes ruled out (atk1-10)"}
- revise: same, "decision": "revise", "new_range": [lo, hi], 0.0764 in.

## Stop when
Agents never write or widen the range. If the ruling is absent at the end
of the run, stop with RESULT: not done — the one remaining action is
Kevin's; do not re-run ruled-out cause checks (that is the loop atk1-12
escaped). atk2 split the TODO line: kevinruling (the human file) + this
agent residue, now gated (after: kevinruling), so agents stop being
dispatched to the human-only part.

## Plan
1. Verify no ruling arrived (master moved? rulings dir?) — done: none.
2. Re-run the verdict as this run's evidence (expect exit 1, unchanged).
3. Update NOTES.md one-liner; commit ticket + NOTES.
4. Report not done with the exact unblock (one file, two forms above).
