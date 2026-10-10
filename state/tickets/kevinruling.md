# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. The dedicated line was meant to
stop re-dispatch but did not (9th dispatch: atk9).

## Question
Does the range author accept 0.0764 (task-clustered CI straddles the 0.10 floor)
or revise the pre-run range? Open iff results/rulings/base05_val.ruling.json is
absent; a committed ruling closes it.

## Evidence (all observed, no guesses)
- atk10: ruling still absent — `results/rulings/` empty, `git log --all`
  has no ruling on any ref; checker and results/ unchanged since atk8, so the
  atk5 E2E stands. Fresh verdict exit 1, single FAIL base05 0.0764 < 0.10,
  unchanged (.agent-evidence/kevinruling10-verdict.log). 10th dispatch; the
  dedicated line did not stop re-dispatch — root-cause ticket (id:
  humandispatch) stands: Kevin, commit the file or stop dispatching this line.
- Raw sidecar recompute exact (11/144 = 0.076389, 0 timeouts): number is real.
- Ten pipeline causes ruled out atk1-12 (.agent-evidence/basecheck*.log).
- atk5: ruling-channel E2E on checker cd681e9 against a copy of the real
  results/: none->exit 1, verbatim accept->exit 0, revise with 0.0764 in
  range->exit 0, revise with 0.0764 out->exit 1 (fail closed)
  (.agent-evidence/kevinruling5-accept-e2e.log).

## Done when
results/rulings/base05_val.ruling.json committed by kevin, and
`.venv/bin/python tools/check_results.py --only baselines` exits 0.
Verbatim forms (malformed rulings fail closed; check_ruling, tools/check_results.py:61):
- accept: {"author": "kevin", "decision": "accept", "rationale": "task-clustered CI [0.014, 0.153] straddles the 0.10 floor; ten pipeline causes ruled out (atk1-10)"}
- revise: same shape, "decision": "revise", "new_range": [lo, hi] with 0.0764 in [lo, hi].

## Stop when
HUMAN-ONLY: agents never write or widen the range (rule 5). If the ruling is
absent at run end, the agent stops with RESULT: not done and does not re-run
ruled-out cause checks. Kevin: commit the one file above; the gated line
(id: rangerule, after: kevinruling) then runs the checker and records the verdict.

## Plan
1. Kevin commits results/rulings/base05_val.ruling.json (one of the two forms).
2. Next agent run (rangerule): checker `--only baselines` → expect exit 0,
   record verdict in NOTES.md, unblock basecheck.
