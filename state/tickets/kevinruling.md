# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. This dedicated line stops agents
being re-dispatched to the human-only part (6th dispatch: atk6).

## Question
Does the range author accept 0.0764 (task-clustered CI straddles the 0.10 floor)
or revise the pre-run range? Open iff results/rulings/base05_val.ruling.json is
absent; a committed ruling closes it.

## Evidence (all observed, no guesses)
- Fresh this run: .agent-evidence/kevinruling1-verdict.log — checker exit 1,
  single FAIL row base05 0.0764 < 0.10, all other rows PASS; rulings/ is
  .gitkeep-only.
- Raw sidecar recompute exact (11/144 = 0.076389, 0 timeouts): number is real.
- Ten pipeline causes ruled out atk1-12 (.agent-evidence/basecheck*.log).
- atk5: ruling-channel E2E re-proven on the current checker (cd681e9) against a
  copy of the real results/: none->exit 1, verbatim accept->exit 0, revise with
  0.0764 in range->exit 0, revise with 0.0764 out of range->exit 1 (fail
  closed) — .agent-evidence/kevinruling5-accept-e2e.log. (The older atk8 log is
  no longer in .agent-evidence/; this run replaces it.)
- atk6: ruling still absent — master tip bae8ab5, rulings/ .gitkeep-only,
  `git log --all` never added the file. Fresh on-site verdict exit 1, single
  FAIL base05 0.0764 < 0.10 (.agent-evidence/kevinruling6-verdict.log). 6th
  dispatch; nothing agent-doable remains (rule 5).

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
