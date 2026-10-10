# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. The dedicated line was meant to
stop re-dispatch but did not (23rd dispatch: atk23).

## Question
Does the range author accept 0.0764 (task-clustered CI straddles the 0.10 floor)
or revise the pre-run range? Open iff results/rulings/base05_val.ruling.json is
absent; a committed ruling closes it.

## Evidence (all observed, no guesses)
- atk13-16 (2026-10-10): ruling absent on all refs and never written on disk
  (find /work /home /tmp: nothing); no git remote exists (the ruling must be
  a local commit here); verbatim forms re-read against check_ruling each
  time (check_results.py:61; malformed fails closed); verdict exit 1, single
  FAIL base05, unchanged every run
  (.agent-evidence/kevinruling{13,14,16}-verdict.log).
- atk17-18 (2026-10-10): unchanged — results/rulings/ holds only .gitkeep,
  no ruling on any ref, no remote, checker unchanged since cd681e9 (forms
  re-verified against check_ruling, tools/check_results.py:61). Fresh verdict
  exit 1, single FAIL base05 0.0764 [0.014, 0.153] < 0.10
  (.agent-evidence/kevinruling{17,18}-verdict.log). 17th/18th dispatch; per
  Stop-when, stopped without re-running ruled-out checks. Root-cause ticket
  (id: humandispatch) stands: the dispatcher keeps re-dispatching this
  human-only line.
- atk19 (2026-10-10): unchanged — ruling absent on disk (.gitkeep only) and
  on all refs, no remote; fresh verdict exit 1, single FAIL base05
  (.agent-evidence/kevinruling19-verdict.log). 19th dispatch. Task dispatcher
  is harness-side: no scheduler code in the repo (only match is the lgbox
  job-queue dispatcher, unrelated). No ruled-out checks re-run. NEW: the
  paste-ready accept command in Plan is E2E-verified on a scratch copy of
  the real results/ against the real checker -> exit 0, all rows PASS
  (.agent-evidence/kevinruling19-accept-e2e.log) — Kevin's unblock is now a
  single paste.
- atk20 (2026-10-10): unchanged — ruling absent on disk and all refs, no
  remote; fresh verdict exit 1, single FAIL base05
  (.agent-evidence/kevinruling20-verdict.log). 20th dispatch. Root-cause
  ticket (id: humandispatch) stands at TODO.md:135; nothing further an
  agent may do (rule 5).
- atk21 (2026-10-10): unchanged — ruling absent on disk (results/rulings/
  = .gitkeep only) and on all refs (git log --all: only cd681e9 touched the
  path), no remote; fresh verdict exit 1, single FAIL base05 0.0764
  [0.014,0.153] < 0.10 (.agent-evidence/kevinruling21-verdict.log). 21st
  dispatch; humandispatch ticket stands at TODO.md:135. Kevin's one paste
  (Plan step 1) is the only unblock.
- atk22 (2026-10-10): unchanged — ruling absent on disk (results/rulings/
  = .gitkeep only) and on all refs (git log --all: only cd681e9 touched the
  path), no remote; fresh verdict exit 1, single FAIL base05 0.0764
  [0.014,0.153] < 0.10 (.agent-evidence/kevinruling22-verdict.log). 22nd
  dispatch; humandispatch ticket stands at TODO.md:135. Rule 5: an agent may
  not write the ruling (it would self-accept its own out-of-range number and
  void the gate). Kevin's one paste (Plan step 1) is the only unblock.
- atk23 (2026-10-10): unchanged — ruling absent on disk (results/rulings/
  = .gitkeep only) and on all refs, no remote; fresh verdict exit 1, single
  FAIL base05 0.0764 [0.014,0.153] < 0.10
  (.agent-evidence/kevinruling23-verdict.log). 23rd dispatch; humandispatch
  ticket stands at TODO.md:135. No agent-side path exists (rule 5); Kevin's
  one paste (Plan step 1) is the only unblock.
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
1. Kevin pastes (accept; E2E-verified exit 0 atk19 — kevinruling19-accept-e2e.log):
   ```sh
   printf %s '{"author": "kevin", "decision": "accept", "rationale": "task-clustered CI [0.014, 0.153] straddles the 0.10 floor; ten pipeline causes ruled out (atk1-10)"}' > results/rulings/base05_val.ruling.json
   git add results/rulings/base05_val.ruling.json && git commit -m "ruling: accept base05 0.0764 (CI straddles 0.10 floor)"
   ```
   (revise form: see Done-when — new_range [lo,hi] with 0.0764 inside.)
2. Next agent run (rangerule): checker `--only baselines` → expect exit 0,
   record verdict in NOTES.md, unblock basecheck.
