# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. The dedicated line was meant to
stop re-dispatch but did not (humandispatch; host dispatcher never calls
sched_guard — root-cause ticket: hostgate, w2 working).

## Question
Does the range author accept 0.0764 (task-clustered CI straddles the 0.10 floor)
or revise the pre-run range? Open iff results/rulings/base05_val.ruling.json is
absent; a committed ruling closes it.

## Evidence (all observed, no guesses)
- Ruling absent everywhere (re-verified every attempt; latest w1 atk38,
  2026-10-10, master 3e4460e docs-only since dcb59cf; atk38 re-ran all three
  checks fresh: gate exit 1 (--line kevinruling), verdict exit 1, paste E2E
  exit 0 5/5 PASS, checker+guard 16/16):
  results/rulings/ = .gitkeep only; `git log --all -- <ruling file>` empty (no
  commit ever touched it — cd681e9 added only the .gitkeep); no stash; reflog =
  harness syncs only. The bound host .git IS /work's .git, so the single-repo
  check is exhaustive.
- Checker + eval byte-identical since cd681e9, so the atk28 E2E (temp copy of
  real results/ + checker `--results-dir`) still holds: none -> exit 1; accept
  verbatim -> exit 0, all baselines PASS; revise with 0.0764 in range -> exit 0;
  out of range / malformed (empty rationale, bad decision) -> exit 1 (fail
  closed). Logs: .agent-evidence/kevinruling{5,19,27,56,72}-accept-e2e.log,
  *-checker-tests.log (11/11).
- Verdict, fresh each run: exit 1, single FAIL base05 0.0764 [0.014, 0.153]
  < 0.10, other rows PASS (.agent-evidence/*-verdict.log; latest w1a34).
- atk35 correction: the "wiped tree" in Kevin's checkout was a sandbox mount
  artifact (only his .git is bound here). atk47: sandbox /opt/agent = bridge.py
  + hooks only — no dispatcher code in the sandbox.
- Attempt history (40+ re-dispatches, all same verdict): git log
  -- state/tickets/kevinruling.md.

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
0. (atk35 correction) the "wiped tree" was a sandbox mount artifact (only
   his .git is bound here). Kevin, on the host: `git status` in
   /home/kevin/Documents/coders/loop/projects/loopgym; if clean, go straight
   to step 1; `git restore .` only if it shows local deletions. (Or paste in
   any checkout with the full tree.)
1. Kevin pastes (accept; E2E re-verified exit 0 atk28 — temp results copy +
   paste -> checker exit 0, all baselines PASS):
   ```sh
   printf %s '{"author": "kevin", "decision": "accept", "rationale": "task-clustered CI [0.014, 0.153] straddles the 0.10 floor; ten pipeline causes ruled out (atk1-10)"}' > results/rulings/base05_val.ruling.json
   git add results/rulings/base05_val.ruling.json && git commit -m "ruling: accept base05 0.0764 (CI straddles 0.10 floor)"
   ```
   (revise form: see Done-when — new_range [lo,hi] with 0.0764 inside.)
2. Next agent run (rangerule): checker `--only baselines` → expect exit 0,
   record verdict in NOTES.md, unblock basecheck.
