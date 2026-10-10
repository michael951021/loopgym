# hostgate — wire the host dispatcher to tools/sched_guard.py

## Issue
The host dispatcher never calls the in-repo gate, so it re-dispatched the
gated kevinruling line 5x (atk2-6) and dispatched this (human-only) line
again today (w2). Each dispatch burns a full agent run on a line no agent
can finish. The gate is built, tested and documented; it is inert until the
host selection loop calls it.

## Question
Does the host dispatcher's line-selection loop call
`tools/sched_guard.py --line <id>` for every candidate line before
dispatch, skipping on non-zero exit? A dispatch of an open human-only line
after wiring refutes the fix.

## Evidence
- Observed (atk48 bwrap audit, re-verified this run): no dispatcher code in
  the sandbox; /opt/agent is read-only (bridge.py + hooks only), task arrives
  via AGENT_TASK env. The edit is Kevin-side → line is (human-only).
- Observed (this run, real TODO.md): hostgate→exit 1, kevinruling→exit 1
  (ruling file absent), checked line→0, unknown id→2; `--list` exit 1
  lists both open human-only lines. pytest tests/test_sched_guard.py 5/5.
- This run is the live repro: the dispatcher sent this (human-only) line to
  w2 without gating it.

## Done when
1. Host dispatcher, in its candidate loop from the loopgym repo root, runs
   the gate per line and skips on any non-zero exit (1 = human-only;
   2 = unknown/stale id):
   `if ! .venv/bin/python tools/sched_guard.py --line "$ID"; then continue; fi`
   (gate is stdlib-only; plain python3 works too.)
2. Live check after wiring: while hostgate/kevinruling stay open, neither
   is dispatched again (no AGENT_TASK containing those ids); the next
   dispatch is an agent-doable line. `--list` output matches what the
   dispatcher skips.
3. No repo change needed; tests/test_sched_guard.py stays green as the
   regression cover for the contract.

## Stop when
Repo side is complete and verified. If the dispatcher's location or
selection loop can't be found on the host, record the blocker and stop.
Out of scope: re-implementing line selection in-repo, editing the host
dispatcher from the sandbox (ro mount — impossible), changing the gate.

## Plan
1. Kevin: add the 2-line gate call to the host dispatcher's candidate loop.
2. Restart the dispatcher; confirm no dispatch of open human-only lines.
3. Kevin checks off TODO:137; a later run records it.
