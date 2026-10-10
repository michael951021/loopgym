# humandispatch — stop dispatching human-only TODO lines

## Issue
The host dispatcher sent agents at the kevinruling line 8x (atk1-8; 68+ total)
though rule 5 makes it Kevin-only. No agent can finish it, so every dispatch
wastes a run. The same trap awaits Kevin-only Scale-out run lines.

## Question
Can the dispatcher be given a machine-readable, in-repo marker (plus a gate
command it can call) so it skips open human-only lines — and treats the
kevinruling line as releasable once the ruling file is committed?

## Evidence
- Observed: no dispatcher code in the sandbox (bwrap audit atk48: /opt/agent =
  bridge.py + hooks only; task arrives via AGENT_TASK env). The fix must be a
  repo-side contract + gate Kevin wires into the host dispatcher.
- Observed: TODO.md:133 (id: kevinruling) is open, untagged, and the only
  line agents were repeatedly sent at; Scale-out lines (TODO.md:149-162) are
  all agent-doable today (write + dry-run), so none are tagged now.
- Check: `.venv/bin/python tools/sched_guard.py --line kevinruling` on the real
  TODO.md → expect exit 1 (skip) before the ruling lands, exit 0 after.

## Done when
1. `tools/sched_guard.py` parses TODO.md; open lines may carry `(human-only)`
   (skip while open) or `(human-only: RELPATH)` (skip while open and RELPATH
   missing). `--line ID` → 0 dispatchable, 1 skip, 2 unknown id; `--list`
   prints open human-only lines, exit 1 iff any must be skipped.
2. `(human-only: results/rulings/base05_val.ruling.json)` added to the
   kevinruling line (TODO.md:133); convention documented in CLAUDE.md.
3. `tests/test_sched_guard.py` passes (synthetic TODO.md + real TODO.md),
   including: tagged+unreleased → skip 1; tagged+released → 0; checked → 0;
   untagged → 0; unknown id → 2. `make check` clean.

## Stop when
Out of scope: editing the host dispatcher (not in sandbox; Kevin-side wiring),
tagging Scale-out lines (none are human-only yet), touching the kevinruling
ticket or w2's work. After two attempts without new evidence: record the
blocker and stop.

## Plan
1. tools/sched_guard.py (stdlib): parse open/checked lines, id + tag regex,
   release-path existence check, `--line`/`--list` exit codes.
2. Tag TODO.md:133; add a short "Scheduling" section to CLAUDE.md.
3. tests/test_sched_guard.py: synthetic matrix + real TODO.md assertion.
4. Run pytest + ruff; NOTES.md update; commit; merge master.
