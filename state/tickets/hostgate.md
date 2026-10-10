# hostgate — wire the host dispatcher to the (human-only) gate

## Issue
The host dispatcher (`lib/sched.py` in the host harness repo) has no
(human-only) concept: `plan()` classified every untagged open line "ready"
and `cmd_next` claimed it. Result: kevinruling was dispatched 5x (atk2-6)
and this line was re-dispatched to w2. Each dispatch burns a full agent
run on a line no agent can finish.

## Question
Does the host scheduler hold open `(human-only[: RELPATH])` lines (never
dispatch them) and release them once RELPATH exists? A dispatch of an open
human-only line after applying the patch refutes the fix.

## Evidence
- Root cause (harness mirror, data/raw/repos/harness.git HEAD 4cb686d,
  2026-10-07): `git grep human-only` empty; `lib/sched.py` plan() has no
  tag; the dispatch point is `loop/agent` :473 -> `python3
  "$ROOT/lib/sched.py" next ...`. In-repo tools/sched_guard.py was never
  in the harness's path — hence the 5x re-dispatches.
- RED (real /work/TODO.md, pristine lib/sched.py via worktree /tmp/hcheck
  at 4cb686d): plan() gave L133 kevinruling and L137 hostgate state
  "ready" -> GATED-LINES-READY count 2 (bug present).
- Patch (below, verified): plan() gates tagged lines to state
  "human-only"; cmd_next returns 10 "no agent tasks" (with held titles)
  when only held lines remain, 11 as before while anything runs; a held
  line's agent-doable neighbors are still claimed (rc 0).
- GREEN (same real TODO.md, patched): L133 and L137 -> "human-only",
  GATED-LINES-READY count 0; every other open id-tagged line unchanged
  (ready/job/waiting).
- Regression tests: tests/test_sched_human_only.py, 4/4 pass (tag held +
  rest ready; (human-only: P) release once P exists; held-only -> rc 10;
  agent line below a held line -> claimed).
- Cross-check vs the in-repo contract: for every open (id:) line in the
  real TODO.md, tools/sched_guard.py --line <id> exit 1 iff patched plan
  state "human-only" (0 mismatches; both hold L133/L137, both release on
  the ruling file). Known deliberate divergence: the patch's release
  check is `files.read(P)` truthiness (works in --ref mode; empty file =
  not released, fails closed) vs sched_guard's os.path.exists.
- Pre-existing, not from this patch: `python3 -m unittest discover` on
  the mirror fails one import (lib/plan.py needs pydantic, absent from
  sandbox python3; the host harness .venv has it). Pristine 4cb686d
  fails identically.
- atk4 (2026-10-10): all mirror refs scanned — host tip still 4cb686d
  (main 7b4cc82 and origin/* are older ancestors); no HUMAN_ONLY in any
  ref (still unapplied). Fresh worktree: apply-check clean; real
  TODO.md pre states ready/ready (RED) -> post human-only/human-only
  (GREEN); 4/4 tests; 0/16 mismatches vs sched_guard. Shipped
  tools/hostgate_apply.sh (host-only, idempotent, does not restart the
  dispatcher); tested on the real worktree: first run applies + 4/4 +
  exit 0, rerun idempotent + exit 0
  (.agent-evidence/hostgate-atk4-apply.log).

## Done when
1. On the host harness repo: `git apply
   <loopgym>/state/tickets/hostgate.patch` (base: mirror HEAD 4cb686d; if
   the host HEAD is newer, confirm lib/sched.py is unchanged around the
   three hunks — plan() else-branch, cmd_next tail, module regexes — or
   rebase the hunks).
2. In the host harness repo: `.venv/bin/python -m unittest
   tests.test_sched_human_only -v` -> 4/4 OK.
3. Restart the dispatcher. Live check while kevinruling/hostgate stay
   open: `next` reports them held (reason "no agent tasks", or waiting
   alongside a run), and no AGENT_TASK containing id kevinruling or
   hostgate is dispatched; an agent-doable line still dispatches.
4. Kevin checks off TODO:137; a later agent run records it.

## Stop when
Patch + tests are done and verified (above). The remaining step is
host-side and human-only by design: the dispatcher is not mounted in the
sandbox. If the host lib/sched.py diverged so much the hunks don't fit,
record the diff and stop; do not fork the scheduler.

## Plan (Kevin, host side)
1. One command, from the host:
   `~/Documents/coders/loop/work/loopgym/tools/hostgate_apply.sh`
   (applies state/tickets/hostgate.patch to the harness checkout, runs
   the 4 regression tests, reports; idempotent — rerun to re-verify).
   Manual fallback: `git apply
   ~/Documents/coders/loop/work/loopgym/state/tickets/hostgate.patch`
   then `.venv/bin/python -m unittest tests.test_sched_human_only -v`
   (4/4). Exit 3 = lib/sched.py diverged: diff the three hunk regions
   (module regexes, plan() else-branch, cmd_next tail) and rebase.
2. Restart the dispatcher process; watch the next cycle: held lines
   reported (rc 10 "no agent tasks" / rc 11 "waiting"), no AGENT_TASK
   for kevinruling or hostgate, agent lines still dispatch.
3. Check off TODO:137 once observed. Same gate now protects all future
   `(human-only)` lines, incl. the Kevin-only Scale-out lines.
