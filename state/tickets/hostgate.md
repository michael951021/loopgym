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
  tools/hostgate_apply.sh (host-only, idempotent); tested on the real
  worktree: first run applies + 4/4 + exit 0, rerun idempotent + exit 0
  (.agent-evidence/hostgate-atk4-apply.log).
- atk5 (2026-10-10): host still unapplied (this 5th dispatch of the line is
  the live proof). Fresh worktree at 4cb686d re-verified: real TODO.md
  kevinruling+hostgate ready (RED) -> human-only (GREEN), 4/4 tests, 0/16
  mismatches vs sched_guard (.agent-evidence/hostgate-atk5-verify.log). New:
  **no dispatcher restart needed** — the gate takes effect on the next poll:
  agent:473 spawns `python3 "$ROOT/lib/sched.py" next` each cycle and
  pool.py:77 `importlib.reload(sched)` each poll ("plan with the scheduler
  on disk"). Also: /work is the host worktree
  ~/Documents/coders/loop/work/loopgym/w2; the stable script path is the
  primary checkout ~/Documents/coders/loop/projects/loopgym.
- atk6 (2026-10-10): this 6th dispatch of the (human-only) line is the live
  proof the gate is still not in the live dispatcher (with it, plan() holds
  L137). No patch re-verification (identical to atk4/atk5); instead
  hardened the operator path: hostgate_apply.sh no longer says "restart
  the dispatcher" (contradicted atk5's no-restart finding). Fresh E2E on a
  /tmp/hg6 clone of the mirror at 4cb686d: first run applies + 4/4 + rc 0,
  rerun idempotent + rc 0; real TODO.md: `next` -> rc 11 waiting (no gated
  line claimed), `status` holds L133/L137 as human-only
  (.agent-evidence/hostgate-atk6-apply.log). In-repo 16/16 (sched_guard +
  check_results); live gate rc: kevinruling 1, hostgate 1, rangerule 0.
- atk7 (2026-10-10): 7th dispatch = live proof the gate is still not in the
  live dispatcher. New: audited the harness dispatch path (mirror 4cb686d)
  for any in-repo extension that could gate dispatch without the host
  patch — none exists: plan() state set {failed, claimed, waiting,
  blocked-failed, job, ready} (sched.py:149-182), cmd_next claims only
  "ready", pool.ready() counts only "ready" (pool.py:53-60), (run:) job
  CMDs run in the same bwrap sandbox (agent cmd_job: `sandbox "$wt" --
  bash -c "$cmd"`), hooks are ro-bound from $ROOT/hooks (harness
  territory), failed.json/MAX_TASK_ATTEMPTS are host state, and the only
  TODO.md lever is (after:) deps (a semantic fudge; cannot hold
  kevinruling, w1's line). Extending .agent-shared to bind the harness
  root would self-apply the patch: violates hard rule 4 + the
  (human-only) tag and grants every agent RW to the dispatcher —
  rejected. So Kevin's one-liner is the only path. Re-verified fresh at
  4cb686d: apply-check OK, 4/4, real TODO.md RED (ready/ready, 2 ready)
  -> GREEN (human-only/human-only, 0 ready; rangerule unchanged
  waiting), in-repo 16/16, guard rc 1/1/0
  (.agent-evidence/hostgate-atk7-verify.log). Note: the mirror updates
  only when host_export.py runs on the host, so a later run cannot
  self-detect Kevin's apply from the mirror; the live dispatcher is
  ground truth.
- atk8 (2026-10-10): 8th dispatch = live proof the gate is still not in
  the live dispatcher. New: delivery-path check (atk4-7 only tested
  /tmp clones): master 18e9b3f contains tools/hostgate_apply.sh +
  state/tickets/hostgate.patch (same blobs as w2), and the primary
  checkout ~/Documents/coders/loop/projects/loopgym sits on master
  (`git worktree list`), so Plan step 1's one-liner is executable as
  written. Mirror still 4cb686d, no HUMAN_ONLY in any ref. Fresh E2E on
  a new 4cb686d worktree: apply rc 0 + 4/4, rerun idempotent rc 0;
  guard 1/1/0; in-repo 16/16 (.agent-evidence/hostgate-atk8-verify.log).
  No new agent-side step; nothing left but Kevin's run.

## Done when
1. On the host harness repo: `git apply
   <loopgym>/state/tickets/hostgate.patch` (base: mirror HEAD 4cb686d; if
   the host HEAD is newer, confirm lib/sched.py is unchanged around the
   three hunks — plan() else-branch, cmd_next tail, module regexes — or
   rebase the hunks).
2. In the host harness repo: `.venv/bin/python -m unittest
   tests.test_sched_human_only -v` -> 4/4 OK.
3. No restart needed (agent:473, pool.py:77 — the gate is live at the next
   poll). Live check while kevinruling/hostgate stay open: `next` reports
   them held (reason "no agent tasks", or waiting alongside a run), and no
   AGENT_TASK containing id kevinruling or hostgate is dispatched; an
   agent-doable line still dispatches.
4. Kevin checks off TODO:137; a later agent run records it.

## Stop when
Patch + tests are done and verified (above). The remaining step is
host-side and human-only by design: the dispatcher is not mounted in the
sandbox. If the host lib/sched.py diverged so much the hunks don't fit,
record the diff and stop; do not fork the scheduler.

## Plan (Kevin, host side)
1. One command, from the host (primary checkout; the agent worktree
   work/loopgym/w2 is transient):
   `~/Documents/coders/loop/projects/loopgym/tools/hostgate_apply.sh`
   (applies state/tickets/hostgate.patch to the harness checkout
   ~/Documents/coders/loop, runs the 4 regression tests, reports;
   idempotent — rerun to re-verify). Manual fallback: `git apply
   ~/Documents/coders/loop/projects/loopgym/state/tickets/hostgate.patch`
   then `.venv/bin/python -m unittest tests.test_sched_human_only -v`
   (4/4). Exit 3 = lib/sched.py diverged: diff the three hunk regions
   (module regexes, plan() else-branch, cmd_next tail) and rebase.
   Commit the lib/sched.py change in the harness repo afterwards so a
   reset cannot drop the gate.
2. No restart needed: the dispatcher spawns (agent:473) or reloads
   (pool.py:77) lib/sched.py every cycle — the gate is live at the next
   poll. Watch the next cycle: held lines reported (rc 10 "no agent
   tasks" / rc 11 "waiting"), no AGENT_TASK for kevinruling or hostgate,
   agent lines still dispatch.
3. Check off TODO:137 once observed. Same gate now protects all future
   `(human-only)` lines, incl. the Kevin-only Scale-out lines.
