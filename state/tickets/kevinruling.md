# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. The dedicated line was meant to
stop re-dispatch but did not (39th dispatch: atk39; humandispatch TODO.md:135).

## Question
Does the range author accept 0.0764 (task-clustered CI straddles the 0.10 floor)
or revise the pre-run range? Open iff results/rulings/base05_val.ruling.json is
absent; a committed ruling closes it.

## Evidence (all observed, no guesses)
- atk13-28 (2026-10-10): ruling absent on disk (results/rulings/ = .gitkeep
  only), all refs (only cd681e9 touched the path), no stash, no remote;
  checker unchanged since cd681e9 (check_ruling, tools/check_results.py:61 —
  schema only; author VALUE unchecked, identity by rule 5 + repo access);
  verdict exit 1 each run, single FAIL base05
  (.agent-evidence/kevinruling{13..28}-verdict.log). Ruling channel E2E on a
  copy of real results/ (atk5/19/27): none->exit 1, verbatim accept->exit 0
  all rows PASS, revise w/ 0.0764 in range->exit 0, out of range->exit 1
  (fail closed) (kevinruling{5,19,27}-accept-e2e.log).
- atk29-31 (2026-10-10): still absent (both repos: no ref/stash/remote).
  Kevin's checkout (projects/loopgym) worktree WIPED (all tracked files
  deleted, results/rulings/ missing): Plan step 0 (git restore .) required
  before the paste; his HEAD harness-synced 151566f->a9ea865->2468727(=w2).
  Verdict exit 1 each run (kevinruling{29,30,31}-verdict.log).
- atk32-34 (2026-10-10): unchanged — ruling absent on disk, all refs, stashes
  in /work and Kevin's repo; his checkout HEAD 2da001b->d87b5f2->969029c (w2
  HEAD), tree still wiped (181 deletions); fresh verdicts exit 1, single FAIL
  base05 0.0764 [0.014, 0.153] < 0.10 (kevinruling{32..34}-verdict.log).
  No agent-side path exists (rule 5).
- atk35 (2026-10-10): ruling still absent, now via a LIVE view of the host
  .git (bind mount): no commit in any ref/stash/reflog touches
  results/rulings/base05_val.ruling.json; reflog = harness fast-forwards
  only (Kevin has made no commits). CORRECTION: "tree wiped (181 del)" atk29-34
  was a mount artifact — the sandbox binds only his .git (his worktree dir
  here holds only .git), and 181 == /work's tracked-file count; the host
  worktree is invisible from the sandbox, so no damage can be inferred.
  Fresh verdict exit 1, unchanged rows (kevinruling35-verdict.log); the
  verbatim accept JSON passes the real check_ruling in memory (malformed
  fails closed).
- atk36-37 (2026-10-10): unchanged — ruling absent (results/rulings/ =
  .gitkeep only; no ref/stash/commit in the bound host .git touches the path;
  HEAD 6ca3875 = harness sync, no Kevin commits). Fresh verdicts exit 1,
  same single FAIL base05 (kevinruling3{6,7}-verdict.log). Nothing
  agent-side can change; 37th dispatch on this human-only line.
- atk38-39 (2026-10-10): absent via full author enumeration — Kevin's only
  5 commits (2026-10-09 bootstrap 7442a3e, 46fa73f, 0a84071, 80a2e1d,
  8506b8f) touch no ruling path; `git log --all -- <ruling>` empty; dir =
  .gitkeep. Fresh verdicts exit 1, same single FAIL base05
  (kevinruling{38,39}-verdict.log).
- atk40 (2026-10-10): unchanged — absent (dir = .gitkeep;
  `git log --all -- <ruling>` = cd681e9 only, the agent commit that created
  the path). Fresh verdict exit 1, single FAIL base05
  0.0764 [0.014, 0.153] < 0.10 (kevinruling40-verdict.log). 40th dispatch on
  this human-only line; root-cause ticket (humandispatch, Kevin-only) stands
  at TODO.md:135; no agent-side path exists (rule 5).
- Raw sidecar recompute exact (11/144 = 0.076389, 0 timeouts): number is real.
- Ten pipeline causes ruled out atk1-12 (.agent-evidence/basecheck*.log).

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
1. Kevin pastes (accept; E2E re-verified exit 0 atk27 — kevinruling27-accept-e2e.log):
   ```sh
   printf %s '{"author": "kevin", "decision": "accept", "rationale": "task-clustered CI [0.014, 0.153] straddles the 0.10 floor; ten pipeline causes ruled out (atk1-10)"}' > results/rulings/base05_val.ruling.json
   git add results/rulings/base05_val.ruling.json && git commit -m "ruling: accept base05 0.0764 (CI straddles 0.10 floor)"
   ```
   (revise form: see Done-when — new_range [lo,hi] with 0.0764 inside.)
2. Next agent run (rangerule): checker `--only baselines` → expect exit 0,
   record verdict in NOTES.md, unblock basecheck.
