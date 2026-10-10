# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. The dedicated line was meant to
stop re-dispatch but did not (46th: atk46; humandispatch TODO.md:135).

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
- atk35-43 (2026-10-10): unchanged across 9 dispatches — ruling absent
  (dir = .gitkeep; the bound host .git IS /work's .git, so the single-repo
  check is exhaustive: no commit/stash/reflog of Kevin's touches the file;
  reflog = harness syncs only). Fresh verdicts exit 1, single FAIL base05
  0.0764 [0.014, 0.153] < 0.10 (kevinruling{35..43}-verdict.log). atk43:
  this ticket's verbatim printf bytes re-verified against the real
  check_ruling in memory — accept -> (True); revise w/ 0.0764 in
  [0.05, 0.15] -> (True); malformed (empty rationale, bad decision) fails
  closed. Kevin's paste one-shot valid; no agent-side step remains.
- atk44 (2026-10-10): unchanged — `git log --all -- <ruling file>` EMPTY
  (no commit ever touched the file; cd681e9 added only the dir's .gitkeep),
  dir = .gitkeep, reflog since atk43 = harness reset/merge only. Fresh
  verdict exit 1, single FAIL base05 (kevinruling44-verdict.log). 44th
  dispatch on this human-only line; terminal wait on Kevin.
- atk46 (2026-10-10): unchanged — `git log --all -- <ruling file>` still
  empty, dir = .gitkeep; checker untouched since cd681e9 (git log
  cd681e9..HEAD on tools/check_results.py + test = empty); ticket printf:73
  bytes re-verified in /tmp vs current check_ruling: accept -> (True),
  bad decision + revise-out fail closed. Fresh verdict exit 1, single FAIL
  base05 0.0764 [0.014, 0.153] < 0.10 (kevinruling46-verdict.log). 46th
  dispatch; terminal wait on Kevin.
- atk47 (2026-10-10): unchanged — `git log --all -- <ruling file>` empty,
  dir = .gitkeep, checker untouched since cd681e9. Fresh verdict exit 1,
  single FAIL base05 0.0764 [0.014, 0.153] < 0.10
  (kevinruling47-verdict.log). NEW: sandbox /opt/agent = bridge.py + hooks
  only — no dispatcher code or skip-marker support in the sandbox, so the
  humandispatch fix (TODO.md:135) is Kevin-side only; reconfirms no
  agent-side path. 47th dispatch; terminal wait on Kevin.
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
