# kevinruling — Range-author ruling: base05 floor (HUMAN-ONLY)

## Issue
base05 val pass@1 0.0764 [0.014, 0.153] is below the pre-run 0.10 floor. Rule 5:
only the range author (Kevin) may accept the number or revise the range — agents
never write rulings. Until this lands, basecheck stays exit 1 and everything
gated after it (grpocfg, grpo05) is blocked. The dedicated line was meant to
stop re-dispatch but did not (58th: atk58; humandispatch TODO.md:135).

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
- atk48 (2026-10-10): unchanged — `git log --all -- <ruling file>` empty,
  dir = .gitkeep, checker untouched since cd681e9 (git log cd681e9..HEAD on
  checker + test empty). Fresh verdict exit 1, single FAIL base05 0.0764
  [0.014, 0.153] < 0.10 (kevinruling48-verdict.log). Full bwrap mount audit
  (parent process args): writable = w2 worktree + shared project dirs
  (data/models/ckpt/runs/.tools/.venv) + host .git + /home/agent; read-only
  = bridge.py, mergelines.py, hooks, mcp, prompts, gh_token. No dispatcher
  code, config or state inside the sandbox; the task arrives via a host-set
  AGENT_TASK env. Zero agent-side levers to stop re-dispatch — the
  humandispatch fix (TODO.md:135) is Kevin-side only. 48th dispatch;
  terminal wait on Kevin.
- atk49 (2026-10-10): unchanged — `git log --all -- <ruling file>` empty,
  dir = .gitkeep, reflog/stash since atk48 = harness syncs only. /opt/agent
  inspected: bridge.py is an Ollama unix-socket forwarder only — no
  notification or dispatcher channel in the sandbox. Fresh verdict exit 1,
  single FAIL base05 0.0764 [0.014, 0.153] < 0.10
  (kevinruling49-verdict.log). 49th dispatch; terminal wait on Kevin.
- atk50 (2026-10-10): unchanged — `git log --all -- <ruling file>` empty,
  no stash, dir = .gitkeep; ticket's one-shot paste intact and E2E-verified.
  Fresh verdict exit 1, single FAIL base05 0.0764 [0.014, 0.153] < 0.10
  (kevinruling50-verdict.log); checker regression tests 11/11 pass
  (kevinruling50-checker-tests.log) — ruling channel ready. 50th dispatch;
  terminal wait on Kevin.
- atk56 (2026-10-10): ruling still absent (dir = .gitkeep; no commit/stash;
  ListAgents = no live peer to message Kevin). Fresh this run: real verdict
  exit 1, single base05 FAIL 0.0764 [0.014,0.153] <0.10
  (kevinruling56-verdict.log); E2E accept ruling on a copy of real results/
  flips the checker to exit 0 (kevinruling56-accept-e2e.log); checker tests
  11/11 (kevinruling56-checker-tests.log). No dispatcher code in repo (lib/
  absent, no AGENT_TASK handler) — the re-dispatch lever is host-side; the
  "kevin:" title prefix is the ready marker for the humandispatch fix
  (TODO.md:135, Kevin-side). 56th dispatch; terminal wait on Kevin.
- w1 atk4 (2026-10-10): unchanged — ruling never committed
  (for-each-ref newest = harness commits; no stash/remote), Kevin's checkout
  HEAD synced to master, tree view = known mount artifact (only his .git
  bound). No live peer (ListAgents); /opt/agent re-audited: prompts =
  agent-run prompts only, line selection is host-side. Gate exit 1; fresh
  verdict exit 1, single base05 FAIL 0.0764 [0.014, 0.153] < 0.10, other
  rows PASS (kevinruling-w1a4-verdict.log). Terminal wait on Kevin.
- atk57 (2026-10-10): unchanged — ruling never committed
  (`git log --all -- <file>` empty), dir = .gitkeep, no stash/remote,
  master not ahead of w2; ListAgents: no live peer. Fresh real verdict
  exit 1, single base05 FAIL 0.0764 [0.014,0.153] <0.10, all other
  baseline rows PASS (kevinruling57-verdict.log). 57th dispatch;
  terminal wait on Kevin.
- atk58 (2026-10-10): unchanged — ruling never committed
  (`git log --all -- <file>` empty; dir = .gitkeep; no stash/remote; master
  = w2 HEAD da7a0ca); ListAgents: no live peer. Checker+test unchanged since
  cd681e9 (git log cd681e9..HEAD empty). Fresh real verdict exit 1, single
  base05 FAIL 0.0764 [0.014,0.153] <0.10, other rows PASS
  (kevinruling58-verdict.log). 58th dispatch; terminal wait on Kevin.
- atk59-60 (2026-10-10): unchanged — ruling still absent (dir = .gitkeep;
  `git log --all -- <ruling file>` = checker-mechanism commit cd681e9 only;
  no stash; master == w2 HEAD; ListAgents = no live peer). Checker+tests
  unchanged since cd681e9. Fresh verdict exit 1, single base05 FAIL
  0.0764 [0.014, 0.153] < 0.10, other rows PASS
  (kevinruling59/60-verdict.log). 60th dispatch; terminal wait on Kevin.
- atk61 (2026-10-10): unchanged — ruling absent on all refs/stashes, no live
  peer (ListAgents), checker+results unchanged since cd681e9 → no re-run of
  the deterministic verdict (identical inputs; Stop-when forbids re-runs).
  One-shot paste verified intact at :136-139 (earlier line refs 114-115/131
  stale — file grew). Root-cause ticket (humandispatch) already queued at
  TODO.md:135. 61st dispatch; terminal wait on Kevin.
- atk72 (2026-10-10): unchanged — ruling never committed
  (`git log --all -- <file>` empty), dir = .gitkeep, checker+tests unchanged
  since cd681e9. Fresh verdict exit 1, single base05 FAIL 0.0764
  [0.014, 0.153] < 0.10 (kevinruling72-verdict.log). NEW: the CURRENT file's
  paste (line 143; file grew past atk61's :136-139) re-verified E2E on a copy
  of real results/ — exact bytes -> checker exit 0, all rows PASS, row cites
  the ruling (kevinruling72-e2e-accept.log); checker tests 11/11
  (kevinruling72-checker-tests.log). Paste valid as of this file state; no
  agent-side step remains. 72nd dispatch; terminal wait on Kevin.
- w1 2026-10-10 (parallel dispatch with w2 atk73): unchanged — ruling absent
  on all refs (git log --all = cd681e9 mechanism only; no stash/remote).
  Gate now shipped: sched_guard.py --line kevinruling -> exit 1 (skip until
  the ruling file exists); host dispatcher still dispatching -> wiring is
  Kevin-side (humandispatch line). Fresh verdict exit 1, single base05 FAIL
  0.0764 [0.014, 0.153] < 0.10, other rows PASS. Paste intact at :152.
  Terminal wait on Kevin.
- Raw sidecar recompute exact (11/144 = 0.076389, 0 timeouts): number is real.
- Ten pipeline causes ruled out atk1-12 (.agent-evidence/basecheck*.log).
- w1 atk5 (2026-10-10): unchanged — ruling absent on all refs/stash/remotes
  (master == 796f533, no fetch delta); ListAgents: no live peer (same as atk4).
  Gate exit 1 (skip); fresh verdict exit 1, single base05 FAIL 0.0764
  [0.014, 0.153] < 0.10, other rows PASS (kevinruling-w1a5-verdict.log).
  humandispatch ticket (TODO:135) already [x]: in-repo gate works, only the
  host-side dispatcher wiring is missing (Kevin's). Paste intact. Terminal
  wait on Kevin.
- w1 atk9 (2026-10-10): unchanged — ruling never committed (for-each-ref:
  no ref beyond harness/job commits; no stash/remote; master = w1 HEAD);
  ListAgents: no live peer; results/rulings/ empty. Fresh verdict exit 1,
  single base05 FAIL 0.0764 [0.014, 0.153] < 0.10, other rows PASS
  (kevinruling-w1a9-verdict.log). Paste intact at :173-174. No agent-side
  step remains (rule 5); re-dispatch root cause ticketed (hostgate,
  TODO:137). Terminal wait on Kevin.
- w1 atk10 (2026-10-10): unchanged — ruling absent everywhere reachable.
  NEW (different approach): checked Kevin's host checkout directly —
  /work/.git is a gitfile into /home/kevin/.../projects/loopgym/.git/
  worktrees/w1: SAME object store (15 refs / 379 commits both sides, no
  stash), so no Kevin-side private state exists and the single-repo
  absence check is exhaustive. His worktree files are not bound into the
  sandbox (only his .git: status shows all files deleted); find over
  /home /work /tmp /var/tmp finds no ruling file. No live peer
  (ListAgents). Gate exit 1; fresh verdict exit 1, single base05 FAIL
  0.0764 [0.014,0.153] <0.10, other rows PASS
  (kevinruling-w1a10-verdict.log). Paste now at :180-181 (file grew).
  10th w1 dispatch; terminal wait on Kevin.

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
