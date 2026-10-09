# lgboxbenchcode

## Issue
No throughput/latency numbers for the two graders. A real RL run grades tens of
thousands of programs; we must know local vs lgbox grades/s and tail latency at
concurrency 1/4/16 before trusting lgbox for the training loop.

## Question
Does the lgbox http runner sustain the local runner's grades/s at concurrency 1/4/16,
with per-grade overhead < 50 ms at c1 (CLAUDE.md plausible range)? A cell whose
grades/s < local's at the same concurrency, or c1 mean overhead > 50 ms, refutes it.

## Evidence
- `results/env_audit.json` + `results/splits.json`: val split, all reference solutions
  score 1.0 on local (known fact from the audit).
- Smallest check: `tools/bench_lgbox.py --quick` (20 grades/cell) — real val exercises,
  real release binary, real server on a free port. Output: `results/lgbox_bench.json`,
  log `.agent-evidence/lgboxbench-quick.log`.
- Known-answer: every grade of a reference solution must score reward 1.0 in every cell
  (audit fact); the script exits 1 otherwise.

## Done when
- `.venv/bin/python tools/bench_lgbox.py --quick` exits 0, wall < 2 min, writes
  `results/lgbox_bench.json` with per-cell grades/s, p50/p95, reward counts.
- Regression: `make check` green (ruff F/E4/E7/E9 + pytest).

## Stop when
- Two failed --quick runs with no new evidence -> record blocker, stop.
- Out of range: c1 mean http−local overhead > 50 ms or any c grades/s < 1 -> record as
  suspected bug in NOTES.md with the JSON, do not tune past scope.
- Out of scope: no server config changes, no new lgbox endpoints, no full run (that is
  the next job).

## Plan
1. Script: load val split, cycle tasks over N grades; ThreadPoolExecutor(max_workers=c)
   per cell (c ∈ 1/4/16) × runner ∈ local/http; per-grade latency; p50/p95 nearest-rank.
2. Server: script spawns `lgbox serve --workers 16 --queue 64` on a free port (release
   binary, already built by the job), polls /health, kills on exit; `--url` reuses one.
3. Reward 1.0 gate + c1 overhead check in the JSON `checks` block.
