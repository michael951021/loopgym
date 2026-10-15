# grpo05ok — Analyse the CPU GRPO run

## Issue
The 40-step 0.5B GRPO run on Exercism (`runs/grpo05_exercism`) and its val
evals exist, but the result is not yet a claim: no CI on the training-reward
curve, no paired grpo05-vs-base05 val comparison, and no record of the
signal-free group fraction that determines whether GRPO had anything to learn.

## Question
Did the grpo05 run lift val pass@1 over base05, and what did the run's own
metrics say about learning signal? A refuting observation for "it learned":
a task-clustered CI on the paired val delta that excludes 0 upward, or a
reward-hack pattern in the completions behind any apparent gain.

## Evidence
- `runs/grpo05_exercism/metrics.jsonl` (40 train rows + 2 eval rows).
  Observed: reward_mean 0.31→~0.06 (slope -0.0031/step, CI [-0.0056, -0.0001]
  excl. 0); kl max 4.1e-4; clip_frac 0 everywhere; frac_groups_all_same 0.675.
- `results/eval/{base05,grpo05}_val.json`: 9 identical tasks, n=16 vs n=4.
  Observed: 0.0764 vs 0.1111; hand delta (3 tasks up 0.25/0.0625→0.25, 0.3125→0.5;
  isogram 0.25→0) = +0.0347; task-cluster bootstrap CI [-0.056, +0.125] → contains 0.
- Hack check: `tools/grpo05_hackcheck.py` regenerates the first 20 grpo05 val
  completions at their eval seeds, grades them, saves text+grade info to
  `results/grpo05_hackcheck.json` (output also logged under .agent-evidence/).

## Done when
`.venv/bin/python tools/check_results.py --only grpo05` exits 0: new `grpo05`
section prints slope CI, paired val delta CI, KL/clip trends,
frac_groups_all_same, hack-screen rows, and writes `results/grpo05_summary.json`.
Known answers: paired delta recomputes by hand from per-task counts (+0.0347);
bootstrap CI uses the same resample-as-task-units method as
`loopgym/eval.stats.bootstrap_ci` (known-test-pinned); hack check rewards must
equal the sidecar rewards (determinism). Pre-run plausible ranges: delta point
estimate in [-0.5, 0.5], CI half-width < 0.5; slope in [-0.05, +0.05]/step;
last-10-step mean kl < 0.5; clip_frac in [0, 1]; 20/20 sidecar matches.
Null (CI contains 0) is a passing result; a gain whose CI excludes 0 fails iff
the hack screen finds reward-hack-like outputs; a significant reward drop or
kl explosion also fails.

## Stop when
Scope: no re-training, no GPU, no new evals beyond the 20 hack-check
completions, no changes to tests/known/ or ranges. Stop/split: if the hack
check cannot reproduce the sidecar rewards (non-determinism beyond suspicion)
or the paired comparison is impossible (task sets differ), record evidence in
NOTES.md and stop. Two attempts at any check without new evidence → stop.

## Plan
1. Background: hack-check generation (20 completions, ~5-10 min).
2. `tools/check_results.py`: `section_grpo05` + summary writer + tests.
3. Review hack-check text; record verdict in NOTES.md.
4. `make check`, NOTES lines, commit.
