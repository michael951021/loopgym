# stats — Bootstrap CIs and pass@k

## Issue
Every reported metric needs a CI over tasks; draftlab's CIs were ~100x too narrow because they
resampled individual samples, not tasks. No `loopgym/eval/` module exists yet.

## Question
Does `bootstrap_ci` resample groups (task) as a unit — i.e. does 10 tasks × 20 identical samples
give a CI as wide as 10 samples, not 200? Refuted if the clustered width ≈ naive width.

## Evidence
- Starting point: `tests/known/test_stats.py` (Kevin's 4 tests, currently ImportError).
- Check: `.venv/bin/python -m pytest -q tests/known/test_stats.py` → fails on import; save a
  quick run under .agent-evidence. Guess: none — math is pinned by known tests.

## Done when
- `.venv/bin/python -m pytest -q tests/known/test_stats.py tests/test_passk.py` passes.
- `pass_at_k` known answers (hand-computed): pass_at_k(5,2,1)=0.4, (5,2,2)=0.7, (5,0,k)=0,
  (5,5,k)=1; product form equals math.comb reference for n≤30 random (n,c,k).
- Plausible range (before run): binomial p=0.5, n=100 CI half-width 0.07–0.13 (1.96·0.05≈0.098);
  width ratio for n=50 vs 800 in (2.8, 5.2) around sqrt(16)=4.

## Stop when
Scope: only `loopgym/eval/{__init__,stats,passk}.py` + `tests/test_passk.py`. No other eval
modules (train/rollout come later). Two failed attempts with no new evidence → record and stop.

## Plan
1. Write failing check (known test), run it.
2. Implement `bootstrap_ci` (vectorized group resample) and `pass_at_k` (product form, no
   factorial overflow).
3. Implement `summarize(records, k)` → per-task pass@k + task-clustered bootstrap CI.
4. Run known + new tests, make check, commit.
