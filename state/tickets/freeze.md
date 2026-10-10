# Freeze loopgym v0

## Issue
Evals and REPORT.md currently read loopgym tasks off the live mining state
(`data/loopgym/verified.jsonl` + `load_tasks()`), which grows as vb tickets land.
The report needs a *fixed, documented* dataset version: which tasks, from which
repos, under which verification rule, with what split and counts. Without a
freeze, "loopgym baseline" numbers are not reproducible.

## Question
Can we freeze exactly the set of verified, representable loopgym tasks into
`data/loopgym/v0/` such that a re-derivation (`--check`) proves the frozen
files match the current verified state, the test split is >= 30% by sha1(id),
and the kill criterion (< 25 tasks -> eval-only) is decided and documented?
Refuting observation: `--check` exit != 0, or test share < 30%, or a frozen
task absent from verified.jsonl.

## Evidence
- `data/loopgym/verified.jsonl`: 41 unique records, all verdict=verified
  (draftlab 36, harness 5); `load_tasks()` currently yields 26 representable
  tasks -> above the 25-task kill threshold (observed 2026-10-10).
- Split h=int(sha1(id),16)%100, test iff h>=70: 13 train / 13 test on the
  current 26 (observed).
- Candidate tickets mentioning apply/contrib-loop (harness-*-16ce269) are not
  among the 26 loadable tasks (observed); no task in the frozen set has
  provenance outside the public draftlab/harness repos.
- Acceptance: `pytest -q tests/test_loopgym_env.py` (baseline log
  `.agent-evidence/freeze-baseline-env.log`) and
  `python -m loopgym.data.freeze --check` (log `.agent-evidence/freeze-check.log`).

## Done when
1. `data/loopgym/v0/tasks.jsonl`: one record per verified representable task
   (id, repo, bad, good, split, kind, fix_msg, files, diff_lines).
2. `data/loopgym/v0/checks/<id>/test_*.py` == `tests/hidden/<id>/` byte-for-byte.
3. `results/loopgym_v0.json`: n_tasks, by_repo, by_kind, split sizes,
   test_frac, mean_diff_lines, kill-criterion verdict. Known answers: split
   rule pinned by a hand-computed sha1 case in tests/test_freeze.py; counts
   re-derived by `--check` (independent of the written files).
4. `data/loopgym/v0/DATASET.md`: source, verification rule (0/3 bad, 3/3 good),
   redaction (draftlab + harness public; nothing from apply/contrib-loop),
   kill-criterion outcome (25+ -> training + eval; else eval-only).
5. `.venv/bin/python -m pytest -q tests/test_loopgym_env.py &&
   .venv/bin/python -m loopgym.data.freeze --check` exits 0.

## Stop when
No GPU/model work; do not re-verify candidates (vb1-3 already done) and do not
touch tests/known/. If < 25 verified tasks, do not manufacture tasks: mark
eval-only in DATASET.md and stop. Two failed `--check` runs with no new
evidence -> record and stop.

## Plan
1. Baseline env tests (background).
2. `loopgym/data/freeze.py`: split_of/kind_of/diff stats, build() to
   data/loopgym/v0 + results, check() re-derives and diffs; `--check` CLI.
3. Build, run --check, add tests/test_freeze.py (hand-computed split case,
   tmp build, redaction strings in DATASET.md).
4. NOTES.md, commit, merge master.
