# Fix cleanclone: non-zero exit (exit-code)

## Issue
The auto-queued job `cleanclone` (TODO.md:167) clones /work to /tmp/lgcc and runs
`make check`; it exits non-zero, so the repo's claim "a fresh clone passes the full
check" is false. Clean clone is the reproducibility gate for the whole stack.

## Question
Why does `make check` fail in a git-only clone, and can it exit 0 there?
Hypothesis: the Makefile hardcodes `$(CURDIR)/.venv/bin/python` (gitignored, absent
in a clone) and several tests load gitignored assets (`.tools/src`, `models/`,
`data/loopgym`) with no skip or fetch. Refutation: a reproduced clone run that
fails for a different reason (e.g. network, disk).

## Evidence
- Makefile:2 `PY := $(CURDIR)/.venv/bin/python`; `.venv/`, `.tools/` are gitignored.
- bootstrap.sh:36 `find .venv -mindepth 1 -delete` → fails when `.venv` dir absent.
- tests/test_vllm_api.py:18 `next(...glob("vllm-*"))` → StopIteration at import
  when `.tools/src` missing (collection error).
- tests/test_loopgym_env.py:11 `TASKS = lg.load_tasks()` at import →
  FileNotFoundError on `data/loopgym/verified.jsonl` (loopgym/env/loopgym.py:308).
- tests/test_rollout.py:17 loads `models/qwen2.5-coder-0.5b-instruct`, no skip.
- tests/test_freeze.py:30-45 read `data/loopgym/v0` + results, no skip.
- fetch_src.sh fetches *latest* PyPI versions, ignoring the pins in
  tools/versions.txt → would rewrite the committed pins in a clone.
- Reproduction (observed, this run): job log is host-side; local clone reproduced at
  `.agent-evidence/cleanclone-*.log`. Facts vs guesses: the .venv absence is certain
  (gitignore); the rest were inferred from code and must be confirmed by the run.
- Shared-tree regression was red on exactly 1 test:
  `test_reference_scores_1[draftlab-7074368-07efa48]` — its hidden check
  `import pandas` (ModuleNotFoundError), then, with pandas installed, failed on a
  missing parquet engine (`to_parquet`/`read_parquet`). Import scan of all of
  tests/hidden: pandas was the only third-party module absent from the venv;
  `pyarrow` is the pandas parquet engine. Mined code (`draftlab/sweep.py`) also
  imports pandas, so it belongs in the `draftlab` extra (comment at pyproject:23).
  `sweep.py`'s `import ray` is never imported by the check -> not added (scope).
  Of the 5 hidden dirs that import pandas, only this task is loaded (26 verified);
  the others are unrepresented candidates, never graded in the suite.
- Fix (scope add, recorded here): pyproject `draftlab` += pandas, pyarrow;
  `uv lock` + `uv sync --extra draftlab` (uv.lock was stale vs the extra);
  bootstrap smoke import += pandas, pyarrow.
- test_loopgym_env.py 106/106 in 173s (.agent-evidence/loopgym-env-full.log).

## Done when
Exact job command exits 0:
`rm -rf /tmp/lgcc && git clone -q /work /tmp/lgcc && cd /tmp/lgcc && PATH=/work/.tools/bin:/work/.tools/cargo/bin:/work/.venv/bin:$PATH make check`
- Expected: bootstrap self-installs toolchain + pinned sources; ruff, pytest
  (tests/known+hidden ignored), cargo test, shellcheck, hadolint, terraform
  validate all pass; tests needing `models/` or `data/` skip with a reason.
- Known-answer checks (already in-suite, run on the real pipeline in the clone):
  toy null control, GRPO hand cases, SHA-1 split vector, reference-solution reward
  (where data present), rollout vs trainer logp 1e-4 (where model present).
- Plausible: full run 10–60 min (downloads + CPU suite) — inside the job's 2 h.
- Regression: `make check` still exits 0 in the shared tree (run before finishing).

## Stop when
Scope (final): Makefile, tools/bootstrap.sh, tools/fetch_src.sh, skip logic in
tests/test_{vllm_api,rollout,loopgym_env,freeze,eval_run,mine,toy,train,
train_dist,train_vllm,gpu_run}.py (all the same absent-asset convention),
NOTES.md condensation (<4 KB, required by the harness). No changes to
tests/known, grading, RL code, or the cleanclone job line itself.
Stop/split after two reproduction attempts without new evidence; record blocker in
NOTES.md. (Each of the 3 clone attempts here found new evidence: 25 -> 1 -> 0
failures.)

## Plan
1. Makefile: `check: bootstrap` (bootstrap is idempotent).
2. bootstrap.sh: `mkdir -p .venv` before the find; call fetch_src.sh.
3. fetch_src.sh: honor `name=version` pins from tools/versions.txt; latest only when unpinned.
4. Add skipif (absent-asset) guards to the four test files above, matching the
   test_exercism.py convention.
5. Reproduce the job command verbatim; iterate until exit 0; re-run shared-tree check.
6. draftlab extra += pandas+pyarrow (lock, sync, smoke import); rerun the clone
   job command on the committed tree; final shared-tree `make check`.
