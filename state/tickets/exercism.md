# Exercism tasks and local grader

## Issue
No code can turn an Exercism exercise into an RL task or score a reply: `loopgym/env/`
does not exist (`ImportError: cannot import name 'env' from 'loopgym`). Without it there
is no reward signal at all and the known-answer reward tests cannot run.

## Question
Does `env.grade` return 1.0 exactly when the pristine hidden tests pass on a fresh copy
containing only the model's solution file, and 0.0 for the planted hacks (exit at import,
os._exit, test overwrite, infinite loop, module-level skip)? Refutation: any hack scoring
1.0, or reference/scoring != 1.0 / stub != 0.0.

## Evidence
- tests/known/test_reward.py (Kevin, read-only) names the exact API.
- Acceptance: `LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_reward.py`
  — currently `ImportError` (observed 2026-10-09); log .agent-evidence/exercism-accept1.log.
- Exercism layout verified: `.meta/config.json` lists solution/test/example; paasio has 2
  test files; some `.docs/` have instructions.append.md (observed).

## Done when
Acceptance command exits 0 (all 33 tests: 3 names x ref/stub, 5 hacks, conftest-exclusion).
Known answers: reference solutions score 1, stubs 0, hacks 0 — by construction, hand-
verifiable per exercise (leap: is_leap_year(2000)=True). Plausible: each local grade < 30 s.

## Stop when
Scope: local runner only (no http/lgbox, no splits, no toy rewards). If a hack cannot be
made to score 0 without changing tests/known/ (forbidden), record why in NOTES.md and stop.
Two attempts without new evidence → record blocker, stop or split.

## Plan
1. `loopgym/env/exercism.py`: Task (stub/reference/prompt/parse) + load_exercise from
   `.meta/config.json`.
2. Local runner: fresh tempdir, pristine test file(s) + only the solution file,
   `python -m pytest -q -p no:cacheprovider --junitxml=r.xml` in a new process group,
   SIGKILL group on timeout, reward from XML (tests>0, no failures/errors/skips).
3. `__init__.py` exports load_exercise/grade/runner_names/RUNNERS; run acceptance,
   `make check`, commit.
