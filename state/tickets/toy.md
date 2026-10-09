# toy — Toy rewards for known answers

## Issue
Before any real GRPO run, the trainer must prove it learns a planted reward and ignores a null
reward (CLAUDE.md known answers). `loopgym/env/toy.py` does not exist; the trainer has no toy
task source, so the planted-effect / null-control check (the gate on all training runs) cannot run.

## Question
Can we define `banana` and `coinflip` behind the same Task/grade interface as exercism so the
trainer cannot tell them apart: banana = 1 iff "banana" (case-insensitive) is in the first 20
completion tokens; coinflip = 1.0 with prob 0.5 from an RNG seeded by (step, index), text-independent?
Refuted if: banana("I like Banana") != 1 or banana("apple") != 0, coinflip 10k mean outside
0.5 ± 0.02, or the same (step, index) scores differently for different texts.

## Evidence
- Interface to match: loopgym/env/exercism.py:22-54 (Task: prompt/parse/stub/reference), :124 (grade → (reward, info)).
- Tokenizer (shared by all three models): models/qwen2.5-coder-0.5b-instruct (loopgym/models.py:17).
- Check: tests/test_toy.py (new). First run: `.venv/bin/python -m pytest -q tests/test_toy.py`
  expected to fail with ModuleNotFoundError (module missing) — observed facts vs guesses: the
  expected values are hand-derived, not measured.

## Done when
`.venv/bin/python -m pytest -q tests/test_toy.py` passes: banana("I like Banana") = 1,
banana("apple") = 0, case-insensitive; "fig "*21 + "banana" = 0 (beyond the 20-token window);
coinflip 10k mean in 0.5 ± 0.02, deterministic per (step, index), independent of text;
`env.grade` dispatches ToyTask with the trainer's call shape (runner/timeout/step/index kwargs).
Known answers are hand-computed; no measured numbers. `make check` stays green (env.grade
gains kwargs with defaults, so the known exercism tests still pass unchanged).

## Stop when
Scope: loopgym/env/toy.py, the dispatch in loopgym/env/__init__.py, tests/test_toy.py, NOTES.md.
No trainer, configs, or runs. After 2 attempts without new evidence (e.g. tokenizer fails to
load in the sandbox), record the blocker in NOTES.md and stop.

## Plan
1. Write tests/test_toy.py; run it (expect ModuleNotFoundError).
2. Implement loopgym/env/toy.py (ToyTask, banana, coinflip, grade, load_toy); make env.grade
   dispatch on ToyTask; keep the exercism path untouched.
3. Run tests/test_toy.py, then `make check`; update NOTES.md; commit.
