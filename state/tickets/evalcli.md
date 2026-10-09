# Eval CLI

## Issue
No single command scores a model/checkpoint on a task split. Baselines (base05, base15
tickets) and all later GRPO evals need one evaluator; each would otherwise re-implement
sampling, parsing, grading, and CI.

## Question
Can `python -m loopgym.eval.run` produce a resumable per-sample JSONL and a pass@1 with
task-clustered CI for any model/adapter on exercism val? Refutation: reference solutions
(not 1.0) or a 2-task 0.5B run > 5 min / non-zero exit.

## Evidence
- `results/splits.json`: val = 9 usable exercises (ref=1.0, stub=0.0 each, per env_audit).
- `loopgym/env/exercism.py`: Task.prompt/parse/reference, grade(task, files, "local").
- `loopgym/rl/rollout.py` HFBackend; `loopgym/eval/passk.py` summarize (task-clustered CI).
- Smallest check: `--reference --source exercism --split val` must print pass@1 1.000
  (graders known answer: every usable reference scores 1). Log: .agent-evidence/evalcli-ref.log.

## Done when
1. `.venv/bin/python -m loopgym.eval.run --reference --source exercism --split val` prints `pass@1 1.000`.
2. `... --model models/qwen2.5-coder-0.5b-instruct --source exercism --split val --limit 2 --n 1 --out /tmp/e.json`
   exits 0 in < 5 min, writes JSON + sidecar JSONL; rerun appends nothing (resume).
Known answer for #1: 1.000 (hand-known: reference of every usable exercise passes its own tests).
Plausible: 2-task model pass@1 in {0.000, 0.500, 1.000} (2 tasks, n=1).
Regression: tests/test_eval_run.py (reference limit-2 pass@1 == 1.0; resume no-op).

## Stop when
Out of scope: vLLM backend, loopgym task source (train.py still NotImplementedError —
no env loader for mined tasks), lgbox runner, GPU. Stop/split after 2 failed attempts
without new evidence.

## Plan
1. `loopgym/eval/run.py`: argparse CLI, exercism split loading, HFBackend sampling
   (seed per task+sample), parse, local grade, sidecar JSONL append, passk.summarize,
   JSON out, `pass@1 x.xxx` print. `--source loopgym` -> NotImplementedError.
2. tests/test_eval_run.py: reference limit-2 == 1.0; resume idempotent.
3. Run both acceptance commands under capture.py; run make check.
