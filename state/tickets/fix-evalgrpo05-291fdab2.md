# Fix evalgrpo05 crash: ValueError in run.py

## Issue
Compute job `evalgrpo05` gave up after its retries with the same crash each time, so every ticket that waits on its
results is blocked. The failure is a Python traceback — a code bug, not a resource limit — so it is fixable here.
Auto-filed from the job log; signature `ValueError@run.py:45`.

## Evidence
Exception: `ValueError: Can't find 'adapter_config.json' at 'ckpt/grpo05_exercism/last'`
Raised through (deepest project frame last):
  <frozen runpy>:198 in _run_module_as_main
  <frozen runpy>:88 in _run_code
  run.py:184 in <module>
  run.py:154 in main
  run.py:45 in load_model
  peft_model.py:492 in from_pretrained
  config.py:327 in _get_peft_type
Fix the deepest project frame: **run.py:45** in `load_model`.
Failing command: `OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.eval.run --model models/qwen2.5-coder-0.5b-instruct --adapter ckpt/grpo05_exercism/last --source exercism --split val --n 4 --out results/eval/grpo05_val.json`
Full log: /home/kevin/Documents/coders/loop/logs/jobs/loopgym-evalgrpo05.log

## Done when
The crash no longer reproduces and a fast test covers the fixed path (so it cannot slip back into a multi-hour
run). Reproduce cheaply FIRST — do not launch the real job to test: drive just the failing path with a tiny
config / small input, or the module's smoke entrypoint, so one iteration is seconds not hours. Add a regression
test in the fast/smoke tier that exercises the SAME code path and config shape the real job takes, then let the
job be retried.

## Stop when
After two attempts with no new evidence, write the blocker in the ticket and stop — the signature then escalates
to a human card. Do not loosen or skip a check to make the crash disappear.

## Plan
1. Reproduce `ValueError` at run.py:45 with the smallest/fastest invocation.
2. Fix the cause; add the fast regression test.
3. Make sure the fix reaches the job's branch so its retry actually picks it up.
