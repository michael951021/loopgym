# Fix gpurun1 crash: RuntimeError in rollout_vllm.py

## Issue
Compute job `gpurun1` gave up after its retries with the same crash each time, so every ticket that waits on its
results is blocked. The failure is a Python traceback — a code bug, not a resource limit — so it is fixable here.
Auto-filed from the job log; signature `RuntimeError@rollout_vllm.py:42`.

## Evidence
Exception: `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
Raised through (deepest project frame last):
  llm_engine.py:111 in __init__
  core_client.py:126 in make_client
  otel.py:175 in sync_wrapper
  core_client.py:917 in __init__
  core_client.py:674 in __init__
  contextlib.py:144 in __exit__
  utils.py:1268 in launch_core_engines
  utils.py:1348 in wait_for_engine_startup
Fix the deepest project frame: **rollout_vllm.py:42** in `build_llm`.
Failing command: `bash scripts/gpu_run.sh`
Full log: /home/kevin/Documents/coders/loop/logs/jobs/loopgym-gpurun1.log

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
1. Reproduce `RuntimeError` at rollout_vllm.py:42 with the smallest/fastest invocation.
2. Fix the cause; add the fast regression test.
3. Make sure the fix reaches the job's branch so its retry actually picks it up.
