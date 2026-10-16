# Fix gpurun1 crash: torch.OutOfMemoryError in __init__.py

## Issue
Compute job `gpurun1` gave up after its retries with the same crash each time, so every ticket that waits on its
results is blocked. The failure is a Python traceback — a code bug, not a resource limit — so it is fixable here.
Auto-filed from the job log; signature `torch.OutOfMemoryError@__init__.py:144`.

## Evidence
Exception: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 892.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 418.75 MiB is free. Including non-PyTorch memory, this process has 1.30 GiB memory in use. Process 1005 has 21.56 GiB memory in use. Of the allocated memory 1.04 GiB is allocated by PyTorch, and 3.58 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)`
Raised through (deepest project frame last):
  train.py:553 in run
  train.py:299 in run_step
  batch.py:207 in grpo_step
  batch.py:160 in _ref_logp
  batch.py:120 in logp_of
  batch.py:80 in _logp_rows
  __init__.py:182 in token_logprobs
  __init__.py:144 in wrapped
Fix the deepest project frame: **__init__.py:144** in `wrapped`.
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
1. Reproduce `torch.OutOfMemoryError` at __init__.py:144 with the smallest/fastest invocation.
2. Fix the cause; add the fast regression test.
3. Make sure the fix reaches the job's branch so its retry actually picks it up.
