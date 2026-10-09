# cppbwd — C++ CPU kernel backward

## Issue
`_CppLogprob.backward` silently reuses `_torch_backward` (Python/torch math) — the
"cpp" impl's gradient path is not actually C++. Training through the fused kernel
on the CPU must stay fast and must be verified, not delegated.

## Question
Can a C++ `logprob_bwd(h, w, t, lse, g)` compute grad_h and grad_w per V-chunk
(exp(s − lse) recomputed from saved per-row lse, target +1 fixup, dS@w_c / dS.T@h)
in f32 and f64, matching the naive log-softmax autograd (f64 1e-10, f32 1e-4) and
gradcheck f64 (atol 1e-4)? Refutation: gradcheck or the direct C++-backward
comparison failing.

## Evidence
- csrc/logprob_cpu.cpp: only `logprob_fwd` exists. `python -c` probe: the extension
  module has no `logprob_bwd` (pre-state check, expected before the change).
- loopgym/kernels/__init__.py:119-122 `_CppLogprob.backward` -> `_torch_backward`
  fallback (the path being replaced).
- tests/known/test_logprob.py:34 gradcheck over IMPLS incl. "cpp" (f64).
- Acceptance already passes pre-change via the torch fallback; the discriminating
  pre-state failure is the probe above plus a new direct-C++-bwd test that
  KeyErrors before the symbol exists. Logs: .agent-evidence/cppbwd-*.log.

## Done when
`LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_logprob.py -k cpp`
passes 5/5 incl. gradcheck, with `_CppLogprob.backward` calling the C++ op (no torch
fallback). Known-answer checks: in-test f64 gradcheck; new tests/test_kernels.py
test comparing `logprob_bwd` output to naive log-softmax autograd (f64 1e-10,
f32 1e-4); real 0.5B HF-loss ±1e-3 (in acceptance). Regression: make check + make known.

## Stop when
Only csrc/logprob_cpu.cpp + _CppLogprob.backward + one regression test in
tests/test_kernels.py. No CUDA/Triton, no changes to _torch_backward (torch impl
keeps it), no tests/known edits. Stop/split after 2 attempts without new evidence.

## Plan
1. Probe: extension lacks logprob_bwd (record). Run acceptance (passes via fallback;
   record as baseline).
2. csrc: templated logprob_bwd_impl (f32/f64), need_h/need_w flags, per-chunk
   dS = g*exp(s−lse) + target fixup, grad_h.addmm_(dS, w_c), grad_w slice = dS.T@h;
   bind logprob_bwd.
3. kernels/__init__.py: _CppLogprob.backward calls logprob_bwd, returns None where
   unneeded. Add tests/test_kernels.py test_cpp_backward_matches_reference.
4. Acceptance command, make check, make known. NOTES + commit + merge master.
