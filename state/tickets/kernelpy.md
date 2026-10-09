# kernelpy — Torch chunked log-probs

## Issue
GRPO needs log-softmax of every sampled token over V=151936; materializing [T, V]
logits is ~1.24 GB at T=2048. Need a chunked reference impl (memory O(T·chunk))
that every faster impl (C++/Triton/CUDA) must match.

## Question
Can `token_logprobs(h, w, t, impl="torch", chunk)` compute log softmax(h @ w.T)[t]
in V-chunks (online max/log-sum-exp) with a custom autograd.Function whose backward
recomputes per-chunk softmax (no [T,V] saved), matching the double reference within
1e-4 (f32) and passing float64 gradcheck on grad wrt hidden?
Refutation: known test test_logprob.py failing any param.

## Evidence
- tests/known/test_logprob.py: reference(), test_matches_reference,
  test_large_logits_are_stable (|logits|~1000, chunk=2), test_gradient_wrt_hidden
  (f64 gradcheck), test_real_model_matches_hf_loss (0.5B, mean -logp == HF loss ±1e-3).
- First run (must fail: module missing):
  `LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_logprob.py`
  → log .agent-evidence/kernelpy-before.log
- Note: gradcheck on f64 input forces the impl to keep input dtype end-to-end;
  casting output to f32 would make the numerical gradient noise.

## Done when
`LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_logprob.py`
passes (4 tests × 1 impl, real model loaded from models/). Known-answer checks:
double-precision reference in-test; real HF loss on Qwen2.5-Coder-0.5B (±1e-3);
gradcheck (f64). Range: f32 outputs within 1e-4 of double reference, finite at
|logits|~1000. Regression: `make check` stays green.

## Stop when
Only loopgym/kernels/ + tests; no C++/Triton/CUDA. Stop/split after 2 attempts
without new evidence; record blocker in NOTES.md.

## Plan
1. Run known test first (expect ImportError).
2. Write loopgym/kernels/__init__.py: IMPLS, token_logprobs, chunked
   autograd.Function (forward: online LSE + target logit; backward: recompute
   per-chunk softmax, grads wrt hidden and weight).
3. Re-run known test; then make check.
