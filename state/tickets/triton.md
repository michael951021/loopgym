# triton — Triton kernel (interpreter tested)

## Issue
GRPO log-probs need a GPU kernel path. The torch chunked impl is Python-looped
matmuls; the triton impl (forward + backward, tiles over V, online softmax) is the
path labs actually ship. Sandbox has no GPU (rule 2), so it is checked under
TRITON_INTERPRET=1 here; gpu_run.sh checks it on the host.

## Question
Does `token_logprobs(h, w, t, impl="triton")` — Triton kernels, V-tiled online
max/LSE, backward recomputing per-tile softmax, dtype-preserving — match the
double reference within the known-test tolerances (f32 1e-4, |logits|~1000
finite, f64 gradcheck) under TRITON_INTERPRET=1? Refutation: any known test
failing, or a NaN where the torch impl is finite.

## Evidence
- loopgym/kernels/__init__.py: IMPLS, register_impl; torch impl is the reference.
- tests/known/test_logprob.py (4 tests × impl incl. real-model HF loss @1e-3).
- Before (expect 0 triton tests, exit 5):
  `TRITON_INTERPRET=1 LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_logprob.py -k triton`
  → log .agent-evidence/triton-before.log
- Guess (to verify): triton not in .venv yet; interpreter f64 support untested.

## Done when
`TRITON_INTERPRET=1 LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_logprob.py -k triton`
passes (5 tests: matches_reference×3, large_logits, gradient, real-model).
Known-answer checks: in-test double reference, HF loss ±1e-3 on 0.5B, f64 gradcheck.
Range: f32 within 1e-4 of double ref; finite at |logits|~1000.
Regression: `make check` green.

## Stop when
Scope: loopgym/kernels/triton_logprob.py, __init__.py registration, pyproject dep.
No CUDA source, no GPU perf tuning. After 2 attempts without new evidence:
record blocker in NOTES.md, stop/split.

## Plan
1. Run acceptance (expect fail: no triton impl).
2. Add pinned triton to pyproject; uv sync; probe interpreter f64 + tiny kernel.
3. Write triton_logprob.py (fwd: per-row programs, V-tile online max/LSE + target
   logit; bwd: bwd_h per-row acc of p·w; bwd_w per-V-tile, static row-tile loop,
   no atomics) + autograd.Function; register "triton" iff CUDA or TRITON_INTERPRET=1.
4. Run acceptance, then make check + make known.
