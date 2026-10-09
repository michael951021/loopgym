# cppfwd — C++ CPU kernel forward

## Issue
The fused log-prob kernel currently has only the torch reference impl. The C++ CPU
impl (csrc/logprob_cpu.cpp) is the CPU-testable half of the C++/CUDA kernel work:
chunked online-max/LSE forward, OpenMP over T, no [T, V] materialization.

## Question
Can a torch C++ extension (JIT, -O3 -fopenmp) compute log softmax(h @ w.T)[t] per row
in single V-chunk passes (online max + sum, per-chunk dot via at::matmul), float32
AND float64, matching the in-test double reference (atol 1e-4 f32, 1e-3 at |logits|~1000)
and the real 0.5B HF loss (1e-3)? Refutation: any known logprob test failing for "cpp".

## Evidence
- loopgym/kernels/__init__.py: _chunked_forward (reference math), _REGISTRY/IMPLS.
- tests/known/test_logprob.py: parametrized over kernels.IMPLS at collection time, so
  "cpp" must register at import once the build succeeds.
- g++ present, ninja absent (JIT falls back to make). First run before the change
  must fail: `... pytest -q tests/known/test_logprob.py -k "cpp and not gradient"`
  collects 0 tests (exit 5). Log: .agent-evidence/cppfwd-before.log.
- NOTE: `make known` runs ALL known tests incl. cpp gradcheck (f64) and the real-model
  test, so this ticket ships f64 support and a Function whose backward reuses the torch
  chunked backward (cppbwd replaces it with C++).

## Done when
`LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_logprob.py -k "cpp and not gradient"`
passes (3 forward params + large-logits stability). Known-answer checks: in-test double
reference; real Qwen2.5-Coder-0.5B HF loss ±1e-3 (full `make known`). Range: f32 within
1e-4 of double reference, finite at |logits|~1000. Regression: make check + make known
green (known grows 26 -> 32).

## Stop when
Only csrc/logprob_cpu.cpp + loopgym/kernels/__init__.py registration; no C++ backward,
no CUDA, no Triton. Stop/split after 2 attempts without new evidence.

## Plan
1. Run acceptance command first (expect exit 5, no cpp tests).
2. Write csrc/logprob_cpu.cpp: templated f32/f64 forward -> (out, lse); OpenMP over T,
   per-row chunk loop, at::matmul dots, double accumulation of m/S.
3. kernels/__init__.py: build via cpp_extension.load (build dir .tools/build,
   -O3 -fopenmp), register "cpp" in a try/except with a clear RuntimeError if requested
   while unavailable; factor torch backward to a module fn reused by the cpp Function
   placeholder backward.
4. Acceptance command, then full make known, then make check.
