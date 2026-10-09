# cuda — CUDA kernel source, compile-checked

## Issue
The fused log-prob kernel has torch (reference) and cpp (CPU) impls. The C++/CUDA
work needs the CUDA half written and compile-checked here so gpu_run.sh can run the
same known tests on the 3090s: fwd (one block per row, warp-shuffle online
max/LSE over V tiles, no [T,V] logits) and bwd (softmax from saved per-row lse).

## Question
Can csrc/logprob_cuda_kernels.cu (pure CUDA, no torch headers) compile with
`nvcc -c -arch=sm_86 -std=c++17` from a pip nvcc (.venv-nvcc), and does the
torch binding (csrc/logprob_cuda.cpp + loopgym/kernels/cuda_ext.py, built only
when torch.cuda is available) register impl "cuda" with the same fwd/bwd API as
cpp? Refutation: acceptance command nonzero, or binding not registering.
Functional correctness is NOT checkable here (no GPU, rule 2); it is pinned for
the host by the existing known tests parametrized over IMPLS (f32 atol 1e-4 vs
double ref, f64 gradcheck, 0.5B HF loss 1e-3).

## Evidence
- loopgym/kernels/__init__.py: _REGISTRY/register_impl, _CppLogprob (API shape);
  csrc/logprob_cpu.cpp: math (online max/S; dS = -g*p + g@tgt; grad_h = dS@w_c,
  grad_w = dS.T@h) — CUDA mirrors this exactly.
- tests/known/test_logprob.py: IMPLS parametrization; chunk values 2..8192, V not
  a multiple of chunk (50, 3001), f64 gradcheck chunk=16, real model V=151936.
- Observed (pre-run): .venv-nvcc absent; acceptance command fails (bootstrap
  script missing). Log: .agent-evidence/cuda-before.log.
- Verified, replacing the cu12-wheel guess above:
  - nvidia-cuda-nvcc-cu12 ships ptxas/nvvm/crt but NO nvcc driver; and CUDA 12.9
    wheels cannot compile here at all (gcc 15 > supported 14; cicc 12.9 chokes on
    gcc-15 libstdc++ builtins; glibc 2.43 mathcalls.h noexcept clash, 31 errors).
    -> CUDA 13.2.86 pip wheels (pinned cuda_nvcc in tools/versions.txt).
  - 13.x wheel layout: nvidia-cuda-nvcc carries the driver (bin/{nvcc,ptxas,crt});
    nvidia-cuda-runtime headers + libcudart; nvidia-nvvm libnvvm, must be pinned to
    the same version (newer nvvm emits PTX 9.4 that ptxas 13.2.86 rejects). The
    13.x driver resolves the toolkit root from its argv[0] path (must be
    <root>/bin/nvcc), hence the 3-line sh wrapper at .venv-nvcc/bin/nvcc.
  - Bonus compile: csrc/logprob_cuda.cpp passes `g++ -c` against torch 2.14.1+cpu
    headers + cu13 wheel includes. Host-JIT facts baked in: -std=c++20 (torch 2.14
    #error otherwise), -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE (cmake-generated header
    absent from wheels), c10/cuda/CUDAStream.h (ATen/cuda/CUDAContext.h pulls
    cusparse.h unconditionally), int64_t for at::kLong (== long on x86_64).
  - Acceptance: `bash tools/bootstrap_cuda.sh && .venv-nvcc/bin/nvcc -c -arch=sm_86
    -std=c++17 csrc/logprob_cuda_kernels.cu -o /tmp/k.o` exit 0, idempotent
    (cuda-accept5.log; benign #20208-D long-double warning). In the sandbox
    IMPLS = (torch, cpp) (torch.cuda.is_available() False); tests/test_kernels.py
    5 passed 1 skipped. make check 88 passed 1 skipped (cuda-mkcheck.log);
    make known 32/32 (cuda-mkknown.log).

## Done when
`bash tools/bootstrap_cuda.sh && .venv-nvcc/bin/nvcc -c -arch=sm_86 -std=c++17 csrc/logprob_cuda_kernels.cu -o /tmp/k.o`
exits 0 (idempotent on re-run). Bonus evidence: binding file compiles against
torch headers + runtime-wheel headers (nvcc -c) here. Regression: make check
green (ruff/pytest/shellcheck), make known 32/32 unchanged ("cuda" not registered
in the sandbox: torch.cuda.is_available() False). Host functional pass is the
next ticket's gate (gpu_run.sh runs known tests incl. cuda impl).

## Stop when
Scope: tools/bootstrap_cuda.sh, csrc/logprob_cuda_kernels.{h,cu},
csrc/logprob_cuda.cpp, loopgym/kernels/cuda_ext.py, registration in
loopgym/kernels/__init__.py, one test in tests/test_kernels.py. No GPU runs, no
benchmark, no Triton (w1), no changes to known tests. Stop/split after 2 attempts
without new evidence (e.g. wheel layout defeats bare-nvcc include resolution).

## Plan
1. Run acceptance command first (expect fail: no script, no .venv-nvcc).
2. tools/bootstrap_cuda.sh: uv venv .venv-nvcc (py3.12), uv pip install the two
   nvidia-cu12 wheels, symlink runtime include/lib into venv root so bare nvcc
   resolves cuda_runtime.h; verify with a trivial .cu.
3. csrc/logprob_cuda_kernels.h (declarations) + .cu (fwd/bwd_h/bwd_w, f32/f64,
   warp-shuffle block reductions, one-writer-per-element determinism).
4. csrc/logprob_cuda.cpp torch binding (same API as cpp) + cuda_ext.py JIT build
   gated on torch.cuda.is_available() + register "cuda"; try nvcc -c of the
   binding with torch + wheel includes.
5. Acceptance command; bonus binding compile; make check; make known; NOTES; commit.
