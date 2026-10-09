# benchlpcode — Log-prob bench script

## Issue
The report's kernel memory/speed claims need one script that measures them: the ticket
says the numbers for the report come from `tools/bench_logprob.py`, which does not exist.

## Question
Do the fused impls (torch, cpp, triton) match the full-softmax reference and stay under
the memory budget on 0.5B shapes (T ∈ {512, 2048}, V=151936, d=896, fp32), and how much
faster/leaner are they than naive? Refutation: any impl off by > 1e-4 (f64 ref) or a
fused impl peaking ≥ 1/8 of naive's extra bytes.

## Evidence
`loopgym/kernels/__init__.py` (`token_logprobs`, IMPLS), `tests/known/test_logprob.py`
(f64 full log_softmax reference, atol 1e-4 — the pattern to reuse). Probes done:
- triton under TRITON_INTERPRET=1: 3.35 s per row at V=151936 (T=1 fwd) → 5 runs at
  T≥128 cannot fit a 3-min budget; interpreter time is not kernel performance.
- peak extra memory on CPU: `torch.profiler.profile(profile_memory=True)`, peak =
  max cumulative `self_cpu_memory_usage` over `prof.profiler.function_events`; validated
  against synthetic alloc/free orderings and naive at T=128 (peak = exactly 2·T·V·4).

## Done when
`.venv/bin/python tools/bench_logprob.py --quick` exits 0 in < 3 min, writes
results/bench_logprob.json (median of 5 runs + peak extra bytes per impl).
Known-answer checks, written before the run:
1. every impl (incl. naive) matches the f64 full log_softmax within 1e-4 at every T
   (same reference/tolerance as tests/known/test_logprob.py);
2. fused peak extra < 1/8 of naive peak extra (CLAUDE.md plausible range) for torch and
   cpp at every T (triton when CUDA). Plausible: naive ≈ (2–4)·T·V·4 bytes; torch/cpp
   peak a few tens of MB (chunk 8192 → ~T·8192·4 + tiles).
Triton without CUDA is benched at T=1, 5 runs, flagged `triton_interpret: true` (Python
interpreter overhead, not a GPU number); on CUDA it is benched like the others.

## Stop when
Two failed attempts at the memory-peak extraction or the time budget with no new
evidence → record the blocker in NOTES.md and stop. Out of scope: changing any kernel,
GPU numbers (host's job), the full-run ticket (separate line, 2 h).

## Plan
1. Write the ticket (this file).
2. Write tools/bench_logprob.py (shapes, timing, memory, checks, JSON), mirroring
   tools/bench_lgbox.py conventions.
3. Run --quick, verify checks pass and runtime < 3 min; log evidence.
4. `make check` regression (new file only), NOTES.md, commit, merge master.
