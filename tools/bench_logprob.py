#!/usr/bin/env python3
"""Bench for fused token log-probs (loopgym/kernels) vs naive full-softmax.

0.5B shapes: V=151936, d=896, fp32 random inputs, T in {512, 2048} (--quick: T=128).
Per impl (naive full logits, torch, cpp, triton when available): median wall time of
5 runs and peak extra memory (torch profiler profile_memory on CPU,
cuda max_memory_allocated on GPU) -> results/bench_logprob.json.

Known-answer checks (CLAUDE.md rule 5), written before the run:
  1. every impl matches the f64 full log_softmax within 1e-4 at every T (same
     reference and tolerance as tests/known/test_logprob.py);
  2. fused peak extra < 1/8 of naive peak extra (CLAUDE.md plausible range) for
     torch and cpp at every T (triton too when CUDA).
Triton without CUDA runs under TRITON_INTERPRET=1: its times are Python-interpreter
overhead, not kernel performance, so it is benched at T=1 and flagged
"triton_interpret": true in the JSON; GPU numbers come from scripts/gpu_run.sh.

Usage: .venv/bin/python tools/bench_logprob.py [--quick]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V, D, CHUNK, RUNS = 151936, 896, 8192, 5
T_FULL = (512, 2048)
T_QUICK = (128,)
TOLERANCE = 1e-4


def make_inputs(t_value):
    g = torch.Generator().manual_seed(t_value * V)
    h = torch.randn(t_value, D, generator=g)
    w = torch.randn(V, D, generator=g) * 0.5
    t = torch.randint(0, V, (t_value,), generator=g)
    return h, w, t


def naive_impl(h, w, t, chunk):
    x = h @ w.T
    return torch.log_softmax(x, dim=-1)[torch.arange(t.shape[0]), t]


def make_fn(impl):
    def fn(h, w, t):
        if impl == "naive":
            return naive_impl(h, w, t, CHUNK)
        from loopgym.kernels import token_logprobs

        return token_logprobs(h, w, t, impl=impl, chunk=CHUNK)

    return fn


def reference_f64(h, w, t):
    return (
        torch.log_softmax((h @ w.T).double(), -1)
        .gather(-1, t.unsqueeze(-1))
        .squeeze(-1)
    )


def peak_extra_bytes(fn):
    """Peak extra memory of one call, in bytes (relative to the call's start)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        out = fn()
        torch.cuda.synchronize()
        del out
        return int(torch.cuda.max_memory_allocated())
    import tempfile
    from torch.profiler import profile

    with profile(profile_memory=True) as prof:
        fn()
    # [memory] instant events carry the cumulative allocation; max is the peak.
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        evs = json.load(open(path))["traceEvents"]
        totals = [
            e["args"]["Total Allocated"]
            for e in evs
            if e.get("name") == "[memory]" and "Total Allocated" in e.get("args", {})
        ]
        return int(max(totals)) if totals else 0
    finally:
        os.unlink(path)


def bench_impl(impl_name, fn, h, w, t):
    fn(h, w, t)  # warmup (JIT, BLAS, allocator)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(h, w, t)
        times.append(time.perf_counter() - t0)
        del out
    return {
        "time_median_s": round(statistics.median(times), 4),
        "time_runs_s": [round(x, 4) for x in times],
        "peak_extra_bytes": peak_extra_bytes(lambda: fn(h, w, t)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="T=128 only")
    args = ap.parse_args()
    Ts = T_QUICK if args.quick else T_FULL
    cuda = torch.cuda.is_available()
    if not cuda:
        os.environ.setdefault("TRITON_INTERPRET", "1")
    from loopgym.kernels import IMPLS  # noqa: E402

    impls = ("naive",) + IMPLS

    print(
        f"bench_logprob: V={V} d={D} T={list(Ts)} chunk={CHUNK} runs={RUNS} "
        f"cuda={cuda} impls={list(impls)}"
    )

    cells = []
    max_diffs = []
    for T in Ts:
        h, w, t = make_inputs(T)
        ref = reference_f64(h, w, t)
        for impl in impls:
            # TRITON_INTERPRET emulates the GPU in Python (~3.4 s/row here): too slow
            # for T>1, and its times are interpreter overhead, not kernel performance.
            T_b = 1 if (impl == "triton" and not cuda) else T
            if T_b != T:
                h_b, w_b, t_b = make_inputs(T_b)
                ref_b = reference_f64(h_b, w_b, t_b)
            else:
                h_b, w_b, t_b, ref_b = h, w, t, ref
            fn = make_fn(impl)
            out = fn(h_b, w_b, t_b)
            diff = float((out.double() - ref_b).abs().max())
            max_diffs.append((impl, T_b, diff))
            del out
            cell = {"impl": impl, "T": T_b}
            cell.update(bench_impl(impl, fn, h_b, w_b, t_b))
            if impl == "triton" and not cuda:
                cell["triton_interpret"] = True
            cells.append(cell)
            print(
                f"  T={T_b:<5d} {impl:7s} {cell['time_median_s']:8.3f}s "
                f"peak_extra={cell['peak_extra_bytes'] / 2**20:9.1f} MiB "
                f"max|diff|={diff:.2e}"
            )
            if T_b != T:
                del h_b, w_b, t_b, ref_b
        del h, w, t, ref

    by = {(c["impl"], c["T"]): c for c in cells}
    corr_ok = all(d <= TOLERANCE for _, _, d in max_diffs)
    mem_ok = True
    mem_details = {}
    for T in Ts:
        naive = by[("naive", T)]["peak_extra_bytes"]
        for impl in sorted(
            set(i for i, _, _ in max_diffs)
            - {"naive"}
            - (set() if cuda else {"triton"})
        ):
            peak = by[(impl, T)]["peak_extra_bytes"]
            ok = peak < naive / 8
            mem_ok &= ok
            mem_details[f"{impl}_T{T}"] = {
                "peak_extra_bytes": peak,
                "naive_peak_bytes": naive,
                "pass": ok,
            }
    checks = {
        "all_impls_match_f64_reference_1e-4": {
            "pass": corr_ok,
            "tolerance": TOLERANCE,
            "max_diffs": {f"{i}_T{T}": d for i, T, d in max_diffs},
        },
        "fused_peak_lt_1_8_naive": {"pass": mem_ok, "detail": mem_details},
    }
    doc = {
        "mode": "quick" if args.quick else "full",
        "T": list(Ts),
        "V": V,
        "d": D,
        "chunk": CHUNK,
        "runs": RUNS,
        "dtype": str(torch.float32),
        "cuda": cuda,
        "triton_interpret": (not cuda) and "triton" in IMPLS,
        "note": (
            "triton_interpret times are TRITON_INTERPRET=1 Python overhead, not "
            "kernel performance; GPU numbers come from scripts/gpu_run.sh"
            if not cuda and "triton" in IMPLS
            else None
        ),
        "cells": cells,
        "checks": checks,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = ROOT / "results" / "bench_logprob.json"
    out_path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out_path}")
    for name, chk in checks.items():
        print(f"check {name}: {'PASS' if chk['pass'] else 'FAIL'}")
    return 0 if corr_ok and mem_ok else 1


if __name__ == "__main__":
    sys.exit(main())
