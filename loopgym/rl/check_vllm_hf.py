"""vLLM-vs-HF sampler logprob check (host only; CLAUDE.md known answer).

Greedy rollouts of the same prompts from HF (transformers) and vLLM must produce
the same tokens, and the mean |Δ| of the per-token log-probs over the common
prefix must be < 0.02 nats in fp32 (bf16 is reported separately, not gated).
Larger means a weight-sync or sampling-parameter bug.

Both backends return the log-prob of each sampled token under p (greedy: HF
passes temperature 1.0 with do_sample=False, vLLM returns the raw-logit
logprob), so the two series compare directly.

Sandbox (rule 2): only --dry-run runs here (no CUDA, no vLLM import)."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from loopgym.rl import rollout_vllm as rv
from loopgym.rl.rollout import HFBackend

GATE = 0.02  # nats, fp32 mean |Δ| (CLAUDE.md known answer)

PROMPTS = [
    "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True iff s is a palindrome, ignoring case and non-alphanumerics.\"\"\"",
    "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] pairs and returns them merged: overlapping or touching intervals become one.",
    "def longest_common_prefix(strs: list[str]) -> str:\n    \"\"\"Return the longest string that is a prefix of every string in strs (\"\" if none).\"\"\"",
    "Write a Python function `two_sum(nums, target)` returning the two indices (i < j) whose values sum to target, or None.",
    "def group_anagrams(words: list[str]) -> list[list[str]]:\n    \"\"\"Group words that are anagrams of each other; order of groups and words is free.\"\"\"",
    "Write a Python function `max_subarray_sum(nums)` returning the largest sum over any non-empty contiguous subarray.",
    "def valid_parentheses(s: str) -> bool:\n    \"\"\"Return True iff s consists of balanced () [] {} pairs.\"\"\"",
    "Write a Python function `flatten(nested)` that flattens arbitrarily nested lists/tuples into a single flat list of the non-list elements.",
]


def _common_prefix(a, b) -> int:
    L = 0
    for x, y in zip(a, b):
        if x != y:
            break
        L += 1
    return L


def _compare_one(hf_backend, vllm_backend, prompts, max_new_tokens, seed=0):
    """Greedy n=1 from both backends; per-prompt common-prefix |Δ| stats."""
    rows, tot, ntok = [], 0.0, 0
    for p in prompts:
        h = hf_backend.generate([p], 1, max_new_tokens, 0.0, 1.0, seed)[0][0]
        v = vllm_backend.generate([p], 1, max_new_tokens, 0.0, 1.0, seed)[0][0]
        L = _common_prefix(h.completion_ids, v.completion_ids)
        d = [abs(h.sampler_logp[i] - v.sampler_logp[i]) for i in range(L)]
        tot += sum(d)
        ntok += len(d)
        rows.append({
            "prompt": p[0]["content"][:60],
            "prefix": L,
            "hf_tokens": len(h.completion_ids),
            "vllm_tokens": len(v.completion_ids),
            "mean_abs_delta": sum(d) / len(d) if d else 0.0,
        })
    return {
        "mean_abs_delta": tot / ntok if ntok else 0.0,
        "min_prefix": min((r["prefix"] for r in rows), default=0),
        "tokens_compared": ntok,
        "prompts": rows,
    }


def _teardown(llm, model):
    """Free both engines before the next dtype pass (3090 is 24 GB, not more)."""
    eng = getattr(llm, "llm_engine", None)
    core = getattr(eng, "engine_core", None) if eng is not None else None
    if core is not None and callable(getattr(core, "shutdown", None)):
        try:
            core.shutdown()
        except Exception as e:  # engine already dead; GC still reclaims
            print(f"engine_core.shutdown: {e}", file=sys.stderr)
    del llm, model
    gc.collect()
    torch.cuda.empty_cache()


def check_dtype(model: str, dtype: str, prompts, max_new_tokens: int) -> dict:
    """HF vs vLLM greedy comparison at one dtype (both engines in the same dtype)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
    m = m.to("cuda:0").eval()
    hf = HFBackend(m, tok)
    # the check needs room for the HF copy on the same 3090: small KV budget
    # base model only: no adapter is loaded, and vLLM's LoRA kernels reject float32
    args = rv.engine_args(model, dtype=dtype, gpu_memory_utilization=0.35,
                          max_model_len=2048, enable_lora=False)
    llm = rv.build_llm(args)
    res = _compare_one(hf, rv.VLLMBackend(llm, tok), prompts, max_new_tokens)
    _teardown(llm, m)
    return res


def plan(args) -> None:
    print("=== check_vllm_hf plan ===")
    print(f" model           : {args.model}")
    print(f" prompts         : {args.n_prompts} built-in coding prompts (greedy, n=1, seed=0)")
    print(f" max_new_tokens  : {args.max_new_tokens}")
    print(f" pass 1 (GATED)  : HF fp32 vs vLLM dtype=float32 -> mean |Δ| < {args.gate} nats")
    print(" pass 2 (report) : HF bf16 vs vLLM dtype=bfloat16 -> recorded, not gated")
    print(f" output          : {args.out}")
    print(" both backends: greedy (T=0), logprob of each sampled token under p")
    print(" gate: fp32 mean |Δ| < threshold AND full prefix match on every prompt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loopgym.rl.check_vllm_hf")
    ap.add_argument("--model", required=True, help="base model path (no LoRA)")
    ap.add_argument("--out", default="results/gpu/vllm_hf_check.json")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gate", type=float, default=GATE, help="fp32 mean |Δ| gate (nats)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no CUDA, no vLLM import)")
    args = ap.parse_args(argv)
    if args.dry_run:
        plan(args)
        return 0

    prompts = [{"role": "user", "content": p} for p in PROMPTS[:args.n_prompts]]
    fp32 = check_dtype(args.model, "float32", prompts, args.max_new_tokens)
    bf16 = check_dtype(args.model, "bfloat16", prompts, args.max_new_tokens)
    gate = (fp32["mean_abs_delta"] < args.gate
            and fp32["min_prefix"] >= args.max_new_tokens)
    payload = {
        "model": args.model,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "gate_nats": args.gate,
        "fp32": fp32,
        "bf16": bf16,
        "gate": {"pass": bool(gate),
                 "reason": "fp32 mean |Δ| < " + str(args.gate)
                 + " and greedy prefix complete" if gate else
                 "fp32 mean |Δ| >= " + str(args.gate)
                 + " or greedy prefix diverged (weight-sync or sampling-param bug)"},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"fp32 mean |Δ| = {fp32['mean_abs_delta']:.5f} nats "
          f"(gate < {args.gate}), min prefix {fp32['min_prefix']}/{args.max_new_tokens}")
    print(f"bf16 mean |Δ| = {bf16['mean_abs_delta']:.5f} nats (reported, not gated)")
    print(f"wrote {out}; gate {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
