"""vLLM rollout backend with per-step LoRA weight sync (GPU host only).

vLLM 0.31.0 is pinned (tools/versions.txt); its source is read-only at
.tools/src/vllm-0.31.0/ and is not installed in this sandbox (rule 2). The
vllm import is try/except at module level (the sandbox binds None; tests
monkeypatch fakes onto the same module globals), tests/test_vllm_api.py
asserts every name/attribute/kwarg used below against the pinned source with
ast, and `python -m loopgym.rl.rollout_vllm --dry-run` prints the exact
LLM(...) engine args without importing vLLM.

Weight sync: the engine is built once with enable_lora=True, max_loras=1.
After each optimizer step the trainer saves the PEFT adapter and calls
sync_lora(dir, name); the next generate() passes a fresh
LoRARequest(load_inplace=True), so the engine reloads the new adapter weights
before sampling.

sampler_logp: SamplingParams(logprobs=0) returns the logprob of the sampled
token at each position; logprobs[i][token].logprob is the log-prob under the
temperature-scaled distribution — the same definition as HFBackend."""
from __future__ import annotations

import argparse
import json
import sys

from loopgym.rl.rollout import Completion

try:  # vLLM is installed only on the GPU host (rule 2)
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
except ModuleNotFoundError:  # sandbox: tests monkeypatch fakes onto these names
    SamplingParams = None
    LoRARequest = None

__all__ = ["VLLMBackend", "engine_args", "main"]


def engine_args(model: str, *, lora_rank: int = 16, dtype: str = "bfloat16",
                gpu_memory_utilization: float = 0.9, seed: int = 0,
                max_model_len: int = 8192, tensor_parallel_size: int = 1) -> dict:
    """Exact vllm.LLM(...) kwargs for the GPU run: base model + LoRA adapters."""
    return {
        "model": model,
        "trust_remote_code": False,
        "dtype": dtype,
        "seed": seed,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": lora_rank,
    }


class VLLMBackend:
    """RolloutBackend on a vLLM LLM; LoRA weights synced per step via sync_lora."""

    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self._lora = None
        self._lora_id = 0

    def sync_lora(self, adapter_dir, name: str) -> None:
        """Point the next generate() at a freshly saved PEFT adapter directory."""
        if LoRARequest is None:
            raise RuntimeError("vLLM not installed; VLLMBackend runs only on the GPU host")
        self._lora_id += 1
        self._lora = LoRARequest(lora_name=name, lora_int_id=self._lora_id,
                                 lora_path=str(adapter_dir), load_inplace=True)

    def generate(
        self,
        prompts: list[list[dict]],
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[Completion]]:
        texts = [
            self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        if SamplingParams is None:
            raise RuntimeError("vLLM not installed; VLLMBackend runs only on the GPU host")
        greedy = temperature <= 0
        params = SamplingParams(
            n=n,
            temperature=0.0 if greedy else temperature,
            top_p=1.0 if greedy else top_p,
            max_tokens=max_new_tokens,  # the pinned default is 16, not None
            seed=seed,
            logprobs=0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
        )
        outs = self.llm.generate(texts, params, use_tqdm=False, lora_request=self._lora)
        groups: list[list[Completion]] = []
        for ro in outs:
            g = []
            for co in sorted(ro.outputs, key=lambda o: o.index):
                ids = [int(t) for t in co.token_ids]
                if co.logprobs is None:
                    raise AssertionError("vLLM returned no logprobs with logprobs=0")
                lp = [float(co.logprobs[i][t].logprob) for i, t in enumerate(ids)]
                g.append(Completion(prompt_ids=list(ro.prompt_token_ids or []),
                                    completion_ids=ids, text=co.text, sampler_logp=lp))
            groups.append(g)
        return groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loopgym.rl.rollout_vllm")
    ap.add_argument("--model", required=True, help="base model path; LoRA adapters apply on top")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--lora-path", default=None, help="also print the LoRARequest for this adapter")
    ap.add_argument("--dry-run", action="store_true", help="print the engine args and exit (no vLLM import)")
    args = ap.parse_args(argv)
    payload = {"engine_args": engine_args(
        args.model, lora_rank=args.lora_rank, dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
        max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size)}
    if args.lora_path:
        payload["lora_request"] = {"lora_name": "step_1", "lora_int_id": 1,
                                   "lora_path": str(args.lora_path), "load_inplace": True}
    print(json.dumps(payload, indent=1))
    if args.dry_run:
        return 0
    from vllm import LLM
    LLM(**payload["engine_args"])
    print("engine built; wire VLLMBackend(llm, tokenizer) into the host training loop",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
