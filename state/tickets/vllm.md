# vLLM rollout backend with weight sync

## Issue
GRPO rollouts run on the HF backend (CPU here, minutes/step at 1.5B). Fast GPU
rollouts are what make the real GRPO run affordable, but there is no
`RolloutBackend` implementation on vLLM, and after each optimizer step the
rollout engine must see the updated LoRA weights — a weight-sync path that does
not exist yet.

## Question
Can `loopgym/rl/rollout_vllm.py` provide a `RolloutBackend` whose calls are
exactly legal on the pinned vLLM 0.31.0 (tools/versions.txt) and whose
sampler_logp is the log-prob of each sampled token under the temperature-scaled
distribution? Refuting observation: any name/attribute/kwarg the backend uses
that is absent from `.tools/src/vllm-0.31.0/` (checked with `ast`, rule 6), or a
FakeLLM run where log-probs/order/sync don't come out right.

## Evidence
Pinned source `.tools/src/vllm-0.31.0/` (verified read-only):
- `LLM.__init__` (entrypoints/llm.py:180) forwards `**kwargs` to EngineArgs;
  `enable_lora`/`max_loras`/`max_lora_rank` are EngineArgs fields
  (engine/arg_utils.py:667-669); `LLM.generate(prompts, sampling_params, *,
  use_tqdm, lora_request, ...)` (llm.py:420).
- `SamplingParams` fields n/temperature/top_p/max_tokens/seed/logprobs/top_k/
  min_p/repetition_penalty (sampling_params.py:211+); `max_tokens` defaults to
  16 → must always be passed; `logprobs=0` = sampled-token logprob only
  (sampling_params.py:646 allows 0).
- `LoRARequest(lora_name, lora_int_id, lora_path, load_inplace)` (lora/request.py).
- `CompletionOutput.index/token_ids/text/logprobs`, `RequestOutput
  .prompt_token_ids/.outputs` (outputs.py), `Logprob.logprob` (logprobs.py).
- vLLM is not importable in the sandbox (ModuleNotFoundError) → lazy import +
  FakeLLM unit tests.
Smallest check: `.venv/bin/python -m pytest -q tests/test_vllm_api.py` (written
first, fails until rollout_vllm.py exists). Output saved under .agent-evidence/.

## Done when
`.venv/bin/python -m pytest -q tests/test_vllm_api.py` passes: (a) ast check —
every vLLM name/attribute/kwarg the backend source uses (extracted from
rollout_vllm.py itself) exists in `.tools/src/vllm-0.31.0/`; (b) FakeLLM unit
tests: shape/order, SamplingParams passthrough, greedy mapping, logprob
extraction, LoRA sync ids; (c) `--dry-run` CLI prints engine args JSON, exit 0.
No measured GPU number exists in the sandbox; the known-answer anchor is the
pinned-source API itself (rule 6) — the vLLM-vs-HF 0.02-nat logprob check stays
for the host run (CLAUDE.md known answers). Plausible range: n/a (no number
produced here).

## Stop when
Out of scope: wiring VLLMBackend into train.py (w1 owns it; GPU run is
gpu_run.sh's), full-weight (non-LoRA) sync, vLLM server mode. Stop after two
attempts if the pinned source contradicts an assumption above — record it.

## Plan
1. Write tests/test_vllm_api.py (ast spec + FakeLLM + CLI), run → fails (no
   rollout_vllm.py).
2. Implement rollout_vllm.py: engine_args(), VLLMBackend.sync_lora/generate,
   lazy vllm imports via module __getattr__, main(--dry-run).
3. Run acceptance, `make check`, NOTES.md, commit, merge master.
