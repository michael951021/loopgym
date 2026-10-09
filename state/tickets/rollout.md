# rollout — HF rollout backend

## Issue
No rollout backend: the GRPO step (next ticket) needs completions plus the sampler's
per-token log-probs. If the sampler's log-probs disagree with the trainer's recomputed
forward, the policy loss trains on the wrong distribution and the run is garbage.
`loopgym/rl/rollout.py` must expose that check now, on the real model, before training.

## Question
Does `HFBackend.generate` return, for each sampled token, the log-prob under the
temperature-scaled sampling distribution that matches a fresh `model.forward` +
`kernels.token_logprobs` within 1e-4 (temperature 1, fp32)? Refuting observation: max
|Δ| > 1e-4 on any sampled token, or the greedy path differs from `model.generate` tokens.

## Evidence
- transformers 5.19 `_sample` (generation/utils.py:3229): `scores += (next_token_scores,)`
  where `next_token_scores = logits_processor(input_ids, next_token_logits)` and
  `next_token_logits = outputs.logits[:, -1].to(dtype=float32)` — scores are post-temperature,
  fp32, on the do_sample and greedy branches alike.
- Check: tests/test_rollout.py on real models/qwen2.5-coder-0.5b-instruct, fp32,
  2 prompts × n=2, 16 tokens, seed fixed. Output: .agent-evidence/rollout-accept.log.

## Done when
`.venv/bin/python -m pytest -q tests/test_rollout.py` passes in < 5 min. Known-answer check:
max |sampler_logp − kernels.token_logprobs(fresh forward)| ≤ 1e-4 per token at T=1 (fp32);
greedy completion tokens equal a direct `model.generate` call; a prompt's greedy completion
is unchanged whether it runs alone or batched after a longer (padding-creating) prompt.

## Stop when
Scope is the HF backend + its test only; no vLLM, no batch.py, no training loop. If the
log-prob gap stays > 1e-4 after matching dtypes and layouts, stop and record the measured
gap with layout (padded vs not) in NOTES.md — do not loosen the tolerance.

## Plan
1. rollout.py: Completion, RolloutBackend protocol, HFBackend (chat template, left pad,
   rows replicated ×n, manual_seed, generate output_scores, per-token log_softmax at the
   sampled id). temperature <= 0 → do_sample=False (greedy).
2. tests/test_rollout.py: the three checks above, module-scoped model fixture (fp32).
3. Run acceptance, record max |Δ| in NOTES.md.
