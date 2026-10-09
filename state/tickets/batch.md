# batch — one GRPO step from completions

## Issue
No code turns rollout completions + rewards into one training step. Without it the train loop
cannot exist and every GRPO bug (mask, old/ref logp, clipping) would only surface in a 10-min
run. This ticket isolates the step so bugs show in a seconds-long test.

## Question
Does `build_batch` produce ids whose completion mask selects exactly the completion tokens'
log-probs (verified against a plain forward+gather), and does `grpo_step` implement the
CLAUDE.md key definitions (recomputed logp/old/ref, policy_loss, clip, step) so that an
on-policy step-0 loss with β=0 is ~0 when per-group advantages sum to 0, and a
positive-advantage completion's log-prob rises after one step? Refuted if the mask includes
any prompt/pad column, the loss is not ~0 on the controlled input, or the rise does not
happen (or the ref path does not actually disable the adapter).

## Evidence
- loopgym/rl/grpo.py: group_advantages, policy_loss (done, known-green).
- loopgym/kernels/__init__.py: token_logprobs(h [T,d], w [V,d], targets [T], impl, chunk) —
  unbatched, so the step must flatten [N, T-1] rows.
- peft 0.21.2 in .venv: PeftModel.disable_adapter() is a context manager (verified in source).
- Smallest check: tests/test_batch.py on tiny_config (fp32 CPU) — first run fails with
  ModuleNotFoundError for loopgym.rl.batch. Log: .agent-evidence/batch-first.log.

## Done when
`.venv/bin/python -m pytest -q tests/test_batch.py` passes. Known-answer checks (per ticket):
(1) mask covers exactly the completion tokens — cross-checked against independent
log_softmax(logits).gather on a tiny model; (2) on-policy step-0 loss, β=0, per-group
advantages sum 0, equal-length rows → |loss| < 1e-6 (CLAUDE.md initial-loss range is
|loss| < 0.05 for the real run); (3) positive-adv completion's masked mean logp rises (and
negative's falls) after one step. Plus: after a step, stats["kl"] > 0 proves the ref forward
ran with the adapter disabled. Plausible range for (3): rise > 1e-4 nats at lr 1e-2, AdamW.

## Stop when
Scope: batch.py + tests/test_batch.py only. No train loop, no vLLM, no kernel impls, no
multi-epoch old_logp storage. Stop/split after two attempts without new evidence (e.g. rise
test numerically flaky on tiny model → record and split the check).

## Plan
1. Write tests/test_batch.py (failing), run it.
2. loopgym/rl/batch.py: Batch, StepCfg, build_batch, grpo_step (+ _forward_logp helper).
3. Run tests, then `make check` (ruff + full suite + known).
