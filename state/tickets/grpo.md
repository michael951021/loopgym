# grpo — GRPO math

## Issue
The trainer's per-step loss has no implementation; `loopgym/rl/grpo.py` does not exist, so no
GRPO step can run and its math is unpinned. Everything downstream (batch.py, train.py, the
planted-effect gate) depends on this exact math.

## Question
Can `group_advantages`, `kl_k3`, `policy_loss` in loopgym/rl/grpo.py match CLAUDE.md "Key
definitions" exactly, so Kevin's hand-computed known answers in tests/known/test_grpo.py hold?
Refuted if any of: constant group gives NaN, k3 < 0 anywhere, on-policy token-mean loss !=
−mean(adv over unmasked tokens), a clipped token still gets gradient, or the KL term fails to
pull logp toward ref.

## Evidence
- Specs: CLAUDE.md "Key definitions" (group advantage, k3 KL, policy loss, agg modes).
- Tests (read-only, Kevin's): tests/known/test_grpo.py — 7 hand-computed cases incl. a
  200k-sample Monte-Carlo check that k3 is an unbiased KL estimator.
- Smallest check: `.venv/bin/python -m pytest -q tests/known/test_grpo.py`
  (log: .agent-evidence/grpo-first.log). Expected first-run failure: ModuleNotFoundError
  (loopgym.rl missing) — that is the failure mode to debug from.

## Done when
`.venv/bin/python -m pytest -q tests/known/test_grpo.py` passes (all 7 tests). Known answers
are hand-computed by Kevin, run on real torch tensors — no measured numbers of my own, so no
plausible-range write-down needed. `make check` stays green (ruff + full pytest; tests/known
stays untouched, `make known` author check intact).

## Stop when
Scope is the three functions in one module — no rollout, batching, or train loop. Stop/split if
a test fails for a reason inside tests/known (rule 1: record in NOTES.md, RESULT: not done) or
after two attempts without new evidence.

## Plan
1. Run the known test first; record the expected ModuleNotFoundError.
2. Implement loopgym/rl/grpo.py (plus rl/__init__.py) per CLAUDE.md definitions:
   per-row (r−mean)/(sample-std+1e-6), scale="none" skips division; k3 = exp(d)−d−1,
   d = ref−logp; ratio = exp(logp−old_logp), obj = min(ratio·A, clip(ratio)·A) − β·k3,
   token-mean = −Σobj·mask/Σmask, seq-mean = per-row masked mean then row mean;
   clip_frac over masked tokens (sign-aware), stats kl/ratio_mean masked means.
3. Re-run known tests, then `make check`; evidence to .agent-evidence/.
