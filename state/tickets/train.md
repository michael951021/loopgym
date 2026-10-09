# Ticket: GRPO training loop and CLI (id: train)

## Issue
There is no loop that chains the existing pieces (rollout → grade → grpo_step) into a
trainable run. CPU jobs get killed and restarted, so exact resume (step counter, data
order, weights, optimizer, RNG) is a first-class requirement, not a nicety.

## Question
Does `python -m loopgym.rl.train --config X.toml [--resume]` produce a deterministic
GRPO run such that a run killed after step 2 and resumed yields bit-identical
deterministic metrics at step 3 as an uninterrupted 3-step run with the same seed?
Refutation: any differing deterministic metric (loss, reward_mean, ...) at the same
step between killed+resumed and uninterrupted runs.

## Evidence
- Building blocks exist and are checked: `loopgym/rl/rollout.py` HFBackend.generate
  (seeded), `loopgym/rl/batch.py` build_batch/grpo_step, `loopgym/env/toy.py`
  grade (banana), `loopgym/env/exercism.py` grade + runner, `loopgym/env/splits.py`.
- Observed: no `loopgym/rl/train.py`, no `configs/`. tiny_config has vocab 512, so the
  toy smoke needs a 512-token chat tokenizer (new: `tiny_tokenizer()` in models.py).
- Check: tests/test_train.py (new) — 3-step smoke via the real CLI subprocess, and
  killed-after-2 + `--resume` vs uninterrupted, on real configs under a tmp dir.

## Done when
`.venv/bin/python -m pytest -q tests/test_train.py` passes:
1. smoke (tiny_config, toy banana, 3 steps) finishes < 3 min via the CLI, writes
   runs/<name>/metrics.jsonl with the ticket's fields and ckpt/<name>/ checkpoints.
2. stopping after step 2 then `--resume` gives the same step-3 metrics as an
   uninterrupted 3-step run (deterministic fields: step, reward_mean, reward_std,
   frac_groups_all_same, loss, kl, clip_frac, grad_norm).
Known-answer checks: determinism pair above (resume == uninterrupted is the
independent reference); metrics step numbers are exactly 1..3.
Plausible range (written before run): tiny smoke < 120 s total on CPU.

## Stop when
Scope: single-process loop + CLI + toy/exercism sources; loopgym source raises
NotImplementedError (mining is w1's ticket). No FSDP, no vLLM, no GPU, no eval sweep.
Stop/split: after two attempts without new evidence on resume determinism (record
which metric differs and stop); if tiny chat tokenizer cannot be built in < 2 attempts,
split off a tokenizer ticket.

## Plan
1. `tiny_tokenizer()` in loopgym/models.py (byte-level BPE, 400 < 512 vocab, Qwen-style
   chat template, pad/eos).
2. `loopgym/rl/train.py`: TrainCfg dataclass + TOML load + CLI overrides; TaskSampler
   (persisted random state); per step: sample B tasks → HFBackend.generate (seed =
   cfg.seed + s) → grade (ThreadPool; toy in-process with (step, row)) → build_batch →
   grpo_step → append metrics.jsonl → save ckpt every save_every; --resume restores
   step, adapter, optimizer, torch+task RNG.
3. configs/smoke.toml (tiny, toy banana, 3 steps).
4. tests/test_train.py; run it, then make check + make known.
