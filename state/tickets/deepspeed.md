# DeepSpeed ZeRO-3 path

## Issue
The only distributed stack wired to the trainer is FSDP2. Postings ask for DeepSpeed
too; there is no `--dist deepspeed` path, no ZeRO-3 config, and no check that the
two stacks consume the same optimizer hyperparameters from one train config.

## Question
Does the DeepSpeed path build a valid pinned-version (0.19.7) ZeRO-3 engine from the
same train config the FSDP path reads, with identical lr / grad-clip / rows-per-step?
Refutation: `DeepSpeedConfig` (vendored) rejects our config, or lr/clip/micro-batch
differ from the values the FSDP path (AdamW(lr=cfg.lr), clip at cfg.max_grad_norm,
one backward over batch*group rows) uses.

## Evidence
- loopgym/rl/dist.py (FSDP path), w2's train.py TrainCfg (agent/w2: lr, max_grad_norm,
  batch, group) — facts; the ds path does not exist yet.
- Pinned source .tools/src/deepspeed-0.19.7/: initialize(model=, optimizer=, config=)
  in __init__.py:108; config keys in runtime/constants.py, zero/config.py,
  precision_config.py (bf16 has `enabled`; extra="forbid" pydantic sections).
- Check: tests/test_deepspeed_config.py, run first (fails: no configs/ds_zero3.json,
  no loopgym/rl/ds.py). Long output -> .agent-evidence/deepspeed.log.

## Done when
`.venv/bin/python -m pytest -q tests/test_deepspeed_config.py` passes (vendored
DeepSpeedConfig accepts configs/ds_zero3.json merged with a train config; every key
in the vendored constant set; negative control: bogus zero-section key rejected;
lr/clip/micro-batch/accum == FSDP-path values for the same doc; engine builds on the
tiny model, gloo world-1, one forward+step). And
`.venv/bin/python -m loopgym.rl.train --config configs/smoke.toml --dist deepspeed
--dry-run` prints the engine config (train.py lands via w2's merge; if absent at run
time, the same check runs through `python -m loopgym.rl.ds --config <toml> --dry-run`).

## Stop when
No GPU, no DeepSpeed install into .venv (pinned source only, rule 6); no changes to
train.py/batch.py (w2's files) beyond the 2-line `--dist deepspeed` wiring if w2 has
merged. Two attempts at the gloo ZeRO-3 engine build without progress -> record
blocker, keep the config-only checks.

## Plan
1. configs/ds_zero3.json: bf16{enabled}, zero_optimization{stage:3, no offload keys},
   placeholder micro-batch/accum/clip (overridden from the train config).
2. loopgym/rl/ds.py: engine_cfg(doc) (micro = batch*group, accum 1, clip =
   max_grad_norm), make_optimizer (same AdamW line as the FSDP path), build_engine
   (deepspeed.initialize, verified names), --dry-run CLI printing the config.
3. tests/test_deepspeed_config.py: vendored-import (4 test-only stubs: cpuinfo, hjson,
   einops, msgpack) else constant-fallback; key check via AST over the vendored
   constants; identity vs FSDP values; negative control; tiny-model engine smoke.
