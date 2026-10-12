# twonode — local two-node rehearsal

## Issue
No multi-node rehearsal exists: FSDP/ZeRO sharding is only proven on 2 local CPU
ranks (tests/test_fsdp_equiv.py, tests/test_deepspeed_config.py) — never through
`torchrun --nnodes 2`. The AWS scaling study (2× g4dn.xlarge) would run
unrehearsed rendezvous, sharding and the train.py multi-rank path.

## Question
Do the generated 1-node / 2-local-node / 2-AWS-node torchrun commands share one
template differing only in node rank, endpoint and device list, and does the
2-rank train.py path reproduce the mirrored single-process 2b step within 1e-5
(params), for both `--dist fsdp` and `--dist deepspeed`?

## Evidence
- `scripts/gpu_run.sh` rejects `--two-node-local` (exit 2, "unknown flag").
- `loopgym/rl/train.py` has no WORLD_SIZE>1 path: no init_dist/wrap_fsdp/engine.
- `loopgym/rl/launch.py` and `configs/scale05.toml` do not exist.
- Known answers: 2 ranks × b ≡ single 2b within 1e-5 (CLAUDE.md, proven in
  fsdp ticket); ZeRO-3 gradient reduce divides by dp_world_size (pinned
  0.19.7 stage_1_and_2.py:1329-1348) → same AVG math as FSDP2.

## Done when
`.venv/bin/python -m pytest -q tests/test_launch_cmds.py && bash scripts/gpu_run.sh --two-node-local --dry-run`
exits 0.

## Stop when
No GPU in the sandbox: the 20-step GPU runs stay in the host script (dry-run
verified here). DeepSpeed world-2 gloo CPU smoke: 2 attempts, then record as
host-only risk in NOTES.md and keep the FSDP result.

## Plan
1. launch.py: node-launch generator (single / two-local / two-aws, c10d
   127.0.0.1:29500, one GPU per rank) + metrics report aggregator, CLI.
2. configs/scale05.toml: 0.5B, B2 G8 256tok, 20 steps, no eval, trainer_device
   cuda:0 (one visible GPU per rank in every topology), logprob_impl=torch.
3. train.py: WORLD_SIZE>1 → init_dist + wrap_fsdp (fsdp) or ds.build_engine
   (deepspeed); per-rank task slice (same seed → same draw); rank-0 metrics and
   save (full_state_dict / save_16bit_model, collective); global token-mean
   loss; per-rank peak mem; engine.backward/step for ZeRO (clips w/ global norm).
4. gpu_run.sh --two-node-local: dry-run = plan + generated commands + file
   checks exit 0; real = stop-agents (required) + .venv-gpu + for fsdp,
   deepspeed: 2 concurrent torchrun, then report →
   results/gpu/scale_local_{dist}.json.
5. tests: test_launch_cmds.py (template invariance + hand-written command),
   test_train_dist.py (2-rank CPU gloo tiny run ≡ mirrored single 2b, 1e-5).
