# fsdp — FSDP2 path and equivalence test

## Issue
No distributed training path exists: `train.py` (w2) will need `--dist fsdp` under torchrun, and the known answer "FSDP2 2 ranks × batch b == single process batch 2b, params within 1e-5" is unverified. Multi-node runs are only as good as this equivalence.

## Question
Does one AdamW step on FSDP2 (fully_shard per decoder layer + root, gloo, 2 ranks, per-rank mean loss) reproduce the single-process step on 2b (tiny_config, fp32) within 1e-5 on every parameter, with both ranks reporting one identical global loss? Falsifier: max param diff > 1e-5, or the two ranks' reported losses differ.

## Evidence
Observed: pinned torch 2.14.1 FSDP2 (`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:_get_gradient_divide_factors`) defaults fp32/world>1 to `ReduceOp.AVG` reduce-scatter → per-rank *mean* loss already equals full-batch mean gradient; no loss scaling needed. tiny_config dims all even (2-way shardable). Final run (fsdp_equiv6.log + direct torchrun): both ranks `max_param_diff=0.000e+00` (bit-identical to the single-process 2b step), loss=loss_ref=6.247735, loss_span=0. Two bugs fixed along the way: (1) worker called `rank()` before the group existed (init happens in `wrap_fsdp`) → both ranks took slice 0, diff 2e-3 = 2×LR; (2) reference backward used the raw 60-token sum while ranks use the per-token mean → Adam near-eps elements diverge (observed 1.253e-4 ≈ lr/6 ceiling). Reference now mirrors the ranks: two b-row forwards, mean loss.

## Done when
`.venv/bin/python -m pytest -q tests/test_fsdp_equiv.py` passes in < 5 min. Worker = torchrun --nproc_per_node=2 over tests/_fsdp_equiv_worker.py: 1 AdamW step, per-rank b=2 vs single-process 2b=4, T=16, seed 1234; known answer = hand-built reference step (independent, non-FSDP model) — max |Δparam| < 1e-5, reported global loss equal across ranks and within 1e-5 of the reference mean loss.

## Stop when
No train.py CLI wiring in this ticket (w2 owns train.py; dist.py exposes init_dist/wrap_fsdp/full_state_dict/is_rank0 for it). Stop/split after 2 attempts if gloo AVG reduce-scatter is unsupported or diff > 1e-5 persists with a correct reference.

## Plan
1. loopgym/rl/dist.py: init_dist (gloo CPU / nccl CUDA, env:// under torchrun), wrap_fsdp(model, cfg) fully_shard per layer + root, is_rank0/rank/world_size, full_state_dict (collective; rank-0 persists), global_mean_loss (all-reduce sum+count).
2. tests/_fsdp_equiv_worker.py + tests/test_fsdp_equiv.py (spawns torchrun, free port, OMP_NUM_THREADS=8, parses max_param_diff from output, asserts < 1e-5 and rc==0).
3. Run test, then `make check`.
