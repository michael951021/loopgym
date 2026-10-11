# grpomem — GRPO step memory fix

## Issue
Job grpo05 (0.5B, B=4 G=8, 384 new tokens, 24 GB MemoryMax) was OOM-killed twice
~5m45s in, before step 1 finished (logs/jobs/loopgym-grpo05.log). The grpocfg 1-step
probe ran uncapped, so the cap never caught it. Blocks the CPU GRPO run.

## Question
Does `logp_of` (loopgym/rl/batch.py) hold the full [rows, T, vocab] fp32 logits of the
model's lm_head forward plus its backward (dlogits), on top of the 32-row forward
activations, for the whole step? Refutation: a capped 2-step run staying under 24 GB
with only row micro-batching would show the logits tensor is not the dominant holder.

## Evidence
- Observed: OOM at 24 GB (job log); `Qwen2ForCausalLM.forward` (transformers 5.19)
  computes `logits = self.lm_head(...)` for all positions ([32, ~800, 151936] fp32
  ≈ 15 GB) and backward materializes dlogits of the same size; HFBackend.generate
  keeps `scores` = [steps, rows, V] fp32 (≈ 7.5 GB at 32 rows × 384 steps).
- Smallest repro: probe in this ticket's run — forward 32×800 ids through the 0.5B
  policy, print `torch.cuda`-free peak via resource.getrusage in a child; saved to
  .agent-evidence/grpomem-probe.log.
- Guess: CPU attention backend (math vs efficient) decides per-layer backward
  activation size; measured by the 2-step run, not assumed.

## Done when
- `tests/test_grpomem.py` passes: micro-batched (micro_batch=1) gradients equal the
  single-big-batch reference within 1e-5 on tiny_config; loss and stats match.
- `.venv/bin/python -m loopgym.rl.train --config configs/grpo05_exercism.toml
  --steps 2 --out runs/grpomem_probe` exits 0 under `systemd-run --user --scope
  -p MemoryMax=24G -p MemorySwapMax=0` (fallback `/usr/bin/time -v`), peak RSS
  recorded in NOTES.md below the grpo05 line's (mem: 24) value.
- Known-answer reference for the fix: single-big-batch gradient accumulation is the
  independent reference; micro-batch loss equals it (token-mean, global denominator).
- Plausible range (written before the run): peak RSS with micro_batch=8 ≤ 20 GB;
  2-step wall time 10–60 min (known CPU step range 5–25 min).

## Stop when
- Out of scope: vLLM backend memory, GPU paths, kernel chunk size, the 40-step run.
- Stop/split: after two attempts without new evidence (e.g. still OOM at mb=1),
  record the dominant allocation from the probe and split.

## Plan
1. Probe: measure current vs row-chunked forward peak on the real 0.5B.
2. batch.py: logp_of via base Qwen2Model (no lm_head), rows in micro-batches;
   grpo_step = gradient accumulation with global loss denominator; policy_loss
   gains optional `denom` (defaults unchanged); StepCfg.micro_batch.
3. rollout.py: generate in row chunks, per-chunk seed, scores freed per chunk.
4. train.py: TrainCfg.micro_batch (default 8), RESUME_CHECK; toml key.
5. tests/test_grpomem.py; make check; capped 2-step run; NOTES.md.
