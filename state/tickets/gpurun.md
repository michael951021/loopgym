# gpurun — Host GPU run script

## Issue
Kevin's GPU window needs one command: stop agents, build .venv-gpu, run the GPU
known tests + vLLM-vs-HF check, train 1.5B GRPO (vLLM rollouts GPU0, trainer
GPU1), then eval → results/gpu/. Today there is no script, no 1.5B config, and
train.py has no vLLM path at all.

## Question
What must `scripts/gpu_run.sh` do, and what code has to exist for the plan to be
honest (vLLM backend wired, GPU device moves, kernel shims, vLLM-vs-HF checker)?

## Evidence
- train.py hardcodes HFBackend (run.py:178,305); rollout_vllm.py exists but is
  unwired (vllm.md deferred wiring to w1/gpu_run).
- transformers 5.19 `generate()` only WARNS on device mismatch → explicit
  .to(device) needed in HFBackend.generate, batch.logp_of, eval load_model.
- tests/known use CPU tensors; cuda ext hard-requires CUDA tensors →
  triton/cuda impls need a transparent CPU→CUDA move on the host.
- vLLM-vs-HF known answer (0.02 nats) has no checker script yet.
- 26 loopgym v0 tasks (13 train / 13 test, no val); loopgym eval = test split.

## Done when
`bash scripts/gpu_run.sh --dry-run` exits 0 and prints the plan + file checks;
`.tools/bin/shellcheck scripts/gpu_run.sh` clean; configs/grpo15_gpu.toml valid;
pytest tests/ green (dry-run, vllm wiring with FakeLLM, check_vllm_hf --dry-run).

## Stop when
Dry-run can't pass without editing tests/known/ or a shared data dir, or two
attempts at the vLLM wiring yield no new evidence (record NOTES.md, RESULT: not
done). No GPU work here (rule 2): host-only paths are written + dry-run only.

## Plan
1. scripts/gpu_run.sh: flags --dry-run/--stop-agents; env-overridable model/data
   paths; dry-run = plan + required-file check (exit 1 if missing).
2. Full run: preflight (2 GPUs) → ./agent stop all --now (if --stop-agents) →
   uv .venv-gpu (torch CUDA, vllm==0.31.0, deepspeed==0.19.7, stamp) → IMPLS
   ⊇{cuda,triton} → pytest tests/known (CUDA_VISIBLE_DEVICES=0) →
   check_vllm_hf → train --config configs/grpo15_gpu.toml --resume-if-any →
   3 evals (exercism val/test, loopgym test) → results/gpu/.
3. Code: TrainCfg.backend/trainer_device; VLLMBackend selection; per-step adapter
   save + sync_lora (fresh name/step); restore re-syncs; device moves; kernel
   shims; loopgym/rl/check_vllm_hf.py (--dry-run plan, real run gate fp32<0.02).
4. Tests: tests/test_gpu_run.py, tests/test_train_vllm.py (FakeLLM), check dry-run.
