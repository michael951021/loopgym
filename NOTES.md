# Notes

## Gotchas
- Bind mounts to Kevin's host (never `rm DIR`): /work, .git, /home/agent, .venv,
  .tools, data, models, ckpt, runs.
- ruff I/EXE pinned, whole tree; known/ I001/F401. .venv: no pip (uv).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.

## Scheduling
- `(human-only[: P])`: tools/sched_guard.py --line <id> (0 ok/1 skip/2 unknown); tagged held; hostgate DONE (030ad80).

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar resume. lgsource: one
  ```diff block; ref = git diff bad..good; empty patch -> 0.0; frozen v0/checks, t=120;
  e2e pass@1 0.0 (range 0-0.25): lgsource-e2e2.log.
- `check_results.py`: PENDING != FAIL; out-of-range only via range-author ruling; base05
  ruling LANDED (64ecd1b); basecheck 5/5 PASS.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  `lgbox.py` http iff LOOPGYM_LGBOX_URL; `splits.py` sha1%100 101/9/27 (137/142, val 9);
  `loopgym.py`: ```path blocks; grade = reply on bad worktree (outside task.files -> 0).

## Kernel
- `token_logprobs(h,w,t,impl,chunk)->[T]`: chunked bwd softmax (no [T,V]); torch,cpp,
  triton (TRITON_INTERPRET=1, C<=1024),cuda host-only. `_host_shim`: CPU test tensors ->
  cuda on a CUDA host (no-op here); triton/cuda untested on real GPU (gpu_run.sh step 5).

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss(denom=None); 7/7 known.
- `batch.py`: grpo_step = grad accumulation over micro_batch row chunks (token-mean:
  global mask denom; seq-mean: scale rows_c/n); logp_of = base Qwen2Model (no lm_head)
  + head weight + chunked kernel per row chunk (never [rows,T,V]); ref=disable_adapter.
- `rollout.py` HFBackend(micro_batch=8): left pad, output_scores fp32, generate in row
  chunks w/ per-chunk seed (seed+7919*chunk); scores freed per chunk; top_k=0, rep=1.0.
- `rollout_vllm.py` (host): bf16 enable_lora r16; sync_lora id+1/step; 0.02-nat gate;
  build_llm() lazy (vLLM never imported in sandbox).
- grpomem DONE: grpo05 OOM'd at 24 G (full-vocab logits [32,~800,151936] fp32 + dlogits
  + rollout scores). micro_batch=8 key (TrainCfg, RESUME_CHECK, toml). 2-step run exit 0,
  peak RSS 25.1 GiB: .agent-evidence/grpomem-2step.log. grpo05 (mem: 24)->33 = peak GiB
  x1.3 rounded up. tests/test_grpomem.py 3/3: mb=1 grads == big batch 1e-5.
- gpurun: scripts/gpu_run.sh = Kevin's GPU window: 2 GPU preflight, --stop-agents,
  .venv-gpu, known tests on CUDA, check_vllm_hf <0.02-nats, 1.5B GRPO, 3 evals -> results/gpu/.
- train.py: toml+--k v; sample->gen->grade->grpo_step->jsonl->ckpt; exact resume, exit 1
  on orphan ckpt; from_pretrained freezes ALL -> requires_grad lora_; vllm: sync_vllm
  adapter+sync_lora, restore re-syncs, first gen unlora.
- max_difficulty caps sampler (67 of 101 train <=3); eval_n stateless draw seed+off+s.
  grpo05_exercism.toml (0.5B r16, diff<=3, B4 G8 384tok, lr2e-5, b0.02, 40 steps,
  eval 20/n2, save 5, micro_batch 8); probe 524s/step, reward 0.25, tok-mean 0.0957
  vs seq-mean 0.0 = length artifact, not a bug.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors, grads AVG, no 1/W.
  `ds.py` ZeRO-3 bf16 stage3; micro=b*group; CPU compatible_ops[deepspeed_shm_comm]=False.
- Toy gates PASS: banana 0.148 -> 1.000; null slope CI contains 0.
- `tools/fetch_src.sh`: vllm/deepspeed/trl -> .tools/src/ (rule 6).

## Data
- `freeze.py`: v0 = 26 verified (draftlab 22/harness 4), train 13/test 13; `--check`
  exit 1 on drift; public draftlab/harness only. `data/verify.py`: 3x/side pristine
  worktrees -> verified.jsonl; tests=0 -> error.
- hidden checks: synthetic fixtures, temp root; assert completion claim; cross-check
  vs results/*.json (lazy discovery: absent at bad).
