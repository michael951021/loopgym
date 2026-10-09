# Notes

State for next run (<4KB). Evidence: .agent-evidence/*.log.

## Foundations
- `make bootstrap` (pins versions.txt): .venv py3.12 (torch 2.14.1+cpu, transformers 5.19.0,
  triton 3.8.0, ninja, ruff, pytest), rust 1.99.0.
- `make check` = ruff + pytest (known out) + cargo + shellcheck + hadolint + tf validate;
  `make known` = kevin-only + strict known.

## Eval
- `bootstrap_ci(values,groups)`; `pass_at_k` (Chen 2021), CI over tasks.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151,643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip; `_xml_counts` shared by BOTH runners. `lgbox.py`:
  RUNNERS["http"] iff LOOPGYM_LGBOX_URL. `toy.py`: banana (first 20 tok ci), coinflip (seeded).
  `splits.py`: sha1%100, 101/9/27, 137/142 usable.

## lgbox
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT, 4 KB tails, mpsc(Q)->503), /health,
  /metrics; Firecracker Vmm+VmPool, --dry-run, FakeVmm-only (rule 4); bench 120/120.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]`: chunked online max/LSE, bwd per-chunk softmax
  (no [T,V] saved); impls torch, cpp, triton. Known: 0.5B==HF 1e-3, f64 gradcheck; peak << 1/8
  naive. Keep input dtype.
  cpp: csrc/logprob_cpu.cpp JIT (ninja, .tools/build, -O3 -fopenmp), OpenMP over T, f32+f64;
  bwd = torch placeholder (cppbwd).
- `triton_logprob.py`: fwd 1 prog/row, V-tile online max/LSE + target logit; bwd_h 1 prog/row
  (acc p·w); bwd_w 1 prog/V-tile (row loop, no atomics); f64 ok. Registered iff CUDA or
  TRITON_INTERPRET=1. GOTCHA: interpret block cap 1048576 -> C<=1024; masked V cols s=-inf;
  target-acc init -inf. strict incl. model: triton-accept2.log.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: Batch (mask slice [p-1,p+c-1)), build_batch, grpo_step (logp via
  token_logprobs(hidden_states[-1][:,:-1]); old=detach; ref=disable_adapter); step-0 β=0 ~0.
  GOTCHA: no last_hidden_state in 5.x; kernel unbatched -> flatten.
- `rollout.py` HFBackend (left pad, output_scores fp32 logp; T<=0 greedy). GOTCHA: pass
  top_k=0, rep=1.0 (gen defaults leak). Real 0.5B |Δ| 2.8e-5.
- `rollout_vllm.py` (host): bf16, enable_lora (rank 16); sync_lora ->
  LoRARequest(load_inplace) id+1/step; logprobs=0 -> sampler_logp; explicit max_tokens
  (pinned def 16); vllm try/except (sandbox fakes). 14/14 ast vs .tools/src (rule 6);
  0.02-nat = host.
- `train.py`: TrainCfg (toml + --k v): sample->gen(seed+s)->grade->grpo_step->
  jsonl->ckpt <out>^2/ckpt/<name>; --dist {fsdp,deepspeed}+--dry-run; exact resume
  (RESUME_CHECK; eval seed+1e9+3+s); killed+resume bit-identical.
  GOTCHA: from_pretrained freezes ALL -> requires_grad=("lora_" in name) before AdamW.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors (.full_tensor()), grads AVG
  (W>1), no 1/W; 2x b == 2b: 0.0.
- `ds.py` ZeRO-3 (0.19.7, not in .venv, lazy): ds_zero3.json (bf16 stage3, offload off);
  micro=batch*group, accum 1; 6/6. CPU:
  ops.__compatible_ops__[deepspeed_shm_comm]=False (no ninja).
- Toy gates: toy_{banana,null}.toml (r16 B4 G8 24tok lr5e-5 β0 60st); toy_check.py:
  means 0-4/55-59 + slope CI, exit 1 unless banana <0.2->>0.8 & null CI ∋ 0. OBS: step-1
  0.281 > 0.2, 60-st run must confirm (rule 5).
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 / trl 1.15.0 -> .tools/src/.

## Data
- tests/known/ = Kevin's (rule 1); data/raw/ = host export.
- `data/mine.py`: keyword fixes in %B x blame -> (bad,good), <=200 lines/4 files + 3 seeds;
  141 cands, 3/3 recall.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`,
  never `rm DIR` (EBUSY).
- rust 1.99.0 (1.100+ 404); `Mutex::lock()` -> Result.
- ruff 0.16 I/EXE pinned in pyproject; known/ I001/F401 (rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork; tests /bin/sleep.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- NOTES.md: merges can re-introduce duplicate sections; condense when editing.
