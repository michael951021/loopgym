# Notes

State for next run (<4KB). Evidence: .agent-evidence/*.log.

## Foundations
- `make bootstrap` (pins versions.txt): .venv py3.12 (torch 2.14.1+cpu, transformers 5.19.0,
  peft, accelerate, triton 3.8.0, ruff, pytest), rust 1.99.0.
- `make check` = ruff (E4/E7/E9/F, known out) + pytest (minus known) + cargo + lints;
  `make known` = kevin-only + strict known.

## Eval
- `eval/stats.py` bootstrap_ci(values, groups); `eval/passk.py` pass_at_k (Chen 2021) + task CI.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151,643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit, pgroup SIGKILL; reward 1 iff
  tests>0 no fail/err/skip; `_xml_counts`/`_reward` shared both runners. `lgbox.py`: LgboxClient +
  RUNNERS["http"] (iff LOOPGYM_LGBOX_URL). `toy.py`: banana (first-20-tok ci), coinflip (seeded).
  `splits.py`: sha1%100, 101/9/27.

## lgbox
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT, mpsc(Q)->503), /health, /metrics. Firecracker
  Vmm+VmPool; FakeVmm-only (rule 4). bench --quick 120/120.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]`: V-tile online max/LSE; backward recomputes
  per-chunk softmax (no [T,V] saved); grads h+w; impls torch, triton. 6/6 known (0.5B==HF 1e-3,
  gradcheck); peak << 1/8 naive. GOTCHA: keep input dtype.
- `triton_logprob.py`: fwd 1 prog/row, V-tile online max/LSE + target logit; bwd_h 1
  prog/row (acc p·w); bwd_w 1 prog/V-tile (row loop, no atomics); f64 ok. Registered iff CUDA or
  TRITON_INTERPRET=1. GOTCHA: interpreter block cap 1048576 -> C<=1024 in interpret; masked V
  cols s=-inf; target-acc init -inf. 6/6 strict incl. model: triton-accept2.log.

## RL
- `rl/grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `rl/batch.py`: build_batch, logp_of, grpo_step (logp via token_logprobs(hidden_states[-1][:,:-1]);
  old=detach; ref=disable_adapter). 5/5: step-0 β=0 loss -9.9e-9. GOTCHA: no last_hidden_state in
  5.x; kernel unbatched -> flatten.
- `rl/rollout.py` HFBackend (left pad, fp32 per-token logp; T<=0 greedy). GOTCHA: pass top_k=0,
  rep=1.0 (gen_config defaults). Real 0.5B |Δ| 2.8e-5.
- `rl/rollout_vllm.py` VLLMBackend (host): bf16, enable_lora, max_loras=1, max_lora_rank=16;
  sync_lora->LoRARequest(load_inplace=True), id+1/step; logprobs=0->sampler_logp; top_k=0 rep=1
  min_p=0; explicit max_tokens (pin=16); import try/except. 14/14 ast vs vllm-0.31.0 (rule 6).
  0.02-nat check = host.
- `rl/train.py`: toml+CLI, Run sample->generate(seed+s)->grade(ThreadPool)->grpo_step->
  metrics.jsonl->ckpt; --dist {fsdp,deepspeed}+--dry-run (ds->rl/ds.py). Exact resume
  ckpt=step+adapter+opt+RNG; --resume checks RESUME_CHECK. 5/5. GOTCHA: from_pretrained leaves
  ALL frozen -> requires_grad=("lora_" in name) before AdamW.
- `rl/dist.py` FSDP2: init_dist, wrap_fsdp, full_state_dict, global_mean_loss. torch 2.14:
  DTensors (.full_tensor()); grads AVG (W>1) -> no 1/W. 2 ranks x b == single 2b: diff 0.0.
- `rl/ds.py` ZeRO-3 (0.19.7, lazy): ds_zero3.json bf16 stage3; micro=batch*group/accum 1. 6/6.
  CPU: no ninja -> ops.__compatible_ops__[deepspeed_shm_comm]=False.
- Toy gates: configs/toy_{banana,null}.toml; `tools/toy_check.py` means steps 0-4/55-59 + slope
  CI; exit 1 unless banana <0.2->>0.8 & null CI ∋ 0. OBS: step-1 0.281 > 0.2; 60-st must confirm.
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 / trl 1.15.0 -> .tools/src/.

## Data
- tests/known/ = Kevin's (rule 1); data/raw/ = host export. `data/mine.py`: fixes %B x blame ->
  (bad,good), <=200 lines/4 files + 3 seeds; 141 cands, 3/3 recall.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`, never `rm`.
- rust 1.99 (1.100+ 404); `Mutex::lock()` -> Result -> .unwrap().
- ruff 0.16 I/EXE pinned in pyproject; known/ I001/F401 (rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient "Cannot fork"; tests use /bin/sleep.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- NOTES.md: merges can re-introduce duplicate sections; condense when editing.
