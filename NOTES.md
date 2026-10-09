# Notes

Current state for next run (keep <4KB). Evidence: .agent-evidence/*.log.

## Foundations
- `make bootstrap` (pins versions.txt): .venv py3.12.14 (torch 2.14.1+cpu, transformers 5.19.0,
  peft, accelerate, ruff, pytest), rust 1.99.0.
- `make check` = ruff (E4/E7/E9/F, known out) + pytest (minus known) + cargo + shellcheck +
  hadolint + terraform validate. `make known` = kevin-only + strict known.

## Eval
- `eval/stats.py` bootstrap_ci(values, groups)->(mean,lo,hi); `eval/passk.py` pass_at_k (Chen
  2021) + task CI. Known 4/4+5/5.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` vocab-400 byte-BPE. GOTCHA: config vocab
  151936 vs tokenizer 151,643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL; reward
  1 iff tests>0 no fail/err/skip; `_xml_counts`/`_reward` shared by BOTH runners. `lgbox.py`:
  LgboxClient + RUNNERS["http"] (iff LOOPGYM_LGBOX_URL). `toy.py`: banana (first 20 tok ci),
  coinflip (seeded, text-blind). `splits.py`: sha1%100, 101/9/27, 137/142 usable.

## lgbox
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT, mpsc(Q)->503), /health, /metrics. Firecracker
  Vmm+VmPool; --dry-run; FakeVmm-only (rule 4). bench --quick 120/120.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]`: chunked online max/LSE; backward recomputes
  per-chunk softmax (no [T,V] saved); grads wrt hidden+weight. 6/6 known (0.5B==HF 1e-3,
  gradcheck); peak << 1/8 naive. GOTCHA: keep input dtype.

## RL
- `rl/grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `rl/batch.py`: Batch (target_mask slice [p-1,p+c-1)), build_batch, logp_of, grpo_step (logp
  via token_logprobs on hidden[-1][:,:-1]; old=detach; ref via disable_adapter). 5/5: step-0
  β=0 loss -9.9e-9. GOTCHA: no last_hidden_state in 5.x; kernel unbatched.
- `rl/rollout.py` HFBackend (left pad, output_scores -> fp32 per-token logp; T<=0 greedy; trim
  eos). GOTCHA: gen_config top_k=20+rep 1.05 -> pass top_k=0, rep=1.0. Real 0.5B |Δ| 2.8e-5.
- `rl/train.py` TrainCfg (toml + CLI --k v), TaskSampler, Run: sample->generate->grade
  (ThreadPool)->grpo_step->metrics.jsonl->ckpt. Exact resume: ckpt=step+adapter+opt+RNG;
  --resume checks RESUME_CHECK. 5/5 (28 s): killed-2 + --resume bit-identical. GOTCHA:
  from_pretrained leaves ALL frozen -> requires_grad=("lora_" in name) before AdamW.
- `rl/dist.py` FSDP2: init_dist (env://; standalone file://), wrap_fsdp (fully_shard per layer
  + root), full_state_dict (collective), global_mean_loss. 2 ranks x b == single 2b: diff 0.0.
- `rl/ds.py` DeepSpeed ZeRO-3 (0.19.7, not in .venv, lazy): ds_zero3.json (bf16, stage3,
  offload off); micro=batch*group/accum 1/clip=max_grad_norm; AdamW(lr); initialize;
  train.py --dist {fsdp,deepspeed} --dry-run. 6/6 (vendored cfg ok, bogus key rejected, tiny
  gloo world-1 steps). CPU: ops.__compatible_ops__[deepspeed_shm_comm]=False (no ninja) +
  LOCAL_RANK/RANK/WORLD_SIZE.
- Toy gates: configs/toy_{banana,null}.toml (0.5b r16 B4 G8 24tok lr5e-5 β0 60st);
  `tools/toy_check.py`: means steps 0-4/55-59 + slope CI; exit 1 unless banana <0.2->>0.8 &
  null CI ∋ 0. OBS: step-1 banana 0.281 > 0.2 -> 60-step run must confirm (rule 5).
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 / trl 1.15.0 -> read-only .tools/src/.

## Data
- tests/known/ = Kevin's (read-only, rule 1); data/raw/ = host export.
- `data/mine.py`: keyword fixes in %B x blame of deleted lines -> (bad,good) pairs, <=200 lines
  /4 files + 3 seeds; 141 cands, 3/3 recall; 5/5 tests.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`, never `rm`.
- rust pin 1.99.0 (1.100/1.101 404); std `Mutex::lock()` -> Result -> .unwrap().
- ruff 0.16 drifted (I/EXE) — pinned in [tool.ruff.lint]; known I001/F401 (rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks -> transient "Cannot fork"; tests use /bin/sleep.
- torch 2.14 FSDP2: get_model_state_dict -> DTensors (.full_tensor() collective); fully_shard
  auto-inits env://; rank() pre-init 0; grads AVG-reduce (world>1) -> backward token-mean, no
  1/W scaling.
