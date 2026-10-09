# Notes

Current state for the next run (keep < 4 KB).

## Foundations
- `make bootstrap` (pins versions.txt): .venv py3.12 (torch 2.14.1+cpu, transformers
  5.19.0, peft, accelerate, ruff, pytest), rust 1.99.0.
- `make check` = ruff (E4/E7/E9/F, known out) + pytest (minus known) + cargo + shellcheck
  + hadolint + terraform validate. `make known` = kevin-only + clean worktree + strict known.

## Eval
- `bootstrap_ci(values,groups)->(mean,lo,hi)`; `pass_at_k` (Chen 2021), CI over tasks.
  4/4+5/5 known.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` vocab-400 byte-BPE, Qwen tags, full
  byte alphabet (no <unk>). GOTCHA: config vocab 151936 (training) vs tok len 151,643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip; `_xml_counts`/`_reward` shared by BOTH runners.
  `lgbox.py`: LgboxClient + RUNNERS["http"] (iff LOOPGYM_LGBOX_URL); fixed read_tail.
  `toy.py`: banana (first 20 tok ci), coinflip (seeded, text-blind).
  `splits.py`: bucket=sha1%100, 101/9/27, 137/142 usable.

## lgbox
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT, 4 KB tails, mpsc(Q)->503), /health,
  /metrics. Firecracker Vmm+VmPool; --dry-run; FakeVmm-only (rule 4). bench --quick 120/120.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]`: chunked online max/LSE; backward
  recomputes per-chunk softmax (no [T,V] saved); grads wrt hidden+weight; register_impl
  cpp/triton/cuda. 6/6 known (0.5B==HF 1e-3, gradcheck); peak << 1/8 naive. GOTCHA: keep
  input dtype end-to-end.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py` Batch (target_mask col j=1 iff col predicts a completion token, slice
  [p-1,p+c-1)), StepCfg, build_batch, logp_of, grpo_step (logp via token_logprobs on
  hidden_states[-1][:,:-1]; old=detach; ref via disable_adapter or ref_fn). 5/5 (2 s):
  step-0 β=0 loss -9.9e-9. GOTCHA: no last_hidden_state in 5.x; kernel unbatched -> flatten.
- `rollout.py` HFBackend (chat, left pad, rows x n, output_scores -> fp32 per-token logp;
  T<=0 greedy; trim eos). GOTCHA: gen_config top_k=20+rep 1.05 -> pass top_k=0, rep=1.0.
  Known real 0.5B |Δ| 2.8e-5 < 1e-4 (T=1).
- `train.py` TrainCfg (toml + CLI --k v), TaskSampler (persisted random), Run: sample B ->
  generate (seed=seed+s) -> grade (ThreadPool) -> grpo_step -> metrics.jsonl -> ckpt
  <out>^2/ckpt/<name>. Exact resume: ckpt=step+adapter+opt+torch/task RNG; --resume checks
  RESUME_CHECK (lora_targets tuple vs JSON list); eval seed=seed+1e9+3+s. 5/5 (28 s): smoke
  <3 min; killed-2 + --resume == uninterrupted bit-identical. GOTCHA: from_pretrained leaves
  ALL frozen -> requires_grad=("lora_" in name) before AdamW.
- `dist.py` FSDP2: init_dist (env://; standalone file://), wrap_fsdp (fully_shard per layer
  + root), full_state_dict (collective), global_mean_loss. Equiv 2 ranks x b == single 2b:
  param diff 0.0.
- Toy gates: configs/toy_{banana,null}.toml (0.5b r16 B4 G8 24tok lr5e-5 β0 60st);
  `tools/toy_check.py [--dir runs]`: means steps 0-4/55-59 + bootstrap slope CI; exit 1
  unless banana <0.2->>0.8 and null CI ∋ 0. Probe 2st 46s (~21 s/step; 60st ≈ 21 min).
  OBS: step-1 banana reward 0.281 > 0.2 gate start -> toybanana run risks start criterion.

## Data
- tests/known/ = Kevin's (rule 1); data/raw/ = host export. `data/mine.py`: keyword fixes x
  blame -> (bad,good) pairs, <=200 lines/4 files + 3 seeds; 141 cands, 3/3 recall.
  .gitignore: /data/ absolute.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`,
  never `rm DIR` (EBUSY).
- rust 1.99.0 (1.100/1.101 404); `Mutex::lock()` -> Result -> .unwrap().
- ruff 0.16 drifted (I/EXE) pinned in pyproject; tests/known/ I001/F401 (rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient "Cannot fork"; tests use /bin/sleep.
- torch 2.14 FSDP2: get_model_state_dict -> DTensors (no StateDictType; .full_tensor()
  collective); fully_shard auto-inits env://; rank() pre-init 0; grads AVG (W>1) ->
  backward per-token-mean loss, no 1/W scaling.
