# Notes

Current state for next run (keep <4KB). Evidence: .agent-evidence/*.log.
Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins versions.txt): uv 0.12.24, .venv py3.12.14
  (torch 2.14.1+cpu, transformers 5.19.0, ruff, pytest), rust 1.99.0, linters in .tools/bin.
- `make check` = ruff (E4/E7/E9/F, known excluded) + pytest (minus known) + cargo test
  + shellcheck + hadolint + terraform validate. `make known` = kevin-only + strict known.
## Foundations
- `make bootstrap` (pins versions.txt): .venv py3.12 (torch 2.14.1+cpu, transformers
  5.19.0, peft, accelerate, ruff, pytest), rust 1.99.0.
- `make check` = ruff (E4/E7/E9/F, known out) + pytest (minus known) + cargo + shellcheck
  + hadolint + terraform validate. `make known` = kevin-only + clean worktree + strict known.

## Eval (done 2026-10-09)
- `eval/stats.py` bootstrap_ci(values, groups) -> (mean, lo, hi); resamples task groups.
  `eval/passk.py` pass_at_k (Chen 2021); summarize() -> per-task pass@k +
  task CI. Known 4/4 + 5/5.
## Eval
- `bootstrap_ci(values,groups)->(mean,lo,hi)`; `pass_at_k` (Chen 2021), CI over tasks.
  4/4+5/5 known.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` vocab-400 byte-BPE, Qwen tags, full
  byte alphabet (no <unk>). GOTCHA: config vocab 151936 (training) vs tok len 151,643.

## Env (done 2026-10-09)
- `exercism.py` local runner: fresh tmpdir, pristine tests + ONLY solution, junit XML,
  pgroup SIGKILL timeout; reward 1 iff tests>0 no fail/err/skip; `_xml_counts`/
  `_reward` shared by BOTH runners.
- `lgbox.py`: LgboxClient + RUNNERS["http"] (active iff LOOPGYM_LGBOX_URL). `toy.py`: banana
  (first 20 tokens ci), coinflip (seeded, text-blind). `splits.py`: bucket=sha1%100,
  101/9/27, 137/142 usable.
## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip; `_xml_counts`/`_reward` shared by BOTH runners.
  `lgbox.py`: LgboxClient + RUNNERS["http"] (iff LOOPGYM_LGBOX_URL); fixed read_tail.
  `toy.py`: banana (first 20 tok ci), coinflip (seeded, text-blind).
  `splits.py`: bucket=sha1%100, 101/9/27, 137/142 usable.

## lgbox server (done 2026-10-09)
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT AS/CPU/NPROC/FSIZE, 4 KB tails,
  mpsc(Q)->503+Retry-After), /health, /metrics. Firecracker: VmPool+Vmm, --dry-run,
  FakeVmm-only (rule 4).
- `tools/bench_lgbox.py`: --quick 11.5 s, 120/120 reward 1.
## lgbox
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT, 4 KB tails, mpsc(Q)->503), /health,
  /metrics. Firecracker Vmm+VmPool; --dry-run; FakeVmm-only (rule 4). bench --quick 120/120.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]`: chunked online max/LSE; backward
  recomputes per-chunk softmax (no [T,V] saved); grads wrt hidden+weight; register_impl
  cpp/triton/cuda. 6/6 known (0.5B==HF 1e-3, gradcheck); peak << 1/8 naive. GOTCHA: keep
  input dtype end-to-end.

## RL
- `rl/grpo.py` group_advantages/kl_k3/policy_loss; known 7/7.
- `rl/batch.py`: Batch (target_mask slice [p-1, p+c-1)), StepCfg, build_batch, logp_of,
  grpo_step (recomputed logp; old=detach; ref via disable_adapter; stats
  loss/kl/clip_frac/ratio/grad_norm/tokens). 5/5 (2 s): step-0 β=0 loss -9.9e-9.
- `rl/rollout.py` HFBackend (left pad, output_scores -> fp32 per-token logp;
  T<=0 greedy; trim at eos). GOTCHA: shipped gen_cfg top_k=20 + rep-pen 1.05 -> pass
  top_k=0, rep-pen=1.0. Known: real 0.5B |Δ| 2.8e-5 < 1e-4 vs fresh forward (T=1 fp32); 3/3.
- `rl/dist.py` FSDP2: init_dist (env://; world-1 file:// store), wrap_fsdp (fully_shard
  per layer + root), full_state_dict (collective), global_mean_loss. 2 ranks x b == single
  2b (tiny fp32): param diff 0.0, one loss; 1/1.
- `rl/ds.py` DeepSpeed ZeRO-3 (0.19.7, not in .venv): configs/ds_zero3.json (bf16, stage3,
  offload off); micro=batch*group/accum 1/clip=max_grad_norm; AdamW(lr); initialize;
  --dry-run. 6/6 (vendored cfg ok, bogus key rejected, tiny gloo world-1 steps).
  Follow-up: --dist deepspeed in train.py (w2's).
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 / trl 1.15.0 -> read-only
  .tools/src/; pins versions.txt. Never in .venv.
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
- tests/known/ = Kevin's (read-only, rule 1); data/raw/ = host export.
- `data/mine.py`: keyword fixes in %B x blame of deleted lines -> (bad, good) pairs,
  defect diff <=200 lines/4 files + 3 seeds; 141 candidates, 3/3 recall; 5/5 tests.
- tests/known/ = Kevin's (rule 1); data/raw/ = host export. `data/mine.py`: keyword fixes x
  blame -> (bad,good) pairs, <=200 lines/4 files + 3 seeds; 141 cands, 3/3 recall.
  .gitignore: /data/ absolute.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth 1
  -delete`, never `rm` (EBUSY).
- rust pin: 1.100/1.101 404; 1.99.0 verified; std `Mutex::lock()` -> Result, `.unwrap()`.
- ruff 0.16 defaults drifted — pinned in [tool.ruff.lint]; tests/known/ I001/F401
  (Kevin's, rule 1).
- bootstrap.sh version probes format-fragile.
- RLIMIT_NPROC=64 counts uid-1000 tasks -> transient "Cannot fork"; tests use
  /bin/sleep, not sh-fork.
- torch 2.14 FSDP2: get_model_state_dict -> DTensors (.full_tensor() collective);
  fully_shard auto-inits env://; rank() pre-init is 0; grads AVG-reduce (world>1) ->
  backward token-mean loss, no 1/W scaling.
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`,
  never `rm DIR` (EBUSY).
- rust 1.99.0 (1.100/1.101 404); `Mutex::lock()` -> Result -> .unwrap().
- ruff 0.16 drifted (I/EXE) pinned in pyproject; tests/known/ I001/F401 (rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient "Cannot fork"; tests use /bin/sleep.
- torch 2.14 FSDP2: get_model_state_dict -> DTensors (no StateDictType; .full_tensor()
  collective); fully_shard auto-inits env://; rank() pre-init 0; grads AVG (W>1) ->
  backward per-token-mean loss, no 1/W scaling.
