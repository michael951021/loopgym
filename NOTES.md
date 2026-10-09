# Notes

Current state for the next run (keep < 4 KB). Evidence logs: .agent-evidence/*.log.

## Foundations (done)
- `make bootstrap` (pins versions.txt): uv, .venv py3.12, rust 1.99.0, linters in
  .tools/bin; venv: torch 2.14.1+cpu, transformers 5.19.0, peft, ruff.
- `make check` = ruff (E4/E7/E9/F, tests/known out) + pytest (minus known) + cargo test
  lgbox/ + shellcheck + hadolint + terraform (last 3 skip until files exist).
  `make known` = kevin-only authors + clean worktree + strict known.
## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins versions.txt): uv 0.12.24, .venv py3.12.14
  (torch 2.14.1+cpu, transformers 5.19.0, peft, accelerate, ruff, pytest),
  rust 1.99.0, linters in .tools/bin.
- `make check` = ruff (E4/E7/E9/F, known excluded) + pytest (minus known) + cargo test
  + shellcheck + hadolint + terraform validate. `make known` = kevin-only + strict
  known. All known green.

## Eval (done)
- `eval/stats.py` bootstrap_ci resamples tasks; `eval/passk.py` pass_at_k (Chen 2021) +
  summarize(records,k) task-clustered CI. 4/4+5/5 known.
## Eval (done 2026-10-09)
- `eval/stats.py` bootstrap_ci(values, groups) -> (mean, lo, hi); resamples task groups.
  `eval/passk.py` pass_at_k product form (Chen 2021); summarize() -> per-task pass@k +
  task CI. Known 4/4 + 5/5.

## Models (done)
- `loopgym/models.py`: fetch/info, tiny_config (2L/d64/4H/2KV/V512); 0.5b 954 MB / 1.5b
  3.09 GB BF16; tokenizers byte-identical. GOTCHA: 151936=config vocab, tok len 151,643.
  `tiny_tokenizer()`: cached vocab-400 byte-BPE chat tokenizer (Qwen role tags), full byte
  alphabet (no <unk>).
## Models (done 2026-10-09)
- `loopgym/models.py` fetch/info CLI, tiny_config (2L/d64/V512); 0.5b/1.5b BF16,
  params == HF. GOTCHA: vocab 151936 vs tokenizer 151,643; training uses 151936.

## Env (done)
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution file, junit XML, new pgroup,
  SIGKILL timeout; reward 1 iff tests>0 no fail/err/skip. `lgbox.py`: LgboxClient (503
  retries) + _http runner. 18/18 strict; 7/7. `toy.py`: banana/coinflip. `splits.py`:
  splits.json, bucket=sha1%100 (train<70,val<85,test); 101/9/27.
## Env (done 2026-10-09)
- `exercism.py` local runner: fresh tmpdir, pristine tests + ONLY solution, junit XML,
  pgroup SIGKILL timeout; reward 1 iff tests>0 no fail/err/skip; `_xml_counts`/
  `_reward` shared by BOTH runners.
- `lgbox.py`: LgboxClient + RUNNERS["http"] (active iff LOOPGYM_LGBOX_URL); fixed
  read_tail underflow (spurious 0). `toy.py`: banana (first 20 tokens ci), coinflip
  (seeded, text-blind). `splits.py`: bucket=sha1%100, 101/9/27, 137/142 usable.

## lgbox (done)
- `lgbox serve --port --workers --queue` (axum 0.8): /run fresh tmpdir, setpgid+RLIMIT,
  killpg, mpsc->503, /health, /metrics. Firecracker Vmm/VmPool (FakeVmm tests); process|
  firecracker; --dry-run. `tools/bench_lgbox.py` -> results/lgbox_bench.json.
## lgbox server (done 2026-10-09)
- `lgbox serve` (axum 0.8): /run (setpgid, RLIMIT AS/CPU/NPROC/FSIZE, 4 KB tails,
  mpsc(Q)->503+Retry-After), /health, /metrics. Firecracker: Vmm + VmPool +
  FirecrackerVmm; --dry-run; FakeVmm-only tests (rule 4).
- `tools/bench_lgbox.py`: --quick 11.5 s, 120/120 reward 1.

## Kernel (done)
- `loopgym/kernels/__init__.py`: token_logprobs(h,w,t,impl,chunk)->[T]; chunked online
  max/LSE; autograd backward recomputes per-chunk softmax (no [T,V] saved); register_impl
  for cpp/triton/cuda. Known 6/6 (0.5B==HF loss 1e-3, f64 gradcheck). GOTCHA: keep input
  dtype end-to-end (f32 cast breaks f64).
## Kernel (done 2026-10-09)
- `kernels/token_logprobs(h,w,t,impl,chunk)` -> [T]; chunked online max/LSE; autograd
  backward recomputes per-chunk softmax (no [T,V] saved); grads wrt hidden AND weight.
  Known 6/6 (0.5B == HF 1e-3, gradcheck); peak extra 156 MB << 1/8 naive.
  GOTCHA: keep input dtype end-to-end.

## RL
- `rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md; known 7/7.
- `rl/batch.py`: Batch (right-pad; target_mask col j=1 iff col predicts a completion
  token, slice [p-1,p+c-1)), StepCfg, build_batch, logp_of, grpo_step (logp via
  token_logprobs on hidden_states[-1][:,:-1]; old=logp.detach(); ref via disable_adapter
  or ref_fn). 5/5 test_batch (2 s): step-0 β=0 loss -9.9e-9<1e-6; +adv logp +0.211/-adv
  -0.105 after one step. GOTCHA: no last_hidden_state in 5.x; kernel unbatched -> flatten.
- `rl/rollout.py`: Completion/HFBackend (chat template, left pad, rows x n, output_scores
  -> sampled-token logp fp32 post-warper); T<=0 greedy; trim at eos. GOTCHA: gen_config
  ships top_k=20+rep 1.05 -> pass top_k=0, rep=1.0. Known T=1 real 0.5B max|Δ| 2.8e-5<1e-4.
- `rl/train.py`: TrainCfg (toml + CLI --k v), load_cfg, TaskSampler (persisted RNG), Run:
  sample B -> generate (seed=seed+s) -> grade (ThreadPool) -> build_batch -> grpo_step ->
  metrics.jsonl -> ckpt. ckpt <out>^2/ckpt/<name> (runs/X->ROOT/ckpt/X). Exact resume:
  ckpt=step+adapter+opt+torch/task RNG; --resume checks RESUME_CHECK (lora_targets tuple
  vs JSON list). eval seed=seed+1000000003+s. 5/5 test_train (28 s): smoke <3 min;
  killed-after-2 + --resume == uninterrupted bit-identical. GOTCHA: from_pretrained leaves
  ALL params frozen -> requires_grad=("lora_" in name) before AdamW.
- `rl/grpo.py` group_advantages/kl_k3/policy_loss; known 7/7.
- `rl/batch.py`: Batch (target_mask slice [p-1, p+c-1)), StepCfg, build_batch, logp_of,
  grpo_step (recomputed logp; old=detach; ref via disable_adapter; stats
  loss/kl/clip_frac/ratio/grad_norm/tokens). 5/5 (2 s): step-0 β=0 loss -9.9e-9.
- `rl/rollout.py` HFBackend (left pad, output_scores -> fp32 per-token logp;
  T<=0 greedy; trim at eos). GOTCHA: shipped generation_config top_k=20 + rep-pen
  1.05 -> pass top_k=0, repetition_penalty=1.0. Known: real 0.5B |Δ| 2.8e-5 < 1e-4
  vs fresh forward (T=1 fp32); 3/3.
- `rl/dist.py` FSDP2: init_dist (torchrun env://; standalone world-1 file:// store),
  wrap_fsdp (fully_shard per layer + root), full_state_dict (collective),
  global_mean_loss. Equiv 2 ranks x b == single 2b (tiny fp32): param diff 0.0, one
  shared loss; 1/1.
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 / trl 1.15.0 -> read-only
  .tools/src/; pins versions.txt. Never in .venv.

## Data
- tests/known/ = Kevin's (read-only, rule 1); data/raw/ = host export.
- `data/mine.py`: keyword fixes in %B x blame of deleted lines -> (bad, good) pairs,
  defect diff <=200 lines/4 files + 3 seeds; 141 candidates, 3/3 recall; 5/5 tests.
  .gitignore: /data/ absolute so loopgym/data is trackable.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`,
  never `rm DIR` (EBUSY).
- rust 1.99.0 (1.100/1.101 404); `Mutex::lock()` -> Result -> .unwrap().
- ruff 0.16 drifted (I/EXE) pinned in pyproject; tests/known/ has I001/F401.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load -> transient "Cannot fork".
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth 1
  -delete`, never `rm` (EBUSY).
- rust pin: 1.100/1.101 404; 1.99.0 verified; std `Mutex::lock()` -> Result, `.unwrap()`.
- ruff 0.16 defaults drifted — pinned in [tool.ruff.lint]; tests/known/ I001/F401
  (Kevin's, rule 1).
- bootstrap.sh version probes format-fragile.
- RLIMIT_NPROC=64 counts all uid-1000 tasks -> transient "Cannot fork"; tests use
  /bin/sleep, not sh-fork.
- torch 2.14 FSDP2: get_model_state_dict(model) -> DTensors (no StateDictType arg;
  .full_tensor() is collective); fully_shard auto-inits env://; rank() pre-init is 0;
  grads reduce by AVG (world>1) -> backward per-token-mean loss, no 1/W scaling.
