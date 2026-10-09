# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done)
- `make bootstrap` (pins versions.txt): uv, .venv py3.12, rust 1.99.0, linters in
  .tools/bin; venv: torch 2.14.1+cpu, transformers 5.19.0, peft, ruff.
- `make check` = ruff (E4/E7/E9/F, tests/known out) + pytest (minus known) + cargo test
  lgbox/ + shellcheck + hadolint + terraform (last 3 skip until files exist).
  `make known` = kevin-only authors + clean worktree + strict known.

## Eval (done)
- `eval/stats.py` bootstrap_ci resamples tasks; `eval/passk.py` pass_at_k (Chen 2021) +
  summarize(records,k) task-clustered CI. 4/4+5/5 known.

## Models (done)
- `loopgym/models.py`: fetch/info, tiny_config (2L/d64/4H/2KV/V512); 0.5b 954 MB / 1.5b
  3.09 GB BF16; tokenizers byte-identical. GOTCHA: 151936=config vocab, tok len 151,643.
  `tiny_tokenizer()`: cached vocab-400 byte-BPE chat tokenizer (Qwen role tags), full byte
  alphabet (no <unk>).

## Env (done)
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution file, junit XML, new pgroup,
  SIGKILL timeout; reward 1 iff tests>0 no fail/err/skip. `lgbox.py`: LgboxClient (503
  retries) + _http runner. 18/18 strict; 7/7. `toy.py`: banana/coinflip. `splits.py`:
  splits.json, bucket=sha1%100 (train<70,val<85,test); 101/9/27.

## lgbox (done)
- `lgbox serve --port --workers --queue` (axum 0.8): /run fresh tmpdir, setpgid+RLIMIT,
  killpg, mpsc->503, /health, /metrics. Firecracker Vmm/VmPool (FakeVmm tests); process|
  firecracker; --dry-run. `tools/bench_lgbox.py` -> results/lgbox_bench.json.

## Kernel (done)
- `loopgym/kernels/__init__.py`: token_logprobs(h,w,t,impl,chunk)->[T]; chunked online
  max/LSE; autograd backward recomputes per-chunk softmax (no [T,V] saved); register_impl
  for cpp/triton/cuda. Known 6/6 (0.5B==HF loss 1e-3, f64 gradcheck). GOTCHA: keep input
  dtype end-to-end (f32 cast breaks f64).

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

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`,
  never `rm DIR` (EBUSY).
- rust 1.99.0 (1.100/1.101 404); `Mutex::lock()` -> Result -> .unwrap().
- ruff 0.16 drifted (I/EXE) pinned in pyproject; tests/known/ has I001/F401.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load -> transient "Cannot fork".
