# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins tools/versions.txt): uv 0.12.24, py3.12.14
  venv (torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate, numpy,
  pydantic, httpx, pytest, ruff; uv.lock), rust 1.99.0 minimal, hadolint/shellcheck/
  terraform/tflint in .tools/bin. Evidence: boot-*.log.
- `make check` = ruff (E4/E7/E9/F, known excluded) + pytest (minus known) + cargo test +
  shellcheck + hadolint + terraform validate (skip until files exist). `make known` =
  check_known.sh + strict tests/known. Still red: test_logprob (kernels, w2).
## Foundations (done)
- `make bootstrap` (tools/bootstrap.sh, pins versions.txt): uv 0.12.24, .venv py3.12.14,
  rust 1.99.0, linters in .tools/bin; venv: torch 2.14.1+cpu, transformers 5.19.0,
  peft, safetensors, accelerate, numpy, ruff.
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest (minus known) +
  cargo test lgbox/ + shellcheck + hadolint + terraform (last three skip until
  files exist). `make known` = kevin-only authors + clean worktree + strict known.
  All known green.

## Eval (done 2026-10-09)
- `eval/stats.py` bootstrap_ci(values, groups, ...) -> (mean, lo, hi); resamples groups
  (tasks) as units. `eval/passk.py`: pass_at_k(n,c,k) product form (Chen 2021);
  summarize() -> per-task pass@k + task-clustered CI. 4/4 + 5/5 known.
  Evidence: stats-makecheck.log.
## Eval (done)
- `eval/stats.py`: bootstrap_ci(values, groups, n_boot, seed, alpha) -> (mean, lo,
  hi); resamples groups (tasks) as units. `eval/passk.py`: pass_at_k(n,c,k) product
  form of 1 - C(n-c,k)/C(n,k) (Chen 2021); summarize(records,k) -> per-task pass@k +
  task-clustered CI. 4/4 known test_stats, 5/5 test_passk.

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI, tiny_config (2L/d64/V512); 0.5b/1.5b BF16,
  params == HF. GOTCHA: vocab 151936 (config/lm_head) vs tokenizer 151,643; training
  uses 151936. Evidence: models-*.log.
## Models (done)
- `loopgym/models.py`: fetch/info CLI, tiny_config (2L/d64/4H/2KV/V512). models/:
  0.5b 954 MB / 1.5b 3.09 GB BF16; params == HF; tokenizers byte-identical.
  GOTCHA: 151936 = config vocab, tokenizer length 151,643.

## Env (done 2026-10-09)
- `exercism.py`: Task + load_exercise; local runner: fresh tmpdir, pristine tests + ONLY
  solution file, junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0, no
  fail/err/skip; `_xml_counts`/`_reward` shared by BOTH runners.
- `lgbox.py`: LgboxClient (503 retries honor Retry-After) + RUNNERS["http"] (pristine
  tests + solution, --junitxml, mem 2048; active iff LOOPGYM_LGBOX_URL). FIXED
  read_tail underflow panic (spurious 0). 7/7. Evidence: lgboxpy-*.log.
- `toy.py`: ToyTask (banana ci in first 20 tokens; coinflip p=0.5 seeded
  (step, index), text-blind). 12/12.
- `splits.py`: audit() of 142 exercises -> results/env_audit.json + splits.json;
  bucket=sha1%100 (train<70, val<85, test); 137/142 usable, splits 101/9/27; 5 unusable
  (env_audit.json). Evidence: splits-*.log.
## Env (done)
- `exercism.py`: local runner fresh tmpdir, pristine tests + ONLY solution file,
  junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0, no fail/err/skip;
  shared `_xml_counts`/`_reward`.
- `lgbox.py`: LgboxClient (retries 503 honoring Retry-After, 120 s budget) + `_http`
  runner; "http" iff LOOPGYM_LGBOX_URL set. strict test_reward 18/18; 7/7 test_lgbox.py.
- `toy.py`: banana ("banana" ci in first 20 tokens), coinflip (seeded, text-blind).
  `splits.py`: audit() 142 -> env_audit.json + splits.json; bucket=sha1%100
  (train<70,val<85,test); 137/142 usable, 101/9/27.

## lgbox server (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run (fresh tmpdir, setpgid,
  RLIMIT AS/CPU/NPROC/FSIZE, 4 KB tails, killpg on timeout, mpsc(Q) -> 503+Retry-After,
  bad path -> 400), /health, /metrics (prometheus 0.14). 10/10 tests.
  Evidence: lgboxmet-*.log.
- Firecracker: Vmm trait + VmPool (lease/reset, re-boot on crash, queue when full) +
  FirecrackerVmm (API sock); --dry-run; docs/firecracker.md. FakeVmm-only tests
  (rule 4). Evidence: lgboxfc-*.log.
- `tools/bench_lgbox.py`: val refs x local/http x c1/4/16 -> lgbox_bench.json;
  --quick 11.5 s, 120/120 reward 1. Evidence: lgboxbench-*.log.
## lgbox (done)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run fresh tmpdir; pre_exec
  setpgid + RLIMIT AS/CPU/NPROC64/FSIZE16MB; 4 KB tails; killpg; mpsc(Q)->503;
  /health; /metrics (prometheus runs_total/run_seconds/queue_depth/workers_busy).
  Firecracker: `Vmm` + `VmPool` (round-robin, reset, reboot crashed, queue) +
  `FirecrackerVmm` (API sock + vsock); process|firecracker; `--dry-run`;
  docs/firecracker.md. 7 process + 8 FakeVmm tests (rule 4, no /dev/kvm).
- `tools/bench_lgbox.py`: val refs x local/http x c1/4/16 -> results/lgbox_bench.json
  (grades/s, p50/p95, reward gate); --quick 11.5 s.

## Kernel (done 2026-10-09)
- `loopgym/kernels/__init__.py`: IMPLS=("torch",), token_logprobs(h,w,t,impl,chunk)
  -> [T] in h's dtype. Chunked online max/LSE over V; autograd.Function backward
  recomputes per-chunk softmax (no [T,V] saved); grads wrt hidden AND weight;
  register_impl() for cpp/triton/cuda. Known 6/6 strict (real 0.5B == HF loss 1e-3,
  f64 gradcheck); 3/3 test_kernels.py. Peak extra << 1/8 naive (156 MB) at
  T=2048,V=151936. GOTCHA: keep input dtype end-to-end (f64 gradcheck breaks if
  output cast to f32).

## RL
- `rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md; known 7/7.
- `rl/batch.py`: Batch (right-pad, target_mask col j = 1 iff col predicts a completion
  token, i.e. slice [p-1, p+c-1)), StepCfg (logprob_impl/clip_eps/kl_coef/agg/
  max_grad_norm/chunk), build_batch(groups,rewards,tokenizer,scale), logp_of,
  grpo_step (logp via kernels.token_logprobs on hidden_states[-1][:, :-1];
  old=logp.detach(); ref via policy.disable_adapter() or custom ref_fn; backward,
  clip, step; stats loss/kl/clip_frac/ratio_mean/grad_norm/tokens). 5/5
  test_batch.py (2 s, tiny_config fp32): mask == independent gather; step-0 β=0
  loss = -9.9e-9 < 1e-6, ratio=1.0; +adv logp +0.211 / -adv -0.105 nats after one
  step (lr 1e-2 AdamW LoRA r=4); kl>0 after step proves adapter disabled in ref.
  GOTCHA: transformers 5.x CausalLMOutput has NO last_hidden_state — use
  output_hidden_states=True, hidden_states[-1]; kernel is unbatched -> flatten rows.
- `rl/rollout.py`: Completion/RolloutBackend/HFBackend (chat template, left pad,
  rows x n, output_scores -> per-token logp of sampled token in fp32 post-warper
  scores); temperature<=0 = greedy; trim completion at first eos. GOTCHA:
  generation_config.json ships top_k=20 + rep-pen 1.05 -> pass top_k=0,
  repetition_penalty=1.0. Known: T=1 fp32 real 0.5B max|Δ vs fresh forward|
  = 2.8e-5 < 1e-4; greedy == generate; padding-invariant. 3/3 (11 s).
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 (sdist) / trl 1.15.0 ->
  read-only .tools/src/<name>-<ver>/; pins versions.txt. Never in .venv.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth 1
  -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100/1.101; 1.99.0 verified.
- rust 1.99: std `Mutex::lock()` returns a Result -> `.unwrap()`.
- ruff 0.16 defaults drifted — pinned in [tool.ruff.lint]; tests/known/ has I001/F401
  (Kevin's, rule 1).
- bootstrap.sh version probes format-fragile.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load can push past 64 -> transient
  "Cannot fork". Tests use /bin/sleep, not sh-fork.

## Data
- tests/known/ = Kevin's (read-only, rule 1); data/raw/ = host export (exercism-python,
  mirrors, ticket_review.json, runs.jsonl).
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth
  1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100/1.101; 1.99.0 verified; `Mutex::lock()`
  returns Result -> `.unwrap()`.
- ruff 0.16 defaults drifted (I/EXE) — pinned in pyproject; tests/known/
  has I001/F401.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load ->
  transient "Cannot fork". Tests use /bin/sleep, not sh-fork.
- `data/mine.py` (2026-10-09): keyword fixes (fix/correct/wrong/bug/audit in %B) x
  blame of deleted lines -> (bad,good) pairs; keep defect diff (bad's lines the fix
  reverts) <=200 lines/<=4 files; +3 seeds unconditional, win collisions. 141 cands,
  3/3 seed recall. Ticket: (id: x) or unique subject-prefix line in bad:TODO.md,
  else TODO: msg line (50/141). 5/5 tests. Evidence: mine.log.
