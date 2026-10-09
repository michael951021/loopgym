# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done)
- `make bootstrap` (tools/bootstrap.sh, pins versions.txt): uv 0.12.24, .venv py3.12.14,
  rust 1.99.0, hadolint/shellcheck/terraform/tflint in .tools/bin; venv:
  torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate, numpy, ruff.
  Evidence: boot-*.log.
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest (minus known) +
  cargo test lgbox/ + shellcheck + hadolint + terraform (last three skip until
  files exist). `make known` = kevin-only authors + clean worktree + strict known.
  All known green (test_stats, test_logprob pass).

## Eval (done)
- `eval/stats.py`: bootstrap_ci(values, groups, n_boot, seed, alpha) -> (mean, lo,
  hi); resamples groups (tasks) as units. `eval/passk.py`: pass_at_k(n,c,k) product
  form of 1 - C(n-c,k)/C(n,k) (Chen 2021); summarize(records,k) -> per-task pass@k +
  task-clustered CI. 4/4 known test_stats, 5/5 test_passk.

## Models (done)
- `loopgym/models.py`: fetch/info CLI, tiny_config (2L/d64/4H/2KV/V512). models/:
  0.5b 954 MB / 1.5b 3.09 GB BF16; params == HF; tokenizers byte-identical.
  GOTCHA: 151936 = config vocab (lm_head dim), tokenizer length 151,643.

## Env (done)
- `exercism.py`: local runner fresh tmpdir, pristine tests + ONLY solution file,
  junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0, no fail/err/skip;
  shared `_xml_counts`/`_reward`.
- `lgbox.py`: LgboxClient (retries 503 honoring Retry-After, 120 s budget) + `_http`
  runner; "http" iff LOOPGYM_LGBOX_URL set. FIXED read_tail panic (`chunk[TAIL-n..]`
  underflow at read >= 4 KB -> reward 0; now `chunk[n-TAIL..n]`). strict test_reward
  18/18; 7/7 test_lgbox.py.
- `toy.py`: banana ("banana" ci in first 20 tokens), coinflip (seeded, text-blind).
  `splits.py`: audit() 142 -> env_audit.json + splits.json; bucket=sha1%100
  (train<70,val<85,test); 137/142 usable, 101/9/27; 5 unusable (env_audit.json).
  Evidence: exercism-*, toy-*, splits-*, lgboxpy-*.log.

## lgbox (done)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run fresh tmpdir; pre_exec
  setpgid + RLIMIT AS/CPU/NPROC64/FSIZE16MB; 4 KB tails; killpg; mpsc(Q)->503;
  /health; /metrics (prometheus runs_total/run_seconds/queue_depth/workers_busy).
  Firecracker backend: `Vmm` + `VmPool` (round-robin, reset-after-run, reboot
  crashed, queue when full) + `FirecrackerVmm` (API sock + vsock agent); Runner
  process|firecracker; `--dry-run`; docs/firecracker.md. 7 process + 8 FakeVmm tests
  (rule 4, no /dev/kvm).
- `tools/bench_lgbox.py`: val refs x local/http x c1/4/16 -> results/lgbox_bench.json
  (grades/s, p50/p95, reward gate, c1 <50ms); --quick 11.5 s.
  Evidence: lgboxmet-*, lgboxfc-*, lgboxbench-*.log.

## Kernel (kernelpy, done 2026-10-09)
- `loopgym/kernels/__init__.py`: IMPLS=("torch",), token_logprobs(h,w,t,impl,chunk)
  -> [T] in h's dtype. Chunked online max/LSE over V; autograd.Function backward
  recomputes per-chunk softmax (no [T,V] saved); grads wrt hidden AND weight;
  register_impl() for cpp/triton/cuda. Known 6/6 strict (real 0.5B == HF loss 1e-3,
  f64 gradcheck); 3/3 test_kernels.py. Peak extra << 1/8 naive (156 MB) at
  T=2048,V=151936. GOTCHA: keep input dtype end-to-end (f64 gradcheck breaks if
  output cast to f32). Evidence: kernelpy-*.log.

## RL + pinned sources (done)
- `rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md; known 7/7.
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 (sdist) / trl 1.15.0 ->
  read-only .tools/src/<name>-<ver>/; pins versions.txt. Never in .venv.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth
  1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100/1.101; 1.99.0 verified; `Mutex::lock()`
  returns Result -> `.unwrap()`.
- ruff 0.16 defaults drifted (I/EXE) — pinned in [tool.ruff.lint]; tests/known/
  has I001/F401.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load can push past 64 ->
  transient "Cannot fork". Tests use /bin/sleep, not sh-fork.