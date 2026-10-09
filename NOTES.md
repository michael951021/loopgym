# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins tools/versions.txt): uv 0.12.24, py3.12.14
  venv (torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate, numpy,
  pydantic, httpx, pytest, ruff; uv.lock), rust 1.99.0 minimal, hadolint/shellcheck/
  terraform/tflint in .tools/bin. Evidence: boot-*.log.
- `make check` = ruff (E4/E7/E9/F, known excluded) + pytest (minus known) + cargo test +
  shellcheck + hadolint + terraform validate (skip until files exist). `make known` =
  check_known.sh + strict tests/known. Still red: test_logprob (kernels not built).

## Eval (done 2026-10-09)
- `eval/stats.py` bootstrap_ci(values, groups, n_boot, seed, alpha) -> (mean, lo, hi);
  resamples groups (tasks) as units when given. `eval/passk.py`: pass_at_k(n,c,k) =
  product form of 1 - C(n-c,k)/C(n,k) (Chen 2021); summarize(records, k) (task/n/c
  dicts) -> per-task pass@k + task-clustered CI. 4/4 known test_stats, 5/5 test_passk.
  Evidence: stats-makecheck.log.

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI (`python -m loopgym.models`), tiny_config
  (2L/d64/4H/2KV/V512); models/ (bind mount) 0.5b/1.5b BF16, params == HF. GOTCHA: 151936
  = config vocab (lm_head dim), tokenizer length 151,643 (training uses 151936).
  Evidence: models-{fetch,info2}.log.

## Env (done 2026-10-09)
- `exercism.py`: Task + load_exercise; local runner: fresh tmpdir, pristine tests + ONLY
  solution file, junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0, no
  fail/err/skip; `_xml_counts`/`_reward` shared by BOTH runners.
- `lgbox.py`: LgboxClient (httpx; 503 retries honor Retry-After) + RUNNERS["http"]
  (pristine tests + solution file, --junitxml, outputs r.xml, mem_mb 2048); active iff
  LOOPGYM_LGBOX_URL set. FIXED read_tail panic (chunk underflow -> spurious reward 0).
  7/7 test_lgbox.py. Evidence: lgboxpy-*.log.
- `toy.py`: ToyTask (banana: 1 iff "banana" ci in first 20 tokens; coinflip: 1 w.p. 0.5
  seeded by (step, index), text-blind). 12/12 test_toy.py.
- `splits.py`: audit() of 142 exercises -> results/env_audit.json + splits.json;
  bucket=sha1%100 (train<70, val<85, test); 137/142 usable, splits 101/9/27; 5 unusable
  (env_audit.json). Evidence: exercism-*, toy-*, splits-*.log.

## lgbox server (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run (fresh tmpdir, setpgid,
  RLIMIT AS/CPU/NPROC/FSIZE, 4 KB tails, killpg on timeout, mpsc(Q) -> 503+Retry-After,
  bad path -> 400), /health, /metrics (prometheus 0.14). 10/10 tests.
  Evidence: lgboxmet-*.log.
- Firecracker backend: Vmm trait + VmPool (lease/reset, crashed slot re-booted, queues
  when full) + FirecrackerVmm (API sock); --dry-run prints VM config; docs/firecracker.md.
  FakeVmm-only tests (rule 4). Evidence: lgboxfc-*.log.
- `tools/bench_lgbox.py`: val refs x local/http x c1/4/16 -> results/lgbox_bench.json
  (spawns release server; --url to reuse). --quick 11.5 s, 120/120 reward 1.
  Evidence: lgboxbench-*.log.

## RL + pinned sources
- `rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md; known 7/7.
  Evidence: grpo-*.log.
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 (sdist) / trl 1.15.0 -> read-only
  .tools/src/<name>-<ver>/; pins tools/versions.txt; never in .venv. Evidence: vsrc-*.log.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth 1
  -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100/1.101; 1.99.0 verified.
- rust 1.99: std `Mutex::lock()` returns a Result -> `.unwrap()`.
- ruff 0.16 defaults drifted (I/EXE) — pinned in [tool.ruff.lint]; tests/known/ has
  I001/F401 (Kevin's, read-only, rule 1).
- bootstrap.sh version probes format-fragile.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load can push past 64 -> transient
  "Cannot fork". Tests use /bin/sleep, not sh-fork.

## Data
- tests/known/ = Kevin's (read-only, rule 1); data/raw/ = host export (exercism-python,
  mirrors, ticket_review.json, runs.jsonl).
