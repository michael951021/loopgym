# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins tools/versions.txt): uv 0.12.24, py3.12.14 venv
  (torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate, numpy, pydantic,
  httpx, pytest, ruff, tomli-w; uv.lock), rust 1.99.0 minimal, hadolint/shellcheck/
  terraform/tflint in .tools/bin.
- `make check` = ruff (E4/E7/E9/F, known excluded) + pytest (minus known) + cargo test +
  shellcheck + hadolint + terraform validate (skips when files absent). `make known` =
  kevin-only authors + strict tests/known. Still red: test_logprob (kernels not built).

## Eval (done 2026-10-09)
- `eval/stats.py` bootstrap_ci(values, groups, n_boot, seed, alpha) -> (mean, lo, hi);
  resamples groups (tasks) as units when given. `eval/passk.py`: pass_at_k(n,c,k) =
  product form of 1 - C(n-c,k)/C(n,k) (Chen 2021); summarize(records, k) (task/n/c dicts)
  -> per-task pass@k + task-clustered CI. 4/4 known test_stats, 5/5 test_passk.
  Evidence: stats-makecheck.log.

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI (`python -m loopgym.models`), tiny_config
  (2L/d64/4H/2KV/V512). models/: 0.5b 954 MB / 1.5b 3.09 GB BF16; params == HF.
- GOTCHA: 151936 = config vocab (lm_head dim), tokenizer length 151,643 (training uses 151936).

## Env (done 2026-10-09)
- `exercism.py`: Task + load_exercise; local runner: fresh tmpdir, pristine tests + ONLY
  solution file, junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0, no
  failures/errors/skips; `_xml_counts`/`_reward` shared by BOTH runners.
- `lgbox.py`: LgboxClient (httpx; 503 retries honor Retry-After, 120 s budget) +
  RUNNERS["http"] (pristine tests + solution file, --junitxml, outputs r.xml, mem_mb 2048);
  active iff LOOPGYM_LGBOX_URL set. FIXED read_tail panic (lib.rs `chunk[TAIL-n..]`
  underflow -> spurious reward 0; now `chunk[n-TAIL..n]`). 7/7 test_lgbox.py.
  Evidence: lgboxpy-*.log.
- `toy.py`: ToyTask (banana: "banana" ci in first 20 tokens; coinflip: 0.5 seeded by
  (step, index), text-blind). 12/12 test_toy.py.
- `splits.py`: audit() of 142 exercises -> results/env_audit.json + splits.json;
  bucket=sha1%100 (train<70, val<85, test); 137/142 usable, splits 101/9/27; 5 unusable
  (list in env_audit.json).

## lgbox server (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run in fresh tmpdir; pre_exec
  setpgid + RLIMIT AS/CPU/NPROC64/FSIZE16MB; 4 KB tails; killpg on timeout; mpsc(Q) ->
  503+Retry-After:1; bad path -> 400; /health; /metrics (prometheus 0.14 text:
  lgbox_runs_total{ok|nonzero|timeout|rejected}, run_seconds, queue_depth, workers_busy;
  all outcomes pre-materialized). 2/2 lib + 8/8 integration tests. Evidence: lgboxmet-*/
  lgboxpy-*.log.
- `tools/bench_lgbox.py`: val refs x local/http x c1/4/16 ->
  results/lgbox_bench.json (grades/s, p50/p95, reward gate, c1 <50ms); spawns release
  server (--url to reuse). --quick 11.5 s, 120/120 reward 1. Evidence: lgboxbench-*.log.

## RL (w2, done 2026-10-09)
- `rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md; known 7/7.
  Evidence: grpo-*.log.

## Pinned sources (done 2026-10-09)
- `tools/fetch_src.sh`: vllm 0.31.0 / deepspeed 0.19.7 (sdist) / trl 1.15.0 -> read-only
  .tools/src/<name>-<ver>/; pins tools/versions.txt. Never in .venv.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth 1
  -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100/1.101; 1.99.0 verified.
- rust 1.99: std `Mutex::lock()` returns a Result (poison-aware) -> `.unwrap()`.
- ruff 0.16 defaults drifted (I/EXE) — pinned in [tool.ruff.lint]; tests/known/ has
  I001/F401 (Kevin's, read-only, rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load can push past 64 -> transient
  "Cannot fork". Tests use /bin/sleep, not sh-fork.

## Data
- tests/known/ = Kevin's tests (read-only); data/raw/ = host export (exercism-python,
  draftlab+harness mirrors, ticket_review.json, runs.jsonl).
