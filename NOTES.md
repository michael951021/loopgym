# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins tools/versions.txt): uv 0.12.24, .venv py3.12.14,
  rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1, shellcheck 0.11.0, terraform 1.16.5,
  tflint 0.64.0 in .tools/bin. venv: torch 2.14.1+cpu ([tool.uv.sources]), transformers 5.19.0,
  peft, safetensors, accelerate, numpy, pydantic, httpx, pytest, ruff, tomli-w; uv.lock.
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest (minus known) + cargo test
  lgbox/ + shellcheck + hadolint + terraform init -backend=false && validate (last three skip
  until their files exist). `make known` = kevin-only authors + clean worktree + strict
  tests/known. Still red: test_logprob, test_stats (import kernels/eval, not built).

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI (`python -m loopgym.models`), tiny_config
  (2L/d64/4H/2KV/V512). models/: 0.5b 954 MB / 1.5b 3.09 GB BF16; params == HF.
- GOTCHA: 151936 = config vocab (lm_head dim), tokenizer length 151,643 (training uses 151936).

## Env (done 2026-10-09)
- `exercism.py`: Task + load_exercise; local runner: fresh tmpdir, pristine tests + ONLY
  solution file, pytest junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0, no
  failures/errors/skips. Shared `_xml_counts`/`_reward` score XML for BOTH runners.
- `lgbox.py`: LgboxClient (httpx; retries 503 honoring Retry-After, 120 s budget) + `_http`
  runner = RUNNERS["http"] (pristine tests + solution file, pytest --junitxml=r.xml,
  outputs ["r.xml"], mem_mb 2048). runner_names() adds "http" iff LOOPGYM_LGBOX_URL set.
  tools/lgbox_smoke.sh: release build, serve on free port, strict test_reward.py both runners
  -> 18/18, zero server panics. FIXED pre-existing read_tail panic (lib.rs: read >=
  4 KB underflowed `chunk[TAIL-n..]` -> spurious reward 0; now `chunk[n-TAIL..n]`,
  regression test kept). 7/7 tests/test_lgbox.py (MockTransport).
  Evidence: lgboxpy-{smoke2,makecheck}.log.
- `toy.py`: ToyTask (banana: "banana" ci in first 20 tokens; coinflip: 0.5 seeded by
  (step, index), text-blind). 12/12 test_toy.py.
- `splits.py`: audit() of 142 exercises -> results/env_audit.json + splits.json;
  bucket=sha1%100 (train<70, val<85, test); 137/142 usable, splits 101/9/27. Unusable:
  alphametics, error-handling, ledger, markdown, parallel-letter-frequency.

## lgbox server (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run in fresh tmpdir; pre_exec
  setpgid + RLIMIT AS/CPU/NPROC64/FSIZE16MB; 4 KB tails; killpg on timeout; mpsc(Q) ->
  503+Retry-After:1; bad path -> 400; /health; /metrics (prometheus 0.14 text:
  lgbox_runs_total{ok|nonzero|timeout|rejected}, run_seconds, queue_depth, workers_busy;
  all outcomes pre-materialized). 2/2 lib + 8/8 integration tests. Evidence: lgboxmet-*,
  lgboxpy-*.log (all exit 0).

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
