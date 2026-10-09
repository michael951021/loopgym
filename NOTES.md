# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins tools/versions.txt): uv 0.12.24, .venv
  py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1, shellcheck
  0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin; evidence boot-cleanslate.log.
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest (minus known) +
  cargo test lgbox/ + shellcheck + hadolint + terraform init -backend=false && validate
  (last three skip until their files exist). `make known` = check_known.sh (kevin-only
  authors, clean worktree) + strict tests/known. Still red: {test_grpo,test_logprob,
  test_stats}.py import loopgym.rl/kernels/eval (not built); test_reward.py 9/9 strict.
- venv: torch 2.14.1+cpu (CPU index [tool.uv.sources]), transformers 5.19.0, peft,
  safetensors, accelerate, numpy, pydantic, httpx, pytest, ruff, tomli-w; uv.lock.

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI (`python -m loopgym.models`), tiny_config
  (2L/d64/4H/2KV/V512). models/: 0.5b 954 MB / 1.5b 3.09 GB, BF16; params 494,032,768 /
  1,543,714,304 == HF; tokenizers byte-identical. GOTCHA: 151936 = config vocab
  (embedding dim), tokenizer length 151,643 (kernel/training use 151936).
  Evidence: models-{fetch,info2}.log, tests/test_models.py.

## Env: exercism + toy + splits (done 2026-10-09)
- `exercism.py`: Task (stub/reference/prompt/parse) + load_exercise; grade local runner:
  fresh tmpdir, pristine tests + ONLY solution file, pytest junit XML, new pgroup,
  SIGKILL on timeout; reward 1 iff tests>0, no failures/errors/skips. 9/9 known strict.
  Evidence: exercism-{accept2,makecheck3}.log.
- `toy.py`: ToyTask ("Name a fruit."→reply.txt); grade(task,files,step,index) in-process.
  banana = 1 iff "banana" (ci) in decode of first 20 tokens; coinflip = 1 w.p. 0.5 from
  random.Random(f"...:{step}:{index}"), ignores text. Known "I like Banana"=1, "apple"=0,
  "fig "*21+"banana"=0; coinflip 10k mean 0.4945. 12/12 test_toy.py. Evidence
  toy-{makecheck2,makeknown}.log.
- `splits.py`: audit() grades ref+stub of 142 practice exercises (4 threads, local);
  writes results/env_audit.json + splits.json; bucket=sha1%100 (train<70,val<85,test);
  137/142 usable, splits 101/9/27; unusable: alphametics, error-handling, ledger,
  markdown (stub leaky), parallel-letter-frequency (timeout). tests/test_splits.py.
  Evidence: splits-{accept,makecheck}.log.

## lgbox (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8/tokio/libc/tempfile): /run in
  fresh tmpdir; pre_exec setpgid(0,0) + RLIMIT AS(mem_mb)/CPU(timeout+1s)/NPROC 64/
  FSIZE 16MB; 4 KB tails; try_wait vs deadline, killpg(SIGKILL); N workers (semaphore),
  mpsc(Q) → 503+Retry-After:1; `..`/abs/empty → 400. /health.
- /metrics (prometheus 0.14 text): lgbox_runs_total{outcome=ok|nonzero|timeout|rejected}
  (rejected=400/503 in run_handler; outcome from RunOut in dispatch), lgbox_run_seconds
  histogram (accepted runs only), lgbox_queue_depth (inc on accept, dec on finish),
  lgbox_workers_busy (around run_blocking). All 4 outcomes pre-materialized (show at 0).
- 8/8 integration tests incl. metrics_counts_exact (3 runs → exact 1/1/1/0, hist count=3,
  gauges 0). Evidence: lgboxmet-{cargo-test,makecheck}.log (exit 0).

## Pinned sources (done 2026-10-09)
- `tools/fetch_src.sh`: latest stable vllm/deepspeed/trl → read-only .tools/src/<name>-
  <ver>/; pins tools/versions.txt (vllm 0.31.0, deepspeed 0.19.7 sdist, trl 1.15.0);
  never in .venv; vllm wheel=aarch64, deepspeed=sdist. Evidence: vsrc-{fetch,rerun,
  makecheck}.log (exit 0).

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR -mindepth 1
  -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100.0/1.101.0; 1.99.0 verified.
- ruff 0.16 defaults drifted (I/EXE) — pinned in [tool.ruff.lint].
- bootstrap.sh version probes format-fragile (shellcheck `version:`, tflint two lines).
- tests/known/ has I001/F401 — Kevin's, read-only, left as-is.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load can push past 64 → transient
  "Cannot fork". Tests avoid sh-fork (/bin/sleep).

## Data
- tests/known/ = Kevin's known-answer tests (read-only, rule 1); data/raw/ = host export:
  exercism-python, draftlab+harness mirrors, ticket_review.json, runs.jsonl.
