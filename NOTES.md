# Notes

Current state for the next run (keep < 4 KB).

## Foundations (boot — done 2026-10-09)
- `make bootstrap` = idempotent toolchain (tools/bootstrap.sh, pins tools/versions.txt):
  uv 0.12.24, .venv py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1,
  shellcheck 0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin.
- `make check` = ruff (E4/E7/E9/F) + pytest (minus known) + cargo test in lgbox/ +
  shellcheck + hadolint + terraform init -backend=false && validate (last three skip
  until their files exist). `make known` = kevin-only worktree check + strict tests/known.
- venv: torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate, numpy,
  pydantic, httpx, pytest, ruff, tomli-w; uv.lock. Evidence: boot-cleanslate.log.
- `make known` fails until rl/kernels/eval exist (pre-existing).

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI, tiny_config (2L/d64/4H/2KV/V512). models/ (bind
  mount): 0.5b 954 MB, 1.5b 3.09 GB, BF16; params 494,032,768 / 1,543,714,304 (== HF).
- GOTCHA: 151936 = config vocab_size; tokenizer length 151,643.

## Env (exercism — done 2026-10-09)
- `loopgym/env/`: Task, load_exercise, grade(task, files, runner="local", timeout=30).
  Local: fresh tmpdir, pristine tests + ONLY solution file, pytest junit XML, new pgroup,
  SIGKILL on timeout; reward 1 iff tests>0 and no failures/errors/skips. 9/9 known strict.
  Evidence: exercism-accept2.log.

## Splits (done 2026-10-09)
- `loopgym/env/splits.py`: audit() grades ref+stub of all 142 exercises; writes
  results/env_audit.json + results/splits.json (bucket by sha1 % 100: train<70, val<85).
  137/142 usable; splits 101/9/27. Unusable: alphametics, error-handling (ref=0),
  ledger, markdown (stub=1 leaky), parallel-letter-frequency (timeout).
- tests/test_splits.py (sha1 value, buckets, 3-fixture mini-audit). Evidence splits-accept.log.

## lgbox (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8/tokio/libc/tempfile): /run in
  fresh tmpdir; pre_exec setpgid(0,0) + RLIMIT AS(mem_mb)/CPU(timeout+1s)/NPROC 64/
  FSIZE 16MB; 4 KB stdout/stderr tails; try_wait vs deadline, killpg(SIGKILL) on timeout;
  tempdir removed; N workers via semaphore, mpsc(Q) → 503+Retry-After:1; `..`/absolute/
  empty paths → 400. /health.
- tests/integration.rs, 7 real-process tests: echo known output; /bin/sleep 30 timeout 1
  (wall 1.0–3.0 s, no /proc + kill -0 survivors); fork bomb and 2 GB alloc under 256 MB
  AS die fast, server healthy; one 503 w/ Retry-After at workers=1 queue=1.
- Evidence: lgbox-cargo-test.log (exit 0), lgbox-makecheck-*.log (make check exit 0).

## Pinned sources (done 2026-10-09)
- `tools/fetch_src.sh`: vllm/deepspeed/trl stable → read-only .tools/src/<name>-<ver>/;
  pins tools/versions.txt (vllm 0.31.0, deepspeed 0.19.7 sdist, trl 1.15.0). Never in .venv.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared): clear with
  `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag); 1.99.0 verified.
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- bootstrap.sh version probes format-fragile (shellcheck `version:`, tflint two lines).
- tests/known/ has unsorted/unused imports — Kevin's, read-only.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared-machine load can push it past 64 and
  untrusted-child forks EAGAIN (transient "Cannot fork"). Tests avoid sh-fork (/bin/sleep).

## Data / known tests
- tests/known/ = Kevin's known-answer tests (read-only, rule 1).
- data/raw/ = host export (tools/host_export.py): exercism-python, draftlab + harness
  mirrors, ticket_review.json, runs.jsonl.
