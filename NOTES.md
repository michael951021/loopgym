# Notes

Current state for the next run (keep < 4 KB).

## Foundations (done 2026-10-09)
- `make bootstrap` (tools/bootstrap.sh, pins tools/versions.txt): uv 0.12.24, .venv
  py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1, shellcheck
  0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin; evidence: boot-*.log.
- `make check` = ruff + pytest (minus known) + cargo test + shellcheck + hadolint
  + terraform (last three skip until files exist). `make known` = check_known.sh +
  strict tests/known; red: test_logprob/test_stats (loopgym.kernels/eval unbuilt).
- venv: torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate,
  numpy, pydantic, httpx, pytest, ruff, tomli-w; uv.lock.

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI, tiny_config (2L/d64/4H/2KV/V512); models/
  (bind mount) 0.5b/1.5b BF16, params == HF; tokenizers byte-identical.
  GOTCHA: 151936 = config vocab (lm_head dim), tokenizer length 151,643 (training
  uses 151936). Evidence: models-{fetch,info2}.log.

## Env: exercism + toy + splits (done 2026-10-09)
- `loopgym/env/exercism.py`: Task + load_exercise; local runner: fresh tmpdir,
  pristine tests + ONLY solution file, pytest junit XML, new pgroup, SIGKILL on
  timeout; reward 1 iff tests>0, no fail/err/skip. Known reward 9/9 strict.
- `loopgym/env/toy.py`: ToyTask (banana: 1 iff "banana" ci in first 20 tokens;
  coinflip: 1 w.p. 0.5 seeded by (step, index), text-blind). Known: "I like
  Banana"=1, "apple"=0, past-20=0; coinflip 10k mean 0.4945. 12/12 test_toy.py.
- `loopgym/env/splits.py`: audit(): ref+stub of 142 exercises →
  results/env_audit.json + splits.json; bucket = sha1%100 (train<70, val<85, test).
  137/142 usable; 101/9/27. Unusable: alphametics, error-handling (ref=0), ledger,
  markdown (stub leaky), parallel-letter-frequency (timeout). Evidence:
  exercism-*, toy-*, splits-*.log.

## lgbox (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum 0.8): /run in fresh tmpdir;
  pre_exec setpgid + RLIMIT AS(mem_mb)/CPU(timeout+1s)/NPROC 64/FSIZE 16MB; 4 KB
  tails; killpg on timeout; N workers, mpsc(Q) → 503; `..`/abs → 400. /health.
- /metrics (prometheus 0.14 text): lgbox_runs_total{outcome=ok|nonzero|timeout|
  rejected} (rejected=400/503; outcome from RunOut), lgbox_run_seconds histogram,
  lgbox_queue_depth, lgbox_workers_busy. All 4 outcomes pre-materialized; 8/8
  integration tests incl. metrics_counts_exact. Evidence: lgboxmet-*.log.
- Firecracker backend: `Vmm` trait + `VmPool` (round-robin lease, reset-after-run,
  crashed slot re-booted on next lease, queues when full) + `FirecrackerVmm` (API
  sock + vsock agent); `Runner` trait: app takes process|firecracker; `--dry-run`
  prints config; docs/firecracker.md. 7 process + 8 FakeVmm tests (rule 4, no
  /dev/kvm). Evidence: lgboxfc-*.log.

## Pinned sources (done 2026-10-09)
- `tools/fetch_src.sh`: vllm/deepspeed/trl stable → read-only .tools/src/<name>-
  <ver>/; pins tools/versions.txt (vllm 0.31.0, deepspeed 0.19.7 sdist, trl
  1.15.0). Never in .venv. Evidence: vsrc-*.log.

## RL (grpo — done 2026-10-09)
- `loopgym/rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md "Key
  definitions"; known 7/7 (tests/known/test_grpo.py). Evidence: grpo-*.log.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: clear with `find DIR
  -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s 1.100.0/1.101.0; 1.99.0 verified.
- ruff 0.16 defaults drifted (I/EXE) — pinned in [tool.ruff.lint].
- bootstrap.sh version probes format-fragile (shellcheck/tflint output).
- tests/known/ has I001/F401 — Kevin's, read-only; left as-is.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared load can push past 64 and
  untrusted-child forks EAGAIN ("Cannot fork"). Tests use /bin/sleep, not sh-fork.

## Data
- tests/known/ = Kevin's (read-only, rule 1); data/raw/ = host export:
  exercism-python, draftlab+harness mirrors, ticket_review.json, runs.jsonl.
