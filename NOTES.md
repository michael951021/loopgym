# Notes

Current state for the next run (keep < 4 KB).

## Foundations (boot — done 2026-10-09)
- `make bootstrap` = idempotent toolchain (tools/bootstrap.sh, pins tools/versions.txt):
  uv 0.12.24, .venv py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1,
  shellcheck 0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin; evidence
  .agent-evidence/boot-cleanslate.log (exit 0).
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest (minus known) + cargo test
  lgbox/ + shellcheck + hadolint + terraform init -backend=false && validate (last three skip
  until files exist). `make known` = kevin-only worktree check + strict tests/known; red until
  loopgym/rl/kernels/eval exist (test_logprob, test_stats).
- venv: torch 2.14.1+cpu (CPU index [tool.uv.sources]), transformers 5.19.0, peft, safetensors,
  accelerate, numpy, pydantic, httpx, pytest, ruff, tomli-w; uv.lock.

## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI, tiny_config (2L/d64/4H/2KV/V512); models/ (bind mount):
  0.5b 954 MB / 1.5b 3.09 GB, BF16; params 494,032,768 / 1,543,714,304 == HF.
- GOTCHA: 151936 = config vocab (lm_head dim), tokenizer length 151,643 (training uses 151936).

## Env: exercism + toy + splits (done 2026-10-09)
- `loopgym/env/exercism.py`: Task + load_exercise; local runner: fresh tmpdir, pristine tests +
  ONLY solution file, pytest junit XML, new pgroup, SIGKILL on timeout; reward 1 iff tests>0,
  no failures/errors/skips. Known reward 9/9 strict.
- `loopgym/env/toy.py`: ToyTask (banana: 1 iff "banana" ci in first 20 tokens; coinflip: 1 w.p.
  0.5 seeded by (step, index), text-blind); env.grade dispatches on ToyTask. Known: banana
  "I like Banana"=1, "apple"=0, past-20=0; coinflip 10k mean 0.4945. 12/12 test_toy.py.
- `loopgym/env/splits.py`: audit() grades ref+stub of 142 exercises → results/env_audit.json +
  splits.json; bucket = sha1 % 100 (train<70, val<85, test). 137/142 usable; 101/9/27.
  Unusable: alphametics, error-handling (ref=0), ledger, markdown (stub=1 leaky),
  parallel-letter-frequency (timeout).

## lgbox (done 2026-10-09)
- `lgbox serve --port P --workers N --queue Q` (axum/tokio/libc/tempfile): /run in fresh tmpdir;
  pre_exec setpgid(0,0) + RLIMIT AS(mem_mb)/CPU(timeout+1s)/NPROC 64/FSIZE 16MB; 4 KB tails;
  killpg on timeout; N workers semaphore, mpsc(Q) → 503+Retry-After; `..`/absolute → 400; /health.
- 7 process tests + 8 FakeVmm tests (lease/reset order, crashed-slot re-boot, exhaustion
  queue). Firecracker backend: `Vmm` trait + `VmPool` + `FirecrackerVmm` (API sock + vsock
  agent); `--dry-run` prints config; docs/firecracker.md. Evidence: lgboxfc-*.log.

## Pinned sources (vsrc — done 2026-10-09)
- `tools/fetch_src.sh`: vllm/deepspeed/trl stable → read-only .tools/src/<name>-<ver>/; pins
  tools/versions.txt (vllm 0.31.0, deepspeed 0.19.7 sdist, trl 1.15.0). Never in .venv.

## RL: GRPO math (grpo — done 2026-10-09)
- `loopgym/rl/grpo.py`: group_advantages/kl_k3/policy_loss per CLAUDE.md "Key definitions";
  known 7/7 (tests/known/test_grpo.py). Evidence: .agent-evidence/grpo-*.log.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared): clear with `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag); 1.99.0 verified.
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- bootstrap.sh version probes format-fragile (shellcheck/tflint output); fix the awk on version bumps.
- tests/known/ has I001/F401 — Kevin's, read-only; left as-is.
- RLIMIT_NPROC=64 counts ALL uid-1000 tasks; shared-machine load can push it past 64 and
  untrusted-child forks EAGAIN ("Cannot fork"). Tests use /bin/sleep, not sh-fork.

## Data
- tests/known/ = Kevin's known-answer tests (read-only, rule 1); data/raw/ = host export:
  exercism-python, draftlab+harness mirrors, ticket_review.json, runs.jsonl.
