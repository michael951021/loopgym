# Task log

<!-- One entry per completed TODO task, appended by the agent. Never edited or condensed — for
     a rolling handoff of current state see NOTES.md instead. -->
## Bootstrap toolchain and make targets
- Goal: Bootstrap toolchain and make targets
- Approach: Pinned real latest versions from upstream release endpoints, wrote the toolchain scripts first, then debugged against the acceptance command: fixed a bind-mount .venv (clear contents, never rm), a 404 rust dist pin (1.99.0), ruff 0.16 default-rule drift (pinned E4/E7/E9/F, excluded read-only tests/known), and two format-fragile version probes.
- Blockers: shared .venv/.tools are bind mounts and rm fails with EBUSY; rust dist server lags its channel toml (1.100.0 404s) — both recorded in NOTES.md gotchas for later tickets.
## Model downloads
- Goal: Model downloads
- Approach: acceptance-first (info failed on missing module, then passed on real downloads); params/dtype read from safetensors headers without loading weights; unit tests pin tiny_config and the header reader
- Blockers: none — one spec subtlety (151936 = config vocab, tokenizer length is 151643) resolved by checking config.json and noted in NOTES.md
## Pinned third-party sources
- Goal: Pinned third-party sources
- Approach: curl PyPI JSON per package, python3 stdlib picks latest non-prerelease + wheel-over-sdist, unzip/tar into .tools/src, sed/append pin; verified with the two rg checks, PyPI sha256 digest, idempotent re-run, make check
- Blockers: none
## Exercism tasks and local grader
- Goal: Exercism tasks and local grader
- Approach: read the known test for the exact API, verified the exercism layout via .meta/config.json (paasio has 2 test files), wrote the runner with start_new_session + SIGKILL process group and XML-based reward, then ran the strict known tests first (ImportError), implemented, and added unit tests for parse/prompt.
- Blockers: none
## Exercise audit and splits
- Goal: Exercise audit and splits
- Approach: wrote failing acceptance first, then audit() with ThreadPoolExecutor over load_exercise/grade, pure sha1-bucket split function, and tests/test_splits.py (published sha1 value, bucket/disjointness checks, 3-fixture mini-audit graded with real pytest).
- Blockers: none (one self-inflicted fixture bug — "bad ref" x+2 actually passed f(2)==4 — caught by the test and fixed).
## lgbox server with process backend
- Goal: lgbox server with process backend
- Approach: std::process::Command in spawn_blocking with pre_exec setpgid+setrlimit, try_wait vs deadline then killpg(SIGKILL); mpsc(Q) queue with semaphore-capped workers; raw-HTTP integration tests on 127.0.0.1:0 covering echo, sleep-30 timeout survivors, fork bomb, 2 GB alloc, 400s, 503 count.
- Blockers: transient shared-machine load pushed uid-1000 task count past RLIMIT_NPROC=64 so sh's fork hit EAGAIN; fixed by execing /bin/sleep directly and documenting the interaction in NOTES.md.
## lgbox metrics
- Goal: lgbox metrics
- Approach: lgbox metrics ticket (id: lgboxmet) complete.
- Blockers: none
## Python client and http runner
- Goal: Python client and http runner
- Approach: (no summary given)
- Blockers: none
## lgbox throughput bench (job)
- Goal: lgbox throughput bench (job)
- Approach: script auto-spawns release lgbox server on a free port, sets LOOPGYM_LGBOX_URL so the real http runner path is graded, ThreadPoolExecutor per cell, nearest-rank p50/p95, reward-1.0 + c1-overhead<50ms checks in the JSON
- Blockers: none
## lgbox bench run
- Goal: lgbox bench run
- Approach: compute job lgboxbench ran bash -c 'cd lgbox && PATH=/work/.tools/cargo/bin:$PATH cargo build --release && cd .. && .venv/bin/python tools/bench_lgbox.py' (exit 0)
- Blockers: none
