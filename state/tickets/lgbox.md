# lgbox server with process backend

## Issue
RL runs must grade tens of thousands of untrusted model-generated programs. Running them
directly risks the grader being fork-bombed, OOM'd, or escaped from (path traversal). Nothing
exists yet to run untrusted commands in a box with hard resource limits and return bounded
results. This endpoint is the foundation the lgbox client/metrics build on.

## Question
Does `lgbox serve` execute a submitted job in a fresh temp dir, a new process group, with
setrlimit (AS/CPU/NPROC/FSIZE) applied pre-exec, kill the whole group on timeout, remove the
temp dir, and bound concurrency (N workers, queue Q, 503+Retry-After when full)? Refutation:
a timed-out child (or grandchild) survives (kill -0 succeeds), a limit is not enforced
(2 GB alloc / fork bomb succeeds or hangs the server), a `..`/absolute path reaches the disk,
or a request beyond Q is accepted instead of 503.

## Evidence
- No lgbox/ dir exists yet (observed): acceptance `cd lgbox && cargo test` fails (missing crate).
- Real-process checks in tests/integration.rs: echo round trip (known output), `sleep 30`
  with timeout 1 (timed_out, no survivor found by /proc scan + kill(pid,0)), python fork bomb
  and 2 GB bytearray under 256 MB RLIMIT_AS (must fail fast, server stays healthy),
  `../x`/absolute → 400, workers=1 queue=1 → exactly one 503 with Retry-After.
- Log: .agent-evidence/lgbox-cargo-test.log.

## Done when
`cd lgbox && PATH=$PWD/../.tools/cargo/bin:$PATH cargo test` passes (all real-process tests).
Known answers: echo output is hand-known; kill -0 on any post-run candidate pid fails (ESRCH);
503 is a count, not a guess. Plausible range written before run: timeout response wall time
1.0–3.0 s (spec says < 3 s); echo run wall_ms < 2000.

## Stop when
Scope: process backend + limits + queue only. No metrics (next ticket), no python client,
no firecracker backend, no Docker. Two attempts without new evidence on a failing limit
(e.g. RLIMIT_NPROC behaving differently in this container) → record observed behavior in
NOTES.md and stop or split.

## Plan
1. Cargo crate lgbox: tokio+axum server, /run /health, manual arg parse (deps per ticket).
2. std::process::Command in spawn_blocking: pre_exec setpgid(0,0) + setrlimit x4, piped
   stdout/stderr with 4 KB tails, try_wait loop vs deadline, killpg(SIGKILL) + reap.
3. Handler validates paths (400), enqueues via mpsc(Q) → 503+Retry-After:1 when full;
   N worker tasks drain it.
4. tests/integration.rs with raw-HTTP client on 127.0.0.1:0; run acceptance; make check; commit.
