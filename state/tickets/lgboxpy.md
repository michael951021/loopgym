# lgboxpy — Python client and http runner

## Issue
The trainer can only grade through the in-process local runner; there is no Python client
for the Rust lgbox /run endpoint, so the Rust sandbox (hard limits, bounded queue) is
unreachable from `loopgym.env.grade`, and the known reward tests can only exercise one runner.

## Question
Can `grade(task, files, runner="http")` produce rewards identical to the local runner when
LOOPGYM_LGBOX_URL points at a running `lgbox serve`? Refuted if any case in
tests/known/test_reward.py gives a different reward for runner="http" vs "local", or if
`bash tools/lgbox_smoke.sh` exits non-zero.

## Evidence
- Server API observed in lgbox/src/lib.rs: POST /run {files, cmd, timeout_s, mem_mb,
  outputs} -> {exit_code, timed_out, outputs{name: text|null}, ...}; 503 + Retry-After when
  queue full; 400 on bad paths/missing fields.
- Acceptance: `bash tools/lgbox_smoke.sh` (build release, serve on a free port, run the full
  known test_reward.py with LOOPGYM_LGBOX_URL set, strict mode). First run expected to fail:
  runner "http" unknown / no client. Evidence in .agent-evidence/lgboxpy-smoke*.log.

## Done when
- `bash tools/lgbox_smoke.sh` exits 0: every test_reward.py case passes for BOTH "local" and
  "http" (strict, no skips).
- `make check` exit 0 (ruff, pytest, cargo test, shellcheck).
- Known answer: the known tests ARE the reference (hand-picked exercises + reward hacks);
  http reward must equal local reward per case (the test asserts 1.0/0.0, not equality —
  both runners asserted against the same fixed values).
- New unit tests tests/test_lgbox.py (503/Retry-After retry, payload shape, shared XML scorer).

## Stop when
Scope is client + runner + smoke script only; no server changes, no new backends, no
throughput bench. After two attempts without new evidence (e.g. server kills pytest under
rlimits), record the blocker in NOTES.md and stop/split.

## Plan
1. Extract XML scoring from _local in exercism.py (`_xml_counts`, `_reward`); share it.
2. loopgym/env/lgbox.py: LgboxClient (httpx; retry 503 honoring Retry-After, bounded),
   `_http` runner (pristine test files + solution file; pytest --junitxml=r.xml; outputs
   ["r.xml"]); register RUNNERS["http"]; runner_names() gates "http" on LOOPGYM_LGBOX_URL.
3. tests/test_lgbox.py with httpx.MockTransport (no server needed).
4. tools/lgbox_smoke.sh; run it, then make check; NOTES.md + commit.
