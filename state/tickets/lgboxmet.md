# lgbox metrics

## Issue
The grading server is the RL run's bottleneck to watch, but nothing is measurable about it:
no throughput, latency, queue backlog or worker saturation. Without metrics we cannot tell
whether lgbox is the thing limiting rollout/grade throughput.

## Question
Does `GET /metrics` return Prometheus text format with
`lgbox_runs_total{outcome=ok|nonzero|timeout|rejected}` (counter), `lgbox_run_seconds`
(histogram), `lgbox_queue_depth` (gauge), `lgbox_workers_busy` (gauge), where the outcome
counts exactly match the runs submitted? Refutation: after 3 runs (ok, nonzero, timeout) the
counts are not exactly ok=1, nonzero=1, timeout=1, rejected=0; or the histogram did not observe
all 3; or a gauge is non-zero after all runs finish.

## Evidence
- lgbox/src/lib.rs: `app()` (route table), `run_handler` (400/503 rejections, try_send),
  `dispatch` (per-job permit + `run_blocking`), `RunOut` (exit_code/timed_out/wall_ms),
  `outcome` not yet classified anywhere.
- Check: new `metrics_counts_exact` in tests/integration.rs — 3 real runs then GET /metrics.
- Log: .agent-evidence/lgboxmet-cargo-test.log.

## Done when
`cd lgbox && PATH=$PWD/../.tools/cargo/bin:$PATH cargo test` passes, incl. the new test.
Known answer: 3 hand-chosen runs → exact integer counts (1/1/1/0), histogram count=3, both
gauges 0 after all responses received (response is sent after the metric updates, so no race).
Plausible range: run_seconds observations ~0 ms (echo) to ~1 s (sleep@timeout 1).

## Stop when
Scope: /metrics + the 4 series only. No client-side metrics, no firecracker, no Python client,
no pushing to a remote server. Two attempts without new evidence (e.g. prometheus 0.14 API
mismatch, label materialization not emitting zero series) → record in NOTES.md and stop/split.

## Plan
1. Add `prometheus = "0.14"` (in cargo cache; offline build) to Cargo.toml.
2. `Metrics::new()` builds Registry + 4 series; pre-materialize all 4 outcome labels so each
   series is present at 0. Store in `AppState`; clone handles into `dispatch`.
3. Record: dispatch → outcome_of(RunOut) counter + run_seconds.observe(wall_ms) + dec gauges
   (busy before/after run_blocking; queue on accept in run_handler, on finish in dispatch);
   run_handler → rejected counter on 400/503.
4. `GET /metrics` → TextEncoder::encode_utf8(registry.gather()).
5. Add test, run acceptance (fail first), implement, cargo test, make check, commit, merge.
