# Hidden checks: batch 1 (candidates 1–12)

<!-- Instantiated from templates/verify_step.md by the vb1 milestone (state/tickets/vb1.md).
     The loop does the first unchecked step per run. -->

## Rules
- One candidate per run. Read only that candidate's line in data/loopgym/candidates.jsonl and its fix diff
  (`git --git-dir=data/raw/repos/<repo>.git show <good> -- <files>`); do not read other candidates.
- The check is a pytest file tests/hidden/<id>/test_<short>.py, ≤ 60 lines, CPU only, < 2 min, no network, no
  model downloads beyond models/ (use tiny configs or the candidate's own small fixtures). It must test the
  behaviour the fix restored (what the ticket asked for), not the fix's exact code: a different correct fix must pass.
- Run `.venv/bin/python -m loopgym.data.verify <id>` and record its printed verdict in the step line.
  Only `verified` (bad 0/3, good 3/3) adds a task. Never edit verify.py or its rules to get a verdict.
- Check a step when its verdict is recorded (verified, flaky, inverted, error, or not-checkable).

## On failure
After 2 attempts at one candidate without a `verified` verdict, append " — not-checkable: <reason in ≤ 12 words>"
to the step, check it, and stop; the next run takes the next candidate. If verify.py itself is broken (every
candidate errors the same way), add a `- [ ]` fix ticket to TODO.md under "loopgym" with `(after: -)`, note it in
NOTES.md, and stop.

## Steps
> why: each verified check turns one real agent-shipped defect into an RL task.

- [x] **Check draftlab-7953937-0c66ede** — draftlab 7953937..0c66ede, files NOTES.md, README.md, REPORT.md, tools/check_report.py: Corrected REPORT: recompute all numbers from results/, run-level CIs, Corrections section. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check harness-520444c-4b0c866** — harness 520444c..4b0c866, files ab/polyglot-review/spec.toml, lib/ab.py, lib/abreview.py: Ticket replays (ab/ticket-review): tasks may live in other projects (task project/base/scratch) and replay a real…. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-fda7335-ca4403c** — draftlab fda7335..ca4403c, files results/alpha/ngram.json: results: hold-out alpha (14 runs, run-level CIs); partial 0.5B sweep on the fixed engine…. Verdict: verified (bad 0/3, good 3/3)
- [ ] **Check draftlab-e8f8ebb-fe264e9** — draftlab e8f8ebb..fe264e9, files NOTES.md, results/alpha/ngram.json, results/alpha/target.json: alpha: regenerate stale results with fixed counting — alpha(T,T)=1.0000. Verdict: (fill in)
- [ ] **Check draftlab-f14caf0-1fc3f9e** — draftlab f14caf0..1fc3f9e, files results/sweep.parquet.part-lora_dpo-8-0.0.jsonl: sweep batch2: complete lora_dpo k=8 cell; check_sweep_batch2 PASS. Verdict: (fill in)
- [ ] **Check draftlab-44cfa02-9b213ef** — draftlab 44cfa02..9b213ef, files NOTES.md, README.md, REPORT.md, tools/check_report.py: Corrected REPORT: recompute all numbers from results/, run-level CIs, Corrections section. Verdict: (fill in)
- [ ] **Check draftlab-5bf9b0d-ca4403c** — draftlab 5bf9b0d..ca4403c, files results/alpha/tiny.json: results: hold-out alpha (14 runs, run-level CIs); partial 0.5B sweep on the fixed engine…. Verdict: (fill in)
- [ ] **Check draftlab-1d2ae92-9b213ef** — draftlab 1d2ae92..9b213ef, files tools/check_report.py: Corrected REPORT: recompute all numbers from results/, run-level CIs, Corrections section. Verdict: (fill in)
- [ ] **Check draftlab-ed3ae6c-1e9f5d3** — draftlab ed3ae6c..1e9f5d3, files draftlab/jaxlm/spec.py, draftlab/tests/test_spec.py: spec: lossless T>0 model drafting, one target forward per step. Verdict: (fill in)
- [ ] **Check draftlab-1d2ae92-44cfa02** — draftlab 1d2ae92..44cfa02, files tools/check_report.py, tools/make_figures.py: report: REPORT.md + README.md; check_report exit 0; fix check temp-filter bug + ruff lint…. Verdict: (fill in)
- [ ] **Check draftlab-3c28deb-5bf9b0d** — draftlab 3c28deb..5bf9b0d, files draftlab/drafters/distill.py, results/distill_curve.json: distill: 1200-step run (val KL 30.6% drop), alpha eval, teacher-init control. Verdict: (fill in)
- [ ] **Check draftlab-a75ef10-1f173d1** — draftlab a75ef10..1f173d1, files NOTES.md, draftlab/jaxlm/spec.py, draftlab/tests/test_spec.py: spec: fix lossless-sampling bias (shared PRNG key) + drafter state leak; add m>=1 chi-square tests. Verdict: (fill in)
