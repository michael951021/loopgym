# Hidden checks: batch 3 (candidates 25–40)

<!-- Instantiated from templates/verify_step.md by the vb3 milestone (state/tickets/vb3.md).
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

- [x] **Check harness-bc2cdd8-97ddf39** — harness bc2cdd8..97ddf39, files lib/jobs.py: jobs: give-up survives failed.json resets; edited line = new job; pool reloads sched each poll. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-35a5d46-1e9f5d3** — draftlab 35a5d46..1e9f5d3, files draftlab/jaxlm/spec.py, draftlab/tests/test_spec.py: spec: lossless T>0 model drafting, one target forward per step. Verdict: verified (bad 0/3, good 3/3)
- [ ] **Check draftlab-bea1633-9b213ef** — draftlab bea1633..9b213ef, files README.md, REPORT.md: REPORT: recompute all numbers from results/, run-level CIs, Corrections section. Verdict: (fill in)
- [ ] **Check draftlab-7074368-07efa48** — draftlab 7074368..07efa48, files NOTES.md, tools/compare_sweep_alpha.py: checks: 393-prompt val set, by_kind key fix, trajectory-scoring tool (engine-vs-metric on same text). Verdict: (fill in)
- [ ] **Check draftlab-8e65e54-ea27462** — draftlab 8e65e54..ea27462, files web/src/pages/Overview.tsx, web/src/tests/overview.test.tsx: Resolve dl-report merge: fix the 3 make-check reds left in the tree. Verdict: (fill in)
- [ ] **Check draftlab-lora** — draftlab 9fa06b3..11b1d16, files draftlab/drafters/lora.py, state/tickets/lora.md: lora a3: fix step_loss off-by-one (HF alignment), motif tiny stream, retrain + re-eval. Verdict: (fill in)
- [ ] **Check draftlab-35a5d46-f0371d7** — draftlab 35a5d46..f0371d7, files NOTES.md: deploy/terraform: GKE as code (VPC+subnet, Standard cluster, L4 GPU pool autoscale 0-2 w/ taint). Verdict: (fill in)
- [ ] **Check draftlab-8a2123e-bdd61a2** — draftlab 8a2123e..bdd61a2, files NOTES.md, draftlab/tests/test_spec.py: spec: fix vacuous chi-square test — drafter uses different params so rejections exercise residual resample. Verdict: (fill in)
- [ ] **Check draftlab-e6895d1-5bf9b0d** — draftlab e6895d1..5bf9b0d, files NOTES.md: distill: 1200-step run (val KL 30.6% drop), alpha eval, teacher-init control. Verdict: (fill in)
- [ ] **Check draftlab-fc210e7-f0371d7** — draftlab fc210e7..f0371d7, files NOTES.md: deploy/terraform: GKE as code (VPC+subnet, Standard cluster, L4 GPU pool autoscale 0-2 w/ taint). Verdict: (fill in)
- [ ] **Check draftlab-fda7335-5bf9b0d** — draftlab fda7335..5bf9b0d, files NOTES.md: distill: 1200-step run (val KL 30.6% drop), alpha eval, teacher-init control. Verdict: (fill in)
- [ ] **Check harness-5279ea0-29cc54f** — harness 5279ea0..29cc54f, files prompts/loop.md, tests/fake_claude.py, tests/smoke.sh: harness close-out: acceptance-check-first and known-answer rules. Verdict: (fill in)
- [ ] **Check draftlab-35a5d46-2c54450** — draftlab 35a5d46..2c54450, files NOTES.md: sweep batch3: T=0.7 grid complete (42 cells); recalibrate model-drafter gates; check PASS. Verdict: (fill in)
- [ ] **Check draftlab-7953937-d5fd1c1** — draftlab 7953937..d5fd1c1, files NOTES.md: NOTES: condense under 4KB, align stale lines with corrected REPORT. Verdict: (fill in)
- [ ] **Check draftlab-44cfa02-2c2ba6b** — draftlab 44cfa02..2c2ba6b, files NOTES.md: NOTES: condense under 4KB, align stale lines with corrected REPORT. Verdict: (fill in)
- [ ] **Check draftlab-cf96577-11b1d16** — draftlab cf96577..11b1d16, files NOTES.md: lora a3: fix step_loss off-by-one (HF alignment), motif tiny stream, retrain + re-eval. Verdict: (fill in)
