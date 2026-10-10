# Hidden checks: batch 2 (candidates 13–24)

<!-- Instantiated from templates/verify_step.md by the vb2 milestone (state/tickets/vb2.md).
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

- [x] **Check draftlab-b7c6e4f-1f173d1** — draftlab b7c6e4f..1f173d1, files NOTES.md: spec: fix lossless-sampling bias (shared PRNG key) + drafter state leak; add m>=1 chi-square tests. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-b7c6e4f-e2004eb** — draftlab b7c6e4f..e2004eb, files NOTES.md: spec fix: structural NGramDrafter rollback sync + empty-prompt guard. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-411f1c8-e2004eb** — draftlab 411f1c8..e2004eb, files NOTES.md, draftlab/jaxlm/spec.py: spec fix: structural NGramDrafter rollback sync + empty-prompt guard. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check harness-5279ea0-429f2af** — harness 5279ea0..429f2af, files agent, ctxreport.py, fleet.py, tests/fake_claude.py: Reports: what became of each task, and why unfinished ones did not finish. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-2c54450-44cfa02** — draftlab 2c54450..44cfa02, files NOTES.md: report: REPORT.md + README.md; check_report exit 0; fix check temp-filter bug + ruff lint in …. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-443c256-2c54450** — draftlab 443c256..2c54450, files tools/check_sweep_batch3.py: sweep batch3: T=0.7 grid complete (42 cells); recalibrate model-drafter gates; check PASS. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-f0371d7-2c54450** — draftlab f0371d7..2c54450, files NOTES.md: sweep batch3: T=0.7 grid complete (42 cells); recalibrate model-drafter gates; check PASS. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check harness-25ebe03-4b0c866** — harness 25ebe03..4b0c866, files lib/ab.py: Ticket replays (ab/ticket-review): tasks may live in other projects (task project/base/scratch) and replay a real …. Verdict: verified (bad 0/3, good 3/3)
- [x] **Check draftlab-673042b-5bf9b0d** — draftlab 673042b..5bf9b0d, files NOTES.md: distill: 1200-step run (val KL 30.6% drop), alpha eval, teacher-init control. Verdict: verified (bad 0/3, good 3/3)
- [ ] **Check draftlab-f67aa44-181ed9d** — draftlab f67aa44..181ed9d, files NOTES.md, draftlab/eval/alpha.py: alpha: fix LoRA-adapter load + JAX steady-state cost ratio c. Verdict: (fill in)
- [ ] **Check draftlab-5bf9b0d-7a84891** — draftlab 5bf9b0d..7a84891, files NOTES.md: alpha: fix off-by-one logit alignment, known-answer tests, 1.5B default target. Verdict: (fill in)
- [ ] **Check draftlab-7953937-d82b39e** — draftlab 7953937..d82b39e, files deploy/terraform/gke/workload_identity.tf, tools/score_engine_trajectory.py, web/src/pages/Overview.tsx, web/src/tests/overview.test.tsx: Resolve dl-report merge: fix the 3 make-check reds left in the tree. Verdict: (fill in)
