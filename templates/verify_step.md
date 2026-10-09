# Hidden checks: {batch} (candidates {range})

<!-- Instantiated by the vb1/vb2/vb3 milestone ticket into state/verify/{batch}.md: copy this file, replace the
     placeholders, and write one step per candidate (in score order) under ## Steps, each in the step format below.
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
<!-- step format:
- [ ] **Check <id>** — <repo> <bad>..<good>, files <files>: <fix message, ≤ 15 words>. Verdict: (fill in)
-->
