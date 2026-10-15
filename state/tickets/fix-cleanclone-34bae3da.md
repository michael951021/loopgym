# Fix cleanclone: non-zero exit (exit-code)

## Issue
Compute job `cleanclone` exited non-zero (`exit-code`, exit 2) with no Python traceback — a failing command, not a crash in the training code. Blocks its dependents.

## Evidence
Exit: `exit-code`, code 2; memory cap 12.0 GB; peak not recorded (systemd MemoryPeak, host-side).
Failing command: `rm -rf /tmp/lgcc && git clone -q /work /tmp/lgcc && cd /tmp/lgcc && PATH=/work/.tools/bin:/work/.tools/cargo/bin:/work/.venv/bin:$PATH make check`
Last console output (the UI does not show this):
/tmp/lgcc/.venv/bin/python -m ruff check .
/bin/bash: line 1: /tmp/lgcc/.venv/bin/python: No such file or directory
make: *** [Makefile:11: check] Error 127
/tmp/lgcc/.venv/bin/python -m ruff check .
/bin/bash: line 1: /tmp/lgcc/.venv/bin/python: No such file or directory
make: *** [Makefile:11: check] Error 127
Full log: /home/kevin/Documents/coders/loop/logs/jobs/loopgym-cleanclone.log

## Done when
The command exits 0 on the real inputs, with a fast check that covers the fixed path.

## Stop when
After two attempts with no new evidence, record the blocker and stop — the signature then escalates to a human
card. Reproduce cheaply first (tiny config / small input), not by re-running the whole job.

## Plan
1. Reproduce the failure with the smallest/fastest invocation.
2. Fix the cause (trim memory / fix the command); add a fast check.
3. Make sure the fix reaches the job's branch so its retry picks it up.
