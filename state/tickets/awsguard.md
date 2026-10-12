# awsguard — AWS spend guardrails

## Outcome (2026-10-11) — DONE
deploy/aws/{terraform,iam,README.md} + scripts/aws_{sweep,launch}.sh +
tests/test_aws_guard.py (15 tests, aws stubbed on PATH — no creds, rule 4).
Done-when passes: pytest 15/15, tflint clean, shellcheck clean; full
`make check` exit 0. Makefile check gate re-pointed at deploy/aws/terraform (old condition looked
for deploy/aws/*.tf and would have silently skipped validate). Known answers
hold: terraform rejects nodes=3,
max_hours=4, instance_type=p3.2xlarge via variable validation on the real
.tf files (plan, -chdir=DIR); `aws_launch.sh --dry-run` refuses
--max-price 5 (estimate 2×3.25×(5+0.0054794+0.005)=$32.57 > $10, hand math),
a stub leftover (sweep), a non-launch-role caller, and a 16 vCPU quota; the
clean dry run prints estimate $2.02 <= $10 and exits 0. Sweep --check exits
1 listing the stub instance/alarm/schedule, --kill removes them and the
post-kill --check exits 0.

## Issue
The AWS scaling study would launch paid spot instances with no cap that acts in
minutes: AWS has no hard spend limit, Budgets data lags up to a day, and money
leaks via hangs, a dead laptop mid-run, lost state, double launches, leaked keys,
or a typo in nodes/hours. Kevin's rule: overspend must be impossible, not unlikely.

## Question
Is there a set of limits, each acting within minutes and none dependent on
terraform state, on the account/instance/config level, such that every leak path
(hung run, dead laptop, double launch, typo, leaked long-lived key) is capped?
Refuting observation: any single component failing (or being bypassed by a
long-lived key) still leaves an unbounded-cost path open.

## Evidence
- `deploy/aws/` does not exist yet (only `deploy/docker/`); `scripts/aws_*.sh`
  absent. `.tools/bin/terraform` 1.16.5 + `tflint` 0.64.0 present; no AWS creds
  in the sandbox (init -backend=false + validate only, per CLAUDE.md rule 4).
- Minimal repro: `tests/test_aws_guard.py` (written first, fails until the
  guardrails exist); terraform variable-validation behavior probed in /tmp/tfprobe
  (scratch config, output in .agent-evidence/awsguard-tfprobe.log).
- Trust boundary: human's launch role (laptop) vs AWS API; kill switches must
  not depend on the laptop or terraform state living.

## Done when
`.venv/bin/python -m pytest -q tests/test_aws_guard.py && .tools/bin/tflint
--chdir=deploy/aws/terraform && .tools/bin/shellcheck scripts/aws_launch.sh
scripts/aws_sweep.sh` exit 0. Known-answer checks: (a) terraform rejects
nodes=3 / max_hours=4 / instance_type=p3.2xlarge via variable validation on the
real .tf files; (c) `aws_launch.sh --dry-run` refuses --max-price 5 (estimate
2×(3.25)×5=$32.5 ≫ $10, hand-computed) and refuses a stub-reported leftover.

## Stop when
No AWS account here: live apply is Kevin's (host). If provider download or
variable-validation surface (validate vs plan vs console) proves unusable after
2 attempts, record in NOTES.md and fall back to the strongest working check.
Do not build the study itself (VPC/SG/3-config run = ticket `aws`).

## Plan
1. tests/test_aws_guard.py first (a: terraform variable validation rejects the
   three bad values; b: .tf contains shutdown-first user-data, terminate
   behavior, delete_on_termination, one-time spot, scheduler + alarm; c:
   launch --dry-run exits 1 on bad price and on stub leftover).
2. deploy/aws/terraform: variables + validation, provider default_tags
   (Project=loopgym, Run=run_id), count=nodes instance (one-time spot,
   terminate on interruption/shutdown, IMDSv2, gp3 50GB delete_on_termination,
   shutdown-first user-data), scheduler at()+max_hours+15min terminate w/ own
   role, idle CPU alarm → arn:aws:automate terminate, budgets budget + 100%
   IAM-policy action.
3. deploy/aws/iam/launch-policy.json (RunInstances only g4dn.xlarge + tag) +
   deny-run-instances.json + README (short-lived creds, MFA, spot quota 8 vCPU).
4. scripts/aws_sweep.sh --check|--kill (tag-based, state-independent) and
   scripts/aws_launch.sh preflight (sweep clean, caller=launch role, quota ≤8
   vCPU, estimate ≤ budget) + dry-run + destroy/sweep trap.
