# AWS scaling study under $10 (id: aws)

## Issue
The twonode/awsguard work gives us the guardrails and the command generator, but no runnable
cloud study: no network (VPC/SG), no bootstrap on the instances, and `aws_launch.sh` has no
3-config study plan. Kevin cannot run the 0.5B 1→2-node scaling study without them.

## Question
Can `aws_launch.sh` launch the 3-config study (1 node, 2 nodes fsdp, 2 nodes deepspeed,
50 steps each) on 2× spot g4dn.xlarge such that (a) `--dry-run` prints the plan, the torchrun
commands and a worst-case estimate ≤ $10, and (b) every awsguard cap (shutdown-first
user_data, one-time spot, scheduler, idle alarm, budget deny, sweep) still holds — i.e. the
study cannot overspend even when a config fails or the script is interrupted?
Refutation: dry-run missing any of plan/commands/estimate; a guardrail attribute lost from the
.tf; a path where destroy/sweep can be skipped after a config failure.

## Evidence
- scripts/aws_launch.sh (preflight + trap exist; no plan, no orchestration),
  deploy/aws/terraform/* (instances + guardrails; no VPC/SG/PG), loopgym/rl/launch.py
  (topologies single/two-local/two-aws; MASTER_IP_AWS=10.0.0.10; CONFIG=configs/scale05.toml),
  tests/test_aws_guard.py (15 stub-aws tests, $2.02 hand math).
- Acceptance check (written first): new tests in tests/test_aws_guard.py
  (dry-run prints 3-config plan + generator commands + estimate; tf has VPC/SG/PG, SSH only
  from var.admin_cidr, no NAT/LB/S3; bootstrap only after the shutdown line) run against the
  real .tf and real scripts with the stub aws. Evidence: .agent-evidence/aws-dryrun.log.

## Done when
`.tools/bin/terraform -chdir=deploy/aws/terraform init -backend=false && .tools/bin/terraform
-chdir=deploy/aws/terraform validate && bash scripts/aws_launch.sh --dry-run &&
.venv/bin/python -m pytest -q tests/test_aws_guard.py` exit 0, and the dry run prints a
worst-case estimate ≤ $10. Known answer: estimate = 2·(3+0.25)·(0.30 + 50·0.08/730 + 0.005)
= $2.02 (hand math, already pinned in test_dry_run_passes_clean); plausible range $1–$10
(CLAUDE.md: expected < $2). Regression: all 15 existing awsguard tests unchanged in meaning.

## Stop when
No AWS calls from the sandbox (rule 4): real mode is untestable here — it is written, dry-run
exercised with stubs, and documented for Kevin. Do not weaken any awsguard cap; do not add
S3/NAT/LB/always-on resources. If bootstrap design (driver install + reboot) cannot be made
reboot-safe for the max_hours shutdown, stop and record it; the scheduler cap still holds.

## Plan
1. state/tickets/aws.md (this file); failing tests first.
2. network.tf + variables (admin_cidr, repo_url, ssh_public_key_path) + instances.tf wiring,
   pinned private IPs (node 0 = 10.0.0.10 = MASTER_IP_AWS), bootstrap after the shutdown line
   (driver install is reboot-resume; re-arm shutdown around the reboot).
3. launch.py: one-aws topology + `--steps`/`--out` on cmds (default output unchanged).
4. aws_launch.sh: dry-run plan; real mode run 3 configs back to back, scp metrics/logs home
   after each config, report to results/gpu/scale_aws_{single,fsdp,deepspeed}.json,
   destroy + sweep in the EXIT trap.
5. README "AWS run": setup, estimate, spot price check, kill switches, recovery.
6. Done-when chain + make check; NOTES.md; commit.
