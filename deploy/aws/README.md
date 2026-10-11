# AWS guardrails (ticket: awsguard)

Spend protection for the AWS scaling study (ticket `aws`). The study itself
(VPC/SG, 3 configs x 50 steps) is not here; this is the cap that makes the
study impossible to overspend, and each leak path has its own cap that acts
within minutes:

| Leak path | Cap | Acts in |
|---|---|---|
| typo in nodes/hours/price | terraform variable validation + `aws_launch.sh` estimate preflight | at launch (refused) |
| hung run, laptop dies mid-run | one-time scheduler `at(launch+max_hours+15m)` terminates the instances, deletes itself | max_hours + 15 min |
| instance up but idle (crash at boot) | CloudWatch CPU < 5% for 6x5 min -> `arn:aws:automate:REGION:ec2:terminate` | ~35 min |
| spot price beaten / interruption | one-time spot, `instance_interruption_behavior = "terminate"` | seconds |
| run finishes | `instance_initiated_shutdown_behavior = "terminate"`; user-data line 1 is `shutdown -h +max_hours` | at finish / max_hours |
| second launch | `aws_launch.sh` refuses unless `aws_sweep.sh --check` is clean | at launch |
| leaked long-lived key | launch role: `ec2:RunInstances` only for g4dn.xlarge in us-east-1 with tag `Project=loopgym`; account spot quota 8 vCPU = exactly 2 instances | at launch |
| total spend | Budgets: e-mail at 50/80% actual, 100% forecast; at 100% actual an IAM deny on `ec2:RunInstances` attaches to the launch role (LAGGING backstop: Budgets data can be ~24 h stale) | hours-days |

## One-time setup (Kevin, on the host)

1. **Launch role.** Create the role `loopgym-launch` trusting your human user
   (or an IAM Identity Center permission set), and attach
   `iam/launch-policy.json` — it is the ONLY policy the launch role gets:
   `ec2:RunInstances` is allowed only for `g4dn.xlarge` in `us-east-1` with
   request tag `Project=loopgym`, plus the specific actions terraform applies.
2. **Short-lived credentials only.** Log in with `aws sso login` (or an STS
   session of at most 4 h, e.g. `aws sts assume-role` on top of the SSO
   session). **Never keep long-lived access keys on disk.** If a key leaks,
   the launch-role policy caps what it can spend, but the key itself should
   not exist.
3. **Root MFA on** for the account.
4. **Spot quota.** Service Quotas -> EC2 -> "All G and VT Spot Instance Requests":
   set to **8 vCPU** (= exactly 2 x g4dn.xlarge). A typo'd third instance or a
   second double launch cannot then be placed.
5. **Automated actions role.** The idle alarm uses
   `arn:aws:automate:us-east-1:ec2:terminate`; create the service-linked role
   once: `aws ram create-service-linked-role --aws-service-name
   automate.ec2.amazonaws.com` (verify the exact service name in the CloudWatch
   docs when running this on the host).

## Running (host only)

```
scripts/aws_launch.sh --run-id <id> --max-hours 3 --max-price 0.30 --nodes 2 \
    --budget-email you@example.com -- <training command>
```

Preflight (also run by `--dry-run`, testable with a stub `aws` on PATH):
sweep clean, caller is `loopgym-launch`, G-spot quota <= 8 vCPU, and worst-case
estimate `nodes x (max_hours + 0.25) x (max_price + EBS $/h + public IPv4
$0.005/h)` <= `budget_usd` ($10). On EXIT/INT/TERM it runs `terraform destroy`
then `aws_sweep.sh --check` and exits 1 loudly (printing
`scripts/aws_sweep.sh --kill`) if anything tagged `Project=loopgym` is left.

Cleanup / emergency:

```
scripts/aws_sweep.sh --check   # list everything tagged Project=loopgym
scripts/aws_sweep.sh --kill    # terminate/delete all of it
```

Sweep is tag-based (`resourcegroupstaggingapi` + `ec2 describe-*`),
independent of terraform state: a lost state file cannot hide resources.
