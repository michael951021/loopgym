# AWS guardrails (ticket: awsguard)

Spend protection for the AWS scaling study (ticket `aws`), which this README
also documents ("AWS run" below). The caps below make the study impossible to
overspend; each leak path has its own cap that acts within minutes:

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

## AWS run: the scaling study (host only)

Two spot g4dn.xlarge (1x T4 each) in a throwaway VPC, one placement group,
run **3 configs back to back, 50 steps each** from `configs/scale05.toml`:
`single` (1 node, FSDP), `fsdp` (2 nodes, FSDP), `deepspeed` (2 nodes,
DeepSpeed). After each config the launcher `scp`s `runs/gpu/scale_<name>/`
(`metrics.jsonl` + per-node stdout) home and writes
`results/gpu/scale_aws_<name>.json`, so a failed config is still debuggable
after the destroy.

```
scripts/aws_launch.sh --run-id <id> --budget-email you@example.com \
    --admin-cidr <your host IP>/32 [--repo-url URL] [--ssh-key PATH] \
    [--max-hours 3] [--max-price 0.30] [--nodes 2]
scripts/aws_launch.sh --dry-run        # plan + commands + estimate, changes nothing
```

Before running:
- **Merge to the default branch** — the workers `git clone <repo-url>` it.
- **SSH key**: default `~/.ssh/loopgym` (+ `.pub`); the public half becomes
  the run's `aws_key_pair`. `--admin-cidr` is the ONLY source of SSH.
- **Spot price**: check current g4dn.xlarge prices in the region,
  `aws ec2 describe-spot-price-history --instance-type g4dn.xlarge \
  --product-desc 'Linux/UNIX' --region <region> --max-items 10`; pick
  `--max-price` at/above the recent price (default 0.30; on-demand is ~0.52).

Spend: worst-case estimate `nodes x (max_hours + 0.25) x (max_price +
50 GB gp3 + public IPv4)` = **$2.02 <= $10** at the defaults (the launcher
prints it and refuses above budget); expected actual is **< $2** — the study
finishes well inside max_hours. Preflight before apply: sweep clean, caller
is the `loopgym-launch` role, G-spot quota <= 8 vCPU, estimate within budget.

Kill switches (all in the table above) and what each does:
- **user-data `shutdown -h +max_hours`** (line 1, re-armed across the
  driver-install reboot): the box stops itself at the hard cap even if every
  other system is down.
- **one-time spot, terminate-on-interruption**: AWS reclaims the instance the
  second the price beats `max_price`.
- **idle alarm** (CPU < 5% for 6x5 min): a node that booted but never trained
  is terminated by `arn:aws:automate` within ~35 min.
- **one-time scheduler** `at(launch + max_hours + 15 min)`: terminates both
  instances from outside, even if their user-data never ran, then deletes
  itself.
- **destroy + sweep trap**: on ANY exit (success, config failure, Ctrl-C) the
  launcher runs `terraform destroy` then `aws_sweep.sh --check`; on leftovers
  it exits 1 and prints the emergency command.
- **budgets**: 50/80% e-mail, 100% actual -> IAM deny on `ec2:RunInstances`
  (lagging backstop only, Budgets data can be ~24 h stale).

Recovery if a sweep reports leftovers (e.g. the host died mid-run and a
`--check` still lists `Project=loopgym` resources):

```
scripts/aws_sweep.sh --check   # list everything tagged Project=loopgym
scripts/aws_sweep.sh --kill    # cancel spot, terminate, delete alarms/schedule,
                               # wait for terminated, delete volumes/SGs, re-check
```

Sweep is tag-based (`resourcegroupstaggingapi` + `ec2 describe-*`),
independent of terraform state: a lost state file cannot hide resources.
