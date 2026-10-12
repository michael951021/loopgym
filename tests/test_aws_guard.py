"""AWS spend guardrails (ticket: awsguard). No AWS creds here (rule 4):
terraform only init+plan-for-validation; the aws CLI is stubbed on PATH."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TF_DIR = REPO / "deploy/aws/terraform"
TF = REPO / ".tools/bin/terraform"
LAUNCH = REPO / "scripts/aws_launch.sh"
SWEEP = REPO / "scripts/aws_sweep.sh"
IAM = REPO / "deploy/aws/iam"
README = REPO / "deploy/aws/README.md"

BASE_VARS = ["-var=run_id=test-run", "-var=budget_email=t@example.com",
             "-var=admin_cidr=203.0.113.7/32",
             "-var=repo_url=git@example.com:kevin/loopgym.git"]

# Stub aws CLI: leftover/kill state via env vars; state dir gets a `killed`
# marker on the first kill-type call, after which describes return empty.
STUB_AWS = """#!/usr/bin/env bash
svc="$1"; cmd="${2:-}"
leftover="${AWS_GUARD_STUB_LEFTOVER:-0}"
state_dir="${AWS_GUARD_STUB_STATE_DIR:-}"
killed=0
if [ -n "$state_dir" ] && [ -f "$state_dir/killed" ]; then killed=1; fi
case "$svc:$cmd" in
  sts:get-caller-identity)
    arn="${AWS_GUARD_STUB_CALLER:-arn:aws:sts::123456789012:assumed-role/loopgym-launch/sess}"
    printf '{"Arn": "%s", "AccountId": "123456789012"}\\n' "$arn" ;;
  service-quotas:list-service-quotas)
    v="${AWS_GUARD_STUB_VCPUS:-8}"
    printf '{"Quotas": [{"QuotaCode": "l-2f1b6a5c", "QuotaName": "All G and VT Spot Instance Requests", "QuotaValue": %s, "Adjustable": true}]}' "$v" ;;
  ec2:describe-instances)
    if [ "$leftover" = 1 ] && [ "$killed" = 0 ]; then
      printf '{"Reservations": [{"Instances": [{"InstanceId": "i-stub0000000000001", "State": {"Name": "running"}, "Tags": [{"Key": "Project", "Value": "loopgym"}]}]}]}'
    else
      printf '{"Reservations": []}'
    fi ;;
  ec2:terminate-instances)
    if [ -n "$state_dir" ]; then touch "$state_dir/killed"; fi
    printf '{"TerminatingInstances": []}' ;;
  ec2:describe-volumes) printf '{"Volumes": []}' ;;
  ec2:delete-volume) printf '{}' ;;
  ec2:describe-spot-instance-requests) printf '{"SpotInstanceRequests": []}' ;;
  ec2:cancel-spot-instance-requests) printf '{}' ;;
  ec2:describe-security-groups) printf '{"SecurityGroups": []}' ;;
  ec2:delete-security-group) printf '{}' ;;
  cloudwatch:describe-alarms)
    if [ "$leftover" = 1 ] && [ "$killed" = 0 ]; then
      printf '{"MetricAlarms": [{"AlarmName": "loopgym-idle-x-0", "Tags": [{"Key": "Project", "Value": "loopgym"}]}]}'
    else
      printf '{"MetricAlarms": []}'
    fi ;;
  cloudwatch:delete-alarms)
    if [ -n "$state_dir" ]; then touch "$state_dir/killed"; fi
    printf '{"OK": true}' ;;
  scheduler:list-schedules)
    if [ "$leftover" = 1 ] && [ "$killed" = 0 ]; then
      printf '{"Schedules": [{"Name": "loopgym-kill-x", "ScheduleArn": "arn:aws:scheduler:us-east-1:123456789012:schedule/loopgym-kill-x", "Tags": [{"Key": "Project", "Value": "loopgym"}]}]}'
    else
      printf '{"Schedules": []}'
    fi ;;
  scheduler:delete-schedule)
    if [ -n "$state_dir" ]; then touch "$state_dir/killed"; fi
    printf '{}' ;;
  resourcegroupstaggingapi:get-resources)
    if [ "$leftover" = 1 ] && [ "$killed" = 0 ]; then
      printf '{"ResourceTagMappingList": [{"ResourceARN": "arn:aws:ec2:us-east-1:123456789012:instance/i-stub0000000000001", "ResourceType": "AWS::EC2::Instance", "Tags": [{"Key": "Project", "Value": "loopgym"}]}]}'
    else
      printf '{"ResourceTagMappingList": []}'
    fi ;;
  *) printf '{}' ;;
esac
exit 0
"""


def run(args, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def out(r):
    return r.stdout + r.stderr


TFARGS = [str(TF), f"-chdir={TF_DIR}"]


@pytest.fixture(scope="module", autouse=True)
def tf_initialized():
    assert TF.exists(), f"{TF} missing"
    assert TF_DIR.is_dir(), "deploy/aws/terraform missing"
    r = subprocess.run(
        [str(TF), "init", "-backend=false", "-input=false"],
        cwd=TF_DIR, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"terraform init failed:\n{out(r)}"


@pytest.fixture()
def aws_stub(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    p = d / "aws"
    p.write_text(STUB_AWS)
    p.chmod(0o755)
    return d


def stub_env(aws_stub, extra=None):
    env = {"PATH": f"{aws_stub}:{os.environ['PATH']}"}
    env.update(extra or {})
    return env


# ---- (a) variable validation on the real .tf files -------------------------

@pytest.mark.parametrize("name,bad", [
    ("nodes", "3"),
    ("max_hours", "4"),
    ("instance_type", "p3.2xlarge"),
])
def test_validation_rejects_bad_values(name, bad):
    r = run([*TFARGS, "plan", "-no-color", *BASE_VARS, f"-var={name}={bad}"], None)
    assert r.returncode != 0, f"plan -var={name}={bad} should fail:\n{out(r)}"
    assert "Invalid value for variable" in out(r), out(r)
    assert name in out(r)


def test_valid_values_pass_validation():
    r = run([*TFARGS, "plan", "-no-color", *BASE_VARS, "-var=nodes=1"], None)
    assert "Invalid value for variable" not in out(r), out(r)


# ---- (b) guardrail attributes in the config --------------------------------

def _tf_text():
    return "\n".join(p.read_text() for p in sorted(TF_DIR.glob("*.tf")))


def test_tf_guardrail_attributes():
    t = _tf_text()
    assert re.search(r'spot_instance_type\s*=\s*"one-time"', t)
    assert re.search(r'instance_interruption_behavior\s*=\s*"terminate"', t)
    assert re.search(r'instance_initiated_shutdown_behavior\s*=\s*"terminate"', t)
    assert re.search(r'delete_on_termination\s*=\s*true', t)
    assert re.search(r'volume_type\s*=\s*"gp3"', t)
    assert re.search(r'http_tokens\s*=\s*"required"', t)
    assert re.search(r'action_after_completion\s*=\s*"DELETE"', t)
    assert "aws-sdk:ec2:terminateInstances" in t
    assert re.search(r'arn:aws:automate:\$\{var\.region\}:ec2:terminate', t)
    assert "CPUUtilization" in t
    assert re.search(r'evaluation_periods\s*=\s*6', t)
    assert re.search(r'period\s*=\s*300', t)
    assert re.search(r'threshold\s*=\s*5', t)
    assert "aws_budgets_budget " in t or re.search(r'aws_budgets_budget\.', t)
    assert re.search(r'aws_budgets_budget_action\s*"', t)
    assert re.search(r'Project\s*=\s*"loopgym"', t)
    assert re.search(r'Run\s*=\s*var\.run_id', t)
    assert "iam:" in t and "ec2:TerminateInstances" in t  # scheduler kill role policy


def test_shutdown_is_first_user_data_action():
    src = (TF_DIR / "instances.tf").read_text()
    m = re.search(r'user_data\s*=\s*<<-?["\']?EOF(.*?)EOF', src, re.S)
    assert m, "user_data heredoc not found in instances.tf"
    lines = [line.strip() for line in m.group(1).strip().splitlines()]
    assert lines[0] == "#!/bin/bash"
    act = [line for line in lines[1:] if line and not line.startswith("#")]
    assert re.match(r'shutdown -h \+\$\(\(.*max_hours.*60.*\)\)', act[0]), act[0]
    # ticket aws: the study bootstrap (clone the repo, build .venv-gpu) is allowed,
    # but only AFTER the shutdown line — a hung bootstrap cannot stop the hard stop.
    boot = "\n".join(act[1:])
    assert "git clone" in boot, "user_data must clone the repo (after the shutdown line)"
    assert ".venv-gpu" in boot, "user_data must build .venv-gpu (after the shutdown line)"


def test_launch_policy_json():
    p = json.loads((IAM / "launch-policy.json").read_text())
    run_inst = [s for s in p["Statement"]
                if s["Effect"] == "Allow" and "ec2:RunInstances" in (
                    s["Action"] if isinstance(s["Action"], list) else [s["Action"]])]
    assert len(run_inst) == 1
    cond = run_inst[0]["Condition"]["StringEquals"]
    assert cond["ec2:InstanceType"] == "g4dn.xlarge"
    assert cond["aws:RequestTag/Project"] == "loopgym"
    assert "s3" not in json.dumps(p).lower()  # nothing else


def test_deny_policy_json():
    d = json.loads((IAM / "deny-run-instances.json").read_text())
    stmts = d["Statement"] if isinstance(d["Statement"], list) else [d["Statement"]]
    assert any(s["Effect"] == "Deny" and "ec2:RunInstances" in (
        s["Action"] if isinstance(s["Action"], list) else [s["Action"]]) for s in stmts)


def test_readme_covers_setup():
    t = README.read_text().lower()
    for needle in ("sso", "mfa", "all g and vt spot instance requests", "8 vcpu", "short-lived"):
        assert needle in t, f"README missing: {needle}"
    assert "aws_sweep.sh --kill" in README.read_text()


# ---- (c) launch preflight + sweep, stub aws on PATH -------------------------

DRY = ["--dry-run", "--nodes", "2", "--max-hours", "3", "--max-price", "0.30",
       "--run-id", "t1", "--budget-email", "t@example.com"]


def test_dry_run_refuses_overspend(aws_stub):
    r = run([str(LAUNCH), *DRY, "--max-price", "5"], stub_env(aws_stub))
    assert r.returncode == 1, out(r)
    assert re.search(r'(?i)estimate.*budget|budget', out(r))


def test_dry_run_refuses_leftover(aws_stub, tmp_path):
    r = run([str(LAUNCH), *DRY],
            stub_env(aws_stub, {"AWS_GUARD_STUB_LEFTOVER": "1",
                                "AWS_GUARD_STUB_STATE_DIR": str(tmp_path)}))
    assert r.returncode == 1, out(r)
    assert re.search(r'(?i)leftover|sweep', out(r))


def test_dry_run_refuses_wrong_caller(aws_stub):
    r = run([str(LAUNCH), *DRY],
            stub_env(aws_stub, {"AWS_GUARD_STUB_CALLER": "arn:aws:iam::123456789012:user/stale-key"}))
    assert r.returncode == 1, out(r)
    assert "launch role" in out(r)


def test_dry_run_refuses_high_quota(aws_stub):
    r = run([str(LAUNCH), *DRY], stub_env(aws_stub, {"AWS_GUARD_STUB_VCPUS": "16"}))
    assert r.returncode == 1, out(r)
    assert "quota" in out(r).lower()


def test_dry_run_passes_clean(aws_stub):
    # 2 * (3 + 0.25) * (0.30 + 50*0.08/730 + 0.005) = 2.018 -> $2.02
    r = run([str(LAUNCH), *DRY], stub_env(aws_stub))
    assert r.returncode == 0, out(r)
    assert "2.02" in out(r), out(r)


def test_sweep_check_and_kill(aws_stub, tmp_path):
    env = stub_env(aws_stub, {"AWS_GUARD_STUB_LEFTOVER": "1",
                              "AWS_GUARD_STUB_STATE_DIR": str(tmp_path)})
    r1 = run([str(SWEEP), "--check"], env)
    assert r1.returncode == 1, out(r1)
    assert "i-stub0000000000001" in out(r1)
    r2 = run([str(SWEEP), "--kill"], env)
    assert r2.returncode == 0, out(r2)
    r3 = run([str(SWEEP), "--check"], env)
    assert r3.returncode == 0, out(r3)


# ---- (d) the study itself (ticket aws): plan, network, bootstrap -------------


def test_tf_network_guardrails():
    t = _tf_text()
    for res in ("aws_vpc", "aws_internet_gateway", "aws_subnet", "aws_route_table",
                "aws_security_group", "aws_placement_group", "aws_key_pair"):
        assert re.search(rf'resource\s+"{res}"\s+"', t), f"missing resource {res}"
    assert re.search(r'strategy\s*=\s*"cluster"', t), "placement group must be cluster"
    assert re.search(r'from_port\s*=\s*22', t) and "var.admin_cidr" in t, \
        "SSH must come from var.admin_cidr only"
    assert re.search(r'self\s*=\s*true', t), "run-internal traffic allowed via self-ingress"
    assert "10.0.0.10" in t, "node 0's private IP must pin the c10d master (MASTER_IP_AWS)"
    for banned in ("aws_nat_gateway", "aws_lb", "aws_alb", "aws_elb", "aws_eip", "aws_s3"):
        assert banned not in t, f"{banned} costs money while idle; not allowed"


def _gen(topo, dist, name):
    g = run([str(REPO / ".venv" / "bin" / "python"), "-m", "loopgym.rl.launch", "cmds",
             "--topology", topo, "--dist", dist, "--steps", "50",
             "--out", f"runs/gpu/scale_{name}"], None)
    assert g.returncode == 0, g.stderr
    return [ln for ln in g.stdout.splitlines() if ln.strip()]


def test_dry_run_prints_plan_commands_estimate(aws_stub):
    r = run([str(LAUNCH), *DRY], stub_env(aws_stub))
    assert r.returncode == 0, out(r)
    o = out(r)
    for needle in ("one-aws", "two-aws", "fsdp", "deepspeed", "50"):
        assert needle in o, f"plan missing {needle!r}\n{o}"
    for topo, dist, name in (("one-aws", "fsdp", "single"), ("two-aws", "fsdp", "fsdp"),
                             ("two-aws", "deepspeed", "deepspeed")):
        for cmd in _gen(topo, dist, name):
            assert cmd in o, f"generated command not printed by --dry-run:\n{cmd}\n{o}"
    # worst-case estimate, hand math: 2 * 3.25 * (0.30 + 50*0.08/730 + 0.005) = 2.02
    assert re.search(r"estimate \$2\.02 <= budget \$10", o), o


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
