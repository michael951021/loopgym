#!/usr/bin/env bash
# Launcher for the AWS scaling study (ticket `aws`). Every spend guardrail from
# deploy/aws/README.md that lives at launch time: preflight refuses unless the
# tag sweep is clean (no double launch), the caller is the loopgym-launch role
# (no long-lived root keys), the account G-spot quota is <= 8 vCPU, and the
# worst-case estimate is within budget. On EXIT/INT/TERM it destroys, re-sweeps,
# and fails loudly if anything tagged Project=loopgym survived.
# No AWS creds in the sandbox: only --dry-run (stub aws on PATH) runs here.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TF_BIN="$REPO_ROOT/.tools/bin/terraform"
TF_DIR="$REPO_ROOT/deploy/aws/terraform"

NODES=2
MAX_HOURS=3
MAX_PRICE=0.30
BUDGET_USD=10
RUN_ID=""
BUDGET_EMAIL=""
REGION="${AWS_REGION:-us-east-1}"
LAUNCH_ROLE="loopgym-launch"
MAX_QUOTA_VCPUS=8
ACCOUNT_BUDGET_USD=10
DRY_RUN=0
TRAIN_CMD=()

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") [--dry-run] [--nodes N] [--max-hours H] [--max-price P]
    [--budget-usd B] [--run-id ID] [--budget-email E] [--region R]
    [--launch-role NAME] -- <training command>
--dry-run: run the full preflight, change nothing, exit 0/1.
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --nodes) NODES="${2:?--nodes needs a value}"; shift 2 ;;
    --max-hours) MAX_HOURS="${2:?--max-hours needs a value}"; shift 2 ;;
    --max-price) MAX_PRICE="${2:?--max-price needs a value}"; shift 2 ;;
    --budget-usd) BUDGET_USD="${2:?--budget-usd needs a value}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
    --budget-email) BUDGET_EMAIL="${2:?--budget-email needs a value}"; shift 2 ;;
    --region) REGION="${2:?--region needs a value}"; shift 2 ;;
    --launch-role) LAUNCH_ROLE="${2:?--launch-role needs a value}"; shift 2 ;;
    --) shift; TRAIN_CMD=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

refuse() { echo "aws_launch: REFUSED — $*" >&2; exit 1; }

preflight_sweep() {
  if ! AWS_REGION="$REGION" "$SCRIPT_DIR/aws_sweep.sh" --check --region "$REGION"; then
    refuse "sweep found leftover loopgym resources; refusing a double launch.
  Inspect: scripts/aws_sweep.sh --check   Delete (yours): scripts/aws_sweep.sh --kill"
  fi
}

preflight_caller() {
  local out arn
  out=$(aws sts get-caller-identity --output json 2>&1) \
    || refuse "sts get-caller-identity failed: $out"
  arn=$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("Arn", ""))')
  case "$arn" in
    *"role/${LAUNCH_ROLE}"*) ;;
    *) refuse "caller is '$arn', not the ${LAUNCH_ROLE} launch role.
  Assume the launch role with a short-lived SSO/STS session; never a long-lived key." ;;
  esac
}

preflight_quota() {
  local page out found token=""
  while :; do
    if [ -n "$token" ]; then
      page=$(aws service-quotas list-service-quotas --service-code ec2 \
        --region "$REGION" --max-results 100 --next-token "$token" --output json 2>&1) \
        || refuse "service-quotas list-service-quotas failed: $page"
    else
      page=$(aws service-quotas list-service-quotas --service-code ec2 \
        --region "$REGION" --max-results 100 --output json 2>&1) \
        || refuse "service-quotas list-service-quotas failed: $page"
    fi
    out=$(printf '%s' "$page" | python3 -c '
import json, sys
d = json.load(sys.stdin)
v = "-"
for q in d.get("Quotas", []):
    if "All G and VT Spot Instance Requests" in q.get("QuotaName", ""):
        v = q["QuotaValue"]
print(v, d.get("NextToken") or "-")
')
    found="${out%% *}"
    token="${out##* }"
    [ "$token" = "-" ] && token=""
    [ -n "$found" ] && [ "$found" != "-" ] && break
    [ -z "$token" ] && break
  done
  [ "$found" = "-" ] && refuse "could not find the 'All G and VT Spot Instance Requests' EC2 quota; set it to ${MAX_QUOTA_VCPUS} vCPU and re-check."
  if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$found" "$MAX_QUOTA_VCPUS"; then
    echo "aws_launch: G-spot quota ${found} vCPU <= ${MAX_QUOTA_VCPUS} vCPU."
  else
    refuse "G-spot quota is ${found} vCPU, above the ${MAX_QUOTA_VCPUS} vCPU cap.
  Service Quotas -> EC2 -> 'All G and VT Spot Instance Requests': set to ${MAX_QUOTA_VCPUS} vCPU."
  fi
}

preflight_estimate() {
  local est ok
  # shellcheck disable=SC2016  # python f-string braces, not shell
  est=$(python3 -c '
import math, sys
nodes, hours, price, budget = (float(x) for x in sys.argv[1:5])
if any(not math.isfinite(v) for v in (nodes, hours, price, budget)) or nodes < 1:
    raise SystemExit(2)
# +0.25 h: the one-time scheduler may fire up to 15 min after max_hours.
ebs_h = 50 * 0.08 / 730.0   # 50 GB gp3 at $0.08/GB-month
ipv4_h = 0.005              # public IPv4 address
print(f"{nodes * (hours + 0.25) * (price + ebs_h + ipv4_h):.2f}")
' "$NODES" "$MAX_HOURS" "$MAX_PRICE" "$BUDGET_USD") \
    || refuse "bad numeric input for nodes/max-hours/max-price/budget-usd."
  python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' \
    "$BUDGET_USD" "$ACCOUNT_BUDGET_USD" \
    || refuse "budget_usd ${BUDGET_USD} exceeds the $${ACCOUNT_BUDGET_USD} account budget (ever)."
  ok=$(python3 -c 'import sys; print(1 if float(sys.argv[1]) <= float(sys.argv[2]) else 0)' "$est" "$BUDGET_USD")
  if [ "$ok" != "1" ]; then
    refuse "worst-case estimate \$${est} exceeds budget \$${BUDGET_USD}."
  fi
  echo "aws_launch: worst-case estimate \$${est} <= budget \$${BUDGET_USD}."
}

run_preflight() {
  preflight_sweep
  preflight_caller
  preflight_quota
  preflight_estimate
}

if [ "$DRY_RUN" -eq 1 ]; then
  run_preflight
  echo "aws_launch: --dry-run passed; nothing was created or changed."
  exit 0
fi

[ -n "$RUN_ID" ] || refuse "--run-id is required (tag + budget name)."
[ -n "$BUDGET_EMAIL" ] || refuse "--budget-email is required."
[ -x "$TF_BIN" ] || refuse "$TF_BIN not found."
command -v aws >/dev/null 2>&1 || refuse "aws CLI not found."

tf() { "$TF_BIN" -chdir "$TF_DIR" "$@"; }
TF_VARS=(
  -var=run_id="$RUN_ID" -var=region="$REGION" -var=nodes="$NODES"
  -var=max_hours="$MAX_HOURS" -var=max_price="$MAX_PRICE"
  -var=budget_usd="$BUDGET_USD" -var=budget_email="$BUDGET_EMAIL"
)

STATE_FILE="$TF_DIR/terraform.tfstate"
CLEANED_UP=0
cleanup() {
  if [ "$CLEANED_UP" -eq 1 ]; then return; fi
  CLEANED_UP=1
  local rc=$?
  set +e
  if [ "$DRY_RUN" -eq 0 ] && [ -s "$STATE_FILE" ]; then
    echo "aws_launch: terraform destroy (exit/signal path)."
    tf destroy -input=false -auto-approve "${TF_VARS[@]}"
  fi
  local sweep_rc=0
  "$SCRIPT_DIR/aws_sweep.sh" --check --region "$REGION" || sweep_rc=1
  if [ "$sweep_rc" -ne 0 ]; then
    echo "aws_launch: WARNING — sweep still finds Project=loopgym leftovers after destroy." >&2
    echo "aws_launch: emergency stop: scripts/aws_sweep.sh --kill" >&2
    rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_preflight

tf init -input=false
tf apply -input=false -auto-approve "${TF_VARS[@]}"

if [ "${#TRAIN_CMD[@]}" -gt 0 ]; then
  echo "aws_launch: running: ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"
else
  echo "aws_launch: instances are up; no training command given."
  echo "aws_launch: hard stops: user-data shutdown at max_hours, scheduler at +15 min, idle alarm, one-time spot."
fi
