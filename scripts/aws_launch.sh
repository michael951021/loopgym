#!/usr/bin/env bash
# Launcher for the AWS scaling study (ticket `aws`): 3 configs back to back
# (1 node fsdp, 2 nodes fsdp, 2 nodes deepspeed, 50 steps each) on
# var.nodes x spot g4dn.xlarge. Every spend guardrail from
# deploy/aws/README.md that lives at launch time: preflight refuses unless
# the tag sweep is clean (no double launch), the caller is the launch role
# (no long-lived keys), the account G-spot quota is <= 8 vCPU, and the
# worst-case estimate is within budget. On EXIT/INT/TERM it destroys,
# re-sweeps, and fails loudly if anything tagged Project=loopgym survived.
# Metrics + node logs are scp'd home after EACH config, so a failed config
# is debuggable after the destroy. No AWS creds in the sandbox: only
# --dry-run (stub aws on PATH) runs here.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TF_BIN="$REPO_ROOT/.tools/bin/terraform"
TF_DIR="$REPO_ROOT/deploy/aws/terraform"
PY="$REPO_ROOT/.venv/bin/python"

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
STEPS=50
ADMIN_CIDR=""
REPO_URL=""
SSH_KEY="${HOME}/.ssh/loopgym"
STUDY_RC=0

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") [--dry-run] [--nodes N] [--max-hours H] [--max-price P]
    [--budget-usd B] [--run-id ID] [--budget-email E] [--region R]
    [--launch-role NAME] [--admin-cidr CIDR] [--repo-url URL] [--ssh-key PATH]
--dry-run: print the 3-config plan + torchrun commands, run the full
  preflight, change nothing, exit 0/1. Real mode (host only, launch role):
  apply, wait for the node bootstrap, run the 3 configs back to back,
  scp metrics/logs home after each, destroy + sweep on every exit path.
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
    --admin-cidr) ADMIN_CIDR="${2:?--admin-cidr needs a value}"; shift 2 ;;
    --repo-url) REPO_URL="${2:?--repo-url needs a value}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:?--ssh-key needs a value}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

refuse() { echo "aws_launch: REFUSED — $*" >&2; exit 1; }

# The study: (name, topology, dist) — one g4dn.xlarge = one T4, so the
# 1-node config is one-aws (single local rank) and runs on node 0.
PLAN_NAMES=(single fsdp deepspeed)
PLAN_TOPOS=(one-aws two-aws two-aws)
PLAN_DISTS=(fsdp fsdp deepspeed)

print_plan() {
  echo "aws_launch: 3-config study plan (config configs/scale05.toml, ${STEPS} steps each):"
  local i
  for i in 0 1 2; do
    echo "  ${PLAN_NAMES[$i]}: topology=${PLAN_TOPOS[$i]} dist=${PLAN_DISTS[$i]} out=runs/gpu/scale_${PLAN_NAMES[$i]}"
    "$PY" -m loopgym.rl.launch cmds --topology "${PLAN_TOPOS[$i]}" --dist "${PLAN_DISTS[$i]}" \
      --steps "$STEPS" --out "runs/gpu/scale_${PLAN_NAMES[$i]}"
  done
}

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
  print_plan
  run_preflight
  echo "aws_launch: --dry-run passed; nothing was created or changed."
  exit 0
fi

[ -n "$RUN_ID" ] || refuse "--run-id is required (tag + budget name)."
[ -n "$BUDGET_EMAIL" ] || refuse "--budget-email is required."
[ "$NODES" = "2" ] || refuse "the 3-config study needs --nodes 2 (the 1-node config runs on node 0)."
[ -n "$ADMIN_CIDR" ] || refuse "--admin-cidr is required (your host IP /32; SSH comes from here only)."
[ -x "$TF_BIN" ] || refuse "$TF_BIN not found."
[ -x "$PY" ] || refuse "$PY not found."
command -v aws >/dev/null 2>&1 || refuse "aws CLI not found."
command -v ssh >/dev/null 2>&1 || refuse "ssh not found."
[ -f "$SSH_KEY" ] || refuse "ssh private key $SSH_KEY not found (pass --ssh-key PATH)."
[ -f "$SSH_KEY.pub" ] || refuse "ssh public key $SSH_KEY.pub not found."
if [ -z "$REPO_URL" ]; then
  REPO_URL=$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)
fi
[ -n "$REPO_URL" ] || refuse "--repo-url is required (no git origin remote found)."
# The nodes clone the default branch: the study code must be on it.
git -C "$REPO_ROOT" rev-parse --verify -q origin/HEAD >/dev/null \
  || refuse "cannot resolve origin/HEAD; push the branch and set the default before the run."

tf() { "$TF_BIN" -chdir "$TF_DIR" "$@"; }
TF_VARS=(
  -var=run_id="$RUN_ID" -var=region="$REGION" -var=nodes="$NODES"
  -var=max_hours="$MAX_HOURS" -var=max_price="$MAX_PRICE"
  -var=budget_usd="$BUDGET_USD" -var=budget_email="$BUDGET_EMAIL"
  -var=admin_cidr="$ADMIN_CIDR" -var=repo_url="$REPO_URL"
  -var=ssh_public_key_path="$SSH_KEY.pub"
)

STATE_FILE="$TF_DIR/terraform.tfstate"
CLEANED_UP=0
# shellcheck disable=SC2329  # invoked via trap cleanup EXIT (this shellcheck version misses that ref)
cleanup() {
  local rc=$?   # capture before anything else assigns
  if [ "$CLEANED_UP" -eq 1 ]; then return; fi
  CLEANED_UP=1
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

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30
          -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
NODE_PRIV=()
NODE_PUB=()

run_preflight
print_plan

tf init -input=false
tf apply -input=false -auto-approve "${TF_VARS[@]}"

discover_nodes() {
  local out parsed
  out=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Run,Values=$RUN_ID" --output json 2>&1) \
    || refuse "ec2 describe-instances failed: $out"
  parsed=$(printf '%s' "$out" | python3 - "$NODES" 2>&1 <<'PY'
import json, sys
n = int(sys.argv[1])
d = json.load(sys.stdin)
insts = []
for r in d.get("Reservations", []):
    for i in r.get("Instances", []):
        tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
        role = tags.get("Role", "")
        if role.startswith("worker-"):
            insts.append((int(role.split("-")[1]), i.get("PrivateIpAddress", ""),
                          i.get("PublicIpAddress", "")))
insts.sort()
assert len(insts) == n, f"expected {n} workers, found {len(insts)}"
print(" ".join(p for _, p, _ in insts))
print(" ".join(p for _, _, p in insts))
PY
) || refuse "node discovery failed: $parsed"
  mapfile -t parsed <<< "$parsed"
  [ "${#parsed[@]}" -eq 2 ] || refuse "node discovery failed: ${parsed[*]}"
  read -r -a NODE_PRIV <<< "${parsed[0]}"
  read -r -a NODE_PUB <<< "${parsed[1]}"
  [ -n "${NODE_PRIV[0]}" ] || refuse "node 0 has no private IP yet; re-run after instances boot."
  [ -n "${NODE_PUB[0]}" ] || refuse "node 0 has no public IP; cannot SSH in."
  [ "${NODE_PRIV[0]}" = "10.0.0.10" ] \
    || refuse "node 0 private IP is ${NODE_PRIV[0]}, expected 10.0.0.10 (c10d master)."
  echo "aws_launch: nodes up — priv ${NODE_PRIV[*]}, pub ${NODE_PUB[*]}."
}

wait_bootstrap() {
  local ip="${NODE_PUB[0]}" deadline=$(( (MAX_HOURS * 60 - 15) * 60 )) waited=0
  echo "aws_launch: waiting for node 0 bootstrap (driver + .venv-gpu + model)..."
  while [ "$waited" -lt "$deadline" ]; do
    if ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" "ubuntu@${ip}" "test -f /work/.aws-ready" 2>/dev/null; then
      echo "aws_launch: node 0 ready after $((waited / 60)) min."
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
  done
  return 1
}

run_config() {
  local name="$1" topo="$2" dist="$3" outdir="runs/gpu/scale_$1" rc=0
  local cmds=() pids=() rcs=() i
  mkdir -p "$outdir"
  mapfile -t cmds < <("$PY" -m loopgym.rl.launch cmds --topology "$topo" --dist "$dist" \
    --steps "$STEPS" --out "$outdir")
  for i in "${!cmds[@]}"; do
    echo "aws_launch: [$name] node $i (pub ${NODE_PUB[$i]}): ${cmds[$i]}"
    ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" "ubuntu@${NODE_PUB[$i]}" \
      "cd /work && ${cmds[$i]}" > "$outdir/node$i.log" 2>&1 &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then rcs+=(0); else rcs+=(1); fi
  done
  rc=${rcs[0]}   # rank 0 decides the config's fate
  # Metrics home before the destroy: a failed config must stay debuggable.
  if scp "${SSH_OPTS[@]}" -i "$SSH_KEY" -q "ubuntu@${NODE_PUB[0]}:${outdir}/metrics.jsonl" \
      "$outdir/metrics.jsonl" 2>/dev/null; then
    "$PY" -m loopgym.rl.launch report --metrics "$outdir/metrics.jsonl" \
      --logs "$outdir"/node*.log --topology "$topo" --dist "$dist" \
      --out "results/gpu/scale_aws_${name}.json" \
      || echo "aws_launch: [$name] report failed (metrics present; see log)."
  else
    echo "aws_launch: [$name] no metrics.jsonl on node 0 (config failed at/before step 1?)."
  fi
  return "$rc"
}

discover_nodes

if ! wait_bootstrap; then
  echo "aws_launch: ERROR — node 0 did not finish bootstrapping before the max_hours hard stop." >&2
  echo "aws_launch: the instances are being destroyed; bootstrap log was /var/log/loopgym-bootstrap.log." >&2
  STUDY_RC=1
else
  for i in 0 1 2; do
    name="${PLAN_NAMES[$i]}"
    if run_config "$name" "${PLAN_TOPOS[$i]}" "${PLAN_DISTS[$i]}"; then
      echo "aws_launch: config $name finished (rc=0)."
    else
      echo "aws_launch: config $name FAILED (rc!=0); continuing with the next config." >&2
      STUDY_RC=1
    fi
  done
fi

if [ "$STUDY_RC" -eq 0 ]; then
  echo "aws_launch: study complete; results in results/gpu/scale_aws_{single,fsdp,deepspeed}.json."
fi
exit "$STUDY_RC"
