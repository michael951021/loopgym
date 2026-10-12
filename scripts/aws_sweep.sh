#!/usr/bin/env bash
# Tag-based AWS sweep for this project: list (--check) or delete (--kill)
# everything tagged Project=loopgym. State-independent (resourcegroupstaggingapi
# + ec2 describe-*): a lost terraform state file cannot hide resources.
# No creds in the sandbox; on the host this is the emergency stop.
set -euo pipefail

TAG_KEY="Project"
TAG_VAL="loopgym"
REGION="${AWS_REGION:-us-east-1}"
MODE=""

usage() { echo "usage: $(basename "$0") --check|--kill [--region R]" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check"; shift ;;
    --kill) MODE="kill"; shift ;;
    --region) REGION="${2:?--region needs a value}"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$MODE" ] || usage
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 2; }

# Print ids (one per line) of every tagged resource of one ec2 type.
ec2_ids() { # $1: describe command, $2: python list expression over var d
  local out
  out=$(aws ec2 "$1" --region "$REGION" \
        --filters "Name=tag:${TAG_KEY},Values=${TAG_VAL}" --output json 2>&1) \
    || { echo "aws ec2 $1 failed:" >&2; echo "$out" >&2; exit 1; }
  printf '%s' "$out" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for line in eval(sys.argv[1], {"__builtins__": {}}, {"d": d}):
    print(line)
' "$2"
}

instances() { ec2_ids describe-instances '[i["InstanceId"] for r in d.get("Reservations", []) for i in r.get("Instances", [])]'; }
volumes()   { ec2_ids describe-volumes '[v["VolumeId"] for v in d.get("Volumes", [])]'; }
spots()     { ec2_ids describe-spot-instance-requests '[s["SpotInstanceRequestId"] for s in d.get("SpotInstanceRequests", [])]'; }
sgs()       { ec2_ids describe-security-groups '[g["GroupId"] for g in d.get("SecurityGroups", [])]'; }

# CloudWatch alarms: describe-alarms tag-filters, plus a client-side re-filter.
alarms() {
  local out
  out=$(aws cloudwatch describe-alarms --region "$REGION" \
        --tag-filters "Key=${TAG_KEY} Values=${TAG_VAL}" --output json 2>&1) \
    || { echo "aws cloudwatch describe-alarms failed:" >&2; echo "$out" >&2; exit 1; }
  printf '%s' "$out" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for a in d.get("MetricAlarms", []):
    if any(t.get("Key") == "Project" and t.get("Value") == "loopgym" for t in a.get("Tags", [])):
        print(a["AlarmName"])
'
}

# EventBridge Scheduler rules: list, re-filter by tag client-side.
schedules() {
  local out
  out=$(aws scheduler list-schedules --region "$REGION" \
        --tag-filters "Key=${TAG_KEY} Values=${TAG_VAL}" --output json 2>&1) \
    || { echo "aws scheduler list-schedules failed:" >&2; echo "$out" >&2; exit 1; }
  printf '%s' "$out" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for s in d.get("Schedules", []):
    if any(t.get("Key") == "Project" and t.get("Value") == "loopgym" for t in s.get("Tags", [])):
        print(s["Name"])
'
}

# Catch-all: anything the per-type describes above could not see.
catchall() {
  local out
  out=$(aws resourcegroupstaggingapi get-resources --region "$REGION" \
        --tag-filters "Key=${TAG_KEY} Values=${TAG_VAL}" --output json 2>&1) \
    || { echo "aws resourcegroupstaggingapi get-resources failed:" >&2; echo "$out" >&2; exit 1; }
  printf '%s' "$out" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for m in d.get("ResourceTagMappingList", []):
    print(m["ResourceARN"])
'
}

# Bounded wait for instances to reach terminated (volumes must outlive them).
wait_terminated() { # $1: newline-separated instance ids
  local ids="$1" tries=0
  if [ -z "$ids" ]; then return 0; fi
  local id_arr
  read -r -a id_arr <<< "$ids"
  while [ "$tries" -lt 20 ]; do
    tries=$((tries + 1))
    if ! aws ec2 describe-instances --region "$REGION" \
          --instance-ids "${id_arr[@]}" --output json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
states = [i["State"]["Name"] for r in d.get("Reservations", []) for i in r.get("Instances", [])]
sys.exit(0 if any(s != "terminated" for s in states) else 1)
'; then
      break
    fi
    sleep 10
  done
}

kill_all() {
  local insts vols spots3 sgs3 alarms3 scheds3 id name
  insts=$(instances) || exit 1
  vols=$(volumes) || exit 1
  spots3=$(spots) || exit 1
  sgs3=$(sgs) || exit 1
  alarms3=$(alarms) || exit 1
  scheds3=$(schedules) || exit 1

  if [ -n "$spots3" ]; then
    while IFS= read -r id; do
      echo "cancelling spot request $id"
      aws ec2 cancel-spot-instance-requests --region "$REGION" \
        --spot-instance-request-ids "$id" >/dev/null
    done <<< "$spots3"
  fi

  if [ -n "$insts" ]; then
    echo "terminating instances: $(printf '%s' "$insts" | tr '\n' ' ')"
    local id_arr
    read -r -a id_arr <<< "$insts"
    aws ec2 terminate-instances --region "$REGION" \
      --instance-ids "${id_arr[@]}" >/dev/null
  fi

  if [ -n "$alarms3" ]; then
    while IFS= read -r name; do
      echo "deleting alarm $name"
      aws cloudwatch delete-alarms --region "$REGION" --alarm-names "$name" >/dev/null
    done <<< "$alarms3"
  fi

  if [ -n "$scheds3" ]; then
    while IFS= read -r name; do
      echo "deleting schedule $name"
      aws scheduler delete-schedule --region "$REGION" --name "$name" >/dev/null
    done <<< "$scheds3"
  fi

  wait_terminated "$insts"

  if [ -n "$vols" ]; then
    while IFS= read -r id; do
      echo "deleting volume $id"
      aws ec2 delete-volume --region "$REGION" --volume-id "$id" >/dev/null
    done <<< "$vols"
  fi

  if [ -n "$sgs3" ]; then
    while IFS= read -r id; do
      echo "deleting security group $id"
      aws ec2 delete-security-group --region "$REGION" --group-id "$id" >/dev/null
    done <<< "$sgs3"
  fi
  return 0
}

check() {
  local found=0
  local insts vols spots3 sgs3 alarms3 scheds3 extra
  insts=$(instances) || exit 1
  vols=$(volumes) || exit 1
  spots3=$(spots) || exit 1
  sgs3=$(sgs) || exit 1
  alarms3=$(alarms) || exit 1
  scheds3=$(schedules) || exit 1
  local -A sections=(
    [instances]="$insts" [spot-requests]="$spots3" [volumes]="$vols"
    [security-groups]="$sgs3" [cloudwatch-alarms]="$alarms3"
    [schedules]="$scheds3"
  )
  local name
  for name in instances spot-requests volumes security-groups cloudwatch-alarms schedules; do
    if [ -n "${sections[$name]}" ]; then
      found=1
      echo "leftover ${name}:"
      printf '%s\n' "${sections[$name]}" | sed 's/^/  /'
    fi
  done
  extra=$(catchall) || exit 1
  if [ -n "$extra" ]; then
    found=1
    echo "tagged resources (catch-all):"
    printf '%s\n' "$extra" | sed 's/^/  /'
  fi
  if [ "$found" -eq 1 ]; then
    echo "sweep: leftover resources tagged Project=loopgym found (region ${REGION})." >&2
    return 1
  fi
  echo "sweep: clean, nothing tagged Project=loopgym in ${REGION}."
  return 0
}

case "$MODE" in
  check)
    check
    ;;
  kill)
    kill_all
    # Re-check: the kill must have removed everything, else fail loudly.
    check
    ;;
esac
