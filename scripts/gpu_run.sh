#!/usr/bin/env bash
# Kevin's GPU window in one command (CLAUDE.md "GPU and AWS", scale-out).
# Host only: rule 2 (no GPU in the sandbox) - written and dry-run here.
#
#   scripts/gpu_run.sh --stop-agents          full run (stop model-server agents first)
#   scripts/gpu_run.sh --two-node-local       0.5B toy multi-node rehearsal on this host
#   scripts/gpu_run.sh --dry-run              plan + required-file checks, nothing runs
#
# Full run: 2-GPU preflight -> .venv-gpu (CUDA torch, pinned vllm/deepspeed) ->
# kernel preflight (cuda+triton in IMPLS) -> tests/known -> vLLM-vs-HF logprob
# check (gate fp32 mean |Δ| < 0.02 nats) -> GRPO 1.5B (vLLM rollouts on GPU0,
# LoRA synced every step, trainer on GPU1) -> eval exercism val/test + loopgym
# test, all into results/gpu/. Logs in runs/gpu_run/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_15=${LOOPGYM_MODEL_15:-models/qwen2.5-coder-1.5b-instruct}
MODEL_05=${LOOPGYM_MODEL_05:-models/qwen2.5-coder-0.5b-instruct}
EXERCISES=${LOOPGYM_EXERCISES:-data/raw/exercism-python/exercises/practice}
STAMP=.venv-gpu/.gpu-stamp
LOGDIR=runs/gpu_run/logs
CKPT=ckpt/grpo15_gpu

usage() {
  cat <<'USAGE'
Usage: scripts/gpu_run.sh [--dry-run] [--stop-agents] [--two-node-local]
  --dry-run         print the plan, check required files and pins, exit
                    (nothing is executed; safe in the sandbox)
  --stop-agents     full run only: ./agent stop all --now before the run
                    (the model server holds ~21 GB of each 3090)
  --two-node-local  0.5B toy multi-node rehearsal on THIS host: two torchrun
                    "nodes" (GPU 0 + GPU 1), 20 steps each for fsdp and deepspeed,
                    reports -> results/gpu/scale_{fsdp,deepspeed}.json
USAGE
}

DRY=0
STOP=0
TWO_LOCAL=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --stop-agents) STOP=1 ;;
    --two-node-local) TWO_LOCAL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

# pin <key> -> value from tools/versions.txt ("" if absent)
pin() { sed -n "s/^$1=//p" tools/versions.txt | head -n 1; }

REQUIRED_FILES=(
  "tools/versions.txt"
  ".tools/bin/uv"
  "tests/known"
  "configs/grpo15_gpu.toml"
  "loopgym/rl/train.py"
  "loopgym/rl/rollout_vllm.py"
  "loopgym/rl/check_vllm_hf.py"
  "loopgym/eval/run.py"
  "csrc/logprob_cuda.cpp"
  "csrc/logprob_cuda_kernels.cu"
  "$MODEL_15"
  "$MODEL_05"
  "$EXERCISES"
  "results/splits.json"
  "data/loopgym/v0/tasks.jsonl"
  "data/loopgym/v0/checks"
  "data/loopgym/candidates.jsonl"
  "data/raw/repos/draftlab.git"
  "data/raw/repos/harness.git"
)

# The two-node rehearsal is 0.5B toy (in-process grading, torch kernel, no eval),
# so it needs far less than the 1.5B exercism/loopgym window.
REQUIRED_FILES_TWOLOCAL=(
  "tools/versions.txt"
  ".tools/bin/uv"
  "configs/scale05.toml"
  "loopgym/rl/train.py"
  "loopgym/rl/launch.py"
  "loopgym/rl/batch.py"
  "loopgym/rl/dist.py"
  "loopgym/rl/ds.py"
  "loopgym/models.py"
  "$MODEL_05"
)

plan() {
  local stopline
  if [ "$STOP" -eq 1 ]; then
    stopline=" 2. ./agent stop all --now"
  else
    stopline=" 2. (agents left running; use --stop-agents for the real window)"
  fi
  cat <<PLAN
=== gpu_run.sh plan ===
 model (1.5B): $MODEL_15
 model (0.5B): $MODEL_05
 exercises   : $EXERCISES
 1. preflight  : nvidia-smi -L (need 2 GPUs)
$stopline
 3. .venv-gpu  : uv venv -p 3.12 + uv pip install torch (CUDA wheels from PyPI)
                vllm==$(pin vllm) deepspeed==$(pin deepspeed) triton ninja ... (stamp: $STAMP)
 4. kernels    : assert kernels.IMPLS contains cuda + triton
 5. known      : CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -m pytest -q tests/known
 6. vllm-vs-hf : CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -m loopgym.rl.check_vllm_hf
                  --model $MODEL_15 --out results/gpu/vllm_hf_check.json
 7. GRPO 1.5B  : CUDA_VISIBLE_DEVICES=0,1 .venv-gpu/bin/python -m loopgym.rl.train
                  --config configs/grpo15_gpu.toml --resume-if-any
 8. eval (n=1, T=0.8) -> results/gpu/:
    - exercism val  (eval_exercism_val.json)
    - exercism test (eval_exercism_test.json)
    - loopgym  test (eval_loopgym_test.json)
PLAN
}

plan_twolocal() {
  cat <<PLAN
=== gpu_run.sh --two-node-local plan (0.5B toy multi-node rehearsal) ===
 config  : configs/scale05.toml (0.5B, toy=banana, B2 G8, 256 tok, 20 steps, no eval)
 topology: 2 torchrun "nodes" on THIS host, 1 rank each; c10d master 127.0.0.1:29500
           node0 -> GPU 0, node1 -> GPU 1 (each node exports exactly one GPU)
 dist    : fsdp then deepspeed (one 20-step rehearsal each)
  1. preflight  : nvidia-smi -L (need 2 GPUs)
  2. .venv-gpu  : uv venv -p 3.12 + pinned install (torch, vllm, deepspeed, triton, ...)
  3. fsdp       : 2 concurrent torchrun -> runs/gpu/scale_fsdp
                  report      -> results/gpu/scale_fsdp.json
  4. deepspeed  : 2 concurrent torchrun -> runs/gpu/scale_deepspeed
                  report      -> results/gpu/scale_deepspeed.json
 (toy rewards grade in-process; this measures rendezvous/sharding/step time, not
  learning)
PLAN
}

ensure_venv_gpu() {
  local vllm_pin ds_pin want
  vllm_pin=$(pin vllm)
  ds_pin=$(pin deepspeed)
  if [ -z "$vllm_pin" ] || [ -z "$ds_pin" ]; then
    echo "tools/versions.txt must pin vllm= and deepspeed=" >&2
    exit 1
  fi
  want="vllm=$vllm_pin deepspeed=$ds_pin"
  if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ] && [ -x .venv-gpu/bin/python ]; then
    echo ".venv-gpu up to date ($want)"
  else
    if [ ! -x .venv-gpu/bin/python ]; then
      .tools/bin/uv venv -p 3.12 .venv-gpu
    fi
    .tools/bin/uv pip install -q --python .venv-gpu/bin/python \
      "torch" "vllm==$vllm_pin" "deepspeed==$ds_pin" "triton" "ninja" \
      "transformers" "peft" "accelerate" "safetensors" "numpy" \
      "pydantic" "httpx" "tomli-w" "pytest"
    printf '%s\n' "$want" > "$STAMP"
  fi
  CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -c \
    "import torch; assert torch.cuda.is_available(), 'torch has no CUDA'; print('torch', torch.__version__, 'GPUs', torch.cuda.device_count())"
}

# Run both two-local rehearsals. For each dist backend, launch the node-0 and
# node-1 torchrun commands (from loopgym.rl.launch) concurrently on this host,
# each pinned to one GPU, then fold the metrics + PEAKMEM logs into one report.
twolocal_run() {
  local dist cmds c0 c1 log0 log1 p0 p1 rc0 rc1
  for dist in fsdp deepspeed; do
    step "two-node-local: $dist (2 torchrun nodes, GPU0 + GPU1)"
    mapfile -t cmds < <(.venv-gpu/bin/python -m loopgym.rl.launch cmds --topology two-local --dist "$dist")
    if [ "${#cmds[@]}" -ne 2 ]; then
      echo "expected 2 node commands for $dist, got ${#cmds[@]}" >&2
      exit 1
    fi
    c0="${cmds[0]}"
    c1="${cmds[1]}"
    log0="$LOGDIR/twolocal_${dist}_node0.log"
    log1="$LOGDIR/twolocal_${dist}_node1.log"
    echo "  node0: $c0"
    echo "  node1: $c1"
    eval "$c0" > "$log0" 2>&1 &
    p0=$!
    eval "$c1" > "$log1" 2>&1 &
    p1=$!
    rc0=0; wait "$p0" || rc0=$?
    rc1=0; wait "$p1" || rc1=$?
    if [ "$rc0" -ne 0 ] || [ "$rc1" -ne 0 ]; then
      echo "two-node-local $dist FAILED (node0 rc=$rc0, node1 rc=$rc1); tail of logs:" >&2
      tail -n 40 "$log0" >&2 || true
      tail -n 40 "$log1" >&2 || true
      exit 1
    fi
    step "two-node-local: $dist report -> results/gpu/scale_${dist}.json"
    .venv-gpu/bin/python -m loopgym.rl.launch report \
      --metrics "runs/gpu/scale_${dist}/metrics.jsonl" \
      --logs "$log0" "$log1" \
      --topology two-local --dist "$dist" \
      --out "results/gpu/scale_${dist}.json"
  done
}

check_required() {
  local f rc=0
  for f in "$@"; do
    if [ -e "$f" ]; then
      echo "OK      $f"
    else
      echo "MISSING $f"
      rc=1
    fi
  done
  return $rc
}

check_pins() {
  local k rc=0
  for k in "$@"; do
    if [ -n "$(pin "$k")" ]; then
      echo "OK      tools/versions.txt: $k=$(pin "$k")"
    else
      echo "MISSING tools/versions.txt: $k="
      rc=1
    fi
  done
  return $rc
}

if [ "$DRY" -eq 1 ]; then
  bad=0
  if [ "$TWO_LOCAL" -eq 1 ]; then
    plan_twolocal
  else
    plan
  fi
  echo "--- required files ---"
  if [ "$TWO_LOCAL" -eq 1 ]; then
    check_required "${REQUIRED_FILES_TWOLOCAL[@]}" || bad=1
  else
    check_required "${REQUIRED_FILES[@]}" || bad=1
  fi
  check_pins vllm deepspeed || bad=1
  if [ "$bad" -ne 0 ]; then
    echo "dry-run FAILED: missing files or pins (above)" >&2
    exit 1
  fi
  echo "dry-run OK: plan executable, all required files present. Nothing ran."
  exit 0
fi

# --- full run (host only) ------------------------------------------------------
mkdir -p "$LOGDIR" results/gpu

step() { echo; echo "### $*"; }

if [ "$TWO_LOCAL" -eq 1 ]; then
  step "two-node-local 1/4 preflight: GPUs"
  nvidia-smi -L
  if [ "$(nvidia-smi -L | wc -l)" -lt 2 ]; then
    echo "two-node-local needs 2 GPUs; see nvidia-smi -L above" >&2
    exit 1
  fi
  if [ "$STOP" -eq 1 ]; then
    step "two-node-local 2/4 stop agents"
    ./agent stop all --now
  else
    step "two-node-local 2/4 agents left running (no --stop-agents)"
  fi
  step "two-node-local 3/4 .venv-gpu"
  ensure_venv_gpu
  step "two-node-local 4/4 rehearsals: fsdp, deepspeed"
  twolocal_run
  step "done"
  echo "results/gpu/:"
  ls -l results/gpu/
  exit 0
fi

step "1/8 preflight: GPUs"
nvidia-smi -L

if [ "$STOP" -eq 1 ]; then
  step "2/8 stop agents"
  ./agent stop all --now
else
  step "2/8 agents left running (no --stop-agents)"
fi

step "3/8 .venv-gpu"
ensure_venv_gpu

step "4/8 kernel preflight (cuda + triton in IMPLS)"
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python - <<'PY'
import torch
from loopgym import kernels
missing = {"cuda", "triton"} - set(kernels.IMPLS)
assert torch.cuda.device_count() >= 2, f"need 2 GPUs, see {torch.cuda.device_count()}"
assert not missing, f"kernels missing {missing}; have {kernels.IMPLS}"
print("IMPLS:", kernels.IMPLS)
PY

step "5/8 known tests (cuda + triton impls)"
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -m pytest -q tests/known \
  2>&1 | tee "$LOGDIR/known_tests.log"

step "6/8 vLLM-vs-HF logprob check (gate: fp32 mean |Δ| < 0.02)"
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -m loopgym.rl.check_vllm_hf \
  --model "$MODEL_15" --out results/gpu/vllm_hf_check.json \
  2>&1 | tee "$LOGDIR/vllm_hf_check.log"

step "7/8 GRPO 1.5B (vLLM rollouts GPU0, trainer GPU1)"
CUDA_VISIBLE_DEVICES=0,1 .venv-gpu/bin/python -m loopgym.rl.train \
  --config configs/grpo15_gpu.toml --resume-if-any \
  2>&1 | tee "$LOGDIR/train.log"

step "8/8 eval -> results/gpu/"
for split in val test; do
  CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -m loopgym.eval.run \
    --model "$MODEL_15" --adapter "$CKPT" --source exercism --split "$split" \
    --n 1 --temperature 0.8 --max-new-tokens 512 \
    --out "results/gpu/eval_exercism_${split}.json" \
    2>&1 | tee "$LOGDIR/eval_exercism_${split}.log"
done
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -m loopgym.eval.run \
  --model "$MODEL_15" --adapter "$CKPT" --source loopgym --split test \
  --n 1 --temperature 0.8 --max-new-tokens 512 \
  --out results/gpu/eval_loopgym_test.json \
  2>&1 | tee "$LOGDIR/eval_loopgym_test.log"

step "done"
echo "results/gpu/:"
ls -l results/gpu/
