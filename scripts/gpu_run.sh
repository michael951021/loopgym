#!/usr/bin/env bash
# Kevin's GPU window in one command (CLAUDE.md "GPU and AWS", scale-out).
# Host only: rule 2 (no GPU in the sandbox) - written and dry-run here.
#
#   scripts/gpu_run.sh --stop-agents   full run (stop model-server agents first)
#   scripts/gpu_run.sh --dry-run       plan + required-file checks, nothing runs
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
Usage: scripts/gpu_run.sh [--dry-run] [--stop-agents]
  --dry-run       print the plan, check required files and pins, exit
                  (nothing is executed; safe in the sandbox)
  --stop-agents   full run only: ./agent stop all --now before the run
                  (the model server holds ~21 GB of each 3090)
USAGE
}

DRY=0
STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --stop-agents) STOP=1 ;;
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

if [ "$DRY" -eq 1 ]; then
  plan
  echo "--- required files ---"
  bad=0
  for f in "${REQUIRED_FILES[@]}"; do
    if [ -e "$f" ]; then
      echo "OK      $f"
    else
      echo "MISSING $f"
      bad=1
    fi
  done
  for k in vllm deepspeed; do
    if [ -n "$(pin "$k")" ]; then
      echo "OK      tools/versions.txt: $k=$(pin "$k")"
    else
      echo "MISSING tools/versions.txt: $k="
      bad=1
    fi
  done
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

step "1/8 preflight: GPUs"
nvidia-smi -L

if [ "$STOP" -eq 1 ]; then
  step "2/8 stop agents"
  ./agent stop all --now
else
  step "2/8 agents left running (no --stop-agents)"
fi

step "3/8 .venv-gpu"
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
