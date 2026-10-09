#!/usr/bin/env bash
# Bootstrap .venv-nvcc: a pip-only CUDA toolkit for COMPILE-CHECKING csrc/*.cu here
# (rule 2: no GPU; the host runs the kernels). Idempotent; pins from tools/versions.txt.
#
# Why CUDA 13 pip wheels (not the cu12 wheels in the original ticket):
# - the nvidia-cuda-nvcc-cu12 wheel ships ptxas/nvvm/crt headers but NO nvcc driver;
# - CUDA 12.9 cannot compile on this box at all (gcc 15 / glibc 2.43 clash with its
#   headers, even with --allow-unsupported-compiler);
# - the 13.x wheels bundle the complete toolkit: nvidia-cuda-nvcc carries the nvcc
#   driver (bin/{nvcc,ptxas,crt/...}), nvidia-cuda-runtime the headers (include/) and
#   libcudart, nvidia-nvvm libnvvm — together a full toolkit root under nvidia/cu13/.
# The 13.x driver locates that root from its invocation path (argv[0], not /proc/self/exe),
# which must be <root>/bin/nvcc, so we assemble .venv-nvcc/cuda in the
# <root>/targets/x86_64-linux layout and expose .venv-nvcc/bin/nvcc as a small wrapper
# (a plain symlink from the venv bin does NOT resolve to the expected form).
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(uname -m)" = x86_64 ] || { echo "FATAL: x86_64 only (pip wheels are arch-specific)" >&2; exit 1; }
UV=.tools/bin/uv
[ -x "$UV" ] || { echo "FATAL: $UV missing (run make bootstrap)" >&2; exit 1; }
V=$(sed -n "s/^cuda_nvcc=//p" tools/versions.txt | head -1)
[ -n "$V" ] || { echo "FATAL: tools/versions.txt has no pin for cuda_nvcc" >&2; exit 1; }

VENV=.venv-nvcc
if [ ! -x "$VENV/bin/python" ]; then
  echo "== venv $VENV (py3.12)"
  "$UV" venv -p 3.12 "$VENV"
fi

echo "== cuda toolkit $V (pip wheels into $VENV)"
# no-op when the exact pins are already installed
"$UV" pip install -q --python "$VENV/bin/python" \
  "nvidia-cuda-nvcc==$V" "nvidia-cuda-runtime==$V" "nvidia-nvvm==$V"

# Assemble the toolkit root the 13.x driver expects, from the installed wheels.
SP=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
[ -x "$SP/bin/nvcc" ] || { echo "FATAL: $SP/bin/nvcc missing after install" >&2; exit 1; }
ROOT=$VENV/cuda
rm -rf "$ROOT" "$VENV/bin/nvcc"
mkdir -p "$ROOT/targets/x86_64-linux"
cp -a "$SP/bin" "$ROOT/targets/x86_64-linux/bin"
# absolute symlinks: the driver follows them; relative ones break (resolved from link dir)
ln -sfn "$SP/include" "$ROOT/targets/x86_64-linux/include"
ln -sfn "$SP/lib" "$ROOT/targets/x86_64-linux/lib"
ln -sfn "$SP/nvvm" "$ROOT/targets/x86_64-linux/nvvm"
ln -sfn targets/x86_64-linux/bin "$ROOT/bin"
ln -sfn targets/x86_64-linux/include "$ROOT/include"
ln -sfn targets/x86_64-linux/lib "$ROOT/lib64"
ln -sfn targets/x86_64-linux/nvvm "$ROOT/nvvm"

cat > "$VENV/bin/nvcc" <<'EOF'
#!/bin/sh
v=$(dirname "$(dirname "$(readlink -f "$0")")")
exec "$v/cuda/bin/nvcc" "$@"
EOF
chmod 0755 "$VENV/bin/nvcc"

# Self-check: a trivial kernel must compile through the venv nvcc.
probe=$(mktemp --suffix=.cu)
trap 'rm -f "$probe"' EXIT
printf '#include <cuda_runtime.h>\n__global__ void k() {}\n' > "$probe"
"$VENV/bin/nvcc" -c -arch=sm_86 -std=c++17 "$probe" -o /tmp/.nvcc_probe.o
rm -f /tmp/.nvcc_probe.o
rel=$("$VENV/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, release \([0-9.]*\).*/\1/p')
echo "bootstrap_cuda done (nvcc release $rel at $VENV/bin/nvcc)"
