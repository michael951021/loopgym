#!/usr/bin/env bash
# Idempotent toolchain bootstrap (CLAUDE.md: Foundations).
# Installs into gitignored dirs only: .tools/{bin,rust,cargo}, .venv.
# Every version comes from tools/versions.txt; re-running is a no-op when pins match.
set -euo pipefail
cd "$(dirname "$0")/.."

T=.tools
B=$T/bin
mkdir -p "$B"

pin() { sed -n "s/^$1=//p" tools/versions.txt | head -1; }
for k in uv hadolint shellcheck terraform tflint rust; do
  [ -n "$(pin "$k")" ] || { echo "FATAL: tools/versions.txt has no pin for $k" >&2; exit 1; }
done
UV=$(pin uv); HADOLINT=$(pin hadolint); SHELLCHECK=$(pin shellcheck)
TF=$(pin terraform); TFLINT=$(pin tflint); RUST=$(pin rust)

fetch() { curl -fsSL --retry 3 "$@"; }

echo "== uv $UV"
have=$("$B/uv" --version 2>/dev/null | awk '{print $2}' || true)
if [ "$have" != "$UV" ]; then
  fetch "https://github.com/astral-sh/uv/releases/download/$UV/uv-x86_64-unknown-linux-gnu.tar.gz" -o "$T/uv.tgz"
  tar -xzf "$T/uv.tgz" -C "$T"
  install -m0755 "$T"/uv-x86_64-unknown-linux-gnu/uv "$B/uv"
  rm -rf "$T/uv.tgz" "$T"/uv-x86_64-unknown-linux-gnu
else
  echo "   already installed"
fi

echo "== python venv (.venv, 3.12)"
venv_ok() { .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; }
if ! venv_ok; then
  # .venv is a bind mount shared between agents (host layout): clear contents, never rm the dir.
  find .venv -mindepth 1 -delete
  "$B/uv" venv -p 3.12 .venv
fi
echo "   $(.venv/bin/python --version 2>&1) at .venv"

echo "== python deps (uv sync; torch from the CPU index)"
"$B/uv" sync -q
if .venv/bin/python -c "import nvidia" 2>/dev/null; then
  echo "FATAL: nvidia packages in .venv — torch did not come from the CPU index" >&2
  exit 1
fi
.venv/bin/python - <<'EOF'
import importlib
for m in ("torch", "transformers", "safetensors", "accelerate", "peft", "numpy", "pydantic", "httpx", "tomli_w"):
    importlib.import_module(m)
print("   imports ok:", __import__("torch").__version__)
EOF

echo "== rust $RUST (RUSTUP_HOME=$T/rust CARGO_HOME=$T/cargo)"
export RUSTUP_HOME="$PWD/$T/rust" CARGO_HOME="$PWD/$T/cargo"
have=$("$CARGO_HOME/bin/rustc" --version 2>/dev/null | awk '{print $2}' || true)
if [ "$have" != "$RUST" ]; then
  fetch https://sh.rustup.rs -o "$T/rustup-init.sh"
  sh "$T/rustup-init.sh" -y --profile minimal --default-toolchain "$RUST" --no-modify-path
else
  echo "   already installed"
fi
"$CARGO_HOME/bin/rustc" --version

echo "== hadolint $HADOLINT"
have=$("$B/hadolint" --version 2>/dev/null | awk '{print $NF}' || true)
if [ "$have" != "$HADOLINT" ]; then
  fetch "https://github.com/hadolint/hadolint/releases/download/v$HADOLINT/hadolint-linux-x86_64" -o "$B/hadolint"
  chmod 0755 "$B/hadolint"
else
  echo "   already installed"
fi

echo "== shellcheck $SHELLCHECK"
have=$("$B/shellcheck" --version 2>/dev/null | awk 'tolower($1) == "version:" {print $2}' || true)
if [ "$have" != "$SHELLCHECK" ]; then
  fetch "https://github.com/koalaman/shellcheck/releases/download/v$SHELLCHECK/shellcheck-v$SHELLCHECK.linux.x86_64.tar.xz" -o "$T/sc.txz"
  tar -xJf "$T/sc.txz" -C "$T"
  install -m0755 "$T"/shellcheck-v"$SHELLCHECK"/shellcheck "$B/shellcheck"
  rm -rf "$T/sc.txz" "$T"/shellcheck-v"$SHELLCHECK"
else
  echo "   already installed"
fi

echo "== terraform $TF"
have=$("$B/terraform" version 2>/dev/null | awk 'NR==1{print $2}' | tr -d v || true)
if [ "$have" != "$TF" ]; then
  fetch "https://releases.hashicorp.com/terraform/$TF/terraform_${TF}_linux_amd64.zip" -o "$T/tf.zip"
  python3 -m zipfile -e "$T/tf.zip" "$T/tf-x/"
  install -m0755 "$T/tf-x/terraform" "$B/terraform"
  rm -rf "$T/tf.zip" "$T/tf-x"
else
  echo "   already installed"
fi

echo "== tflint $TFLINT"
have=$("$B/tflint" --version 2>/dev/null | awk 'NR == 1 {print $3}' || true)
if [ "$have" != "$TFLINT" ]; then
  fetch "https://github.com/terraform-linters/tflint/releases/download/v$TFLINT/tflint_linux_amd64.zip" -o "$T/tflint.zip"
  python3 -m zipfile -e "$T/tflint.zip" "$T/tflint-x/"
  install -m0755 "$T/tflint-x/tflint" "$B/tflint"
  rm -rf "$T/tflint.zip" "$T/tflint-x"
else
  echo "   already installed"
fi

echo "bootstrap done"
