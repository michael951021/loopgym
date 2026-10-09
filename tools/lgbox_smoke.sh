#!/usr/bin/env bash
# End-to-end smoke: build lgbox (release), serve on a free port, run the known reward
# tests with LOOPGYM_LGBOX_URL set so BOTH the local and the http runner are exercised.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_HOME="$PWD/.tools/cargo"
export RUSTUP_HOME="$PWD/.tools/rust"
export PATH="$PWD/.tools/cargo/bin:$PWD/.tools/bin:$PATH"

BIN="$PWD/lgbox/target/release/lgbox"
echo "== cargo build --release (lgbox)"
cargo build --release --manifest-path lgbox/Cargo.toml

PORT=$(.venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
echo "== lgbox serve on 127.0.0.1:$PORT"
"$BIN" serve --port "$PORT" --workers 4 --queue 8 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null   # hard fail if it never came up

export LOOPGYM_LGBOX_URL="http://127.0.0.1:$PORT"
echo "== pytest tests/known/test_reward.py (runner=local + runner=http)"
LOOPGYM_KNOWN_STRICT=1 .venv/bin/python -m pytest -q tests/known/test_reward.py
echo "== lgbox smoke OK"
