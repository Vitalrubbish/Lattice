#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONC=${CONC:-8}
PLEN=${PLEN:-128}
NEW=${NEW:-64}
MODEL=${MODEL:-dummy}
LOADER=${LOADER:-read}

cargo build --release --bin baseline-server --example client

RUST_LOG=${RUST_LOG:-info} ./target/release/baseline-server \
    --listen 127.0.0.1:8000 \
    --model-path "$MODEL" \
    --loader "$LOADER" \
    --max-batch "$CONC" \
    --max-seq-len $((PLEN + NEW + 16)) &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null && break
    sleep 0.5
done

./target/release/examples/client \
    --addr 127.0.0.1:8000 \
    --concurrency "$CONC" \
    --prompt-len "$PLEN" \
    --max-new-tokens "$NEW"
