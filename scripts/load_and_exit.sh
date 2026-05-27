#!/bin/bash
# Wrapper: run the server, kill it once model is loaded ("listening" on stdout).
# Used by bpftrace -c so the END block fires after loading completes.
# Usage: ./scripts/load_and_exit.sh [--loader read|mmap|direct] [--model-type tinyllama|llama7b]

LOADER="${1:-read}"
MODEL_TYPE="${2:-tinyllama}"
MODEL_PATH="${3:-${MODEL_PATH:-./models/tinyllama}}"

./target/release/baseline-server \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --loader "$LOADER" \
    2>&1 | while IFS= read -r line; do
    echo "$line"
    if echo "$line" | grep -q "listening"; then
        kill $$ 2>/dev/null
        exit 0
    fi
done &
PID=$!
sleep 20
kill $PID 2>/dev/null
wait $PID 2>/dev/null
