#!/bin/bash
# Wrapper: run the server for a fixed duration, used by bpftrace -c.
# Usage: ./scripts/load_and_exit.sh [loader] [model-type] [model-path]
# The server is run in the background; the script keeps running for FIXED_DURATION
# so bpftrace can collect data, then kills the server and exits cleanly.

LOADER="${1:-read}"
MODEL_TYPE="${2:-tinyllama}"
MODEL_PATH="${3:-${MODEL_PATH:-./models/tinyllama}}"
FIXED_DURATION="${4:-25}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# Start server in background
"$PROJ_DIR/target/release/baseline-server" \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --loader "$LOADER" \
    > /tmp/baseline_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (or timeout)
DEADLINE=$((SECONDS + FIXED_DURATION))
while [ $SECONDS -lt $DEADLINE ]; do
    if grep -q "listening" /tmp/baseline_server.log 2>/dev/null; then
        echo "Server ready — keeping alive for bpftrace data collection..."
        break
    fi
    sleep 0.5
done

# Keep running until fixed duration ends
while [ $SECONDS -lt $DEADLINE ]; do
    sleep 1
done

# Cleanup
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
cat /tmp/baseline_server.log
rm -f /tmp/baseline_server.log
exit 0
