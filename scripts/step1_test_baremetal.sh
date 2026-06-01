#!/usr/bin/env bash
# ==============================================================================
# step1_test_baremetal.sh — Run step1 trace + loader-comparison tests on bare metal
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# ---- config ----
SUDO_PASS="${SUDO_PASS:-}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/tinyllama}"
RESULTS_DIR="$PROJ_DIR/results/baremetal/$(date +%Y%m%d_%H%M%S)"
TMP_DIR=$(mktemp -d /tmp/bpftrace_baremetal.XXXXXX)

cleanup() {
    rm -rf "$TMP_DIR"
    # Kill any leftover bpftrace or baseline-server processes
    sudo pkill -f "bpftrace.*trace_all" 2>/dev/null || true
    sudo pkill -f "baseline-server" 2>/dev/null || true
}
trap cleanup EXIT

# ---- privileged execution helper ----
IS_ROOT=false
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=true
fi

run_priv() {
    if $IS_ROOT; then
        "$@"
    else
        sudo "$@"
    fi
}

run_priv_bg() {
    if $IS_ROOT; then
        "$@" &
    else
        sudo "$@" &
    fi
}

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo " Step 1 Test Suite — Bare Metal"
echo " Model:  $MODEL_PATH"
echo " Output: $RESULTS_DIR"
echo "=============================================="

# ---- build ----
echo ""
echo ">>> Building..."
source "$HOME/.cargo/env" 2>/dev/null || true
source /etc/profile 2>/dev/null || true
cargo build --release --features gds --bin baseline-server --example bench_loaders 2>&1 | tail -3

# ---- cold-cache trace ----
echo ""
echo ">>> Cold-cache trace test..."
sync
run_priv sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sleep 1

# Run bpftrace with -c; use absolute paths to avoid bpftrace PATH resolution bugs
run_priv bpftrace "$PROJ_DIR/scripts/trace_all.bt" \
    -c "/usr/bin/timeout 30 /usr/bin/bash $PROJ_DIR/scripts/load_and_exit.sh read tinyllama $MODEL_PATH" \
    > "$RESULTS_DIR/trace_cold.log" 2>&1 || true

echo "  -> $RESULTS_DIR/trace_cold.log"

# ---- warm-cache trace ----
echo ""
echo ">>> Warm-cache trace test..."

run_priv bpftrace "$PROJ_DIR/scripts/trace_all.bt" \
    -c "/usr/bin/timeout 30 /usr/bin/bash $PROJ_DIR/scripts/load_and_exit.sh read tinyllama $MODEL_PATH" \
    > "$RESULTS_DIR/trace_warm.log" 2>&1 || true

echo "  -> $RESULTS_DIR/trace_warm.log"

# ---- loader comparison ----
echo ""
echo ">>> Loader comparison test..."

run_priv env \
    MODEL_PATH="$MODEL_PATH" \
    "$PROJ_DIR/target/release/examples/bench_loaders" \
    > "$RESULTS_DIR/loader_comparison.log" 2>&1

echo "  -> $RESULTS_DIR/loader_comparison.log"

# ---- summary ----
echo ""
echo "=============================================="
echo " Tests complete. Results in $RESULTS_DIR/"
echo "=============================================="
echo ""
echo "Quick stats:"

if grep -q "vfs_read calls:" "$RESULTS_DIR/trace_cold.log" 2>/dev/null; then
    echo "  [trace cold]"
    grep -E 'vfs_read (calls|bytes)|filemap_get_pages|submit_bio calls|block_rq_issue|block_rq_complete|cuMemcpyHtoD_v2:|cuMemAlloc_v2:' \
        "$RESULTS_DIR/trace_cold.log" | sed 's/^/    /'
fi

if grep -q "vfs_read calls:" "$RESULTS_DIR/trace_warm.log" 2>/dev/null; then
    echo "  [trace warm]"
    grep -E 'vfs_read (calls|bytes)|cuMemcpyHtoD_v2:|cuMemAlloc_v2:' \
        "$RESULTS_DIR/trace_warm.log" | sed 's/^/    /'
fi

if grep -q "Done." "$RESULTS_DIR/loader_comparison.log" 2>/dev/null; then
    echo "  [loader comparison]"
    grep -E '^\s+\[(cold|warm)\]' "$RESULTS_DIR/loader_comparison.log" | sed 's/^/    /'
fi
