#!/usr/bin/env bash
# ==============================================================================
# step1_test_wsl2.sh — Run step1 trace + loader-comparison tests on WSL2
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

SUDO_PASS="your_sudo_password_here"  # <-- REPLACE THIS with your actual sudo password (or set via env var)
MODEL_PATH="${MODEL_PATH:-/home/vitalrubbish/models/tinyllama}"
BPFTRACE_FLAGS="--unsafe"
RESULTS_DIR="$PROJ_DIR/results/wsl2/$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo " Step 1 Test Suite — WSL2"
echo " Model:  $MODEL_PATH"
echo " Output: $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# ---- build ----
echo ""
echo ">>> Building..."
cargo build --release --bin baseline-server --example bench_loaders 2>&1 | tail -3

# ---- cold trace ----
echo ""
echo ">>> Cold-cache trace test..."
sync
echo "$SUDO_PASS" | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
sleep 1

echo "$SUDO_PASS" | sudo -S bpftrace $BPFTRACE_FLAGS scripts/trace_all.bt \
    -c "timeout 20 bash scripts/load_and_exit.sh read tinyllama $MODEL_PATH" \
    > "$RESULTS_DIR/trace_cold.log" 2>&1

echo "  -> $RESULTS_DIR/trace_cold.log"

# ---- warm trace ----
echo ""
echo ">>> Warm-cache trace test..."

echo "$SUDO_PASS" | sudo -S bpftrace $BPFTRACE_FLAGS scripts/trace_all.bt \
    -c "timeout 20 bash scripts/load_and_exit.sh read tinyllama $MODEL_PATH" \
    > "$RESULTS_DIR/trace_warm.log" 2>&1

echo "  -> $RESULTS_DIR/trace_warm.log"

# ---- loader comparison ----
echo ""
echo ">>> Loader comparison test..."

echo "$SUDO_PASS" | sudo -S env \
    MODEL_PATH="$MODEL_PATH" \
    SUDO_PASS="$SUDO_PASS" \
    ./target/release/examples/bench_loaders \
    > "$RESULTS_DIR/loader_comparison.log" 2>&1

echo "  -> $RESULTS_DIR/loader_comparison.log"

# ---- summary ----
echo ""
echo "=============================================="
echo " Tests complete. Results in $RESULTS_DIR/"
echo "=============================================="
echo ""
echo "Quick stats:"

# extract key numbers from cold trace
if grep -q "vfs_read calls:" "$RESULTS_DIR/trace_cold.log"; then
    echo "  [trace cold]"
    grep -E 'vfs_read (calls|bytes)|filemap_get_pages|submit_bio calls|block_rq_issue|block_rq_complete|cuMemcpyHtoD_v2:' \
        "$RESULTS_DIR/trace_cold.log" | sed 's/^/    /'
fi

if grep -q "Done." "$RESULTS_DIR/loader_comparison.log"; then
    echo "  [loader comparison]"
    grep -E '^\s+\[(cold|warm)\]' "$RESULTS_DIR/loader_comparison.log" | sed 's/^/    /'
fi
