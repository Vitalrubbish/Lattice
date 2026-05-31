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
MODEL_PATH="${MODEL_PATH:-/root/models/tinyllama}"
RESULTS_DIR="$PROJ_DIR/results/baremetal/$(date +%Y%m%d_%H%M%S)"
TMP_DIR=$(mktemp -d /tmp/bpftrace_baremetal.XXXXXX)

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ---- setup temp bpftrace scripts with bare-metal CUDA path ----
CUDA_WSL="/usr/lib/wsl/lib/libcuda.so.1.1"
CUDA_BARE="/usr/lib/x86_64-linux-gnu/libcuda.so.1"

for script in trace_all.bt trace_cuda_memcpy.bt; do
    sed "s|$CUDA_WSL|$CUDA_BARE|g" "$PROJ_DIR/scripts/$script" > "$TMP_DIR/$script"
done

echo "=============================================="
echo " Step 1 Test Suite — Bare Metal"
echo " Model:  $MODEL_PATH"
echo " Output: $RESULTS_DIR"
echo "=============================================="

# ---- privileged execution helper ----
IS_ROOT=false
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=true
    echo "Running as root — skipping sudo."
else
    if [ -z "$SUDO_PASS" ]; then
        echo ""
        echo "Not running as root. Set SUDO_PASS env var to your sudo password, e.g.:"
        echo "  SUDO_PASS='mypassword' ./scripts/step1_test_baremetal.sh"
        exit 1
    fi
fi

run_priv() {
    if $IS_ROOT; then
        "$@"
    else
        echo "$SUDO_PASS" | sudo -S "$@" 2>/dev/null
    fi
}

mkdir -p "$RESULTS_DIR"

# ---- apply sed fix permanently if not already done ----
if grep -q "$CUDA_WSL" "$PROJ_DIR/scripts/trace_all.bt" 2>/dev/null; then
    echo ""
    echo ">>> Fixing CUDA uprobe paths for bare metal..."
    sed -i "s|$CUDA_WSL|$CUDA_BARE|g" \
        "$PROJ_DIR/scripts/trace_all.bt" \
        "$PROJ_DIR/scripts/trace_cuda_memcpy.bt"
    echo "  Done. (These changes will remain in the working tree.)"
fi

# ---- build ----
echo ""
echo ">>> Building..."
cargo build --release --bin baseline-server --example bench_loaders 2>&1 | tail -3

# ---- cold trace ----
echo ""
echo ">>> Cold-cache trace test..."
sync
run_priv sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sleep 1

run_priv bpftrace "$TMP_DIR/trace_all.bt" \
    -c "timeout 20 bash scripts/load_and_exit.sh read tinyllama $MODEL_PATH" \
    > "$RESULTS_DIR/trace_cold.log" 2>&1

echo "  -> $RESULTS_DIR/trace_cold.log"

# ---- warm trace ----
echo ""
echo ">>> Warm-cache trace test..."

run_priv bpftrace "$TMP_DIR/trace_all.bt" \
    -c "timeout 20 bash scripts/load_and_exit.sh read tinyllama $MODEL_PATH" \
    > "$RESULTS_DIR/trace_warm.log" 2>&1

echo "  -> $RESULTS_DIR/trace_warm.log"

# ---- loader comparison ----
echo ""
echo ">>> Loader comparison test..."

run_priv env \
    MODEL_PATH="$MODEL_PATH" \
    SUDO_PASS="${SUDO_PASS:-}" \
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

if grep -q "vfs_read calls:" "$RESULTS_DIR/trace_cold.log"; then
    echo "  [trace cold]"
    grep -E 'vfs_read (calls|bytes)|filemap_get_pages|submit_bio calls|block_rq_issue|block_rq_complete|cuMemcpyHtoD_v2:' \
        "$RESULTS_DIR/trace_cold.log" | sed 's/^/    /'
fi

if grep -q "vfs_read calls:" "$RESULTS_DIR/trace_warm.log"; then
    echo "  [trace warm]"
    grep -E 'vfs_read (calls|bytes)|cuMemcpyHtoD_v2:' \
        "$RESULTS_DIR/trace_warm.log" | sed 's/^/    /'
fi

if grep -q "Done." "$RESULTS_DIR/loader_comparison.log"; then
    echo "  [loader comparison]"
    grep -E '^\s+\[(cold|warm)\]' "$RESULTS_DIR/loader_comparison.log" | sed 's/^/    /'
fi
