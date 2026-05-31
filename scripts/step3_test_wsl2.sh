#!/usr/bin/env bash
# ==============================================================================
# scripts/step3_test_wsl2.sh — Step 3 benchmark for WSL2
#
# NOTE: vLLM DOES NOT WORK on Blackwell GPUs (RTX 5070, sm_120) due to
# FlashInfer not supporting compute capability 12.x yet.
# This script supports baseline-only mode on WSL2.
# For full vLLM comparison, use step3_test_baremetal.sh on an A30 server.
#
# Modes:
#   ./scripts/step3_test_wsl2.sh baseline     # Baseline server + GPU tests
#
# Config (env vars):
#   NUM_REQUESTS            Requests per run (default: 50)
#   CONCURRENCY             Concurrent connections (default: 4)
#   MAX_NEW_TOKENS          Max tokens to generate (default: 64)
#   MAX_BATCH               Max concurrent seqs (default: 32)
#   MAX_SEQ_LEN             Max sequence length (default: 512)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──
MODE="${1:-baseline}"
NUM_REQUESTS="${NUM_REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_BATCH="${MAX_BATCH:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BASELINE_PORT=8000
VLLM_PORT=8001
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJ_DIR/results/wsl2/step3_${MODE}_${TIMESTAMP}"

# ── Sanity ──
if [ "$MODE" = "vllm" ] || [ "$MODE" = "compare" ]; then
    echo "=============================================================="
    echo " WARNING: vLLM does not work on Blackwell GPUs (sm_120)."
    echo " FlashInfer lacks pre-compiled kernels for compute capability 12.x."
    echo ""
    echo " To run vLLM benchmarks, use an A30 (sm_80) bare-metal server with"
    echo " step3_test_baremetal.sh instead."
    echo "=============================================================="
    echo ""
    if [ "$MODE" = "vllm" ]; then
        echo "ERROR: vllm mode not supported on this GPU. Exiting."
        exit 1
    fi
    echo "Falling back to baseline-only mode."
    MODE="baseline"
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown') MiB"

# ── Proxy killers (curl/nc/ss get hijacked by http_proxy on this host) ──
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*" no_proxy="*"

# ── Helper: wait for TCP port ──
wait_port() {
    local port="$1" timeout="${2:-60}"
    local i
    for i in $(seq 1 "$timeout"); do
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# ── Helper: kill process listening on a given port ──
kill_port() {
    local port="$1"
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep -Po ":$port\s+.*pid=\K\d+" | sort -u || true)
    if [ -n "$pids" ]; then
        echo "   (killing existing process on port $port: $pids)"
        for pid in $pids; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in $pids; do
            kill -9 "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        done
    fi
}

# ── Helper: send SIGTERM, then SIGKILL ──
graceful_kill() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# ── Cleanup trap ──
cleanup() {
    echo ""
    echo ">>> Cleaning up..."
    [ -n "${BASELINE_PID:-}" ] && graceful_kill "$BASELINE_PID"
    [ -n "${VLLM_PID:-}" ] && graceful_kill "$VLLM_PID"
    # Fallback: kill anything still listening on our ports.
    kill_port "$BASELINE_PORT"
    kill_port "$VLLM_PORT"
    pkill -9 -f "EngineCore" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════
echo "=============================================="
echo " Step 3 Benchmark (WSL2) — mode: $MODE"
echo "=============================================="
echo " Requests:    $NUM_REQUESTS"
echo " Concurrency: $CONCURRENCY"
echo " Gen tokens:  $MAX_NEW_TOKENS"
echo " Max batch:   $MAX_BATCH"
echo " Max seq len: $MAX_SEQ_LEN"
echo " Results:     $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# ── Build ──
echo ""
echo ">>> Building baseline server + bench tool..."
cd "$PROJ_DIR"
cargo build --release --bin baseline-server --example bench_throughput 2>&1 | tail -4

# ── Run Rust GPU tests ──
echo ""
echo ">>> Running Rust GPU tests (fragmentation, max concurrency, cuMemMap overhead)..."
cargo test --release --package baseline-llm-os -- \
    step3_max_concurrent_requests \
    step3_runtime_fragmentation \
    step3_cumemmap_overhead \
    --test-threads=1 \
    --nocapture 2>&1 | tee "$RESULTS_DIR/baseline_gpu_tests.txt"

# ═══════════════════════════════════════════════════════════════
run_baseline() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo " Baseline Server  (port $BASELINE_PORT)"
    echo "──────────────────────────────────────────────"

    # Kill any stale process on the port from a previous run.
    kill_port "$BASELINE_PORT"

    RUST_LOG=error "$PROJ_DIR/target/release/baseline-server" \
        --listen "127.0.0.1:$BASELINE_PORT" \
        --model-path dummy \
        --max-batch "$MAX_BATCH" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --continuous \
        &> "$RESULTS_DIR/baseline_server.log" &
    BASELINE_PID=$!
    echo "   PID: $BASELINE_PID"

    if ! wait_port "$BASELINE_PORT" 30; then
        echo "ERROR: baseline server did not start (port not listening)"
        cat "$RESULTS_DIR/baseline_server.log"
        exit 1
    fi

    # Verify the PID we started is still alive (not a stale listener).
    if ! kill -0 "$BASELINE_PID" 2>/dev/null; then
        echo "ERROR: baseline server PID $BASELINE_PID died after start"
        cat "$RESULTS_DIR/baseline_server.log"
        exit 1
    fi
    echo "   -> Ready"

    echo ""
    echo ">>> Running throughput benchmark..."
    timeout 600 "$PROJ_DIR/target/release/examples/bench_throughput" \
        --addr "127.0.0.1:$BASELINE_PORT" \
        --num-requests "$NUM_REQUESTS" \
        --concurrency "$CONCURRENCY" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-csv "$RESULTS_DIR/baseline_results.csv" \
        2>&1 | tee "$RESULTS_DIR/baseline_output.txt"

    echo ""
    echo ">>> Stopping baseline..."
    graceful_kill "$BASELINE_PID"
    unset BASELINE_PID
}

# ═══════════════════════════════════════════════════════════════
collect_gpu_test_metrics() {
    local file="$RESULTS_DIR/baseline_gpu_tests.txt"
    if [ ! -f "$file" ]; then
        echo "  (no GPU test output)"
        return
    fi

    echo ""
    echo ">>> Baseline GPU Test Metrics:"
    echo ""

    local max_conc
    max_conc=$(grep -Po 'max concurrent requests:\s+\K\d+' "$file" 2>/dev/null || echo "N/A")
    echo "  max_concurrent_requests:  $max_conc"

    local int_frag
    int_frag=$(grep -Po 'internal_fragmentation:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  internal_fragmentation:   $int_frag"

    local phys_waste
    phys_waste=$(grep -Po 'physical memory waste ratio:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  physical_memory_waste:    $phys_waste"

    local map_overhead
    map_overhead=$(grep -Po 'avg per 2MB map/unmap:\s+\K[\d.]+' "$file" 2>/dev/null || echo "N/A")
    echo "  avg_2MB_map_unmap_us:     $map_overhead"

    local phys_mem
    phys_mem=$(grep -Po 'physical memory:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  physical_memory_mib:      $phys_mem"

    local map_calls
    map_calls=$(grep -Po 'total cuMemMap calls:\s+\K\d+' "$file" 2>/dev/null || echo "N/A")
    echo "  total_cuMemMap_calls:     $map_calls"

    local runtime_frag_avg
    runtime_frag_avg=$(grep -Po 'avg runtime fragmentation ratio:\s+\K[\d.]+' "$file" 2>/dev/null || echo "N/A")
    echo "  runtime_frag_avg_ratio:   $runtime_frag_avg"

    local runtime_frag_peak
    runtime_frag_peak=$(grep -Po 'peak \(worst\):\s+\K[\d.]+' "$file" 2>/dev/null || echo "N/A")
    echo "  runtime_frag_peak_ratio:  $runtime_frag_peak"

    local runtime_frag_stddev
    runtime_frag_stddev=$(grep -Po 'stddev:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  runtime_frag_stddev:      $runtime_frag_stddev"
}

# ═══════════════════════════════════════════════════════════════
summarize() {
    local label="$1" file="$2"
    if [ ! -f "$file" ]; then echo "  [$label] (no data)"; return; fi
    echo "  [$label]"
    grep -E 'requests_completed|requests_failed|output_throughput_tok_s|total_throughput_tok_s|total_mean_ms|total_p50_ms|total_p95_ms|total_p99_ms' \
        "$file" 2>/dev/null | sed 's/^/    /' || true
}

# ═══════════════════════════════════════════════════════════════
run_baseline
collect_gpu_test_metrics
echo ""
summarize "baseline" "$RESULTS_DIR/baseline_output.txt"

echo ""
echo "=============================================="
echo " Done."
echo " Results: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
echo " NOTE: vLLM comparison requires bare-metal A30 server."
echo "       Run step3_test_baremetal.sh for full comparison."
echo "=============================================="
