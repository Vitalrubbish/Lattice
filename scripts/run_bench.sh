#!/usr/bin/env bash
# ==============================================================================
# scripts/run_bench.sh — Unified benchmark entry point for Step 3 comparison.
#
# Runs fragmentation (UFS), max concurrency, and throughput benchmarks for
# baseline and/or vLLM. Automatically handles server lifecycle, build, and
# results collection. Auto-detects GPU and environment.
#
# Usage:
#   ./scripts/run_bench.sh [--target baseline|vllm|compare] \
#                          [--mode max_concurrency|throughput|fragmentation|all] \
#                          [--num-requests 100] [--concurrency 4]
#
# Config (env vars):
#   MODEL_PATH              Path to TinyLlama safetensors (default: auto-detect)
#   MODEL_TYPE              Model config preset: tinyllama | llama7b
#   NUM_REQUESTS            Requests per run (default: 100)
#   CONCURRENCY             Concurrent connections (default: 4)
#   MAX_NEW_TOKENS          Max tokens to generate (default: 64)
#   MAX_BATCH               Max concurrent seqs (default: 128)
#   MAX_SEQ_LEN             Max sequence length (default: 512)
#   BASELINE_PORT           Baseline server port (default: 8000)
#   VLLM_PORT               vLLM server port (default: 8001)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──
TARGET="${1:-compare}"
MODE="${2:-all}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-tinyllama}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_BATCH="${MAX_BATCH:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BASELINE_PORT="${BASELINE_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8001}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJ_DIR/results/run_${TARGET}_${TIMESTAMP}"

# ── Proxy killers ──
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*" no_proxy="*"

# ── Auto-detect model path ──
if [ -z "$MODEL_PATH" ]; then
    for candidate in \
        /home/vitalrubbish/models/tinyllama \
        /root/models/tinyllama \
        "$PROJ_DIR/models/tinyllama"; do
        if [ -f "$candidate/model.safetensors" ]; then
            MODEL_PATH="$candidate"
            break
        fi
    done
fi

# ── Auto-detect GPU ──
detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        local vram
        vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "0")
        echo "GPU: $gpu_name"
        echo "VRAM: ${vram} MiB"

        # Check if Blackwell (sm_120) — vLLM FlashInfer won't work
        if echo "$gpu_name" | grep -qi "RTX 50"; then
            echo "WARNING: Blackwell GPU detected (sm_120). vLLM FlashInfer may not work."
            echo "         vLLM benchmarks will be skipped."
            return 1
        fi
    else
        echo "WARNING: nvidia-smi not found. Is the NVIDIA driver installed?"
        return 1
    fi
    return 0
}

# ── Helpers ──
wait_port() {
    local port="$1" timeout="${2:-60}"
    for i in $(seq 1 "$timeout"); do
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

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

graceful_kill() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    echo ""
    echo ">>> Cleaning up..."
    [ -n "${BASELINE_PID:-}" ] && graceful_kill "$BASELINE_PID"
    [ -n "${VLLM_PID:-}" ] && graceful_kill "$VLLM_PID"
    kill_port "$BASELINE_PORT"
    kill_port "$VLLM_PORT"
    pkill -9 -f "EngineCore" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

echo "=============================================="
echo " Step 3 Benchmark — target: $TARGET, mode: $MODE"
echo "=============================================="
echo " Model:       $MODEL_PATH"
echo " Model type:  $MODEL_TYPE"
echo " Requests:    $NUM_REQUESTS"
echo " Concurrency: $CONCURRENCY"
echo " Gen tokens:  $MAX_NEW_TOKENS"
echo " Max batch:   $MAX_BATCH"
echo " Max seq len: $MAX_SEQ_LEN"
echo " Results:     $RESULTS_DIR"
echo "=============================================="

detect_gpu || true
mkdir -p "$RESULTS_DIR"

# ── Determine if vLLM is available ──
VLLM_AVAILABLE=true
if ! command -v vllm &>/dev/null; then
    echo "vLLM not found in PATH; vLLM benchmarks will be skipped."
    VLLM_AVAILABLE=false
fi

# ═══════════════════════════════════════════════════════════════
# Baseline server lifecycle
# ═══════════════════════════════════════════════════════════════

start_baseline() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo " Baseline Server (port $BASELINE_PORT)"
    echo "──────────────────────────────────────────────"

    kill_port "$BASELINE_PORT"

    echo ">>> Building baseline server..."
    cd "$PROJ_DIR"
    cargo build --release --bin baseline-server 2>&1 | tail -4

    RUST_LOG=error "$PROJ_DIR/target/release/baseline-server" \
        --listen "127.0.0.1:$BASELINE_PORT" \
        --model-path "$MODEL_PATH" \
        --model-type "$MODEL_TYPE" \
        --max-batch "$MAX_BATCH" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --continuous \
        --llama \
        &> "$RESULTS_DIR/baseline_server.log" &
    BASELINE_PID=$!
    echo "   PID: $BASELINE_PID"

    if ! wait_port "$BASELINE_PORT" 120; then
        echo "ERROR: Baseline server did not start"
        cat "$RESULTS_DIR/baseline_server.log"
        exit 1
    fi

    if ! kill -0 "$BASELINE_PID" 2>/dev/null; then
        echo "ERROR: Baseline server PID $BASELINE_PID died after start"
        cat "$RESULTS_DIR/baseline_server.log"
        exit 1
    fi
    echo "   -> Ready"
}

stop_baseline() {
    if [ -n "${BASELINE_PID:-}" ]; then
        echo ">>> Stopping baseline server..."
        graceful_kill "$BASELINE_PID"
        unset BASELINE_PID
    fi
}

# ═══════════════════════════════════════════════════════════════
# vLLM server lifecycle
# ═══════════════════════════════════════════════════════════════

start_vllm() {
    if ! $VLLM_AVAILABLE; then
        echo ">>> Skipping vLLM server (not available)"
        return 1
    fi

    echo ""
    echo "──────────────────────────────────────────────"
    echo " vLLM Server (port $VLLM_PORT)"
    echo "──────────────────────────────────────────────"

    kill_port "$VLLM_PORT"
    pkill -9 -f "EngineCore" 2>/dev/null || true

    vllm serve "$MODEL_PATH" \
        --port "$VLLM_PORT" \
        --block-size 16 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs "$MAX_BATCH" \
        --max-model-len "$MAX_SEQ_LEN" \
        --enforce-eager \
        &> "$RESULTS_DIR/vllm_server.log" &
    VLLM_PID=$!
    echo "   PID: $VLLM_PID"

    if ! wait_port "$VLLM_PORT" 180; then
        echo "ERROR: vLLM server did not start"
        cat "$RESULTS_DIR/vllm_server.log"
        return 1
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server PID $VLLM_PID died after start"
        cat "$RESULTS_DIR/vllm_server.log"
        return 1
    fi
    echo "   -> Ready"
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ]; then
        echo ">>> Stopping vLLM server..."
        graceful_kill "$VLLM_PID"
        unset VLLM_PID
    fi
}

# ═══════════════════════════════════════════════════════════════
# Benchmark runners
# ═══════════════════════════════════════════════════════════════

BENCH_DIR="$SCRIPT_DIR/bench"

run_bench_max_concurrency() {
    local target="$1"
    local port="$2"
    local host="${3:-127.0.0.1}"

    echo ""
    echo ">>> Running max_concurrency benchmark ($target)..."
    timeout 900 python3 "$BENCH_DIR/bench_max_concurrency.py" \
        --target "$target" \
        --host "$host" \
        --port "$port" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$RESULTS_DIR/$target/max_concurrency" \
        --server-ready \
        2>&1 | tee "$RESULTS_DIR/${target}_max_concurrency_output.txt"
}

run_bench_throughput() {
    local target="$1"
    local port="$2"
    local host="${3:-127.0.0.1}"

    echo ""
    echo ">>> Running throughput benchmark ($target)..."
    timeout 600 python3 "$BENCH_DIR/bench_throughput.py" \
        --target "$target" \
        --host "$host" \
        --port "$port" \
        --num-requests "$NUM_REQUESTS" \
        --concurrency "$CONCURRENCY" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$RESULTS_DIR/$target/throughput" \
        --server-ready \
        2>&1 | tee "$RESULTS_DIR/${target}_throughput_output.txt"
}

run_bench_fragmentation() {
    local target="$1"
    local port="$2"
    local host="${3:-127.0.0.1}"

    echo ""
    echo ">>> Running fragmentation benchmark ($target)..."
    timeout 1800 python3 "$BENCH_DIR/bench_fragmentation.py" \
        --target "$target" \
        --host "$host" \
        --port "$port" \
        --num-requests "$NUM_REQUESTS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$RESULTS_DIR/$target/fragmentation" \
        --vllm-log-path "$RESULTS_DIR/vllm_server.log" \
        --server-ready \
        2>&1 | tee "$RESULTS_DIR/${target}_fragmentation_output.txt"
}

run_all_benches() {
    local target="$1"
    local port="$2"
    local host="${3:-127.0.0.1}"

    run_bench_max_concurrency "$target" "$port" "$host"
    run_bench_throughput "$target" "$port" "$host"
    run_bench_fragmentation "$target" "$port" "$host"
}

# ═══════════════════════════════════════════════════════════════
# Execute
# ═══════════════════════════════════════════════════════════════

case "$TARGET" in
    baseline)
        start_baseline
        case "$MODE" in
            max_concurrency) run_bench_max_concurrency baseline "$BASELINE_PORT" ;;
            throughput)      run_bench_throughput baseline "$BASELINE_PORT" ;;
            fragmentation)   run_bench_fragmentation baseline "$BASELINE_PORT" ;;
            all)             run_all_benches baseline "$BASELINE_PORT" ;;
            *) echo "Unknown mode: $MODE"; exit 1 ;;
        esac
        stop_baseline
        ;;
    vllm)
        start_vllm || { echo "vLLM start failed"; exit 1; }
        case "$MODE" in
            max_concurrency) run_bench_max_concurrency vllm "$VLLM_PORT" ;;
            throughput)      run_bench_throughput vllm "$VLLM_PORT" ;;
            fragmentation)   run_bench_fragmentation vllm "$VLLM_PORT" ;;
            all)             run_all_benches vllm "$VLLM_PORT" ;;
            *) echo "Unknown mode: $MODE"; exit 1 ;;
        esac
        stop_vllm
        ;;
    compare)
        # Run both in sequence
        start_baseline
        case "$MODE" in
            max_concurrency) run_bench_max_concurrency baseline "$BASELINE_PORT" ;;
            throughput)      run_bench_throughput baseline "$BASELINE_PORT" ;;
            fragmentation)   run_bench_fragmentation baseline "$BASELINE_PORT" ;;
            all)             run_all_benches baseline "$BASELINE_PORT" ;;
            *) echo "Unknown mode: $MODE"; exit 1 ;;
        esac
        stop_baseline

        start_vllm
        if [ $? -eq 0 ]; then
            case "$MODE" in
                max_concurrency) run_bench_max_concurrency vllm "$VLLM_PORT" ;;
                throughput)      run_bench_throughput vllm "$VLLM_PORT" ;;
                fragmentation)   run_bench_fragmentation vllm "$VLLM_PORT" ;;
                all)             run_all_benches vllm "$VLLM_PORT" ;;
            esac
            stop_vllm
        fi
        ;;
    *)
        echo "Usage: $0 [baseline|vllm|compare] [max_concurrency|throughput|fragmentation|all]"
        echo ""
        echo "Examples:"
        echo "  $0 baseline max_concurrency     # Baseline capacity test only"
        echo "  $0 vllm fragmentation            # vLLM fragmentation stress test only"
        echo "  $0 compare all                   # Full comparison (all 3 benchmarks)"
        exit 1
        ;;
esac

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

echo ""
echo "=============================================="
echo " Done."
echo " Results: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/" 2>/dev/null || true
echo "=============================================="
