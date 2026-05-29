#!/usr/bin/env bash
# ==============================================================================
# scripts/run_step3_bench_baremetal.sh — Step 3 throughput benchmark (bare-metal)
#
# Designed for bare-metal servers with A30 GPU (Compute Capability 8.0).
# If your GPU has a different compute capability, override FLASHINFER_CUDA_ARCH.
#
# Modes:
#   ./scripts/run_step3_bench_baremetal.sh baseline     # Baseline server only
#   ./scripts/run_step3_bench_baremetal.sh vllm          # vLLM only
#   ./scripts/run_step3_bench_baremetal.sh compare       # Both, side by side
#
# Config (env vars or edit defaults below):
#   MODEL_PATH              Path to TinyLlama safetensors (default: /root/models/tinyllama)
#   NUM_REQUESTS            Requests per run (default: 50)
#   CONCURRENCY             Concurrent connections (default: 4)
#   MAX_NEW_TOKENS          Max tokens to generate (default: 64)
#   FLASHINFER_CUDA_ARCH    GPU compute capability for FlashInfer JIT (default: 8.0 for A30)
#   CUDA_HOME               Path to CUDA toolkit (default: /usr/local/cuda-12.2)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──
MODE="${1:-baseline}"
MODEL_PATH="${MODEL_PATH:-/root/models/tinyllama}"
NUM_REQUESTS="${NUM_REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_BATCH="${MAX_BATCH:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BASELINE_PORT=8000
VLLM_PORT=8001
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJ_DIR/results/baremetal/step3_${MODE}_${TIMESTAMP}"

# ── Bare-metal A30 defaults (override via env) ──
FLASHINFER_CUDA_ARCH="${FLASHINFER_CUDA_ARCH:-8.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"

# ── Sanity ──
if [ "$MODE" != "baseline" ] && [ "$MODE" != "vllm" ] && [ "$MODE" != "compare" ]; then
    echo "Usage: $0 {baseline|vllm|compare}"
    exit 1
fi

if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "WARNING: model not found at $MODEL_PATH (no safetensors file)."
    echo "  baseline mode uses --model-path dummy so this is fine for baseline,"
    echo "  but vllm mode needs real weights."
fi

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
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════
echo "=============================================="
echo " Step 3 Benchmark (bare-metal A30) — mode: $MODE"
echo "=============================================="
echo " Model:       $MODEL_PATH"
echo " Requests:    $NUM_REQUESTS"
echo " Concurrency: $CONCURRENCY"
echo " Gen tokens:  $MAX_NEW_TOKENS"
echo " GPU arch:    sm_${FLASHINFER_CUDA_ARCH}"
echo " CUDA_HOME:   $CUDA_HOME"
echo " Baseline:    port $BASELINE_PORT"
echo " vLLM:        port $VLLM_PORT"
echo " Results:     $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# ── Build ──
if [ "$MODE" = "baseline" ] || [ "$MODE" = "compare" ]; then
    echo ""
    echo ">>> Building baseline server + bench tool..."
    cd "$PROJ_DIR"
    cargo build --release --bin baseline-server --example bench_throughput 2>&1 | tail -4
fi

# ═══════════════════════════════════════════════════════════
run_baseline() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo " Baseline Server  (port $BASELINE_PORT)"
    echo "──────────────────────────────────────────────"

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
        echo "ERROR: baseline server did not start"
        cat "$RESULTS_DIR/baseline_server.log"
        exit 1
    fi
    echo "   -> Ready"

    echo ""
    echo ">>> Running benchmark..."
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

# ═══════════════════════════════════════════════════════════
run_vllm() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo " vLLM Server  (port $VLLM_PORT)"
    echo "──────────────────────────────────────────────"

    # Activate the conda env with vLLM + FlashInfer
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate vllm-bench 2>/dev/null || {
        echo "ERROR: conda env 'vllm-bench' not found."
        echo "  Run setup_cloudlab.sh first, or create manually:"
        echo "    conda create -n vllm-bench python=3.12 -y"
        echo "    conda activate vllm-bench"
        echo "    pip install vllm"
        exit 1
    }

    # Ensure FlashInfer CCCL patch is applied (harmless if already patched)
    local SITE_PACKAGES
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

    local CCCL_FILE="$SITE_PACKAGES/flashinfer/compilation_context.py"
    if [ -f "$CCCL_FILE" ] && ! grep -q "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK" "$CCCL_FILE" 2>/dev/null; then
        echo ">>> Applying FlashInfer CCCL patch..."
        sed -i 's/COMMON_NVCC_FLAGS = \[/COMMON_NVCC_FLAGS = ["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",/' "$CCCL_FILE"
    fi

    export FLASHINFER_CUDA_ARCH_LIST="$FLASHINFER_CUDA_ARCH"
    export CUDA_HOME="$CUDA_HOME"

    vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --block-size 16 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs "$MAX_BATCH" \
        --max-model-len "$MAX_SEQ_LEN" \
        --enforce-eager \
        &> "$RESULTS_DIR/vllm_server.log" &
    VLLM_PID=$!
    echo "   PID: $VLLM_PID"

    echo ">>> Waiting for vLLM to load model (may take 2-5 min on first run)..."
    if ! wait_port "$VLLM_PORT" 300; then
        echo "ERROR: vLLM did not start in 5 min"
        tail -30 "$RESULTS_DIR/vllm_server.log"
        exit 1
    fi
    echo "   -> Ready"

    sleep 2

    # Warmup request (first request triggers Triton JIT)
    echo ">>> Warmup..."
    python3 -c "
import http.client, json
conn = http.client.HTTPConnection('127.0.0.1', $VLLM_PORT, timeout=60)
conn.request('POST', '/v1/completions',
    body=json.dumps({'model': '$MODEL_PATH', 'prompt': 'Hello', 'max_tokens': 3}),
    headers={'Content-Type': 'application/json'})
r = conn.getresponse(); r.read()
print('   warmup status:', r.status)
" 2>&1

    # Run benchmark via Python (same prompt-length distribution as baseline)
    echo ""
    echo ">>> Running vLLM benchmark..."
    python3 "$SCRIPT_DIR/bench_vllm.py" \
        --port "$VLLM_PORT" \
        --model "$MODEL_PATH" \
        --num-requests "$NUM_REQUESTS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-csv "$RESULTS_DIR/vllm_results.csv" \
        2>&1 | tee "$RESULTS_DIR/vllm_output.txt"

    echo ""
    echo ">>> Stopping vLLM..."
    graceful_kill "$VLLM_PID"
    unset VLLM_PID
}

# ═══════════════════════════════════════════════════════════
summarize() {
    local label="$1" file="$2"
    if [ ! -f "$file" ]; then echo "  [$label] (no data)"; return; fi
    echo "  [$label]"
    grep -E 'requests_completed|requests_failed|output_throughput_tok_s|total_throughput_tok_s|total_mean_ms|total_p50_ms|total_p95_ms|total_p99_ms' \
        "$file" 2>/dev/null | sed 's/^/    /' || true
}

# ═══════════════════════════════════════════════════════════
case "$MODE" in
    baseline)
        run_baseline
        echo ""
        summarize "baseline" "$RESULTS_DIR/baseline_output.txt"
        ;;
    vllm)
        run_vllm
        echo ""
        summarize "vllm" "$RESULTS_DIR/vllm_output.txt"
        ;;
    compare)
        run_baseline
        run_vllm
        echo ""
        echo "=============================================="
        echo " Comparison Summary"
        echo "=============================================="
        summarize "baseline" "$RESULTS_DIR/baseline_output.txt"
        echo ""
        summarize "vllm"     "$RESULTS_DIR/vllm_output.txt"
        ;;
esac

echo ""
echo "=============================================="
echo " Done."
echo " Results: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo "=============================================="
