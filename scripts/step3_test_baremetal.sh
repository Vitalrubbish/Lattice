#!/usr/bin/env bash
# ==============================================================================
# scripts/step3_test_baremetal.sh — Step 3 benchmark for bare-metal A30
#
# Runs the full comparison suite:
#   baseline  — Rust GPU tests (frag, max-concurrency, cuMemMap overhead) +
#               Rust HTTP server throughput benchmark
#   vllm      — vLLM server + comprehensive benchmark (frag, concurrency, throughput)
#   compare   — Both, produce side-by-side report
#
# Modes:
#   ./scripts/step3_test_baremetal.sh baseline
#   ./scripts/step3_test_baremetal.sh vllm
#   ./scripts/step3_test_baremetal.sh compare
#
# Config (env vars):
#   MODEL_PATH              Path to TinyLlama safetensors (default: /root/models/tinyllama)
#   NUM_REQUESTS            Requests per run (default: 100)
#   CONCURRENCY             Concurrent connections (default: 4)
#   MAX_NEW_TOKENS          Max tokens to generate (default: 64)
#   MAX_BATCH               Max concurrent seqs (default: 128)
#   MAX_SEQ_LEN             Max sequence length (default: 512)
#   FLASHINFER_CUDA_ARCH    GPU compute capability for FlashInfer (default: 8.0 for A30)
#   CUDA_HOME               Path to CUDA toolkit (default: /usr/local/cuda-12.2)
#   VLLM_GPU_MEM_UTIL       vLLM GPU memory utilization (default: 0.85)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──
MODE="${1:-compare}"
MODEL_PATH="${MODEL_PATH:-/root/models/tinyllama}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_BATCH="${MAX_BATCH:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
BASELINE_PORT=8000
VLLM_PORT=8001
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJ_DIR/results/baremetal/step3_${MODE}_${TIMESTAMP}"

# ── Bare-metal A30 defaults ──
FLASHINFER_CUDA_ARCH="${FLASHINFER_CUDA_ARCH:-8.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"

# ── Sanity ──
if [ "$MODE" != "baseline" ] && [ "$MODE" != "vllm" ] && [ "$MODE" != "compare" ]; then
    echo "Usage: $0 {baseline|vllm|compare}"
    exit 1
fi

if [ "$MODE" = "vllm" ] || [ "$MODE" = "compare" ]; then
    if [ ! -f "$MODEL_PATH/model.safetensors" ]; then
        echo "ERROR: model not found at $MODEL_PATH (model.safetensors missing)."
        echo "  Set MODEL_PATH= env var or download the model."
        exit 1
    fi
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown') MiB"

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
    # vLLM V1 engine uses multiprocessing; kill remaining engine processes
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════
echo "=============================================="
echo " Step 3 Benchmark (bare-metal A30) — mode: $MODE"
echo "=============================================="
echo " Model:       $MODEL_PATH"
echo " Requests:    $NUM_REQUESTS"
echo " Concurrency: $CONCURRENCY"
echo " Gen tokens:  $MAX_NEW_TOKENS"
echo " Max batch:   $MAX_BATCH"
echo " Max seq len: $MAX_SEQ_LEN"
echo " Block size:  $BLOCK_SIZE"
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

    echo ">>> Running Rust GPU tests (fragmentation, max concurrency, cuMemMap overhead)..."
    cargo test --release --package baseline-llm-os -- \
        step3_max_concurrent_requests \
        step3_fragmentation_rate \
        step3_cumemmap_overhead \
        step3_internal_fragmentation_analysis \
        --nocapture 2>&1 | tee "$RESULTS_DIR/baseline_gpu_tests.txt"
fi

# ═══════════════════════════════════════════════════════════════
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
    echo ">>> Running throughput benchmark (concurrent)..."
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
run_vllm() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo " vLLM Server  (port $VLLM_PORT)"
    echo "──────────────────────────────────────────────"

    # Activate the conda env with vLLM
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

    # Use a larger max-num-seqs for max concurrency testing
    vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --block-size "$BLOCK_SIZE" \
        --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
        --max-num-seqs "$MAX_BATCH" \
        --max-model-len "$MAX_SEQ_LEN" \
        --enforce-eager \
        &> "$RESULTS_DIR/vllm_server.log" &
    VLLM_PID=$!
    echo "   PID: $VLLM_PID"

    echo ">>> Waiting for vLLM to load model (may take 2-5 min on first run)..."
    if ! wait_port "$VLLM_PORT" 300; then
        echo "ERROR: vLLM did not start in 5 min"
        echo "--- Last 50 lines of vLLM log: ---"
        tail -50 "$RESULTS_DIR/vllm_server.log"
        exit 1
    fi
    echo "   -> Ready"

    sleep 2

    # ── Run comprehensive vLLM benchmark (fragmentation, max concurrency, throughput) ──
    echo ""
    echo ">>> Running comprehensive vLLM benchmark..."
    python3 "$SCRIPT_DIR/bench_vllm_comprehensive.py" \
        --port "$VLLM_PORT" \
        --model "$MODEL_PATH" \
        --mode all \
        --num-requests "$NUM_REQUESTS" \
        --concurrency "$CONCURRENCY" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/vllm_output.txt"

    echo ""
    echo ">>> Stopping vLLM..."
    graceful_kill "$VLLM_PID"
    unset VLLM_PID
}

# ═══════════════════════════════════════════════════════════════
collect_gpu_test_metrics() {
    # Extract key metrics from the Rust GPU test output
    local file="$RESULTS_DIR/baseline_gpu_tests.txt"
    if [ ! -f "$file" ]; then
        echo "  (no GPU test output)"
        return
    fi

    echo ""
    echo ">>> Baseline GPU Test Metrics:"
    echo ""

    # Max concurrent requests
    local max_conc
    max_conc=$(grep -Po 'max concurrent requests:\s+\K\d+' "$file" 2>/dev/null || echo "N/A")
    echo "  max_concurrent_requests:  $max_conc"

    # Internal fragmentation
    local int_frag
    int_frag=$(grep -Po 'internal_fragmentation:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  internal_fragmentation:   $int_frag"

    # cuMemMap overhead
    local map_overhead
    map_overhead=$(grep -Po 'avg per 2MB map/unmap:\s+\K[\d.]+' "$file" 2>/dev/null || echo "N/A")
    echo "  avg_2MB_map_unmap_us:     $map_overhead"

    # Physical memory used
    local phys_mem
    phys_mem=$(grep -Po 'physical memory:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  physical_memory_mib:      $phys_mem"

    # Total cuMemMap calls
    local map_calls
    map_calls=$(grep -Po 'total cuMemMap calls:\s+\K\d+' "$file" 2>/dev/null || echo "N/A")
    echo "  total_cuMemMap_calls:     $map_calls"
}

# ═══════════════════════════════════════════════════════════════
summarize() {
    local label="$1" file="$2"
    if [ ! -f "$file" ]; then echo "  [$label] (no data)"; return; fi
    echo "  [$label]"
    grep -E 'requests_completed|requests_failed|request_throughput_req_s|output_throughput_tok_s|total_throughput_tok_s|total_mean_ms|total_p50_ms|total_p95_ms|total_p99_ms|max_concurrent|internal_fragmentation|external_frag' \
        "$file" 2>/dev/null | sed 's/^/    /' || true
}

# ═══════════════════════════════════════════════════════════════
write_comparison_report() {
    local report="$RESULTS_DIR/comparison_report.md"

    cat > "$report" << 'REPORT_HEADER'
# Step 3 Benchmark Comparison Report

**Date:** REPORT_HEADER
    date >> "$report"
    cat >> "$report" << REPORT_HEADER2
**Server:** Bare-metal A30
**Model:** TinyLlama-1.1B

---

## 1. Maximum Concurrent Requests

| Metric | Baseline (Rust + CUDA VMM) | vLLM |
|--------|---------------------------|------|
REPORT_HEADER2

    # Extract values
    local base_max_conc="N/A"
    local vllm_max_conc="N/A"
    base_max_conc=$(grep -Po 'max concurrent requests:\s+\K\d+' "$RESULTS_DIR/baseline_gpu_tests.txt" 2>/dev/null || echo "N/A")
    vllm_max_conc=$(grep -Po '"max_concurrent_requests":\s+\K\d+' "$RESULTS_DIR/max_concurrency.json" 2>/dev/null || echo "N/A")

    echo "| max_concurrent_requests | $base_max_conc | $vllm_max_conc |" >> "$report"

    cat >> "$report" << 'REPORT_MID'

---

## 2. Memory Fragmentation Rate

| Metric | Baseline (Rust + CUDA VMM) | vLLM |
|--------|---------------------------|------|
REPORT_MID

    local base_int_frag="N/A"
    local vllm_int_frag="N/A"
    local vllm_ext_frag="N/A"
    base_int_frag=$(grep -Po 'internal_fragmentation:\s+\K[\d.]+' "$RESULTS_DIR/baseline_gpu_tests.txt" 2>/dev/null | tail -1 || echo "N/A")
    vllm_int_frag=$(grep -Po '"internal_frag_ratio":\s+\K[\d.]+' "$RESULTS_DIR/fragmentation.json" 2>/dev/null || echo "N/A")
    vllm_ext_frag=$(grep -Po '"external_frag_ratio":\s+\K[\d.]+' "$RESULTS_DIR/fragmentation.json" 2>/dev/null || echo "N/A")

    echo "| internal_fragmentation | $base_int_frag | $vllm_int_frag |" >> "$report"
    echo "| external_frag_proxy | N/A (block-level allocator) | $vllm_ext_frag |" >> "$report"

    cat >> "$report" << 'REPORT_MID2'

---

## 3. Throughput

| Metric | Baseline (Rust + CUDA VMM) | vLLM |
|--------|---------------------------|------|
REPORT_MID2

    local base_tp="N/A" base_lat="N/A" base_p95="N/A"
    local vllm_tp="N/A" vllm_lat="N/A" vllm_p95="N/A"

    base_tp=$(grep -Po 'total_throughput_tok_s:\s+\K[\d.]+' "$RESULTS_DIR/baseline_output.txt" 2>/dev/null || echo "N/A")
    base_lat=$(grep -Po 'total_mean_ms:\s+\K[\d.]+' "$RESULTS_DIR/baseline_output.txt" 2>/dev/null || echo "N/A")
    base_p95=$(grep -Po 'total_p95_ms:\s+\K[\d.]+' "$RESULTS_DIR/baseline_output.txt" 2>/dev/null || echo "N/A")
    vllm_tp=$(grep -Po '"total_throughput_tok_s":\s+\K[\d.]+' "$RESULTS_DIR/throughput.json" 2>/dev/null || echo "N/A")
    vllm_lat=$(grep -Po '"total_mean_ms":\s+\K[\d.]+' "$RESULTS_DIR/throughput.json" 2>/dev/null || echo "N/A")
    vllm_p95=$(grep -Po '"total_p95_ms":\s+\K[\d.]+' "$RESULTS_DIR/throughput.json" 2>/dev/null || echo "N/A")

    echo "| total_throughput_tok_s | $base_tp | $vllm_tp |" >> "$report"
    echo "| total_mean_ms | $base_lat | $vllm_lat |" >> "$report"
    echo "| total_p95_ms | $base_p95 | $vllm_p95 |" >> "$report"

    cat >> "$report" << 'REPORT_MID3'

---

## 4. cuMemMap/cuMemUnmap Overhead (Baseline only)

vLLM uses PyTorch's CUDA caching allocator and does not use the CUDA VMM API directly, so
cuMemMap/cuMemUnmap overhead is measured only for the baseline implementation.

REPORT_MID3

    local map_overhead="N/A"
    local map_gran="N/A"
    local map_calls="N/A"
    map_overhead=$(grep -Po 'avg per 2MB map/unmap:\s+\K[\d.]+' "$RESULTS_DIR/baseline_gpu_tests.txt" 2>/dev/null || echo "N/A")
    map_gran=$(grep -Po 'GPU map granularity:\s+\K\d+' "$RESULTS_DIR/baseline_gpu_tests.txt" 2>/dev/null || echo "N/A")
    map_calls=$(grep -Po 'total cuMemMap calls:\s+\K\d+' "$RESULTS_DIR/baseline_gpu_tests.txt" 2>/dev/null || echo "N/A")

    echo "| map_granularity_bytes | $map_gran | N/A |" >> "$report"
    echo "| avg_2MB_map_unmap_us | $map_overhead | N/A |" >> "$report"
    echo "| total_cuMemMap_calls | $map_calls | N/A |" >> "$report"
    echo "| strategy | batch mapping via superblocks | PyTorch caching allocator |" >> "$report"

    cat >> "$report" << 'REPORT_FOOTER'

---

## 5. Notes

- **Baseline** uses CUDA VMM API (`cuMemCreate`/`cuMemMap`/`cuMemUnmap`) with 2MB superblock batch mapping
- **vLLM** uses PyTorch's CUDA caching allocator with block-based KV cache management
- Both use `block_size=16` tokens
- The baseline's `cuMemMap`/`cuMemUnmap` calls are batched at superblock granularity (2MB) rather than per-block
- For detailed cuMemMap overhead analysis, see the `step3_cumemmap_overhead` GPU test output

---
*Report generated by step3_test_baremetal.sh*
REPORT_FOOTER

    echo ""
    echo ">>> Comparison report written to: $report"
}

# ═══════════════════════════════════════════════════════════════
case "$MODE" in
    baseline)
        run_baseline
        collect_gpu_test_metrics
        echo ""
        summarize "baseline" "$RESULTS_DIR/baseline_output.txt"
        ;;
    vllm)
        run_vllm
        echo ""
        echo "vLLM results in: $RESULTS_DIR/"
        ls -la "$RESULTS_DIR/"*.json 2>/dev/null || true
        ;;
    compare)
        run_baseline
        run_vllm
        collect_gpu_test_metrics
        echo ""
        echo "=============================================="
        echo " Comparison Summary"
        echo "=============================================="
        summarize "baseline" "$RESULTS_DIR/baseline_output.txt"
        echo ""
        summarize "vllm"     "$RESULTS_DIR/vllm_output.txt"
        echo ""
        write_comparison_report
        ;;
esac

echo ""
echo "=============================================="
echo " Done."
echo " Results: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo "=============================================="
