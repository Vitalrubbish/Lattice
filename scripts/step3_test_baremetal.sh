#!/usr/bin/env bash
# ==============================================================================
# scripts/step3_test_baremetal.sh — Step 3 benchmark for bare-metal A30
#
# Runs the full comparison suite:
#   baseline  — Rust GPU tests + llama_transformer throughput benchmark
#   vllm      — vLLM server + comprehensive benchmark (frag, concurrency, throughput)
#   compare   — Both baseline + vLLM tests
#
# Modes:
#   ./scripts/step3_test_baremetal.sh baseline
#   ./scripts/step3_test_baremetal.sh vllm
#   ./scripts/step3_test_baremetal.sh compare
#
# Config (env vars):
#   MODEL_PATH              Path to TinyLlama safetensors (default: /root/models/tinyllama)
#   MODEL_TYPE              Model config preset: tinyllama | llama7b (default: tinyllama)
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
MODEL_TYPE="${MODEL_TYPE:-tinyllama}"
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

# ── Helper: wait for GPU memory to be mostly free ──
wait_gpu_free() {
    local max_wait="${1:-30}" min_free_mib="${2:-20480}"  # 20 GiB free
    local i
    for i in $(seq 1 "$max_wait"); do
        local free_mib
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1 | grep -oP '\d+')
        if [ -n "$free_mib" ] && [ "$free_mib" -ge "$min_free_mib" ]; then
            return 0
        fi
        sleep 1
    done
    echo "   (warning: GPU memory may not be fully freed)"
    return 1
}

# ── Helper: aggressively kill all vLLM-related processes ──
kill_vllm_procs() {
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    sleep 2
}

# ── Cleanup trap ──
cleanup() {
    echo ""
    echo ">>> Cleaning up..."
    [ -n "${BASELINE_PID:-}" ] && graceful_kill "$BASELINE_PID"
    [ -n "${VLLM_PID:-}" ] && graceful_kill "$VLLM_PID"
    kill_vllm_procs
    echo "Done."
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════
echo "=============================================="
echo " Step 3 Benchmark (bare-metal A30) — mode: $MODE"
echo "=============================================="
echo " Model:       $MODEL_PATH"
echo " Model type:  $MODEL_TYPE"
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
        step3_runtime_fragmentation \
        step3_cumemmap_overhead \
        --test-threads=1 \
        --nocapture 2>&1 | tee "$RESULTS_DIR/baseline_gpu_tests.txt"
fi

# ═══════════════════════════════════════════════════════════════
run_baseline_llama() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo " Baseline LlamaTransformer + Continuous (port $BASELINE_PORT)"
    echo "──────────────────────────────────────────────"

    if [ ! -f "$MODEL_PATH/model.safetensors" ]; then
        echo "  WARNING: model not found at $MODEL_PATH, skipping llama_transformer test"
        return
    fi

    kill_port "$BASELINE_PORT"

    RUST_LOG=error "$PROJ_DIR/target/release/baseline-server" \
        --listen "127.0.0.1:$BASELINE_PORT" \
        --model-path "$MODEL_PATH" \
        --model-type "$MODEL_TYPE" \
        --max-batch "$MAX_BATCH" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --continuous \
        --llama \
        &> "$RESULTS_DIR/baseline_llama_server.log" &
    BASELINE_PID=$!
    echo "   PID: $BASELINE_PID"

    if ! wait_port "$BASELINE_PORT" 60; then
        echo "ERROR: baseline llama server did not start (port not listening)"
        cat "$RESULTS_DIR/baseline_llama_server.log"
        exit 1
    fi

    if ! kill -0 "$BASELINE_PID" 2>/dev/null; then
        echo "ERROR: baseline llama server PID $BASELINE_PID died after start"
        cat "$RESULTS_DIR/baseline_llama_server.log"
        exit 1
    fi
    echo "   -> Ready"

    echo ""
    echo ">>> Running throughput benchmark (llama_transformer + continuous)..."
    timeout 600 "$PROJ_DIR/target/release/examples/bench_throughput" \
        --addr "127.0.0.1:$BASELINE_PORT" \
        --num-requests "$NUM_REQUESTS" \
        --concurrency "$CONCURRENCY" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-csv "$RESULTS_DIR/baseline_llama_results.csv" \
        2>&1 | tee "$RESULTS_DIR/baseline_llama_output.txt"

    echo ""
    echo ">>> Stopping baseline llama..."
    graceful_kill "$BASELINE_PID"
    unset BASELINE_PID
    # Wait for GPU memory to be released before next step
    wait_gpu_free 30 || true
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
    kill_vllm_procs
    wait_gpu_free 30 || true
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

    # UFS metrics from runtime fragmentation test
    local ufs_ifr
    ufs_ifr=$(grep -Po 'IFR avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  [UFS] ifr_avg:            $ufs_ifr"

    local ufs_bu
    ufs_bu=$(grep -Po 'BU avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  [UFS] bu_avg:             $ufs_bu"

    local ufs_pme
    ufs_pme=$(grep -Po 'PME avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  [UFS] pme_avg:            $ufs_pme"

    local ufs_rfi
    ufs_rfi=$(grep -Po 'RFI avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "  [UFS] rfi_avg:            $ufs_rfi"

    # Legacy (backward compat)
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

# ── Helper: extract UFS metrics from baseline throughput output ──
extract_ufs_from_baseline() {
    local file="$1"
    if [ ! -f "$file" ]; then echo "N/A"; return; fi

    local ifr_avg bu_avg pme_avg rfi_avg ifr_peak rfi_peak
    ifr_avg=$(grep -Po 'ifr_avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    bu_avg=$(grep -Po 'bu_avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    pme_avg=$(grep -Po 'pme_avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    rfi_avg=$(grep -Po 'rfi_avg:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    ifr_peak=$(grep -Po 'ifr_peak:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    rfi_peak=$(grep -Po 'rfi_peak:\s+\K[\d.]+' "$file" 2>/dev/null | tail -1 || echo "N/A")
    echo "$ifr_avg $bu_avg $pme_avg $rfi_avg $ifr_peak $rfi_peak"
}

# ── Helper: extract UFS metrics from vLLM JSON output ──
extract_ufs_from_vllm() {
    local json_file="$1"
    if [ ! -f "$json_file" ]; then echo "N/A"; return; fi

    python3 -c "
import json, sys
try:
    with open('$json_file') as f:
        data = json.load(f)
    ufs = data.get('ufs_summary', {})
    vals = [
        ufs.get('ifr_avg', 0), ufs.get('bu_avg', 0), ufs.get('pme_avg', 0),
        ufs.get('rfi_avg', 0), ufs.get('ifr_peak', 0), ufs.get('rfi_peak', 0),
    ]
    print(' '.join(f'{v:.4f}' if isinstance(v, (int, float)) else str(v) for v in vals))
except Exception as e:
    print('N/A', file=sys.stderr)
" 2>/dev/null || echo "N/A"
}

# ── Helper: print UFS comparison table ──
print_ufs_comparison() {
    local baseline_label="$1" baseline_file="$2"
    local vllm_label="$3" vllm_file="$4"

    local b_ufs v_ufs
    b_ufs=$(extract_ufs_from_baseline "$baseline_file")
    v_ufs=$(extract_ufs_from_vllm "$vllm_file")

    local b_ifr_avg b_bu_avg b_pme_avg b_rfi_avg b_ifr_peak b_rfi_peak
    read -r b_ifr_avg b_bu_avg b_pme_avg b_rfi_avg b_ifr_peak b_rfi_peak <<< "$b_ufs"
    local v_ifr_avg v_bu_avg v_pme_avg v_rfi_avg v_ifr_peak v_rfi_peak
    read -r v_ifr_avg v_bu_avg v_pme_avg v_rfi_avg v_ifr_peak v_rfi_peak <<< "$v_ufs"

    echo ""
    echo "=============================================="
    echo " Unified Fragmentation Comparison (UFS)"
    echo "=============================================="
    printf " %-22s | %-10s | %-10s\n" "Metric" "$baseline_label" "$vllm_label"
    printf " %-22s-+-%-10s-+-%-10s\n" "----------------------" "----------" "----------"
    printf " %-22s | %-10s | %-10s\n" "IFR avg" "$b_ifr_avg" "$v_ifr_avg"
    printf " %-22s | %-10s | %-10s\n" "IFR peak" "$b_ifr_peak" "$v_ifr_peak"
    printf " %-22s | %-10s | %-10s\n" "BU avg" "$b_bu_avg" "$v_bu_avg"
    printf " %-22s | %-10s | %-10s\n" "PME avg" "$b_pme_avg" "$v_pme_avg"
    printf " %-22s | %-10s | %-10s\n" "RFI avg" "$b_rfi_avg" "$v_rfi_avg"
    printf " %-22s | %-10s | %-10s\n" "RFI peak" "$b_rfi_peak" "$v_rfi_peak"
    echo "=============================================="
    echo " (IFR/BU are directly comparable across systems)"
    echo " (PME/RFI use system-specific actual_physical_bytes)"
    echo "=============================================="
}

# ═══════════════════════════════════════════════════════════════
case "$MODE" in
    baseline)
        run_baseline_llama
        collect_gpu_test_metrics
        echo ""
        summarize "baseline (llama+continuous)" "$RESULTS_DIR/baseline_llama_output.txt"
        ;;
    vllm)
        run_vllm
        echo ""
        echo "vLLM results in: $RESULTS_DIR/"
        ls -la "$RESULTS_DIR/"*.json 2>/dev/null || true
        ;;
    compare)
        run_baseline_llama
        run_vllm
        collect_gpu_test_metrics
        echo ""
        summarize "baseline (llama+continuous)" "$RESULTS_DIR/baseline_llama_output.txt"
        echo ""
        # Print UFS comparison table
        print_ufs_comparison \
            "Baseline" "$RESULTS_DIR/baseline_llama_output.txt" \
            "vLLM" "$RESULTS_DIR/throughput.json"
        ;;
esac

echo ""
echo "=============================================="
echo " Done."
echo " Results: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo "=============================================="
