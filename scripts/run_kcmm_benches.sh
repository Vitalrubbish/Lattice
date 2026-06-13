#!/usr/bin/env bash
# ==============================================================================
# scripts/run_kcmm_benches.sh — Run all KCMM swap policy benchmarks from
# docs/report/WSL2/kcmm-swap-policy-benchmark-report.md
#
# Usage:
#   ./scripts/run_kcmm_benches.sh              # Run all benchmarks (release)
#   ./scripts/run_kcmm_benches.sh --debug      # Run with debug build (faster compile)
#   ./scripts/run_kcmm_benches.sh --filter <name>  # Run only tests matching <name> (e.g. "alloc" or "tiering" or "batch_eviction")
#
# Output:
#   results/kcmm_bench_<timestamp>/  — per-test log files + summary
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ── Argument parsing ──
PROFILE="release"
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            PROFILE="debug"
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--debug] [--filter <test-name-substring>]"
            echo ""
            echo "Runs all KCMM swap policy benchmarks and saves results to disk."
            echo ""
            echo "Options:"
            echo "  --debug    Build and run with debug profile (faster compile, slower runtime)"
            echo "  --filter   Only run tests whose name contains this substring"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Prerequisites check ──
echo "=== KCMM Benchmark Runner ==="
echo ""

# Check CUDA device
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA device required for benchmarks."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "0")
echo "GPU: $GPU_NAME"
echo "VRAM: ${VRAM} MiB"
echo ""

# ── Timestamp and output directory ──
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJ_DIR/results/kcmm_bench_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results: $RESULTS_DIR"
echo "Profile: $PROFILE"
echo ""

# ── Build flags ──
FEATURES="--features kcmm"
CARGO_TEST="cargo test $FEATURES"
if [ "$PROFILE" = "release" ]; then
    CARGO_TEST="$CARGO_TEST --release"
fi

# ── Test definitions ──
# Format: "test_name::test_file::requires_kcmm_feature::description"
# We use arrays of "name|file|desc" for readability.
declare -a BENCHMARKS=(
    # Benchmark 1: Allocation / Free Throughput
    "kcmm_bench_alloc_throughput|kcmm_bench_alloc|Benchmark 1a — Block Size Sweep (alloc/free throughput vs block size)"
    "kcmm_bench_alloc_pool_size_sweep|kcmm_bench_alloc|Benchmark 1b — Pool Size Sweep (alloc/free vs pool capacity)"
    "kcmm_bench_alloc_concurrent_sequences|kcmm_bench_alloc|Benchmark 1c — Multi-Sequence Concurrent Allocation"

    # Benchmark 2: Single-Block Eviction / Restoration + Batch + cuMemMap + Integrity
    "kcmm_bench_single_block_evict_restore|kcmm_bench_tiering|Benchmark 2a — Single-Block Evict/Restore (block size sweep)"
    "kcmm_bench_batch_eviction_amortization|kcmm_bench_tiering|Benchmark 2b — Batch Eviction Amortisation (batch size sweep)"
    "kcmm_bench_cumemmap_latency|kcmm_bench_tiering|Benchmark 2c — Standalone cuMemMap/cuMemUnmap Latency"
    "kcmm_bench_tiering_roundtrip_data_integrity|kcmm_bench_tiering|Benchmark 2d — Evict→Restore Data Integrity (roundtrip)"
    "kcmm_bench_batch_restore_amortization|kcmm_bench_tiering|Benchmark 2e — Batch Restore Amortisation (batch size sweep)"

    # Benchmark 3: CUDA Stream Interference
    "kcmm_bench_stream_interference|kcmm_bench_tiering|Benchmark 3 — CUDA Stream Interference (default vs dedicated streams)"

    # Benchmark 4 (Step 3): cuMemMap overhead per layer
    "step3_cumemmap_overhead|step3_benchmarks|Benchmark 4 — Per-Layer cuMemMap/cuMemUnmap Overhead (22 layers)"

    # Benchmark 6 (Step 3): Max Concurrent Requests
    "step3_max_concurrent_requests|step3_benchmarks|Benchmark 6 — Maximum Concurrent Requests (capacity at workload)"

    # Benchmark 5 (Phase 1c): Memory Pressure — Tiering Capacity Benefit
    "kcmm_bench_memory_pressure_single|kcmm_bench_memory_pressure|Benchmark 5a — Memory Pressure Single Config (baseline vs KCMM tiering)"
    "kcmm_bench_memory_pressure_sweep|kcmm_bench_memory_pressure|Benchmark 5b — Memory Pressure Sweep (block size × pool capacity × prompt dist)"

    # Benchmark §1.6: Engine Integration — LlamaTransformer + KcmmPool continuous batching
    "kcmm_engine_integration_single|kcmm_bench_engine_integration|Benchmark §1.6a — Engine Integration Single Config (throughput, P50/P99, thrashing)"
    "kcmm_engine_integration_sweep|kcmm_bench_engine_integration|Benchmark §1.6b — Engine Integration Sweep (4 configs, OFF vs ON)"
)

# If filter is set, only keep matching tests
if [ -n "$FILTER" ]; then
    echo "Filter: '$FILTER'"
    echo ""
    NEW_BENCHMARKS=()
    for entry in "${BENCHMARKS[@]}"; do
        if [[ "$entry" == *"$FILTER"* ]]; then
            NEW_BENCHMARKS+=("$entry")
        fi
    done
    if [ ${#NEW_BENCHMARKS[@]} -eq 0 ]; then
        echo "ERROR: No tests match filter '$FILTER'"
        echo "Available tests:"
        for entry in "${BENCHMARKS[@]}"; do
            IFS='|' read -r name file desc <<< "$entry"
            echo "  $name"
        done
        exit 1
    fi
    BENCHMARKS=("${NEW_BENCHMARKS[@]}")
    echo "Running ${#BENCHMARKS[@]} test(s) matching filter"
    echo ""
fi

# ── Compile everything once ──
echo "=== Building (this may take a while for release) ==="

# Build all three test binaries in one cargo invocation to save time
TEST_NAMES=$(for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name file desc <<< "$entry"
    echo "--test $file"
done | sort -u | tr '\n' ' ')

echo "Building test binaries: $TEST_NAMES"
# shellcheck disable=SC2086
if [ "$PROFILE" = "release" ]; then
    cargo test $FEATURES --release $TEST_NAMES --no-run 2>&1 | tail -10
else
    cargo test $FEATURES $TEST_NAMES --no-run 2>&1 | tail -10
fi

echo ""
echo "=== Running ${#BENCHMARKS[@]} benchmark(s) ==="
echo ""

# ── Run each test individually ──
PASSED=0
FAILED=0
declare -a FAILED_TESTS=()

for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r test_name test_file desc <<< "$entry"

    echo "──────────────────────────────────────────────────────"
    echo "[${test_name}]"
    echo "  $desc"
    echo "──────────────────────────────────────────────────────"

    LOG_FILE="$RESULTS_DIR/${test_name}.log"

    set +e
    if [ "$PROFILE" = "release" ]; then
        cargo test $FEATURES --release --test "$test_file" "$test_name" -- --nocapture \
            > "$LOG_FILE" 2>&1
    else
        cargo test $FEATURES --test "$test_file" "$test_name" -- --nocapture \
            > "$LOG_FILE" 2>&1
    fi
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  => PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  => FAILED (exit code $EXIT_CODE)"
        echo "  => Log: $LOG_FILE"
        # Print last 20 lines of log for quick diagnosis
        echo "  => Last 20 lines:"
        tail -20 "$LOG_FILE" | sed 's/^/       /'
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name")
    fi
    echo ""
done

# ── Summary ──
{
    echo "=============================================="
    echo " KCMM Benchmark Run Summary"
    echo "=============================================="
    echo " Date:       $(date)"
    echo " GPU:        $GPU_NAME"
    echo " VRAM:       ${VRAM} MiB"
    echo " Profile:    $PROFILE"
    echo " Features:   $FEATURES"
    echo " Results:    $RESULTS_DIR"
    echo ""
    echo " Results:"
    echo "   Passed: $PASSED"
    echo "   Failed: $FAILED"
    echo ""
    if [ $FAILED -gt 0 ]; then
        echo " Failed tests:"
        for t in "${FAILED_TESTS[@]}"; do
            echo "   - $t"
        done
    fi
    echo ""
    echo " Per-test logs:"
    for entry in "${BENCHMARKS[@]}"; do
        IFS='|' read -r name file desc <<< "$entry"
        echo "   ${name}.log"
    done
    echo "=============================================="
} | tee "$RESULTS_DIR/summary.txt"

echo ""
echo "All results saved to: $RESULTS_DIR"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
