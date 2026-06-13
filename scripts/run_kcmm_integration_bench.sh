#!/usr/bin/env bash
# ==============================================================================
# scripts/run_kcmm_integration_bench.sh — KCMM §1.6 Engine Integration Benchmark
#
# Exercises LlamaTransformer + KvCacheBackend through a simulated
# continuous-batching workload, comparing PagedKvCache (baseline) against
# KcmmPool (tiering ON).  Measures throughput, per-step latency distribution,
# eviction/restore counts, and capacity ratio.
#
# Usage:
#   ./scripts/run_kcmm_integration_bench.sh              # Run both tests (release)
#   ./scripts/run_kcmm_integration_bench.sh --debug      # Use debug build (faster compile)
#   ./scripts/run_kcmm_integration_bench.sh --release    # Optimised build (default)
#   ./scripts/run_kcmm_integration_bench.sh --single     # Single config only
#   ./scripts/run_kcmm_integration_bench.sh --sweep      # Sweep only
#   ./scripts/run_kcmm_integration_bench.sh --filter <s> # Run tests matching <s>
#
# Output:
#   results/kcmm_engine_integration_<timestamp>/  — per-test logs + summary
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# ── Argument parsing ──
PROFILE="release"
MODE="both"   # both | single | sweep
FILTER=""
POLICY=""     # eviction policy override (lru, lfu, arc, sink_window)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            PROFILE="release"
            shift
            ;;
        --debug)
            PROFILE="debug"
            shift
            ;;
        --single)
            MODE="single"
            shift
            ;;
        --sweep)
            MODE="sweep"
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--release|--debug] [--single|--sweep] [--filter <substring>] [--policy <name>]"
            echo ""
            echo "KCMM §1.6 Engine Integration Benchmark"
            echo "  LlamaTransformer + KcmmPool continuous batching workload"
            echo ""
            echo "Options:"
            echo "  --release   Build with optimisations (default, slower compile, faster runtime)"
            echo "  --debug     Build without optimisations (faster compile, slower runtime)"
            echo "  --single    Run single-config benchmark only (~4 min; 5 repeats)"
            echo "  --sweep     Run parameter sweep only (~10-12 min; 5 repeats/config)"
            echo "  --filter    Only run tests whose name contains this substring"
            echo "  --policy    Eviction policy override: lru (default), lfu, arc, sink_window"
            echo ""
            echo "Examples:"
            echo "  $0                            # Full benchmark (single + sweep), release build (default)"
            echo "  $0 --single                   # Quick single-config run, release build (default)"
            echo "  $0 --policy arc --single      # Single-config with ARC policy"
            echo "  $0 --debug                    # Full benchmark, debug build"
            echo "  $0 --single --debug           # Quick single-config run, debug build"
            echo "  $0 --filter single            # Only the single test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Prerequisites ──
echo "=== KCMM §1.6 Engine Integration Benchmark ==="
echo ""

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA device required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "0")
echo "GPU:    $GPU_NAME"
echo "VRAM:   ${VRAM} MiB"
echo ""

# ── Timestamp and output directory ──
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJ_DIR/results/kcmm_engine_integration_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

if [ -n "$POLICY" ]; then
    export KCMM_POLICY="$POLICY"
    echo "Policy:  $POLICY"
fi
echo "Results: $RESULTS_DIR"
echo "Profile: $PROFILE"
echo "Mode:    $MODE"
echo ""

# ── Build flags ──
FEATURES="--features kcmm"
TEST_FILE="kcmm_bench_engine_integration"
CARGO_TEST="cargo test $FEATURES"
if [ "$PROFILE" = "release" ]; then
    CARGO_TEST="$CARGO_TEST --release"
fi

# ── Test list ──
declare -a TESTS=()

if [ "$MODE" = "both" ] || [ "$MODE" = "single" ]; then
    TESTS+=("kcmm_engine_integration_single|Single Config — detailed P50/P90/P95/P99, eviction metrics")
fi
if [ "$MODE" = "both" ] || [ "$MODE" = "sweep" ]; then
    TESTS+=("kcmm_engine_integration_sweep|Parameter Sweep — 4 configs × 5 repeats, block size × pool capacity × prompt dist")
fi

# Apply filter
if [ -n "$FILTER" ]; then
    echo "Filter:  '$FILTER'"
    NEW_TESTS=()
    for entry in "${TESTS[@]}"; do
        if [[ "$entry" == *"$FILTER"* ]]; then
            NEW_TESTS+=("$entry")
        fi
    done
    if [ ${#NEW_TESTS[@]} -eq 0 ]; then
        echo "ERROR: No tests match filter '$FILTER'"
        echo "Available tests:"
        for entry in "${TESTS[@]}"; do
            IFS='|' read -r name desc <<< "$entry"
            echo "  $name — $desc"
        done
        exit 1
    fi
    TESTS=("${NEW_TESTS[@]}")
fi

echo "Tests:   ${#TESTS[@]} test(s)"
echo ""

# ── Compile ──
echo "=== Building ($PROFILE) ==="

if [ "$PROFILE" = "release" ]; then
    cargo test $FEATURES --release --test "$TEST_FILE" --no-run 2>&1 | tail -5
else
    cargo test $FEATURES --test "$TEST_FILE" --no-run 2>&1 | tail -5
fi
echo ""

# ── Run ──
echo "=== Running ${#TESTS[@]} test(s) ==="
echo ""

PASSED=0
FAILED=0
declare -a FAILED_TESTS=()

for entry in "${TESTS[@]}"; do
    IFS='|' read -r test_name desc <<< "$entry"

    echo "──────────────────────────────────────────────────────────────"
    echo "[$test_name]"
    echo "  $desc"
    echo "──────────────────────────────────────────────────────────────"

    LOG_FILE="$RESULTS_DIR/${test_name}.log"

    set +e
    if [ "$PROFILE" = "release" ]; then
        cargo test $FEATURES --release --test "$TEST_FILE" "$test_name" -- --nocapture \
            > "$LOG_FILE" 2>&1
    else
        cargo test $FEATURES --test "$TEST_FILE" "$test_name" -- --nocapture \
            > "$LOG_FILE" 2>&1
    fi
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  => PASSED"

        # Extract key metrics for quick summary
        echo "  => Quick metrics:"
        grep -E '(Throughput ratio|Capacity ratio|Per-step overhead|KCMM tiering active|✅|⚡|❌|Best throughput)' \
            "$LOG_FILE" | sed 's/^/       /' || true

        PASSED=$((PASSED + 1))
    else
        echo "  => FAILED (exit code $EXIT_CODE)"
        echo "  => Log: $LOG_FILE"
        echo "  => Last 30 lines:"
        tail -30 "$LOG_FILE" | sed 's/^/       /'
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name")
    fi
    echo ""
done

# ── Summary ──
{
    echo "=============================================="
    echo " KCMM §1.6 Engine Integration Benchmark"
    echo " Summary"
    echo "=============================================="
    echo " Date:       $(date)"
    echo " GPU:        $GPU_NAME"
    echo " VRAM:       ${VRAM} MiB"
    echo " Profile:    $PROFILE"
    echo " Mode:       $MODE"
    echo " Results:    $RESULTS_DIR"
    echo ""
    echo " Tests run:"
    for entry in "${TESTS[@]}"; do
        IFS='|' read -r name desc <<< "$entry"
        echo "   - $name  ($desc)"
    done
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
    for entry in "${TESTS[@]}"; do
        IFS='|' read -r name desc <<< "$entry"
        echo "   ${name}.log"
    done
    echo "=============================================="
} | tee "$RESULTS_DIR/summary.txt"

echo ""
echo "All results saved to: $RESULTS_DIR"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
