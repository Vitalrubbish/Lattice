# KCMM Next Steps Without Bare-Metal Server

**Date:** 2026-06-09
**Status:** Draft
**Related documents:**
- `docs/dev/kcmm-baremetal-plan.md` — Phase 1 plan (1a/1b/1c)
- `docs/dev/kcmm-measurement-methodology-fix.md` — measurement methodology revision
- `docs/task/kcmm-implementation-analysis.md` — implementation targets

---

## 0. Context

The d7525 bare-metal server (2× AMD EPYC 7302, NVIDIA A30 24GB) is currently
unavailable. This document analyses which Phase 1 tasks are blocked and proposes
a concrete execution plan using only the WSL2 development environment.

**Current working tree state** (commit `0b23fbb`, branch `kcmm`):

| Change | Status |
|--------|--------|
| Phase 1a — Threshold 4→8 (`tiering.rs`) | Done (uncommitted) |
| Measurement methodology — stats helpers, increased sampling, warmup | Mostly done (uncommitted) |
| `tests/bench_utils.rs` — shared statistics module | Created (uncommitted) |
| Phase 1b — Bare-metal benchmarks on d7525 | **Blocked** |
| Phase 1c — Memory pressure integration benchmark | Not started |

---

## 1. Priority Stack

### P0-1 — Commit Working Tree Changes

**Effort:** ~30 min
**Feasible on WSL2:** Yes

The working tree has ~127 lines of coherent changes across two logical units:

1. **Threshold fix** — `src/kcmm/tiering.rs`: `MIN_BATCH_FOR_GATHER` and
   `MIN_BATCH_FOR_SCATTER` raised from 4 to 8.
2. **Measurement methodology** — `tests/bench_utils.rs` (new) + updates to
   `tests/kcmm_bench_tiering.rs`, `tests/kcmm_bench_alloc.rs`, and
   `tests/step3_benchmarks.rs`.

These should be committed as two separate commits so the threshold fix and the
methodology changes are independently bisectable.

### P0-2 — Re-Run Amortisation Benchmarks to Verify Threshold Fix

**Effort:** ~30 min
**Feasible on WSL2:** Yes

After committing, confirm the threshold fix works as expected:

```bash
cargo test --features kcmm --release \
    kcmm_bench_batch_eviction_amortization \
    kcmm_bench_batch_restore_amortization \
    -- --nocapture
```

Expected outcomes:
- **batch=4** now uses the sequential path — per-block cost should match batch=1
  (no regression).
- **batch=16 and batch=64** still use the batched path — amortisation factors
  unchanged (2.03× and 2.18× respectively).
- The new output format (`mean ± stddev [min, P50, P99, max] n=N SE=±X`) lets
  us assess statistical significance directly from the report.

### P0-3 — Phase 1c: Memory Pressure Integration Benchmark

**Effort:** 3–5 days
**Feasible on WSL2:** Yes

This is the highest-value remaining P0 task. The implementation plan is in
`docs/dev/kcmm-baremetal-plan.md` §3.

**What it does:**
- Compares `PagedKvCache` (baseline, no swap) vs `KcmmPool` (tiering ON) under
  a synthetic inference workload.
- Measures the maximum number of concurrent sequences that can be admitted and
  grown to completion without OOM-capping.
- Primary metric: `capacity_ratio = KCMM_admitted / Baseline_admitted`, target ≥1.3×.

**Key design decisions:**
1. New file: `tests/kcmm_bench_memory_pressure.rs` (~400–600 lines).
2. Both configs use the same model geometry (TinyLlama: 22 layers, 4 kv_heads, 64 head_dim).
3. Uses tempfile-backed CPU buffer (same pattern as existing benchmarks).
4. No GPU kernel execution — tests the allocator and tiering engine only.
5. Sweeps across block sizes, pool capacities, and prompt length distributions.

**Why it matters:**
This benchmark directly tests KCMM's value proposition: "how many more
concurrent requests does tiering enable?" The existing micro-benchmarks
measure component-level latency, not end-to-end capacity improvement under
memory pressure. This is the benchmark that answers the core evaluation
question.

### P1-1 — Add batch=8 Data Point

**Effort:** ~1 hour
**Feasible on WSL2:** Yes

The current amortisation benchmarks sweep [1, 4, 16, 64]. Adding batch=8
verifies it's above breakeven with the new threshold and provides a finer-grained
amortisation curve. This is a small addition to the existing test loops in
`tests/kcmm_bench_tiering.rs`.

Expected result per the plan:
```
evict_batch=8:  ~150 µs/block  (modest improvement over 201 µs baseline)
restore_batch=8: ~120 µs/block  (modest improvement over 155 µs baseline)
```

### P1-2 — Run Full Suite & Document WSL2 Baseline

**Effort:** ~2 hours
**Feasible on WSL2:** Yes

After P0 items are complete, run the full 11-test benchmark suite and capture
a definitive WSL2 baseline with proper statistics (mean/stddev/SE for every
metric). This provides:
- A reference point for whenever bare-metal becomes available.
- Statistical confidence in the threshold fix.
- Initial memory pressure results.

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/kcmm_bench_wsl2_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

cargo test --features kcmm --release \
    --test kcmm_bench_alloc \
    --test kcmm_bench_tiering \
    --test step3_benchmarks \
    -- --nocapture 2>&1 | tee "$RESULTS_DIR/full_suite.log"
```

### P1-3 — Write Phase 1 Report

**Effort:** ~2 hours
**Feasible on WSL2:** Yes

Write `docs/dev/kcmm-phase1-report.md` covering all available results:
- Threshold fix verification
- Updated WSL2 benchmarks with full statistics
- Memory pressure sweep results
- Bare-metal section as a placeholder for future d7525 data
- Go/no-go recommendation for Phase 2

---

## 2. Long-Term: Bridging the Bare-Metal Gap

Phase 1b requires a bare-metal Linux machine with an NVIDIA GPU. The key
questions it answers:

| Question | Why WSL2 isn't enough |
|----------|----------------------|
| Is 128KB restore P50 < 200µs? | WSL2 GPU-PV adds ~50µs overhead per cuMemMap call |
| What is native `cuMemMap` latency? | WSL2: 167µs; native expected ~100–150µs |
| Are P99 tails representative? | WSL2 GPU-PV introduces spurious jitter |
| Is stream isolation overhead <1%? | WSL2 may mask or inflate real PCIe contention |

### Option A — Colleague with d7525 Access

The simplest path. The plan (`kcmm-baremetal-plan.md` §2.2–2.3) has exact
copy-paste commands. Someone with SSH access can run the 10-minute suite and
send back the log file.

### Option B — Any Bare-Metal Linux + NVIDIA GPU

The benchmarks are portable across any CUDA-capable GPU. Even a desktop with
a consumer card (RTX 3060+, 12GB+) would provide better numbers than WSL2's
GPU-PV layer. The absolute latencies will differ (different GPU generation),
but the critical ratios — amortisation factors, stream interference overhead,
memory pressure capacity improvement — will translate.

### Option C — Cloud GPU Instance

Instances with GPU passthrough (not nested virtualization):

| Provider | Instance | GPU | VRAM | ~Cost/hr |
|----------|----------|-----|------|----------|
| AWS | `p3.2xlarge` | V100 | 16 GB | ~$3.00 |
| AWS | `g4dn.xlarge` | T4 | 16 GB | ~$0.50 |
| GCP | `a2-highgpu-1g` | A100 | 40 GB | ~$3.50 |
| Lambda Labs | `a10-pcie-1` | A10 | 24 GB | ~$0.75 |

The full benchmark suite runs in ~10 minutes once built. Total cloud cost for
a benchmark run: < $5.

### Option D — Start Phase 2 Without Bare-Metal Data

Phase 2 (C FFI API + vLLM integration) is mostly software engineering and can
begin on WSL2. Bare-metal is needed only for final performance validation, not
for development. The WSL2 numbers already validate the core architecture.

---

## 3. What NOT to Do

1. **Don't wait for d7525 to start Phase 1c.** The memory pressure benchmark
   is independently valuable — it quantifies KCMM's core value proposition.
   WSL2 numbers will show the direction and rough magnitude; bare-metal will
   only improve absolute latencies.

2. **Don't leave the working tree uncommitted.** The threshold fix +
   measurement methodology changes are coherent and tested. They should be
   committed so further work builds on clean history.

3. **Don't skip the measurement methodology.** The new statistics (mean ±
   stddev, SE, unified output) are essential for interpreting results with
   confidence. Without them, it is impossible to determine whether a measured
   difference is signal or noise.

4. **Don't gate Phase 2 planning on bare-metal data.** The architecture is
   validated. The C API design, vLLM integration strategy, and Phase 2
   work breakdown can all proceed from the WSL2 baseline.

---

## 4. Summary

| Priority | Task | Feasible on WSL2? | Effort |
|----------|------|-------------------|--------|
| **P0** | Commit working tree changes | Yes | 30 min |
| **P0** | Re-run amortisation benchmarks (verify threshold fix) | Yes | 30 min |
| **P0** | Phase 1c: Memory pressure integration benchmark | Yes | 3–5 days |
| **P1** | Add batch=8 data point | Yes | 1 hour |
| **P1** | Run full suite, document WSL2 baseline | Yes | 2 hours |
| **P1** | Write Phase 1 report (bare-metal as placeholder) | Yes | 2 hours |
| **P2** | Phase 1b: Bare-metal benchmarks on d7525 | **No** — blocked | 2–4 hours |
| **P2** | Phase 2: C API + vLLM integration planning | Yes | TBD |

**~80% of Phase 1 work is unblocked.** The bare-metal data is important for
the final evaluation report but is not a prerequisite for forward progress.
