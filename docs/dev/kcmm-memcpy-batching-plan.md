# KCMM Memcpy Batching — Implementation Plan

**Date:** 2026-06-07
**Branch:** `kcmm`
**Based on:** Commit `f188f7f` (three-phase eviction/restore with batch synchronize amortisation)
**Reference:** [Phase E Benchmark Analysis §7.1](../report/kcmm-phase-e-benchmark-analysis.md#71-immediate-this-week)

---

## 1. Motivation

The synchronize batching fix (`f188f7f`) eliminated the per-block `cuStreamSynchronize`
overhead (~15% of per-block eviction cost), delivering a 31% improvement at batch=4 (262
→ 180 µs/block). However, the amortisation curve **plateaus beyond batch=4** because the
dominant bottleneck — per-memcpy CUDA driver call overhead (~53%, ~108 µs/block) — is
unaffected by synchronize batching alone.

The current approach issues `4 × batch_size` individual `cuMemcpyDtoHAsync` calls (K0, V0,
K1, V1 per block). Each call incurs ~27 µs of CUDA driver overhead on WSL2. This plan
describes how to reduce this to a **fixed 4 calls** regardless of batch size, by gathering
same-layer KV data from scattered GPU addresses into contiguous staging buffers before each
transfer.

### 1.1 Projected Impact

| Batch Size | Current (µs/block) | Target (µs/block) | Improvement |
|---|---|---|---|
| 4 | 180 | ~90 | −50% |
| 16 | 205 | ~45 | −78% |
| 64 | 199 | ~25 | −87% |

---

## 2. Technical Approach

### 2.1 The Core Problem

KV data for different blocks resides at **non-contiguous GPU virtual addresses** (different
superblocks, different offsets within superblocks). A single `cuMemcpyDtoHAsync` can only
transfer from a contiguous source range. To issue fewer, larger transfers, we must first
**gather** scattered block data into a contiguous GPU region.

### 2.2 Architecture: Gather Kernel + Batched Transfer

```
┌─────────────────────────────────────────────────────────────────┐
│  Eviction (D2H): Batch of N blocks                               │
│                                                                  │
│  Phase 1 — GPU Gather + Async D2H:                              │
│    For each layer ∈ {K0, V0, K1, V1}:                           │
│      a. Launch gather_kernel: copy same-layer data from         │
│         N scattered GPU VAs → contiguous GPU staging buffer      │
│      b. memcpy_d2h_async: staging buffer → CPU staging buffer   │
│                                                                  │
│  Phase 2 — Synchronize:                                         │
│    One cuStreamSynchronize on evict stream                       │
│                                                                  │
│  Phase 3 — CPU Scatter + Finalize:                              │
│    memcpy: CPU staging → each block's CPU slot                   │
│    release_block_physical(), set_block_location(CpuResident)     │
└─────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────┐
│  Restore (H2D): Batch of N blocks                                │
│                                                                  │
│  Phase 1 — CPU Gather + Async H2D:                              │
│    For each layer ∈ {K0, V0, K1, V1}:                           │
│      a. memcpy: scatter from each block's CPU slot →             │
│         contiguous CPU staging buffer                            │
│      b. memcpy_h2d_async: CPU staging → GPU staging buffer      │
│      c. Launch scatter_kernel: GPU staging → each block's       │
│         allocated GPU VA                                         │
│                                                                  │
│  Phase 2 — Synchronize:                                         │
│    One cuStreamSynchronize on restore stream                     │
│                                                                  │
│  Phase 3 — Finalize:                                            │
│    set_block_location(GpuResident), free_cpu_slot(), etc.       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Staging Buffer Sizing

- **GPU staging buffer**: `max_batch_blocks × block_bytes` bytes (single-layer data)
  - At configurable `max_batch = 64`, `block_bytes = 65536`: **4 MiB**
  - Allocated once at `TieringEngine` construction, reused across batches
  - Only needs to hold one layer at a time (we process layers sequentially)

- **CPU staging buffer**: same size (4 MiB), also allocated once at construction
  - Used for both eviction (D2H target) and restore (H2D source)

### 2.4 Gather/Scatter CUDA Kernel

A simple element-wise copy kernel that gathers from scattered sources to a contiguous
destination, or scatters from contiguous source to scattered destinations.

```cuda
// Gather: scattered sources → contiguous destination
// Each block contributes block_bytes bytes from its source VA
extern "C" __global__ void gather_kv_layer(
    const __half * __restrict__ const *src_ptrs,  // N pointers to block K/V data
    __half *dst,                                    // contiguous staging buffer
    int block_bytes,                                // bytes per block
    int num_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = num_blocks * (block_bytes / sizeof(__half));
    if (tid >= total_elems) return;

    int blk_idx = tid / (block_bytes / sizeof(__half));
    int offset  = tid % (block_bytes / sizeof(__half));
    dst[tid] = src_ptrs[blk_idx][offset];
}

// Scatter: contiguous source → scattered destinations
extern "C" __global__ void scatter_kv_layer(
    const __half *src,                               // contiguous staging buffer
    __half * __restrict__ const *dst_ptrs,           // N pointers to block GPU VAs
    int block_bytes,
    int num_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = num_blocks * (block_bytes / sizeof(__half));
    if (tid >= total_elems) return;

    int blk_idx = tid / (block_bytes / sizeof(__half));
    int offset  = tid % (block_bytes / sizeof(__half));
    dst_ptrs[blk_idx][offset] = src[tid];
}
```

**Kernel launch overhead**: ~3–5 µs per launch (× 4 launches per batch = ~20 µs total),
amortised to < 1 µs/block at batch ≥ 16. Negligible compared to the ~108 µs/block saved.

### 2.5 Why This Beats Per-Block Memcpy

| Approach | Driver calls (batch=N) | Driver overhead (batch=64) |
|---|---|---|
| Per-block memcpy (current) | 4N | 256 × 27 µs = **6,912 µs** |
| Gather kernel + batch D2H | 4 | 4 × 27 µs = **108 µs** |
| **Saving** | | **6,804 µs** (106 µs/block) |

The gather kernel's data movement is GPU-internal (GDDR6 bandwidth ~400+ GB/s). For a
batch=64 × 64 KiB layer, that's 4 MiB copied at ~400 GB/s ≈ 10 µs. The kernel launch
overhead (~5 µs) plus device-to-device copy (~10 µs) is far cheaper than 64 individual
driver calls (64 × 27 = 1,728 µs per layer).

---

## 3. Task Breakdown

### Task 1: Write gather/scatter CUDA kernels

**File:** `src/cuda/kernels/kv_gather.cu` (new)

- Implement `gather_kv_layer` kernel: `N` scattered `__half*` sources → contiguous destination
- Implement `scatter_kv_layer` kernel: contiguous source → `N` scattered `__half*` destinations
- Handle edge cases: empty batch, misaligned sizes

**Effort:** Small (simple element-wise copy kernels, ~30 lines each)

### Task 2: Integrate kernels into `GpuKernels`

**Files:** `src/cuda/kernels/mod.rs`, `src/cuda/mod.rs`

- Add `gather_kv` and `scatter_kv` to `GpuKernels` struct
- Add NVRTC compilation entries
- Add `launch_kv_gather()` and `launch_kv_scatter()` wrapper functions
- The `src_ptrs`/`dst_ptrs` parameter is an array of device pointers — pass as a
  `CudaSlice<u64>` (CUdeviceptr values)

**Effort:** Small (follow existing kernel patterns)

### Task 3: Add staging buffers to `TieringEngine`

**File:** `src/kcmm/tiering.rs`

- Add fields to `TieringEngine`:
  - `gpu_staging: CudaSlice<u8>` — GPU staging buffer (max 4 MiB)
  - `cpu_staging: Vec<u8>` — CPU staging buffer mirror
  - `gather_kernel: CudaFunction` — reference to the gather kernel
  - `scatter_kernel: CudaFunction` — reference to the scatter kernel
- Initialize in `TieringEngine::new()` — allocate both staging buffers
- Sizing: `max_batch_blocks × block_bytes`, where `max_batch_blocks` is a new config
  field (default 64)

**Effort:** Small (add fields + init)

### Task 4: Implement batched D2H eviction path

**File:** `src/kcmm/tiering.rs`

- New function `evict_batch_d2h_batched()`:
  - Phase 1a: For each layer (K0, V0, K1, V1):
    - Build `src_ptrs: Vec<u64>` of GPU device pointers for that layer across all blocks
    - Upload `src_ptrs` to a temporary GPU slice
    - Launch `gather_kv_layer` kernel (on evict stream)
    - Issue **one** `memcpy_d2h_async` from GPU staging → CPU staging
    - Wait for D2H (sync or event), then CPU-side memcpy scatter to each block's slot
  - Phase 2: One `cuStreamSynchronize` (already done)
  - Phase 3: Finalize blocks (already done)
- Integrate into `evict_blocks()` — use batched path when `pending.len() >= MIN_BATCH`,
  fall back to per-block path for small batches

**Effort:** Medium (core logic ~150 lines)

### Task 5: Implement batched H2D restore path

**File:** `src/kcmm/tiering.rs`

- New function `restore_batch_h2d_batched()`:
  - Phase 1a: For each layer:
    - CPU-side memcpy gather from each block's CPU slot → CPU staging buffer
    - **One** `memcpy_h2d_async` from CPU staging → GPU staging
    - Build `dst_ptrs` array of GPU VAs for each block's layer
    - Launch `scatter_kv_layer` kernel (on restore stream)
  - Phase 2: One `cuStreamSynchronize`
  - Phase 3: Finalize blocks
- Mirror of Task 4, with H2D direction and scatter instead of gather

**Effort:** Medium (mirror of Task 4, ~120 lines)

### Task 6: Add `KcmmConfig` field for max batch size

**File:** `src/config.rs`

- Add `max_batch_blocks: usize` field (default 64)
- Controls staging buffer allocation size
- Used by both eviction and restore batched paths

**Effort:** Trivial

### Task 7: Add unit tests for gather/scatter kernels

**File:** `src/kcmm/streams.rs` (or new test file)

- Test `gather_kv_layer`: scattered sources → contiguous, verify with readback
- Test `scatter_kv_layer`: contiguous → scattered destinations, verify per-destination
- Test roundtrip: gather → scatter → verify identity
- Test edge cases: single block, max batch, misaligned sizes

**Effort:** Small (~100 lines)

### Task 8: Re-run Benchmark 2b and compute amortisation factor

**File:** `tests/kcmm_bench_tiering.rs`

- Add amortisation factor computation to Benchmark 2b output:
  `amortisation = per_block_latency_at_batch_1 / per_block_latency_at_batch_N`
- Re-run with batched memcpy enabled
- Validate that the amortisation curve is now monotonically decreasing
- Document results

**Effort:** Small (formula is straightforward; re-running is automated)

### Task 9: Update benchmark report

**File:** `docs/report/kcmm-phase-e-benchmark-report.md`

- Add a "Batch 2b (Rerun with Memcpy Batching)" section
- Compare old vs new amortisation curves
- Update success criteria traceability matrix

**Effort:** Small

---

## 4. Execution Order and Dependencies

```
Task 1 (kernels) ──→ Task 2 (GpuKernels) ──→ Task 3 (staging buffers)
                                                   │
                          ┌─────────────────────────┤
                          ▼                         ▼
                   Task 4 (evict D2H)        Task 5 (restore H2D)
                          │                         │
                          └─────────┬───────────────┘
                                    ▼
                             Task 7 (unit tests)
                                    │
                                    ▼
                             Task 8 (benchmark re-run)
                                    │
                                    ▼
                             Task 9 (report update)

Task 6 (config) can be done in parallel with Tasks 1–3.
```

**Critical path:** Tasks 1 → 2 → 3 → 4 → 7 → 8 (Tasks 4 and 5 can be parallelized between
two developers).

---

## 5. Estimated Effort

| Task | Description | Est. Hours | Risk |
|---|---|---|---|
| 1 | Write gather/scatter kernels | 2 | Low |
| 2 | Integrate into GpuKernels | 1.5 | Low |
| 3 | Add staging buffers | 1.5 | Low |
| 4 | Batched D2H eviction | 4 | Medium |
| 5 | Batched H2D restore | 3 | Medium |
| 6 | KcmmConfig field | 0.5 | Low |
| 7 | Unit tests | 2 | Low |
| 8 | Benchmark re-run | 1.5 | Low |
| 9 | Report update | 1 | Low |
| **Total** | | **17 hours** (~3 working days) | |

**Risk factors:**
- CUDA kernel debugging on WSL2 may require additional iteration cycles
- Staging buffer size must be carefully coordinated with `max_batch_blocks` config
- Synchronize between gather kernel and D2H within the same stream requires correct
  stream ordering (kernels and memcpys on the same stream are implicitly ordered)

---

## 6. Design Decisions

### 6.1 Why GPU gather kernel instead of `cuMemcpyDtoD`?

Individual `cuMemcpyDtoD` calls are still CUDA driver calls (~27 µs each). Using D2D to
gather scattered data before a batched D2H would still incur `N` driver calls per layer
(just D2D instead of D2H), providing no reduction in driver call count.

A single GPU kernel launch (~3–5 µs) replaces `N` driver calls (~27N µs), delivering the
driver call reduction that is the primary goal of this optimization.

### 6.2 Why per-layer sequential instead of all layers at once?

Processing layers sequentially (K0 → V0 → K1 → V1) keeps the staging buffer to
`batch × block_bytes` (4 MiB) instead of `batch × 4 × block_bytes` (16 MiB). The
sequential overhead is negligible (< 1 µs CPU time per iteration) compared to the
memory savings.

### 6.3 Why separate GPU and CPU staging buffers?

The GPU staging buffer enables batched D2H/H2D calls. The CPU staging buffer provides a
contiguous target/source for these calls. The alternative — having each block's CPU slot
be contiguous within the batch — would require restructuring the `CpuSlotAllocator` to
support batch-aware allocation, which is more invasive and limits flexibility.

### 6.4 Fallback behavior

For small batches (1–3 blocks), the kernel launch overhead (~5 µs × 4 layers = 20 µs) may
exceed the savings from reduced driver calls. The batched path should be used when
`batch_size >= MIN_BATCH_FOR_GATHER` (configurable, default 4), falling back to the
existing per-block path for smaller batches.

---

## 7. Success Criteria

| Criterion | Target | Measurement |
|---|---|---|
| Eviction per-block P50 at batch=4 | < 100 µs | Benchmark 2b |
| Eviction per-block P50 at batch=16 | < 60 µs | Benchmark 2b |
| Eviction per-block P50 at batch=64 | < 40 µs | Benchmark 2b |
| Amortisation curve monotonic | per-block latency ↓ as batch ↑ | Benchmark 2b |
| Restore per-block P50 | Not regressed (> 200 µs) | Benchmark 2a |
| Data integrity roundtrip | 100% pass rate | Benchmark 2d |
| No regression in single-block path | Within 5% of current | Benchmark 2a |
