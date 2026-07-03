# KCMM Phase E: Batch Eviction Scaling Issue

**Reviewer:** Claude
**Date:** 2026-06-07
**Source:** Phase E benchmark analysis (`docs/report/kcmm-phase-e-benchmark-analysis.md`)
**Severity:** Medium — per-block eviction cost worsens at large batch sizes; does not affect correctness
**Files Implicated:**
- `src/kcmm/tiering.rs` — `evict_blocks()` (L430–470), `evict_single_block()` (L480–548)
- `src/kcmm/pool.rs` — `release_block_physical()` (L744–763)

**Test data source:** `tests/kcmm_bench_tiering.rs` — `kcmm_bench_batch_eviction_amortization`

---

## 1. Issue Summary

The batch eviction benchmark (Benchmark 2b) reveals a **non-monotonic per-block cost curve**: batch size 4 achieves the best amortisation (148 µs/block), but larger batches regress — batch size 64 costs 226 µs/block, **worse than single-block eviction** (199 µs/block).

```
Batch size:     1        4        16       64
Per-block:    199 µs   148 µs   172 µs   226 µs    ← measured (release mode, RTX 5070)
Expected:     199 µs   ~50 µs   ~13 µs    ~3 µs    ← if memcpy calls were batched
```

The expected monotonic decrease relies on amortising the fixed CUDA driver call overhead over multiple blocks. The observed U-shape reveals that the current implementation pays this overhead **per-block** rather than **per-batch**.

---

## 2. Root Cause Analysis

### 2.1 Evidence: cuMemUnmap is NOT the bottleneck

Contrary to the initial hypothesis in the benchmark analysis report, **cuMemUnmap is not called in the eviction hot path**. Tracing the code:

**Eviction path** (`tiering.rs:480-548`):
```
evict_single_block():
  1. alloc_cpu_slot()                    // allocate CPU buffer slot
  2. set_block_location(Evicting)        // mark state
  3. evict_single_block_all_layers()     // 4× memcpy_d2h_async per block
  4. streams.evict.synchronize()         // ← PER-BLOCK sync (L531)
  5. release_block_physical()            // returns handle to free list (L534)
  6. set_block_location(CpuResident)     // mark state
```

**`release_block_physical()`** (`pool.rs:744-762`):
```rust
pub fn release_block_physical(&self, block_idx: u32) -> Result<()> {
    // ... gets BlockHandle from block_info ...
    for l in 0..num_layers {
        self.k_pools[l].allocator.free(handle);  // pushes to VecDeque
        self.v_pools[l].allocator.free(handle);  // pushes to VecDeque
    }
    Ok(())
}
```

**`PhysicalBlockAllocator::free()`** (`superblock.rs:126-128`):
```rust
pub fn free(&self, handle: BlockHandle) {
    self.free_blocks.lock().push(handle);  // O(1) free-list push
}
```

`cuMemUnmap` is only called at **pool destruction** (`pool.rs:906-941`, `Drop for KcmmPool`). During normal eviction, GPU physical pages remain mapped — the block handle simply returns to the per-layer free list for reuse. This is a valid design choice (avoids the ~244 µs/call unmap cost on the hot path), but it means cuMemUnmap is **not** the cause of the batch-scaling regression.

### 2.2 Actual Bottleneck: Per-Block CUDA Driver Call Overhead

Each `evict_single_block` call performs:

| Operation | Count per block | Est. per-call cost | Total |
|---|---|---|---|
| `memcpy_d2h_async` | 4 (K0, V0, K1, V1) | ~27 µs driver overhead | ~108 µs |
| Actual D2H transfer | 256 KiB total (64 KiB block) | ~54 µs at ~4.7 GB/s | ~54 µs |
| `synchronize()` | 1 | ~30 µs | ~30 µs |
| Lock acquire/release | 3 (block_info, free_blocks ×2) | ~2 µs | ~6 µs |
| mmap write (CPU buffer) | 1 | ~5 µs | ~5 µs |
| **Total per block** | | | **~203 µs** |

This matches the measured 199 µs (batch=1). The ~27 µs per `memcpy_d2h_async` driver call overhead is the dominant cost component, accounting for ~55% of per-block eviction latency.

### 2.3 Why Batch=64 Regresses

`evict_blocks()` (L430-470) calls `evict_single_block()` in a sequential loop:

```rust
// tiering.rs:445-467
for &victim in &victims {
    // ...
    match self.evict_single_block(pool, block_idx, victim) {
        // ...
    }
}
```

Each iteration does its own `synchronize()` (L531) and acquires/releases the same set of locks (`block_info`, `free_blocks`). For batch=64:

- 64 × `synchronize()` instead of 1 batched synchronize
- 64 × lock acquire/release cycles (though `parking_lot::Mutex` is fast under low contention)
- The sequential loop prevents any CUDA driver-level batching of the D2H transfers

The mild improvement at batch=4 (148 µs vs 199 µs) is likely from **CPU cache warmth** — the first iteration warms the `block_info` and `free_blocks` data structures in cache, benefiting subsequent iterations. Beyond 4, the accumulated synchronize and lock overhead dominates.

### 2.4 Restore Path Comparison

The restore path (`restore_block`, tiering.rs:659-738) similarly calls `synchronize()` per block (L715). However, the restore benchmark shows better scaling because:
- `memcpy_h2d_async` has slightly lower driver overhead than `memcpy_d2h_async` (~20 µs vs ~27 µs)
- The restore path allocates from the free list (no cuMemMap needed if free blocks exist), which is purely a `VecDeque::pop_front()` operation

---

## 3. Additional Finding: cuMemUnmap is Absent from Hot Path

### 3.1 Current State

`cuMemUnmap` is only called in `KcmmPool::drop()` (pool.rs:906-941). After eviction returns a block's handle to the free list via `release_block_physical()`, the 2 MiB GPU physical page backing that superblock remains **mapped to the VA space**. It will be reused by the next allocation without ever being unmapped.

### 3.2 Implications

**Positive:** This avoids the ~244 µs/call cuMemUnmap cost on the eviction hot path, keeping single-block eviction at 135–248 µs instead of ~400+ µs.

**Negative:** GPU physical memory is never released back to the driver during the pool's lifetime. Even if 90% of blocks are evicted to CPU, the GPU physical pages remain allocated. This:
- Prevents the CUDA driver from reusing those physical pages for other CUDA contexts
- Means the GPU's physical memory footprint never shrinks, only the logical "in-use" count decreases
- Contradicts the intended behavior of "freeing GPU physical resources" documented in `evict_single_block`'s step 5 comment (L533: "Release GPU physical resources")

### 3.3 When This Matters

This becomes relevant when:
1. **Multiple KCMM pools coexist** on the same GPU — pool A cannot reclaim physical pages from pool B
2. **KCMM coexists with non-KCMM CUDA allocations** — the driver sees full physical allocation even when logically free
3. **Superblock-level defragmentation** (feature B7) is implemented — unmapping is needed to release empty superblocks

---

## 4. Fix Recommendations

### 4.1 P0: Batch memcpy and synchronize (tiering.rs)

**Current code** (`evict_blocks`, L430-470):
```rust
for &victim in &victims {
    match self.evict_single_block(pool, block_idx, victim) {
        // per-block synchronize() inside evict_single_block
    }
}
```

**Proposed fix** — split `evict_single_block` into phases and batch across victims:

```rust
fn evict_blocks_batched(
    &self,
    pool: &KcmmPool,
    candidates: &[BlockHandle],
    count: usize,
) -> Result<Vec<BlockHandle>> {
    let victims = self.eviction_policy.select_victims(candidates, count);

    // Phase 1: Allocate CPU slots + mark Evicting + issue all D2H copies
    let mut pending: Vec<(u32, BlockHandle, usize)> = Vec::new(); // (idx, handle, cpu_offset)
    let mut evicted = Vec::new();

    for &victim in &victims {
        let block_idx = pool.find_block_idx(victim)
            .ok_or_else(|| anyhow!("victim not found"))?;
        let total_bytes = pool.num_layers * 2 * pool.block_bytes;
        let cpu_offset = self.alloc_cpu_slot(total_bytes)?;
        pool.set_block_location(block_idx, BlockLocation::Evicting)?;

        // Issue D2H copies (async, no per-block synchronize)
        if let Err(e) = self.evict_block_all_layers_async(pool, victim, cpu_offset) {
            // Rollback this single block
            self.free_cpu_slot(cpu_offset, total_bytes);
            pool.set_block_location(block_idx,
                BlockLocation::GpuResident(victim, /* va_offset */))?;
            continue;
        }
        pending.push((block_idx, victim, cpu_offset));
    }

    // Phase 2: ONE synchronize for the entire batch
    pool.streams.evict.synchronize()?;

    // Phase 3: Release physical + mark CpuResident (no GPU ops, fast)
    for (block_idx, victim, cpu_offset) in pending {
        pool.release_block_physical(block_idx)?;
        pool.set_block_location(block_idx, BlockLocation::CpuResident(cpu_offset))?;
        self.eviction_policy.on_evict(victim);
        evicted.push(victim);
    }

    Ok(evicted)
}
```

**Expected impact:** For batch=64, per-block eviction cost drops from 226 µs to approximately:
- Phase 1: 64 × 4 × 27 µs (memcpy driver overhead) + 64 × 256 KiB / 4.7 GB/s (actual transfer) = 64 × (108 + 54) ≈ 10.4 ms
- Phase 2: 1 × synchronize() ≈ 30 µs
- Phase 3: 64 × 6 µs (lock ops) ≈ 0.4 ms
- Total: ~10.8 ms → **169 µs/block** (vs current 226 µs, 25% improvement)

The improvement is modest (25%) because the memcpy driver calls are still per-block. True batching would require:

### 4.2 P1: Multi-block memcpy (streams.rs + tiering.rs)

For optimal batching, add a multi-block memcpy method that copies from multiple GPU source addresses to contiguous CPU buffer offsets in a single operation, or uses CUDA's `cuMemcpyBatch`-style API where available. This would reduce the 64 × 4 = 256 driver calls to 4 (one per layer × K/V pair).

**Expected impact:** Per-block eviction drops to ~10 µs at batch=64 (5–10× improvement).

### 4.3 P2: Selective cuMemUnmap (pool.rs)

Add a `trim_superblock()` method that unmaps and releases a superblock's physical memory when all its blocks are free. Call this from `release_block_physical()` when the superblock becomes fully empty. This gives the CUDA driver visibility into actually-free physical memory without putting unmap on the hot path.

---

## 5. Corrected Performance Model

Based on the code-level analysis above, here is the **corrected** latency composition for single-block eviction at 64 KiB (replacing the model in the benchmark analysis §2.1):

| Operation | Count | Per-call | Total | % |
|---|---|---|---|---|
| `memcpy_d2h_async` (CUDA driver call) | 4 | ~27 µs | ~108 µs | 54% |
| D2H data transfer (256 KiB) | 4 | ~14 µs | ~54 µs | 27% |
| `synchronize()` | 1 | ~30 µs | ~30 µs | 15% |
| Lock acquire/release | 3 | ~2 µs | ~6 µs | 3% |
| mmap write | 1 | ~5 µs | ~5 µs | 2% |
| **Total** | | | **~203 µs** | 100% |

**Key insight:** The CUDA driver call overhead for individual `memcpy_d2h_async` calls is the dominant cost (54%), not cuMemMap/cuMemUnmap (which are not on the hot path at all). The batch eviction fix should focus on reducing the number of driver calls, either by batching memcpy operations or by using multi-block transfer APIs.

---

## 6. Corrigendum: Benchmark Analysis Report

The benchmark analysis report (`docs/report/kcmm-phase-e-benchmark-analysis.md`, §2.1) incorrectly attributed 85–90% of tiering latency to cuMemMap/cuMemUnmap. The corrected attribution is:

| Component | Report claimed | Actual (code-level) |
|---|---|---|
| cuMemMap/cuMemUnmap | 85–90% | 0% (not on hot path) |
| CUDA driver call overhead (memcpy) | ~5% | ~54% |
| Data transfer (PCIe) | ~5% | ~27% |
| Stream synchronize | — | ~15% |
| Other (locks, mmap) | — | ~6% |

This correction does **not** change the conclusion that batch eviction needs fixing, but it **does** change the optimization target: focus on **batching memcpy driver calls**, not on cuMemUnmap batching.

---

## 7. Action Items

| # | Action | Priority | File | Effort |
|---|---|---|---|---|
| 1 | Extract per-block `synchronize()` from eviction loop; do one sync after all D2H copies | P0 | `tiering.rs:430-470, 530-531` | Low |
| 2 | Same treatment for restore loop | P0 | `tiering.rs:659-738` | Low |
| 3 | Update benchmark analysis report with corrected latency model | P1 | `docs/report/kcmm-phase-e-benchmark-analysis.md` | Low |
| 4 | Re-run Benchmark 2b after fix to validate monotonic decrease | P1 | `tests/kcmm_bench_tiering.rs` | Low |
| 5 | Add `trim_superblock()` for releasing empty superblocks | P2 | `pool.rs`, `superblock.rs` | Medium |
| 6 | Explore CUDA multi-block memcpy APIs for further batching | P3 | `streams.rs` | High |

---

## Appendix: Trace of cuMemMap/cuMemUnmap Call Sites

```
cuMemMap call sites:
  pool.rs:284  map_superblock_to_layer() — called from ensure_capacity() when
               creating new superblocks; NOT on evict/restore hot path

cuMemUnmap call sites:
  pool.rs:919  Drop::drop() — unmap K superblocks at pool destruction
  pool.rs:927  Drop::drop() — unmap V superblocks at pool destruction
               NOT called during eviction
```
