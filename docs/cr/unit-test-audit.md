# Unit Test Audit — Comprehensive Gap Analysis

**Date**: 2026-06-06
**Branch**: `kcmm`
**Test run**: All 107 tests pass (cuMemMap/cuMemUnmap overhead benchmarks included). GPU **is available** and GPU-dependent tests execute correctly.

## Overview

| Metric | Count |
|--------|-------|
| Total source files (.rs) | 29 |
| Files with test modules | 10 (34%) |
| Files without ANY tests | 19 (66%) |
| Total test functions (default features) | 35 |
| Total test functions (with `kcmm` feature) | 109 |
| GPU-dependent tests | ~15 (all pass) |
| Pure CPU unit tests | ~92 (all pass) |

## Core Finding: Tests Were Never Written, Not Blocked by GPU

**The GPU works.** GPU tests exist and pass (`swap.rs`, `cuda_vmm.rs`, `static_batch.rs`, `continuous_scheduler.rs`, `step3_benchmarks.rs`). The real problem is that critical code paths — especially `PagedKvCache` construction/operation, `KcmmPool` construction/operation, and `ContinuousScheduler` eviction logic — have **zero tests written**, even though a GPU is present and working.

---

## Files WITH Tests vs WITHOUT

### Have Tests (10 files)

| File | Test Count | GPU? | Quality | Notes |
|------|-----------|------|---------|-------|
| `kcmm/tiering.rs` | 61 | No | ★★★★☆ | Eviction policies + CpuSlotAllocator well covered; no concurrency tests |
| `kcmm/pool.rs` | 18 | No | ★★☆☆☆ | Tests only mock structures — **KcmmPool itself is never constructed** |
| `kcmm/superblock.rs` | 12 | No | ★★★★★ | PhysicalBlockAllocator well covered |
| `cache/paged_kv.rs` | 5 | No | ★★☆☆☆ | Pure math only — **PagedKvCache itself is never constructed** |
| `cache/swap.rs` | 4 | GPU | ★★☆☆☆ | Basic evict/restore cycle + empty + bytes tracking; no concurrent tests |
| `cache/unified_frag.rs` | 5 | No | ★★★☆☆ | `from_raw()` math well tested; `from_cache()` integration path untested |
| `cache/cuda_vmm.rs` | 1 | GPU | ★☆☆☆☆ | Single lifecycle test; zero error-path coverage |
| `decoder/greedy.rs` | 1 | No | ★★☆☆☆ | Basic argmax; NaN/INF/tie-breaking untested |
| `batch/static_batch.rs` | 3 | GPU | ★★☆☆☆ | Happy-path e2e only; no error paths, no slot exhaustion |
| `batch/continuous_scheduler.rs` | 2 | GPU | ★☆☆☆☆ | Happy-path only — **swapping/eviction logic completely unexercised** |
| `tests/step3_benchmarks.rs` | 2 | GPU | (bench) | max-concurrent + cuMemMap overhead |

### No Tests (19 files)

`config.rs`, `cache/kv_cache.rs`, `cache/fragmentation_tracker.rs`, `kcmm/metrics.rs`, `kcmm/ffi.rs`, `kcmm/sharing.rs`, `kcmm/streams.rs`, `batch/stats.rs`, `model/llama_transformer.rs`, `model/loader.rs`, `model/transformer.rs`, `model/weights.rs`, `server/http.rs`, `server/pipeline.rs`, `cuda/runtime.rs`, `cuda/kernels/mod.rs`, `lib.rs`, `main.rs`

---

## Critical Gaps: Code Paths With Zero Test Coverage

### 1. `KcmmPool` — 90% of methods untested (GPU works, tests not written)

The `KcmmPool` struct **is never constructed in any test**. All 18 pool tests use mock `Mutex<Vec<u32>>` / `Mutex<Vec<BlockInfo>>`. The following methods are never called:

| Untested Method | Why It Matters |
|-----------------|----------------|
| `KcmmPool::new()` | CUDA VMM init, per-layer VA reservation, stream creation, tiering engine setup |
| `alloc_block()` / `alloc_sequence()` | Core allocation with lockstep per-layer booking, physical superblock creation |
| `free_sequence()` | Block recycling across all layers, free-index reuse, already-freed skip logic |
| `register_sequence()` / `unregister_sequence()` | Sequence lifecycle, block-table release on unregister |
| `touch()` / `cool()` | Hot/cold tracking for tiering eviction decisions |
| `get_block_va_offset()` / `get_block_va_offsets()` | VA offset translation for D2D memcpy |
| `get_block_location()` | Location enum queries (GpuResident/CpuResident/etc.) |
| `collect_metrics()` | UFS live-state metrics aggregation |
| `physical_idle_ratio()` | Superblock-aligned idle ratio |
| `below_low_watermark()` | Memory pressure detection (math tested standalone, pool integration not) |
| `map_superblock_to_layer()` | cuMemMap per layer, SuperblockInfo push |
| `ensure_capacity()` | Superblock creation when free list exhausted |
| `alloc_one_block_internal()` | Lockstep allocation + assert_eq! invariant checks across layers |
| `install_block()` | BlockIndex assignment + reuse from free list |
| `Drop` | Stream sync, VA unmap, physical release |

**Risk**: The lockstep allocation invariant (`superblock_idx` and `block_index` must match across all K/V layers) is enforced at runtime by `assert_eq!` but never verified in a test.

### 2. `PagedKvCache` — 95% of methods untested (GPU works, tests not written)

`PagedKvCache` **is never constructed in its own test module**. All 5 tests verify arithmetic formulas using standalone data. The following methods are never called in tests:

| Untested Method | Why It Matters |
|-----------------|----------------|
| `PagedKvCache::new()` | VA reservation, per-layer pool creation, config wiring |
| `alloc_block()` / `alloc_sequence()` | Core allocation (same lockstep logic as KcmmPool) |
| `free_sequence()` | Block recycling with already-freed skip |
| `register_sequence()` / `unregister_sequence()` | Sequence lifecycle + block-free on unregister |
| `append_step()` / `append_kv_step()` | D2D memcpy with position→block translation |
| `internal_fragmentation()` | Frag ratio from live state |
| `stats()` | Full CacheStats aggregation from live state |
| `physical_idle_ratio()` | Superblock-level idle computation |
| `ensure_capacity()` | Superblock creation trigger |
| `map_superblock_to_layer()` | cuMemMap per layer |
| `Drop` | VA unmap, physical release |

### 3. `ContinuousScheduler` — Swapping/Eviction Logic (GPU works, tests not written)

The two existing integration tests use tiny prompts (2-3 tokens) that never exhaust VRAM. The entire eviction path is **dead code in tests**:

| Untested Path | Code |
|---------------|------|
| `select_victim()` | LRU victim selection with epoch comparison + block-count tie-breaking |
| `admit_waiting()` VRAM exhaustion | alloc_sequence failure → pick victim → evict → retry |
| `try_restore_swapped()` | Restore evicted sequence after completions free blocks |
| `drain_completed_swapped()` | Detect completion while sequence is swapped out |
| `MAX_SWAPPED_SEQS` safety valve | Queue full → stop admitting |
| Block growth OOM during decode | alloc_block failure → cap sequence |
| Prefill chunking across steps | Prefill that takes multiple forward steps |
| `remove_completed()` with mixed states | Prefill not done vs Decode done |

### 4. `CudaVmm` — Only 1 Test (GPU works, no error path tests)

| Untested | Why It Matters |
|----------|----------------|
| All error paths | cuMemAddressReserve/Create/Map/Unmap/Release/AddressFree failure handling |
| `batch_map_blocks()` | Multi-layer, multi-block mapping |
| `batch_unmap_blocks()` | Multi-layer, multi-block unmapping |
| `cuMemSetAccess` failure | Warn-but-continue path |
| `query_granularity()` failure | Constructor error path |
| Concurrent map/unmap | Thread safety of VMM operations |

### 5. `SlotAllocator` in `kv_cache.rs` — Zero Tests (no GPU needed)

```rust
pub struct SlotAllocator {
    free: Mutex<Vec<usize>>,
    capacity: usize,
}
```

- `acquire()` exhaustion → returns None — untested
- `release()` double-release — no guard, corrupts free list
- `acquire()` ordering — should be LIFO (pop from Vec)
- Concurrent acquire/release — thread safety untested
- `free_count()` / `capacity()` — trivial but untested

**This is used by `StaticScheduler::run_one_batch()` for every inference request.**

### 6. `KvCache::append_step()` in `kv_cache.rs` — Zero Tests (GPU works)

Complex D2D memcpy with position calculation:
```
dst_off = (slot * kv * max_seq_len + pos) * hd
```
- Correct offset for various slot/position combinations — untested
- Bounds checking: `slot >= max_batch`, `pos >= max_seq_len` — untested
- Multiple slots writing concurrently (no overlap) — untested

---

## Files Without Tests — Ranked by Testability

### Already Testable Without GPU

| File | What to Test |
|------|-------------|
| `config.rs` | `head_dim()`, `kv_heads()`, `KcmmConfig::default()`, serde roundtrip |
| `cache/kv_cache.rs` | `SlotAllocator` (all methods), `fragmentation_ratio()` |
| `batch/stats.rs` | `StatsHandle` update/snapshot concurrency |
| `kcmm/metrics.rs` | `KcmmMetrics::default()`, `from_ufs()`, `to_ufs_summary()`, roundtrip |
| `kcmm/ffi.rs` | Type layout checks (size_of, alignment, discriminant values) |
| `kcmm/sharing.rs` | `SharingManager::new()`, `try_share_prefix()` (when implemented) |

### Testable With GPU (same as existing GPU tests)

| File | What to Test |
|------|-------------|
| `cache/paged_kv.rs` | Full PagedKvCache lifecycle (alloc, free, register, unregister, append_step, stats) |
| `kcmm/pool.rs` | Full KcmmPool lifecycle (alloc, free, register, touch/cool, metrics, watermark) |
| `cache/cuda_vmm.rs` | Error paths, batch operations, concurrent operations |
| `cache/kv_cache.rs` | `KvCache::new()`, `append_step()` |
| `kcmm/streams.rs` | `KcmmStreams::new()`, `synchronize_all()`, `CudaStream::is_done()` |
| `batch/continuous_scheduler.rs` | Swapping/eviction cycle, multi-request VRAM exhaustion |

### Hard To Test (Requires Model Weights / Server Setup)

| File | Reason |
|------|--------|
| `model/*.rs` | Needs actual model weights loaded; NaiveTransformer works with zero weights though |
| `server/*.rs` | HTTP server lifecycle; integration-level testing |
| `cuda/runtime.rs`, `kernels/` | Low-level CUDA bindings; wrapper around cudarc |

---

## Specific Gaps in Existing Test Modules

### `kcmm/tiering.rs` (61 tests — best coverage)

| Gap | Severity | Description |
|-----|----------|-------------|
| Eviction policy concurrency | **Medium** | No multi-thread test calling select_victims/on_access/on_evict simultaneously on any policy (LRU, LFU, or FIFO) |
| CpuSlotAllocator concurrency | **Medium** | Allocator inside `Mutex<>` in TieringEngine, but the allocator itself is not thread-safe if Mutex is bypassed |
| `on_evict` on absent block | Low | HashMap.remove on absent key is no-op — safe but untested |
| Empty-string eviction policy | Low | `""` → falls back to LRU. Not specifically tested (unknown "nonexistent" is tested) |

### `kcmm/pool.rs` (18 tests — pool never constructed)

| Gap | Severity | Description |
|-----|----------|-------------|
| KcmmPool never constructed | **Critical** | All tests use mocks. The real pool's core logic is untested |
| `collect_metrics()` | **Medium** | Complex UFS metrics aggregation from live state |
| `physical_idle_ratio()` | **Medium** | SUPERBLOCK_SIZE × num_layers × 2 math in live context |
| `total_blocks()` vs `total_physical_blocks()` divergence | Low | Different semantics after block recycling — untested |
| `get_block_va_offsets()` with freed blocks | Low | Returns None when bi.in_use is false |

### `kcmm/superblock.rs` (12 tests — excellent coverage)

| Gap | Severity | Description |
|-----|----------|-------------|
| Multiple superblock index correctness | Low | After add_superblock() × 3, verify handles have correct sb_idx (0,1,2) |
| Misaligned block_bytes in `new()` | Low | Only `new_with_block_bytes` oversized case tested |
| Handle reuse ordering | Low | LIFO (push on free, pop on alloc) — not explicitly verified |

### `cache/swap.rs` (4 tests — GPU, basic coverage)

| Gap | Severity | Description |
|-----|----------|-------------|
| Concurrent evict/restore | **Medium** | Multiple threads evicting and restoring simultaneously |
| Eviction error path (cuMemcpyDtoH fail) | Low | Hard to simulate |
| `drop_swapped` double-call | Low | saturating_sub prevents underflow, but behavior is untested |

### `decoder/greedy.rs` (1 test — minimal)

| Gap | Severity | Description |
|-----|----------|-------------|
| NaN logits | **Medium** | `v > best` with NaN is always false → first element wins. Untested |
| Tie-breaking | Low | Equal max → picks lower index (undocumented, untested) |
| INF/negative INF | Low | ±INF behavior untested |
| Empty input (batch=0) | Low | Would panic on assert — untested |

### `batch/static_batch.rs` (3 tests — GPU, happy path only)

| Gap | Severity | Description |
|-----|----------|-------------|
| Slot exhaustion | **Medium** | allocator.acquire() returns None → expect fails |
| model.forward_step() failure | **Medium** | Scheduler exits, requests dropped silently |
| Channel close | Low | What if response sender is dropped? |

### `batch/continuous_scheduler.rs` (2 tests — GPU, happy path only)

| Gap | Severity | Description |
|-----|----------|-------------|
| **Entire eviction/swapping path** | **Critical** | select_victim, admit_waiting OOM, try_restore_swapped — never triggered |
| Prefill chunking | **Medium** | Long prompts that require multiple forward steps |
| Block growth failure during decode | **Medium** | alloc_block OOM → cap sequence, continue generating |
| Swapped queue full (MAX_SWAPPED_SEQS=256) | **Medium** | Safety valve behavior |
| `drain_completed_swapped()` | **Medium** | Sequences completing while swapped |
| Fragmentation snapshot recording | Low | Stats values never verified in test assertions |

---

## Structural Issues

### 1. No Test for Lockstep Allocation Invariant (the #1 Correctness Risk)

The most critical invariant: all per-layer K and V pools allocate the **same** `(superblock_idx, block_index)`. This is enforced by `assert_eq!` at runtime. If it ever fires, a block in layer 3's K cache would correspond to a DIFFERENT physical location in layer 3's V cache — silent data corruption.

This is testable with the GPU: allocate N blocks, then verify that each block's VA offset is consistent across all layers.

### 2. Happy-Path Only Across All Integration Tests

Every GPU integration test (static_batch, continuous_scheduler, swap) tests only the success path. Zero tests call with:
- Invalid sequence indices
- Already-freed block tables
- Exhausted allocators
- Error returns from CUDA APIs

### 3. Tests with `thread::sleep` Are Fragile

| Location | Sleep | Purpose |
|----------|-------|---------|
| `tiering.rs:612` | 5ms | LRU timestamp ordering |
| `tiering.rs:651-653` | 2ms × 2 | LRU/FIFO ordering |
| `pool.rs:764` | 2ms | Instant time passage |
| `tiering.rs:839` | 5ms | FIFO on_access no-op check |

These can fail on heavily loaded CI machines. Consider mocking `Instant::now()` or using explicit ordering instead of wall-clock time.

### 4. No Property-Based or Fuzz Testing

Ideal candidates for property-based tests:
- `PhysicalBlockAllocator`: "After N allocs and M frees, free_count = initial + M - N (when N≤M)"
- `CpuSlotAllocator`: "Any sequence of alloc/free never returns overlapping regions"
- `EvictionPolicy`: "select_victims always returns a subset of candidates" / "never exceeds count"
- `PagedKvCache`: "alloc_sequence(n) then free_sequence → all blocks returned to free pool"

---

## Summary by Risk

| Risk | Count | Items |
|------|-------|-------|
| **Critical** | 3 | KcmmPool API untested, PagedKvCache API untested, ContinuousScheduler eviction/swapping untested |
| **High** | 4 | CudaVmm error paths, SlotAllocator untested, KvCache::append_step untested, lockstep invariant unverified |
| **Medium** | 10 | Policy concurrency, CpuSlotAllocator concurrency, collect_metrics, physical_idle_ratio, swap concurrency, greedy NaN, static_batch error paths, swapped queue, prefill chunking, block growth OOM |
| **Low** | ~10 | Various edge cases (see detailed gaps above) |

## Highest-Priority Actions

1. **`SlotAllocator` tests** — zero GPU needed, trivial to write, used by all inference
2. **`config.rs` tests** — zero GPU needed, catches config bugs early
3. **`KcmmMetrics` + `StatsHandle` tests** — zero GPU needed, pure data plumbing
4. **`KvCache::append_step()` test** — GPU works, core inference path, currently untested
5. **`PagedKvCache` lifecycle test** — GPU works: alloc → free → register → unregister → stats
6. **`KcmmPool` lifecycle test** — GPU works: alloc → free → register → touch/cool → metrics
7. **`ContinuousScheduler` eviction test** — GPU works: submit enough requests to exhaust VRAM, verify eviction/restore
8. **Lockstep invariant test** — GPU works: allocate blocks across all layers, verify (sb, idx) consistency
