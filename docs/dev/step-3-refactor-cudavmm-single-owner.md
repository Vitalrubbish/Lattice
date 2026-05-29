# Step 3: Continuous Batching & Paged KV Cache — Implementation

**2026-05-28**

## Changes

### 1. CudaVmm single-owner refactor

Removed `vmm: CudaVmm` field from `PhysicalBlockAllocator`. The `allocate()` method now takes `&CudaVmm` borrowed from `PagedKvCache` — the sole VMM owner.

**Why:** Both structs owned separate `CudaVmm` instances wrapping the same `device: usize`. Since `CudaVmm` is stateless, the duplication was architectural confusion with no benefit.

### 2. Bug fix: physical offset in cuMemMap

`CudaVmm::map()` previously hardcoded physical offset to 0, causing every block carved from a superblock to map to the same physical start address. Now accepts and passes `phys_offset` through to `cuMemMap`, so each block within a superblock maps to the correct physical sub-range.

### 3. Bug fix: K/V separation

`append_step()` previously wrote K and V to the same VA address, causing V to clobber K. Now:
- Separate VA regions per layer: `va_k` and `va_v`
- `append_step` does two `cuMemcpyDtoDAsync_v2` calls — one to K region, one to V region
- Matches the old `KvCache::append_step` pattern

### 4. PagedKvCache completion

- **BlockInfo tracking**: `block_info: Mutex<Vec<BlockInfo>>` maps `physical_block_idx → (va_offset, superblock_phys, block_index, in_use)`
- **Free index recycling**: `free_indices` reclaims physical block indices from freed sequences
- **Free list reuse**: `PhysicalBlockAllocator.phys_handles` tracks created superblock handles
- **Sequence management**: `register_sequence()`, `unregister_sequence()`, `update_seq_len()`
- **`Drop` impl**: releases all physical handles and VA regions on teardown

### 5. ContinuousScheduler (`src/batch/continuous_scheduler.rs`)

New scheduler implementing the Step 3 plan:

```
loop {
    drain incoming requests → waiting queue
    admit waiting requests if budget allows (block allocation)
    if nothing running, block on queue
    run one forward step for all running requests
    remove completed requests, free KV blocks
}
```

Request states: `Prefill { prompt_pos }` → `Decode` → done.
Each `RunningRequest` owns a block table and a `seq_idx` into `PagedKvCache.seq_metadata`.

### 6. NaiveTransformer paged support

Added `forward_step_paged()` and `prefill_paged()` methods that take `&PagedKvCache` instead of `&mut KvCache`. Same logic as the original methods but using the paged cache's `append_step` with block-table addressing.

### 7. main.rs integration

Added `--continuous` CLI flag that toggles between:
- `StaticScheduler` + contiguous `KvCache` (default, baseline)
- `ContinuousScheduler` + `PagedKvCache` (new, experimental)

### Files changed/added

| File | Status |
|------|--------|
| `src/cache/cuda_vmm.rs` | Modified: `map()` signature (phys_offset param) |
| `src/cache/paged_kv.rs` | Modified: bug fixes, K/V sep, block tracking, tests |
| `src/cache/mod.rs` | Modified: added `pub mod paged_kv` |
| `src/batch/continuous_scheduler.rs` | **New** |
| `src/batch/mod.rs` | Modified: added module + re-export |
| `src/model/transformer.rs` | Modified: added paged methods |
| `src/main.rs` | Modified: `--continuous` flag, dual-scheduler support |

### Tests

11 unit tests in `paged_kv.rs` verifying: allocator sizing, free reuse, block address formulas, logical→physical translation, BlockInfo tracking, SeqMetadata, align_up, superblock carving. All pass. GPU VMM lifecycle test skipped (no CUDA driver in current env).
