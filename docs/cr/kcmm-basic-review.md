# KCMM Step 3 Code Review Fix Document

> **Review Date**: 2026-06-05
> **Review Scope**: All 8 files under `src/kcmm/`, cross-referenced with `docs/task/kcmm-implementation-analysis.md`
> **Target Milestone**: Week 13 — Module skeleton + extraction + generalization

## Completeness Assessment

| Requirement | Status |
|------------|--------|
| Module skeleton (8 files) | ✅ Complete |
| `PhysicalBlockAllocator` moved to `superblock.rs` | ✅ Complete |
| `paged_kv.rs` imports types from `kcmm::superblock` | ✅ Complete |
| `KcmmPool` created with basic allocation logic | ✅ Complete |
| `BlockLocation` enum defined | ✅ Complete |
| `SequenceState` tracking fields | ✅ Complete |
| `CudaStream` / `KcmmStreams` wrappers | ✅ Complete |
| `EvictionPolicy` trait defined | ✅ Complete |
| `KcmmConfig` in `config.rs` | ✅ Complete |
| `cdylib` target + `kcmm` feature flag in `Cargo.toml` | ✅ Complete |
| Code compiles | ✅ Pass |

**Week 13 overall completeness: ~55-60%.** Structure is in place, but key integration points are missing.

---

## 🔴 Critical Issues (Correctness / Deadlock / Data Race)

### Issue 1: Deadlock — Inconsistent Lock Acquisition Order

**Location**: `src/kcmm/pool.rs:348-420`, `src/cache/paged_kv.rs:250-299`

**Description**:
`install_block` and `free_sequence` acquire the `free_block_indices` and `block_info` Mutexes in **opposite order**. `parking_lot::Mutex` allows concurrent locking through shared references, so two threads calling these two functions simultaneously will deadlock.

**`install_block`** — locks `free_block_indices` first, then `block_info`:

```rust
// pool.rs:356-366, paged_kv.rs:251-261
fn install_block(&self, va_offset: usize, sb_idx: u32, blk_in_sb: u32) -> u32 {
    let mut free = self.free_block_indices.lock();   // Lock A
    if let Some(idx) = free.pop() {
        let mut info = self.block_info.lock();        // Lock B
        // ...
    }
}
```

**`free_sequence`** — locks `block_info` first, then `free_block_indices`:

```rust
// pool.rs:398-420, paged_kv.rs:299-320
fn free_sequence(&self, block_table: &[u32]) {
    let mut info = self.block_info.lock();            // Lock B
    // ...
    self.free_block_indices.lock().push(block_idx);   // Lock A
}
```

**Deadlock Scenario**:
```
Thread 1: install_block   → acquires free_block_indices → waits for block_info
Thread 2: free_sequence   → acquires block_info         → waits for free_block_indices
```

**Fix**: Unify the lock order. In `free_sequence`, collect the block_idx values to recycle into a temporary Vec, release the `block_info` lock, then push them all at once to `free_block_indices`.

```rust
fn free_sequence(&self, block_table: &[u32]) {
    let mut info = self.block_info.lock();
    let mut recycled = Vec::new();
    let num_layers = self.num_layers;

    for &block_idx in block_table {
        let bi = &mut info[block_idx as usize];
        if !bi.in_use { continue; }
        bi.in_use = false;

        let handle = BlockHandle {
            superblock_idx: bi.superblock_idx,
            block_index: bi.block_index_in_sb,
        };
        for l in 0..num_layers {
            self.k_pools[l].allocator.free(handle);
            self.v_pools[l].allocator.free(handle);
        }
        recycled.push(block_idx);
    }
    drop(info);                                      // Release Lock B
    self.free_block_indices.lock().extend(recycled); // Acquire Lock A
}
```

**Affected Files**: `src/kcmm/pool.rs`, `src/cache/paged_kv.rs`

---

### Issue 2: Massive Code Duplication Between `paged_kv.rs` and `kcmm/pool.rs`

**Location**: `src/kcmm/pool.rs` and `src/cache/paged_kv.rs`

**Description**: Both files contain **near-verbatim identical** core allocation logic, violating the DRY principle and contradicting the design document's goal of "extract from paged_kv.rs into kcmm."

| Function | `paged_kv.rs` | `kcmm/pool.rs` |
|---------|--------------|----------------|
| `map_superblock_to_layer` | L134-170 | L236-270 |
| `ensure_capacity` | L174-204 | L273-297 |
| `alloc_one_block_internal` | L211-247 | L302-346 |
| `install_block` | L250-272 | L348-378 |
| `alloc_block` | L276-278 | L381-383 |
| `alloc_sequence` | L290-298 | L387-394 |
| `append_block_to_sequence` | L282-287 | L480-485 |

`paged_kv.rs` retains its own separate `BlockInfo` (only `in_use: bool`), while `kcmm/pool.rs` has its own `BlockInfo` (with `BlockLocation`). The design document explicitly states:

> "Keep `PagedKvCache` as a compatibility layer, delegating to `KcmmPool` when KCMM is enabled"

**Fix**: Extract the shared allocation core into a private module (e.g., `src/kcmm/allocator.rs`) used by both `PagedKvCache` and `KcmmPool`. Alternatively, make `PagedKvCache` compose a `KcmmPool` instance internally and operate through its API. At minimum, add a `from_kcmm_pool` constructor on `UnifiedFragMetrics` in `unified_frag.rs` to avoid duplicating metric computation.

**Affected Files**: `src/kcmm/pool.rs`, `src/cache/paged_kv.rs`, `src/cache/unified_frag.rs`

---

### Issue 3: `TieringEngine` Ignores `cpu_cache_path` Configuration

**Location**: `src/kcmm/tiering.rs:111`

**Description**: `KcmmConfig::cpu_cache_path` defaults to `"/dev/shm/kcmm_swap"`, but `TieringEngine::new` uses `MAP_ANONYMOUS`, completely ignoring this configuration:

```rust
let ptr = unsafe {
    libc::mmap(
        std::ptr::null_mut(),
        cpu_buffer_size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED | libc::MAP_ANONYMOUS, // ← Anonymous mapping, ignores cpu_cache_path
        -1,                                      // ← fd = -1
        0,
    )
};
```

**Impact**:
- Cannot share CPU swap buffer across processes (violates KCMM's core multi-engine memory management design)
- Cannot persist swap data on disk
- `config.cpu_buffer_size` is not used correctly (a rough estimate of `max_blocks * block_size * 2` is used instead)

**Fix**: Use `OpenOptions` to create/open a file, then `mmap` via the file descriptor:

```rust
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

let file = OpenOptions::new()
    .read(true).write(true).create(true)
    .open(&config.cpu_cache_path)?;
file.set_len(cpu_buffer_size as u64)?;
let ptr = unsafe {
    libc::mmap(
        std::ptr::null_mut(),
        cpu_buffer_size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        file.as_raw_fd(),
        0,
    )
};
```

**Affected File**: `src/kcmm/tiering.rs`

---

## 🟠 High Priority (Design Flaws / Missing Features)

### Issue 4: `EvictionPolicy` Implementations Are All Stubs

**Location**: `src/kcmm/tiering.rs:39-85`

**Description**: `LruPolicy::select_victims`, `LfuPolicy::select_victims`, and `FifoPolicy::select_victims` all return empty `Vec::new()`. If called, they silently evict nothing — neither erroring nor panicking. Under memory pressure, this leads to **silent OOM**.

```rust
// tiering.rs:42-45
fn select_victims(&self, _candidates: &[BlockHandle], _count: usize) -> Vec<BlockHandle> {
    // Placeholder — full implementation in Week 14.
    Vec::new()  // ← Tells the caller: nothing selected, but caller doesn't know this is a stub
}
```

**Fix**: LRU is KCMM's default policy. At minimum implement its basic logic (sort by `last_access`, return the N oldest). Other policies can use `unimplemented!()` or fall back to LRU with a warn log:

```rust
impl EvictionPolicy for LruPolicy {
    fn select_victims(&self, candidates: &[(BlockHandle, Instant)], count: usize)
        -> Vec<BlockHandle>
    {
        let mut sorted: Vec<_> = candidates.iter().collect();
        sorted.sort_by_key(|(_, ts)| *ts);
        sorted.truncate(count);
        sorted.into_iter().map(|(h, _)| *h).collect()
    }
}
```

**Affected File**: `src/kcmm/tiering.rs`

---

### Issue 5: Missing Async `cudaMemcpy` Operations

**Location**: Missing feature — `src/kcmm/streams.rs` or `src/cache/cuda_vmm.rs`

**Description**: `KcmmStreams` exists with three dedicated CUDA streams (`evict`, `restore`, `prefetch`), but **nowhere** are `cuda_memcpy_d2h_async()` or `cuda_memcpy_h2d_async()` functions implemented. The design document (Functional Module D, D6) explicitly requires:

> "Async memcpy operations `cuda_memcpy_d2h_async()` / `cuda_memcpy_h2d_async()`"

Without these functions, `TieringEngine` cannot perform actual GPU↔CPU data transfers — the **core mechanism** of tiered storage.

**Fix**: Add methods on `KcmmStreams` (or `CudaStream`) wrapping `cuMemcpyDtoHAsync` / `cuMemcpyHtoDAsync`:

```rust
impl CudaStream {
    pub fn memcpy_d2h_async(&self, dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<()> {
        let result = unsafe {
            sys::lib().cuMemcpyDtoHAsync_v2(
                dst as sys::CUdeviceptr,
                src,
                bytes,
                self.inner,
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyDtoHAsync failed: {:?}", result));
        }
        Ok(())
    }

    pub fn memcpy_h2d_async(&self, dst: CUdeviceptr, src: *const u8, bytes: usize) -> Result<()> {
        let result = unsafe {
            sys::lib().cuMemcpyHtoDAsync_v2(
                dst,
                src as sys::CUdeviceptr,
                bytes,
                self.inner,
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyHtoDAsync failed: {:?}", result));
        }
        Ok(())
    }
}
```

**Affected File**: `src/kcmm/streams.rs`

---

### Issue 6: Missing CUDA Event Inter-Stream Synchronization

**Location**: Missing feature — `src/kcmm/streams.rs`

**Description**: The design document (D5) requires:
> "Use CUDA Events for synchronization between the evict/restore/prefetch streams and coordination with the inference stream"

Currently `KcmmStreams` only has `synchronize_all()` (a coarse-grained synchronization barrier) with no event recording/waiting. Without events, there is no guarantee that:
- A block won't be evicted **before** the inference kernel finishes writing to it
- Physical pages won't be released **before** eviction memcpy completes
- A block won't be marked `GpuResident` **before** restore memcpy completes

**Fix**: Add `CudaEvent` wrapper:

```rust
pub struct CudaEvent {
    pub(crate) inner: sys::CUevent,
}

impl CudaEvent {
    pub fn new() -> Result<Self> {
        let mut event: sys::CUevent = std::ptr::null_mut();
        let result = unsafe { sys::lib().cuEventCreate(&mut event, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuEventCreate failed: {:?}", result));
        }
        Ok(Self { inner: event })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        let result = unsafe { sys::lib().cuEventRecord(self.inner, stream.inner) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuEventRecord failed: {:?}", result));
        }
        Ok(())
    }
}

impl CudaStream {
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        let result = unsafe { sys::lib().cuStreamWaitEvent(self.inner, event.inner, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuStreamWaitEvent failed: {:?}", result));
        }
        Ok(())
    }
}
```

**Affected File**: `src/kcmm/streams.rs`

---

### Issue 7: KCMM Module Is Not Feature-Gated

**Location**: `src/lib.rs:6`

**Description**: `Cargo.toml` defines a `kcmm` feature flag, but `lib.rs` unconditionally includes the KCMM module:

```rust
pub mod kcmm;  // ← Always compiled, ignores the feature flag
```

The design document (Backward Compatibility Constraint 0.5) explicitly requires:
> "KCMM enabled as an optional feature flag: `cargo build --features kcmm` compiles the KCMM module and produces `libkcmm.so`"

**Fix**:
```rust
#[cfg(feature = "kcmm")]
pub mod kcmm;
```

The `pub use config::KcmmConfig` export in `lib.rs` must also be controlled accordingly.

**Affected File**: `src/lib.rs`

---

## 🟡 Medium Issues (API Design / Maintainability)

### Issue 8: `KcmmPool` Breaks Encapsulation — Nearly All Fields Are `pub`

**Location**: `src/kcmm/pool.rs:101-154`

**Description**: Almost all fields of `KcmmPool` are marked `pub`, allowing external callers to directly mutate internal state without validation:

```rust
pub struct KcmmPool {
    pub config: KcmmConfig,        // ← pub, caller can mutate config
    pub ctx: Arc<CudaContext>,      // ← pub
    pub max_batch: usize,           // ← pub
    pub tiering: Option<TieringEngine>,  // ← pub
    pub streams: KcmmStreams,       // ← pub
    pub fragmentation_tracker: RuntimeFragmentationTracker,  // ← pub
    pub num_layers: usize,          // ← pub
    pub elem_per_block: usize,      // ← pub
    pub block_bytes: usize,         // ← pub
    pub max_blocks_total: usize,    // ← pub
    // ...
}
```

A caller could modify `max_batch` without recomputing `max_blocks_total`, leading to inconsistency. This violates KCMM's design goal as a "standalone OS service."

**Fix**: Change fields to `pub(crate)` or make them private with read-only getters. Add setter methods with validation logic for fields that need mutation.

**Affected File**: `src/kcmm/pool.rs`

---

### Issue 9: `eviction_policy` Configuration Uses String Instead of Enum

**Location**: `src/config.rs:109`

**Description**:
```rust
pub eviction_policy: String,  // "lru", "lfu", or "fifo"
```

String-typed configuration is fragile — typos are only caught at runtime and IDEs cannot autocomplete. The design document (C5) mentions "runtime switching," which is fully compatible with compile-time enums.

**Fix**: Define an enum and implement `Serialize/Deserialize`:
```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EvictionPolicyType { Lru, Lfu, Fifo }
```

Synchronize changes across `KcmmConfig`, `TieringEngine` initialization logic, and `KcmmPool::new`.

**Affected Files**: `src/config.rs`, `src/kcmm/tiering.rs`, `src/kcmm/pool.rs`

---

### Issue 10: Duplicate Logic Between `collect_metrics()` and `unified_frag.rs::from_cache()`

**Location**: `src/kcmm/pool.rs:620-677`, `src/cache/unified_frag.rs:99-180`

**Description**: `KcmmPool::collect_metrics` reimplements **nearly identical** UFS metric calculations as `UnifiedFragMetrics::from_cache`, but with slightly different formulas. Any fix or formula change requires synchronization across both locations.

The design document (G4) suggests:
> "Add `from_kcmm_pool` method or generalize existing methods to support `KcmmPool`"

**Fix**: Add a `from_kcmm_pool` constructor on `UnifiedFragMetrics`, or implement a `AsRawMetrics` trait on `KcmmPool` that exposes raw counts to a unified metric computation function. `collect_metrics` should delegate to `UnifiedFragMetrics::from_kcmm_pool(self)`.

**Affected Files**: `src/kcmm/pool.rs`, `src/cache/unified_frag.rs`

---

### Issue 11: `BlockLocation` Lacks State Transition Validation

**Location**: `src/kcmm/pool.rs:32-44`

**Description**: `BlockLocation` is just a `#[derive(Debug, Clone)]` enum — no `transition_to()` method, no transition validation. The design document risk table (0.6) explicitly mentions:

> "`BlockLocation` state machine correctness: 5-state transitions must handle concurrency. Fine-grained locks + state transition assertions; proptest fuzz testing covering all transitions"

Currently, illegal transitions are allowed (e.g., `Evicting → Evicting` or `Restoring → NvmeResident` without going through `GpuResident`).

**Fix**: Add transition validation methods:
```rust
impl BlockLocation {
    fn can_transition_to(&self, target: &BlockLocation) -> bool {
        matches!((self, target),
            (GpuResident(..), Evicting) |
            (Evicting, CpuResident(_)) | (Evicting, NvmeResident(_)) |
            (CpuResident(_), Restoring) | (NvmeResident(_), Restoring) |
            (Restoring, GpuResident(..))
        )
    }

    fn transition_to(&mut self, target: BlockLocation) -> Result<()> {
        if !self.can_transition_to(&target) {
            return Err(anyhow!("illegal BlockLocation transition: {:?} → {:?}", self, target));
        }
        *self = target;
        Ok(())
    }
}
```

**Affected File**: `src/kcmm/pool.rs`

---

### Issue 12: FFI Module Has Type Definitions but No Exported Functions

**Location**: `src/kcmm/ffi.rs:61-75`

**Description**: C-compatible type definitions are complete (`kcmm_pool_t`, `kcmm_metrics_t`, `kcmm_hint_t`), but all `extern "C"` functions are commented out:

```rust
// extern "C" {
//     pub fn kcmm_pool_create(...) -> *mut kcmm_pool_t;
//     ...
// }
```

Without `#[no_mangle] extern "C" fn`, the `cdylib` target will produce a shared library with **zero exported symbols**, making it unlinkable by external engines.

**Fix**: Implement at least the P0-priority C API functions (design document F1-F6):
- `kcmm_pool_create` / `kcmm_pool_destroy`
- `kcmm_alloc_blocks` / `kcmm_free_blocks`
- `kcmm_touch` / `kcmm_cool`

Each function receives/returns `*mut kcmm_pool_t` (effectively an opaque pointer to `Box<Arc<KcmmPool>>`) and converts internally.

**Affected File**: `src/kcmm/ffi.rs`

---

### Issue 13: `TieringEngine` Has No Evict/Restore/Prefetch Methods

**Location**: `src/kcmm/tiering.rs:103-144`

**Description**: The `TieringEngine` struct only contains a CPU buffer pointer and size, with no methods for:
- `evict_blocks(count)` — GPU→CPU swap-out
- `restore_blocks(handles)` — CPU→GPU restore
- `prefetch_tick()` — background prefetch

`KcmmPool::below_low_watermark()` (pool.rs:683) documents that callers should "trigger tiering eviction" — but there is no eviction method to call. The `KcmmPool`'s `tiering` field exists but is unused by any code path.

**Fix**: Add at least method signatures (stubs with `todo!()`) before Week 14 begins, so callers can write integration code:

```rust
impl TieringEngine {
    pub fn evict_blocks(&self, pool: &KcmmPool, count: usize) -> Result<usize> {
        todo!("Week 14: select victims → D2H memcpy → unmap → mark CpuResident")
    }
    pub fn restore_block(&self, pool: &KcmmPool, block_idx: u32) -> Result<()> {
        todo!("Week 14: alloc physical → map → H2D memcpy → mark GpuResident")
    }
}
```

**Affected File**: `src/kcmm/tiering.rs`

---

## 🟢 Low Priority Issues

### Issue 14: `SharingManager` Methods Accept Data but Don't Store It

**Location**: `src/kcmm/sharing.rs:50-52`

`register_prefix` accepts `Vec<BlockHandle>` (taking ownership via move) and then discards it. Callers may expect the data to be stored. At minimum, store it in an internal structure (even if lookup logic is a stub).

**Affected File**: `src/kcmm/sharing.rs`

---

### Issue 15: `KcmmPool::Drop` Does Not Wait for Outstanding CUDA Stream Operations

**Location**: `src/kcmm/pool.rs:696-727`

`Drop` unmaps physical handles without first synchronizing CUDA streams. If there are outstanding eviction memcpy operations, physical memory could be reclaimed during an in-flight transfer. Fix: add `self.streams.synchronize_all().ok();` at the beginning of Drop.

**Affected File**: `src/kcmm/pool.rs`

---

### Issue 16: Build Dependency — Verify `libc` in Cargo.toml

**Location**: `src/kcmm/tiering.rs:111-121`

`tiering.rs` uses `libc::mmap`, `libc::munmap`, `libc::MAP_FAILED`. Confirm that the `libc` crate is listed in `Cargo.toml` under `[dependencies]`. If already an indirect dependency through other project crates, compilation will succeed, but it should be explicitly listed as a direct dependency.

**Affected File**: `Cargo.toml`

---

### Issue 17: Missing GPU-Free Unit Test Path

**Location**: `src/kcmm/pool.rs:753-758`

```rust
#[test]
fn test_below_low_watermark_empty_pool() {
    // We can't create a full KcmmPool without GPU, but test the logic
    // is covered by integration tests.
}
```

Substantial CPU-only logic (allocate/free, sequence lifecycle, watermark computation, BlockLocation transitions) should have unit test coverage. Consider abstracting GPU dependencies behind a trait (`trait GpuBackend`) so unit tests can use mock implementations.

**Affected Files**: `src/kcmm/pool.rs`, `src/kcmm/superblock.rs`, `src/kcmm/tiering.rs`

---

### Issue 18: Bytes-per-Token Calculation in `collect_metrics` Relies on Integer Division

**Location**: `src/kcmm/pool.rs:647`

```rust
let bpt_all = self.elem_per_block / self.block_size * num_layers * 2;
```

If `elem_per_block` is not evenly divisible by `block_size` (integer division truncation), the formula produces incorrect results. Consider storing a precomputed `bytes_per_token_all_layers` constant or computing with `f64` and then rounding.

**Affected File**: `src/kcmm/pool.rs`

---

## Fix Priority Ranking

| Priority | Issue | Severity | Estimated Effort |
|----------|-------|----------|-----------------|
| **P0** | #1 — Deadlock fix (unify lock order) | 🔴 Critical | 30 min |
| **P0** | #2 — Eliminate paged_kv.rs code duplication | 🔴 Critical | 2-4 hours |
| **P0** | #3 — TieringEngine file-based mmap | 🔴 Critical | 1 hour |
| **P0** | #4 — LruPolicy basic implementation | 🟠 High | 1-2 hours |
| **P1** | #5 — Add async cudaMemcpy operations | 🟠 High | 2-3 hours |
| **P1** | #6 — Add CUDA Event inter-stream sync | 🟠 High | 1-2 hours |
| **P1** | #7 — KCMM feature-gating | 🟠 High | 30 min |
| **P1** | #10 — Unify metric computation | 🟡 Medium | 2 hours |
| **P2** | #8 — Encapsulate KcmmPool fields | 🟡 Medium | 1-2 hours |
| **P2** | #9 — eviction_policy to enum | 🟡 Medium | 1 hour |
| **P2** | #11 — BlockLocation state transition validation | 🟡 Medium | 1-2 hours |
| **P2** | #12 — FFI extern "C" function stubs | 🟡 Medium | 2-3 hours |
| **P2** | #13 — TieringEngine method stubs | 🟡 Medium | 1 hour |
| **P3** | #14-#18 — Low-priority items | 🟢 Low | 3-4 hours |

- **P0**: Must fix before entering Week 14 (blocks subsequent development)
- **P1**: Complete during Week 14
- **P2**: Complete during Weeks 14-15
- **P3**: Fix as time allows

---

## Design Document Gap Summary

| Design Document Requirement | Current Status |
|----------------------------|----------------|
| `PhysicalBlockAllocator` extracted from paged_kv.rs | Type extracted, logic still duplicated |
| `PagedKvCache` as compatibility layer delegating to `KcmmPool` | Delegation not implemented; two independent codebases |
| `BlockInfo.in_use: bool` → `BlockLocation` enum | Implemented on KCMM side; paged_kv side not synced |
| `SeqMetadata` → `SequenceState` extension | Implemented on KCMM side; paged_kv retains old structure |
| CUDA Stream wrappers | Implemented |
| Async memcpy operations | Not implemented |
| CUDA Event inter-stream sync | Not implemented |
| Pluggable EvictionPolicy (LRU/LFU/FIFO) | Trait defined, implementations are stubs |
| CPU buffer mapped via `/dev/shm` file | Uses anonymous mapping |
| `#[cfg(feature = "kcmm")]` feature gating | Not used |
| C API `extern "C"` functions | Types defined, functions not implemented |
| Backward compatibility — original inference path unchanged | Needs verification (kcmm module compiles unconditionally) |
