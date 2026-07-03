# KCMM Swapping Policy: Design & Test Review

> **Review Date**: 2026-06-05
> **Review Scope**: `src/kcmm/tiering.rs`, `src/kcmm/pool.rs`, `src/kcmm/streams.rs`, test modules
> **Context**: Week 13 skeleton complete; entering Week 14-15 core implementation
> **References**:
> - `docs/task/kcmm-implementation-analysis.md`
> - `docs/dev/kcmm-swap-policy-impl.md`
> - `docs/dev/kcmm-week13-fixes.md`
> - `docs/cr/kcmm-code-review-fixes.md`

---

## 1. Current State

Week 13 delivered: `KcmmPool` with alloc/free, `TieringEngine` with file-backed mmap, `EvictionPolicy` trait with LRU/LFU/FIFO implementations, dedicated CUDA streams. 76 tests pass.

**Intentionally deferred to Week 14-15**: GPU↔CPU data movement, async memcpy, CPU buffer slot management, `BlockLocation` state transitions, policy-pool integration, batch optimization, NVMe tier, C FFI bodies.

---

## 2. Issues Requiring Fix Before Week 14

These are small design adjustments. If left unaddressed, they force rework or cause bugs when Week 14 implementation begins.

### 2.1 `EvictionPolicy` Trait Missing `on_allocate` Callback

**Severity**: Design Defect
**File**: `src/kcmm/tiering.rs:27-37`

The trait has `on_access` and `on_evict` but no `on_allocate`:

```rust
pub trait EvictionPolicy: Send + Sync {
    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle>;
    fn on_access(&mut self, block: BlockHandle);
    fn on_evict(&mut self, block: BlockHandle);
}
```

This creates semantic problems:

| Policy | What allocation needs | Current workaround | Viable? |
|--------|----------------------|-------------------|---------|
| LRU | Record first-access time | `on_access` suffices | Acceptable |
| LFU | Initialize count to 1 | `on_access` increments 0→1 | Acceptable |
| FIFO | Record **allocation** time | `on_access` records first-**access** time | **Semantically wrong** |

For FIFO, a block allocated but never accessed is never tracked, and therefore never evictable. The trait conflates "allocation" with "first access." Additionally, `FifoPolicy` only inserts on first call via `or_insert_with` — if `on_access` is never called, the block is invisible to `select_victims`.

**Fix**:

```rust
pub trait EvictionPolicy: Send + Sync {
    fn on_allocate(&mut self, block: BlockHandle);
    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle>;
    fn on_access(&mut self, block: BlockHandle);
    fn on_evict(&mut self, block: BlockHandle);
}
```

- `LruPolicy::on_allocate`: insert `Instant::now()` (same as current `on_access`)
- `LfuPolicy::on_allocate`: insert count = 1
- `FifoPolicy::on_allocate`: insert `Instant::now()` (move from `on_access`); `on_access` becomes no-op

**Estimated effort**: 30 min.

---

### 2.2 `on_access` / `on_evict` Use `&mut self` Inconsistently

**Severity**: API Design
**File**: `src/kcmm/tiering.rs:33,36`

`select_victims` takes `&self` but `on_access`/`on_evict` take `&mut self`. All internal state is behind `parking_lot::Mutex`, which only needs `&self` to lock. The `&mut self` requirement:

- Forces callers to hold a mutable reference to `TieringEngine`
- Prevents concurrent `on_access` from multiple inference threads
- Is inconsistent with the Mutex-based interior mutability pattern

**Fix**: Change all three mutation methods to `&self`:

```rust
fn on_allocate(&self, block: BlockHandle);
fn on_access(&self, block: BlockHandle);
fn on_evict(&self, block: BlockHandle);
```

**Estimated effort**: 5 min.

---

### 2.3 `eviction_policy` Field Is `pub`

**Severity**: Encapsulation
**File**: `src/kcmm/tiering.rs:197`

```rust
pub eviction_policy: Box<dyn EvictionPolicy>,
```

External code can replace the policy at runtime without going through `TieringEngine`. When Week 14 adds in-flight eviction operations, a surprise policy swap could leave blocks in inconsistent states.

**Fix**:

```rust
pub(crate) eviction_policy: Box<dyn EvictionPolicy>,
```

Add a setter later if runtime policy switching is needed.

**Estimated effort**: 2 min.

---

### 2.4 `unsafe impl Sync` for `TieringEngine` — No Internal Synchronization for `cpu_buffer`

**Severity**: Thread Safety (latent — triggers when Week 14 adds reads/writes to mmap)
**File**: `src/kcmm/tiering.rs:279-280`

```rust
unsafe impl Send for TieringEngine {}
unsafe impl Sync for TieringEngine {}
```

`cpu_buffer: *mut u8` is neither `Send` nor `Sync`. The manual `unsafe impl Sync` asserts it is safe for concurrent access, but:

- The mmap region has no Mutex or atomic protection
- Week 14 Phase B will write D2H data; Phase C will read H2D data
- Concurrent evict (write) + restore (read) to overlapping regions = **data race (UB)**
- `MAP_SHARED` extends the risk to cross-process races

**Fix**: Wrap CPU buffer slot management in a `Mutex<CpuSlotAllocator>`:

```rust
pub struct TieringEngine {
    cpu_buffer: *mut u8,
    cpu_buffer_size: usize,
    cpu_buffer_path: String,
    nvme_enabled: bool,
    eviction_policy: Box<dyn EvictionPolicy>,
    // NEW: serializes access to the mmap'd CPU buffer
    slot_allocator: Mutex<CpuSlotAllocator>,
}
```

`CpuSlotAllocator` tracks free/busy byte ranges, ensuring no two operations use the same region concurrently.

**Estimated effort**: 1-2 hours.

---

## 3. Issues Resolved by Planned Week 14-15 Development

These appeared problematic in a static review but are covered by the implementation plan. They need **no separate fix now** — only verification that the plan addresses them.

### 3.1 KcmmPool ↔ EvictionPolicy Disconnection

**Current**: `touch()`/`cool()` update `SequenceState` but never call `eviction_policy`.

**Week 14 Plan**:
- Phase B.2 (`evict_single_block`): calls `self.eviction_policy.on_evict(block)` after eviction
- Phase C.1 (`restore_block`): calls `self.eviction_policy.on_access(block)` after restore

**Missing from plan**: `KcmmPool::touch()` must call `eviction_policy.on_access()` for each block in the sequence. This integration point should be explicitly added to the Phase B/C task list.

**Verdict**: Mostly resolved. One integration call needs explicit inclusion.

### 3.2 `select_victims` Silently Returns Fewer Than Requested

**Current**: `filter_map` skips untracked blocks; caller cannot distinguish "no eligible victims" from "empty candidates."

**Week 14 Plan**: Phase B.2 `evict_blocks()` iterates over whatever `select_victims` returns. The `Evicting` intermediate state prevents double-eviction.

**Verdict**: Resolved by caller semantics. Add a `tracing::warn!` when `returned < requested` during Phase B.

### 3.3 Block-Level vs. Sequence-Level Granularity Mismatch

**Current**: Policies operate on `BlockHandle`; real-world eviction operates on sequences.

**Plan**: Design doc Section 1.3 explicitly documents the granularity change. Section 0.6 risk mitigation: "start with single-block, validate, then batch."

**Verdict**: Intentional architectural decision. Resolved by phased approach.

### 3.4 Two Swap Mechanisms Coexisting

**Current**: `cache/swap.rs` (SwapManager, working) vs. `kcmm/tiering.rs` (TieringEngine, skeleton).

**Plan**: Design doc Section 3.4: "SwapManager evolves into TieringEngine; preserve sequence-level API as backward-compatible layer." Week 15 includes swap.rs refactoring.

**Verdict**: Gradual migration. Not an issue at current stage.

### 3.5 `cpu_buffer_size` Estimation

**Current**: `config.max_blocks * config.block_size * 2` with `// TODO`.

**Plan**: Week 14 Phase B.1 (CPU slot management) requires correct `block_bytes`, available from `KcmmPool`. Either add `cpu_buffer_size` to `KcmmConfig` or compute from model dimensions.

**Verdict**: Known TODO. Resolved in Phase B.1.

### 3.6 Missing `BlockLocation` State Transition Validation

**Current**: Plain enum, no transition enforcement.

**Plan**: Phase B.2/C.1 explicitly use `Evicting`/`Restoring` intermediate states. Transition validation added via `set_block_location()` on `KcmmPool`.

**Verdict**: Resolved in Phase B/C.

### 3.7 Missing Async memcpy + CUDA Events

**Current**: `KcmmStreams` exists but has no memcpy methods.

**Plan**: Phase D explicitly adds `memcpy_d2h_async()` / `memcpy_h2d_async()` to `CudaStream`. CUDA Events are P1.

**Verdict**: Resolved in Phase D.

---

## 4. Test Coverage Gaps

### 4.1 Should Fill in Week 14

| Gap | Rationale | When |
|-----|----------|------|
| Concurrent policy access (multi-thread `on_access` + `select_victims`) | Validates Mutex correctness | After Phase A |
| `touch()` → `eviction_policy.on_access()` integration | Verifies the integration point | After Phase B/C |
| `select_victims` with realistic block counts (16K+) | Default `max_blocks` = 16384; tests use <10 | After Phase A |
| `below_low_watermark()` via actual `KcmmPool` | Currently only raw math | Requires GPU |

### 4.2 Minor (Fix When Convenient)

| Gap | File |
|-----|------|
| LFU missing `count > candidates.len()` test | `tiering.rs` (LRU and FIFO have it) |
| No test for `select_victims` returning fewer than requested | `tiering.rs` |
| `cpu_buffer_path` field stored but never read | `tiering.rs:193` |

---

## 5. Action Plan Summary

### Before Week 14 (~2 hours)

```
1. Add on_allocate to EvictionPolicy trait + all impls        (30 min)
2. Change on_access/on_evict/on_allocate to &self             (5 min)
3. Change eviction_policy to pub(crate)                       (2 min)
4. Add Mutex<CpuSlotAllocator> for cpu_buffer safety          (1-2 hr)
```

### During Week 14 (integration verification)

```
5. In KcmmPool::touch(): call eviction_policy.on_access()
6. In evict_blocks(): warn when select_victims returns < requested
7. Resolve cpu_buffer_size TODO (use actual block_bytes)
8. Add concurrent policy access tests
```

### During Week 15 (hardening)

```
9. Add BlockLocation::can_transition_to() validation
10. Add LFU count-exceeds-candidates test
11. Use cpu_buffer_path in tracing or remove it
```

### Verification

```bash
cargo test --features kcmm
cargo check --features kcmm
cargo check                              # without kcmm feature
```

---

## 6. Summary

| Category | Count | Action |
|----------|-------|--------|
| Must fix before Week 14 | 4 | ~2 hours — blocks clean Week 14 start |
| Resolved by Week 14-15 plan | 7 | Verify integration points during implementation |
| Test gaps for Week 14 | 4 | Add during Phase A/B/C |
| Minor cleanup | 3 | Any time |

The Week 13 skeleton is sound. The four pre-Week-14 fixes are small, well-scoped design adjustments. Everything else is already planned or a test gap fillable incrementally.
