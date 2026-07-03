# KCMM Phase 1 Code Review — Rust Engine Integration

**Reviewer:** Claude
**Date:** 2026-06-11
**Specification:** `docs/dev/kcmm-ffi-roadmap.md` — §3 Phase 1 (Week 1–4)
**Files Reviewed:**
- `src/cache/backend.rs` (new, 106 lines) — `KvCacheBackend` trait
- `src/cache/mod.rs` (+1 line) — re-exports `backend` module
- `src/cache/paged_kv.rs` (+80 lines) — `impl KvCacheBackend for PagedKvCache`
- `src/cache/swap.rs` (modified) — `SwapManager` now uses `&dyn KvCacheBackend`
- `src/kcmm/pool.rs` (+250 lines) — `impl KvCacheBackend for KcmmPool`, `append_kv_step`, accessors
- `src/kcmm/tiering.rs` (baseline, ~3100 lines) — `TieringEngine::evict_blocks`, `EvictionPolicy` trait
- `src/model/transformer.rs` (modified) — `forward_step_paged` uses `&dyn KvCacheBackend`
- `src/model/llama_transformer.rs` (modified) — ditto, uses `cache.get_block_table()` etc.
- `src/batch/continuous_scheduler.rs` (+150 lines) — `CacheBackend` enum, KCMM touch/cool hooks
- `src/batch/mod.rs` (+1 line) — re-exports `CacheBackend`
- `src/main.rs` (+20 lines) — `--kcmm` CLI flag, `CacheBackend` factory

**Commit context:** Working tree on branch `kcmm`; builds on weeks 13–15 skeleton (pool, tiering, superblock, streams, metrics).

---

## 1. Executive Summary

Phase 1 introduces a `KvCacheBackend` trait that abstracts over `PagedKvCache` and `KcmmPool`, enabling the `Transformer` trait and `ContinuousScheduler` to work with either backend through a single code path. The trait design, `Transformer` refactoring, and `CacheBackend` enum infrastructure are well-executed and follow the roadmap specification closely.

**However, the KCMM eviction/restore integration in `ContinuousScheduler` is critically incomplete.** Three code paths unconditionally call `self.swap_manager.as_ref().unwrap()` which will **panic at runtime** when KCMM mode is enabled (`swap_manager` is `None`). This means the scheduler cannot handle OOM conditions in KCMM mode — it will crash on the first memory-pressure event.

**Verdict: NEEDS WORK — 3 critical blockers, 3 design recommendations, 2 test-gap observations.**

---

## 2. Completeness Assessment Against Roadmap

| Sub-task | Roadmap Spec | Status | Notes |
|----------|-------------|--------|-------|
| 1.1 `append_kv_step` in `KcmmPool` | §1.1 L72 | ✅ Complete | `pool.rs:879-942`, ported from `PagedKvCache` |
| 1.1 `get_all_block_offsets_f16` in `KcmmPool` | §1.1 L73 | ✅ Complete | `pool.rs:578-587` |
| 1.1 `with_seq_metadata` / accessors | §1.1 L74 | ✅ Complete (design change) | Individual accessors instead of closure — see §4.1 |
| 1.2 `KvCacheBackend` trait | §1.2 L83-118 | ✅ Complete | `backend.rs`, 22 methods |
| 1.2 `impl KvCacheBackend for PagedKvCache` | §1.2 L121 | ✅ Complete | `paged_kv.rs:662-736` |
| 1.2 `impl KvCacheBackend for KcmmPool` | §1.2 L122 | ✅ Complete | `pool.rs:1097-1171` |
| 1.3 `Transformer` trait → `&dyn KvCacheBackend` | §1.3 L129 | ✅ Complete | `transformer.rs:44-48` |
| 1.3 `NaiveTransformer` update | §1.3 L130 | ✅ Complete | `transformer.rs:134-161` |
| 1.3 `LlamaTransformer` update | §1.3 L130 | ✅ Complete | `llama_transformer.rs:262-388` |
| 1.4 `--kcmm` CLI flag | §1.4 L143 | ✅ Complete | `main.rs:56` (gated on `kcmm` feature) |
| 1.4 `CacheBackend` enum / factory | §1.4 L145 | ✅ Complete | `continuous_scheduler.rs:25-66`, `main.rs:106-141` |
| 1.4 `ContinuousScheduler` → `CacheBackend` | §1.4 L148 | ✅ Complete | Field type changed to `CacheBackend` |
| 1.4 KCMM eviction path | §1.4 L150-151 | ❌ **MISSING** | See §3.1 — panics in KCMM mode |
| 1.4 KCMM restore path | §1.4 L152 | ❌ **MISSING** | See §3.2 — panics in KCMM mode |
| 1.4 `touch()` / `cool()` calls | §1.4 L153-155 | ✅ Complete | `run_step` L521, `remove_completed` L429 |
| 1.4 Backward compatibility | §1.4 L158-160 | ✅ Complete | `#[cfg(feature = "kcmm")]` gates |
| 1.5 Correctness verification | §1.5 L164-168 | ⚠️ Partial | Pool/backend unit tests exist; no KCMM scheduler integration tests |
| 1.6 Integration benchmark | §1.6 L172-178 | ❌ Not started | No `kcmm_engine_integration` benchmark |

---

## 3. 🔴 Critical Issues

### 3.1 CRITICAL: KCMM OOM eviction path will panic (`admit_waiting`)

**Location:** `src/batch/continuous_scheduler.rs:290`

**Description:** When KCMM mode is enabled, `swap_manager` is set to `None` (L141-145). However, `admit_waiting` unconditionally calls `self.swap_manager.as_ref().unwrap()` on the OOM path:

```rust
// L268-321: admit_waiting OOM path
Err(_e) => {
    if free_blocks_available(self.backend.as_trait()) {
        // ...
        i += 1;
        continue;
    }

    // VRAM exhausted — try to evict a running sequence
    if let Some(victim_idx) = self.select_victim(running) {
        // ...
        match self.swap_manager.as_ref().unwrap().evict_sequence(  // ← PANICS
            self.backend.as_trait(),
            victim.seq_idx,
        ) {
```

**Impact:** The scheduler will panic on the first OOM event in KCMM mode. This is not a theoretical edge case — it triggers whenever `alloc_sequence()` fails due to pool exhaustion.

**Roadmap requirement (§1.4 L150-151):**
> "替换 `select_victim()` → 当 `kcmm_enabled` 时调用 `TieringEngine::evict_blocks()`"

**Required fix:** The OOM path must branch on `self.backend.is_kcmm()`:

```rust
Err(_e) => {
    if free_blocks_available(self.backend.as_trait()) {
        i += 1;
        continue;
    }

    #[cfg(feature = "kcmm")]
    if self.backend.is_kcmm() {
        // KCMM path: use TieringEngine to evict blocks
        if let Some(pool) = self.backend.kcmm_pool() {
            if let Some(ref tiering) = pool.tiering {
                // Collect candidate blocks from cooled/inactive sequences,
                // call tiering.evict_blocks(), retry alloc
                // ...
            }
        }
        i += 1; // fallback if eviction fails
        continue;
    }

    // Baseline path: use SwapManager
    if let Some(victim_idx) = self.select_victim(running) {
        // ... existing code ...
    }
}
```

### 3.2 CRITICAL: KCMM restore path will panic (`try_restore_swapped`)

**Location:** `src/batch/continuous_scheduler.rs:361-364`

**Description:** Same root cause — `try_restore_swapped` unconditionally calls `self.swap_manager.as_ref().unwrap()`:

```rust
match self
    .swap_manager
    .as_ref()
    .unwrap()                    // ← PANICS in KCMM mode
    .restore_sequence(self.backend.as_trait(), &sw.kv_data)
```

**Roadmap requirement (§1.4 L152):**
> "KCMM 模式调用 `pool.restore_evicted_blocks()`"

**Required fix:** Add KCMM branch:

```rust
#[cfg(feature = "kcmm")]
if self.backend.is_kcmm() {
    if let Some(pool) = self.backend.kcmm_pool() {
        match pool.restore_evicted_blocks(&sw.evicted_block_indices) {
            Ok(()) => {
                // re-register sequence, update seq_len, etc.
            }
            Err(_) => break,
        }
    }
    continue;
}
// Existing baseline path ...
```

### 3.3 CRITICAL: `drain_completed_swapped` will panic in KCMM mode

**Location:** `src/batch/continuous_scheduler.rs:469`

**Description:** Same unconditional `.unwrap()`:

```rust
self.swap_manager.as_ref().unwrap().drop_swapped(&sw.kv_data);  // ← PANICS
```

**Required fix:** Guard with `is_kcmm()` check — in KCMM mode, swapped data uses the tiering engine's CPU swap buffer, so cleanup should call the appropriate tiering free path instead of `SwapManager::drop_swapped`.

---

## 4. 🟡 Design Observations

### 4.1 `get_block_table()` clones on the hot path

**Location:** `src/model/llama_transformer.rs:329`

```rust
if let Some(bt) = cache.get_block_table(seq_idx) {
    for (j, &blk) in bt.iter().enumerate() {
```

`get_block_table()` returns `Option<Vec<u32>>` — it clones the entire block table on every call. This is invoked **per layer** (22× for TinyLLaMA, 32× for LLaMA-7B) inside `forward_step_paged`. For a sequence with 128 blocks, that's 128 × 4 bytes = 512 bytes cloned per layer, which is not enormous but does add up under high concurrency.

**Recommendation:** Consider a `with_block_table(seq_idx, |table| { ... })` callback approach or return `Option<&[u32]>` with appropriate lifetime management. Alternatively, cache the block tables on the stack once before the per-layer loop — the block table doesn't change during a forward step.

**Severity:** Low. Performance, not correctness.

### 4.2 `append_kv_step` code duplication

**Location:** `paged_kv.rs:465-530` vs `pool.rs:879-942`

The two implementations are nearly identical (95%+ overlap). Both:
1. Lock the sequence table and block_info
2. Iterate over batch, computing `logical_block` and `offset_in_block`
3. Compute source/destination VA offsets
4. Issue `cuMemcpyDtoDAsync_v2` for K and V

The only difference is `self.seq_metadata.lock()` vs `self.sequences.lock()`.

**Observation:** The roadmap acknowledges this is a "port" (§1.1 L72). This duplication is acceptable for now given that the two types have different internal state (`SeqMetadata` vs `SequenceState`). A future refactoring could extract the common loop logic into a free function parameterised by a `Fn(usize) -> (&[u32], usize)` closure, but this is low priority.

### 4.3 `free_blocks_available` doesn't account for KCMM tiering

**Location:** `src/batch/continuous_scheduler.rs:665-667`

```rust
fn free_blocks_available(cache: &dyn KvCacheBackend) -> bool {
    cache.has_free_blocks()
}
```

In KCMM mode, `has_free_blocks()` checks the allocator's free list. But if the pool is full with `CpuResident` blocks, the allocator has 0 free blocks yet the tiering engine could restore them. The function returns `false`, causing `admit_waiting` to enter the OOM path even though GPU blocks could theoretically be made available by restoring evicted blocks.

**Recommendation:** This is acceptable as-is because the OOM path in KCMM mode should trigger `restore_evicted_blocks()` rather than eviction (see §3.2). The function name is misleading in a KCMM context — consider renaming to `allocator_has_free_blocks` or adding a KCMM-aware variant.

**Severity:** Low. Currently masked by the critical issues in §3.1–3.3.

---

## 5. 🟢 Architecture Highlights

### 5.1 Trait design avoids deadlock-prone closure pattern

**Roadmap §1.2 L125** flagged `with_seq_metadata` closures as a potential deadlock risk:

> "`with_seq_metadata` 的闭包方式持有锁期间暴露引用，可能导致死锁"

The implementation chose individual accessor methods (`get_block_table()`, `get_seq_len()`, `get_block_va_offsets()`) instead of the closure-based `with_seq_metadata`. Each accessor acquires and releases the lock internally, returning owned data (`Vec<u32>`, `Vec<usize>`, `usize`). This eliminates the deadlock risk entirely. **Good design decision.**

### 5.2 `CacheBackend` enum enables clean feature-gating

The enum approach (`continuous_scheduler.rs:25-66`) provides a single-owner handle that:
- Avoids `Any` downcasting
- Provides typed accessors (`kcmm_pool()`, `paged_kv()`) for backend-specific operations
- Gates KCMM variants behind `#[cfg(feature = "kcmm")]` so the baseline binary has zero KCMM overhead
- `is_kcmm()` enables branching at runtime without feature-flag divergence

This is a clean pattern that should be extended to the eviction/restore paths.

### 5.3 `SwapManager` already migrated to `&dyn KvCacheBackend`

**Location:** `src/cache/swap.rs:49-55, 145-147`

The `SwapManager` was refactored to accept `&dyn KvCacheBackend` instead of `&PagedKvCache`:

```rust
pub fn evict_sequence(&self, cache: &dyn KvCacheBackend, seq_idx: usize) -> Result<EvictedSeqData>
pub fn restore_sequence(&self, cache: &dyn KvCacheBackend, data: &EvictedSeqData) -> Result<Vec<u32>>
```

This means `SwapManager` is already polymorphic and could theoretically work with `KcmmPool` through the trait — it's only the scheduler's KCMM branch routing that's missing.

### 5.4 KCMM `touch()`/`cool()` integration is in the right places

The `touch()` call in `run_step` (L521) marks all running sequences as recently accessed, and `cool()` in `remove_completed` (L429) marks completed sequences as eviction candidates. These are correctly placed and feature-gated.

---

## 6. Test Coverage Gaps

### 6.1 No KCMM-mode scheduler integration tests

**Location:** `src/batch/continuous_scheduler.rs:669-1013`

The existing integration tests (`e2e_continuous_single_request`, `e2e_multiple_requests`, etc.) all use `CacheBackend::Baseline`. There are no tests that:

1. Construct a `ContinuousScheduler` with `CacheBackend::Kcmm(...)` and verify it runs without panicking
2. Exercise the KCMM touch/cool paths through a full request lifecycle
3. Test the OOM eviction path in KCMM mode (once implemented)
4. Test the restore path in KCMM mode (once implemented)

**Roadmap requirement (§1.5 L164-168):**
> "验证 KCMM 路径下序列生命周期（admit → touch → decode → cool → evict → restore）不丢失 block、不 double-free"

**Recommendation:** After fixing the critical issues in §3, add at minimum:
- `e2e_kcmm_single_request` — basic lifecycle with `KcmmPool` backend
- `e2e_kcmm_oom_eviction` — fill pool, trigger eviction, verify restore
- `e2e_kcmm_lockstep_invariant` — verify per-layer pool invariants hold through evict/restore

### 6.2 `KvCacheBackend` trait lacks documentation tests

The trait definition in `backend.rs` has no doc comments on individual methods explaining:
- Expected behaviour for out-of-bounds indices
- Thread-safety guarantees (all methods take `&self` implying internal synchronisation)
- Whether methods can block (e.g., `alloc_block` may trigger `ensure_capacity` which calls `cuMemCreate`)

**Recommendation:** Add `///` doc comments to the trait methods, particularly for error semantics.

---

## 7. Detailed Per-File Notes

### 7.1 `src/cache/backend.rs`

- **L12**: `Send + Sync` bound is correct — both `PagedKvCache` and `KcmmPool` use `parking_lot::Mutex` internally and are safe to share across threads.
- **L49**: `get_block_va_offsets` returns `Option<Vec<usize>>` — consistent with the pattern of returning owned data to avoid holding locks across call boundaries.
- **L67**: `get_all_block_offsets_f16` returns `Vec<u64>` — allocated per call; the caller in `llama_transformer.rs:278` uploads this to GPU. This is acceptable for the current decode-batch size but could be optimised with a reusable GPU buffer later.
- **L105**: `active_sequences` — used for monitoring. Not on the hot path.

### 7.2 `src/cache/paged_kv.rs` (trait impl)

- **L662-736**: The `impl KvCacheBackend for PagedKvCache` block delegates each trait method to the corresponding inherent method. The inherent methods remain `pub` for direct callers that don't go through the trait, preserving backward compatibility.
- **L697-698**: `va_k()` and `va_v()` delegate to inherent methods which return the stored `u64` values directly (no locking). Correct since VA regions are immutable after construction.

### 7.3 `src/kcmm/pool.rs` (trait impl + new methods)

- **L578-587**: `get_all_block_offsets_f16()` — correctly handles inactive blocks (returns 0), matching `PagedKvCache` semantics.
- **L879-942**: `append_kv_step()` — correct port. Uses `self.sequences.lock()` + `self.block_info.lock()`. Note that this holds **two locks simultaneously** — the ordering is `sequences` then `block_info`, which must be consistent with all other callers. The existing lock-ordering test (L1420-1524) only covers `free_block_indices` + `block_info`, not `sequences` + `block_info`. **This is not a deadlock risk in practice** because `sequences` is never acquired while holding `block_info` elsewhere (verified by code review: all other `sequences` lock sites only access `sequences`).
- **L1097-1171**: Trait impl — clean delegation. All methods are thin wrappers.
- **L1095**: `use crate::cache::backend::KvCacheBackend;` — import is inside the file rather than at the top. This is fine for conditional compilation (the `kcmm` module is feature-gated) but slightly unusual style.

### 7.4 `src/model/transformer.rs`

- **L44-48**: `forward_step_paged` signature changed from `cache: &PagedKvCache` to `cache: &dyn KvCacheBackend`. Breaking change for any external callers — but since this is the project's own trait, that's fine.
- **L134-161**: `NaiveTransformer::forward_step_paged` — uses `cache.append_kv_step(i, seq_indices, positions, hidden, hidden)` with identical K and V sources (zero-weight model). Matches the `append_step` convenience pattern.
- **L49-65**: `prefill_paged` default implementation — unchanged logic, now uses `&dyn KvCacheBackend` through the updated `forward_step_paged` signature.

### 7.5 `src/model/llama_transformer.rs`

- **L278**: `cache.get_all_block_offsets_f16()` — allocated per forward step. Acceptable for now; see §7.1 note.
- **L327**: `cache.get_seq_len(seq_idx)` — called per batch element, lightweight integer return.
- **L329**: `cache.get_block_table(seq_idx)` — clones Vec per call; see §4.1.
- **L317**: `cache.append_kv_step(li, seq_indices, positions, &k, &v)` — correct: K and V are separate post-projection, post-RoPE tensors.

### 7.6 `src/batch/continuous_scheduler.rs`

- **L25-66**: `CacheBackend` enum — clean design. `as_trait()` is the key dispatch method. `kcmm_pool()` and `paged_kv()` return `Option<&T>` for backend-specific operations.
- **L141-145**: `swap_manager` construction — correctly sets `None` for KCMM. But the rest of the scheduler doesn't handle `None` (see §3.1–3.3).
- **L427-430**: KCMM `cool()` call — correctly gated behind `#[cfg(feature = "kcmm")]`. Called before `unregister_sequence`, which is correct: the blocks are still valid at this point and can be marked as eviction candidates.
- **L519-522**: KCMM `touch()` call — correctly gated. Called on every `run_step` for all running sequences.
- **L584**: `record_fragmentation_snapshot` — in KCMM mode, the tracker records nothing (only records for `PagedKvCache`). `KcmmPool` has its own `collect_metrics()` but it's not wired into the stats handle. This is a minor observability gap.

### 7.7 `src/main.rs`

- **L54-56**: `--kcmm` CLI flag — correctly gated behind `#[cfg(feature = "kcmm")]`.
- **L106-141**: Backend factory — clean conditional compilation. The `#[cfg(not(feature = "kcmm"))]` block provides a fallback that ignores the `--kcmm` flag (it simply isn't available as a CLI option).
- **L117**: `tracing::info!("using KCMM pool backend (tiering={})", pool.tiering.is_some())` — good: logs whether tiering is active.

---

## 8. Recommendations Summary

| # | Severity | Description | Section |
|---|----------|-------------|---------|
| C1 | 🔴 Critical | `admit_waiting` panics in KCMM mode on OOM | §3.1 |
| C2 | 🔴 Critical | `try_restore_swapped` panics in KCMM mode | §3.2 |
| C3 | 🔴 Critical | `drain_completed_swapped` panics in KCMM mode | §3.3 |
| D1 | 🟡 Medium | `get_block_table()` clones on hot path (per-layer) | §4.1 |
| D2 | 🟡 Low | `append_kv_step` duplicated between `PagedKvCache` and `KcmmPool` | §4.2 |
| D3 | 🟡 Low | `free_blocks_available` semantics misleading in KCMM context | §4.3 |
| T1 | 🟡 Medium | No KCMM-mode scheduler integration tests | §6.1 |
| T2 | 🟢 Low | `KvCacheBackend` trait methods need doc comments | §6.2 |
| O1 | 🟢 Info | KCMM metrics not wired into `record_fragmentation_snapshot` | §7.6 |

---

## 9. Migration Path

The critical issues (§3.1–3.3) should be fixed before any KCMM-mode testing can proceed. The recommended implementation order:

1. **Fix C1** — Add KCMM branch in `admit_waiting`: when `is_kcmm()` and OOM, call `pool.tiering.as_ref().unwrap().evict_blocks()` to free GPU blocks, then retry allocation.
2. **Fix C2** — Add KCMM branch in `try_restore_swapped`: when `is_kcmm()`, call `pool.restore_evicted_blocks()` for sequences with `CpuResident` blocks.
3. **Fix C3** — Guard `drain_completed_swapped` with `is_kcmm()` check; in KCMM mode, the tiering engine manages CPU buffers.
4. **Add T1** — Write integration tests for the KCMM scheduler path.
5. **Address D1** — Cache block tables on the stack before the per-layer loop in `LlamaTransformer::forward_step_paged`.

Items D2, D3, O1, and T2 can be deferred to later phases.

---

## 10. Phase 1 Readiness for Phase 2

Phase 2 (C FFI) depends on Phase 1 being functionally complete — the scheduler must be able to run end-to-end with the KCMM backend to validate that the FFI-exposed pool operations (alloc/free/touch/cool) behave correctly under load.

**Current status:** Phase 1 is **not ready** to gate Phase 2. The critical scheduler gaps mean a `--kcmm --continuous` run will panic on the first OOM event, making it impossible to validate FFI behaviour in a realistic multi-request scenario.

**Estimated effort to unblock:** 2–3 days (fix C1-C3 + basic integration test).
