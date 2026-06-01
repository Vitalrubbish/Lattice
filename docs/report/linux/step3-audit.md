# Step 3 Audit — Misleading Metrics & Glossary Verification

**Date:** 2026-06-01

Audit of all fragmentation/metrics code paths against the CONTEXT.md glossary,
following the discovery that `step3_max_concurrent_requests` used hardcoded worst-case
pre-allocation rather than workload-driven admission.

---

## Found Issues

### 1. 🔴 `CacheStats.total_blocks_allocated` uses wrong source

**File:** `src/cache/paged_kv.rs:699`

```rust
let allocated = self.total_blocks();  // ← block_info.len(), peak distinct indices
```

`total_blocks()` returns `block_info.lock().len()` — the number of entries in the block-info
vector. This is the **peak number of distinct block indices ever assigned**, not the number
of blocks with physical backing.

When blocks are freed and indices recycled:

```
1 superblock (256 physical blocks)
Allocate 100 blocks  → block_info.len() = 100
Free 50 blocks       → block_info.len() = 100  (indices 0-49 recycled)
Allocate 60 blocks   → 50 reuse recycled indices, 10 new → block_info.len() = 110
→ CacheStats.total_blocks_allocated = 110
→ Actual physical blocks = 256
```

**Fix:** Change line 699 to `self.total_physical_blocks()`, which returns
`superblock_count × blocks_per_superblock`. The UFS `from_cache()` already uses
`total_physical_blocks()` correctly (line 122 of `unified_frag.rs`).

### 2. 🟡 `fragmentation_ratio()` still misnamed

**File:** `src/cache/paged_kv.rs:631-643`

Called "fragmentation ratio" but computes `(total_physical - physical_used) / total_physical` —
the proportion of allocated superblock capacity sitting idle in the free list.
This is allocator-idle-rate, not fragmentation.

**Current usage:** Only in `step3_max_concurrent_requests` test output, printed as
"physical memory waste ratio" — the test already renamed it in the print string.
The function name is the only thing wrong.

**Fix:** Rename to `physical_idle_ratio()` or `superblock_idle_ratio()`.
Low priority — not used in UFS or any comparison pipeline.

### 3. 🟡 Legacy ratio published alongside UFS metrics

**Files:** `src/batch/stats.rs:24-30`, `src/batch/continuous_scheduler.rs:524-526`

`StatsHandle` carries both:
- UFS metrics (`unified: Option<UnifiedFragMetrics>`, RFI avg/peak/stddev)
- Legacy ratio (`legacy_ratio_avg/peak/stddev` from `RuntimeFragmentationTracker`)

These come from the same data source and produce similar-but-different numbers
(legacy uses one-layer K bytes, UFS uses all-layer K+V bytes). Publishing both
creates confusion: which one is authoritative?

The legacy ratio was **fixed** in `record()` (now uses `active_superblocks × 2 MiB`
instead of hardcoded `BLOCK_BYTES`), so it's no longer buggy — but it's redundant.

**Fix:** Keep only the UFS path. Remove `legacy_ratio_*` from `StatsHandle` and
`StatsSnapshot`, or mark them `#[deprecated]` and stop updating them.

### 4. 🟡 `record()` and `record_unified()` both public

**File:** `src/cache/fragmentation_tracker.rs`

`record_unified()` internally calls `record()`, so both sample sets are maintained.
But `record()` is still a public method — callers could use it without the unified
counterpart. The only safe call site is `record_unified()`.

**Fix:** Make `record()` private, or fold its body into `record_unified()`.

### 5. 🔴 step3 "max concurrent requests" uses hardcoded blocks=16

**File:** `tests/step3_benchmarks.rs:52`

```rust
match cache.alloc_sequence(max_blocks_per_seq) {  // = 16, hardcoded
```

Every sequence pre-allocates 16 blocks (256 tokens) regardless of actual prompt length.
This measures "worst-case capacity" rather than "real workload capacity," and is
incomparable with vLLM's benchmark which uses small prompts (8-32 tokens).

**Covered by the existing fix plan (item 2).**

### 6. 🔴 vLLM `total_blocks_allocated` estimated from `nvidia-smi diff`

**File:** `scripts/bench_vllm_comprehensive.py:278-297`

```python
kv_cache_mib = max(0.0, used_mib - self._baseline_gpu_mem_mib)  # ≈ 0
total_blocks_allocated = max(estimated_from_tokens, kv_cache_bytes / ...)
```

The `nvidia-smi` baseline subtraction hides vLLM's pre-allocated pool, and the
`estimated_from_tokens` fallback makes `total_blocks_allocated ≈ blocks_in_use`,
giving a false BU ≈ 1.0.

**Covered by the existing fix plan (item 1).**

### 7. 🟢 IFR computed identically across all code paths — confirmed correct

Checked all three computation sites:

| Site | Method | Input |
|------|--------|-------|
| `internal_fragmentation()` | `(total_slots - total_tokens) / total_slots` | `seq.block_table.len()`, `seq.seq_len` |
| `CacheStats.internal_fragmentation` | same formula | same inputs |
| `UnifiedFragMetrics::from_cache().internal_frag_rate` | same formula | same inputs |

All three use the same formula and same data sources. No divergence. ✓

---

## Glossary Verification

Each CONTEXT.md term checked against actual code behavior:

| Term | Definition matches code? | Notes |
|------|:---:|-------|
| Paged KV Cache | ✓ | `PagedKvCache` + block_table translation |
| Block | ✓ | BLOCK_SIZE=16, block_bytes varies by model |
| BlockHandle | ✓ | Newly added; clarifies distinction from block index |
| Block Index | ✓ | Newly added; `block_info` vec index, recycled on free |
| Superblock | ✓ | 2 MiB, `cuMemCreate` |
| Block Table | ✓ | `SeqMetadata.block_table: Vec<u32>` |
| Lockstep Allocation | ✓ | Newly added; all pools allocate/free together |
| Free List | ✓ | Newly added; `PhysicalBlockAllocator.free_blocks` |
| Allocator Granularity | ✓ | 2 MiB baseline, full pool vLLM |
| Total Blocks Allocated | ⚠️ | Definition is correct (target); `CacheStats` uses wrong source (item 1) |
| IFR | ✓ | Formula confirmed at 3 sites |
| BU | ✓ | Definition correct; comparability note added |
| PME | ✓ | Definition correct |
| RFI | ✓ | Definition correct; all-layer BPT used |
| Pre-allocation | ✓ | vLLM gpu_memory_utilization |
| Grow-on-Demand | ✓ | Baseline cuMemCreate |
| Capacity-at-Workload | ⚠️ | Definition correct (target); not yet implemented (item 5) |
| Continuous Batching | ✓ | `ContinuousScheduler` |
| Loader | ✓ | Four variants in `bench_loaders.rs` |

---

## Summary

| Severity | Count | Items |
|----------|:---:|-------|
| 🔴 Bug (wrong value) | 2 | CacheStats source (1), vLLM measurement (6) |
| 🔴 Test design | 1 | Hardcoded blocks=16 (5) |
| 🟡 Naming/clarity | 3 | fragmentation_ratio name (2), legacy ratio duplication (3), public record() (4) |
| 🟢 Confirmed correct | 1 | IFR consistency (7) |

Items 1, 5, and 6 should be fixed before producing the next comparison report.
Items 2, 3, and 4 are cleanup that can follow.
