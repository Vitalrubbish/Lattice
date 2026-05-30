# PagedKvCache Refactoring for Step 3 Consistency

**Date:** 2026/05/30

## Changes

### 1. Remove epoch/LRU tracking (→ continuous_scheduler.rs)
- Remove `use super::swap::current_epoch`
- Remove `seq_last_epoch` field, `record_epoch()`, `get_seq_epoch()`
- Scheduler tracks epochs in its own `HashMap<usize, u64>`

### 2. Fix per-layer physical memory (data corruption bug)
- Each (layer, K/V) pair gets its own `PhysicalBlockAllocator` and superblock list
- All layers allocate physical blocks in lockstep → same `block_idx` across layers
- `map_superblock_to_layer()` maps a physical handle to exactly one layer's VA region
- `SuperblockInfo` removed; per-pool tracking via `Vec<LayerKvPool>`

### 3. Move RuntimeFragmentationTracker → `src/cache/fragmentation_tracker.rs`

### 4. Move GPU tests → `tests/step3_benchmarks.rs`

### 5. Deduplicate allocation: extract `alloc_one_block_internal()`

### 6. Fix `get_block_table()`: change swap.rs to iterate without cloning
