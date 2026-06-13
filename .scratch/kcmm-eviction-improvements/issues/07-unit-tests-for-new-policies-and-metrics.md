# Write unit tests for new eviction policies and metrics

Status: todo
Type: AFK

## What to build

Commit `9755b31` added three new `EvictionPolicy` implementations (`ArcPolicy`, `SinkWindowPolicy`), a `SequencePriority` enum, and eviction metrics (`LatencyHistogram`, `PolicyStats`). These are pure Rust logic — no CUDA dependency — but have zero dedicated tests.

Add unit tests covering:

### ArcPolicy
- `on_allocate` inserts into T1 on first access, T2 on B1 ghost hit
- `on_access` promotes from T1 to T2, moves to MRU within T2
- `select_victims` evicts from T1 before T2, LRU within each tier
- `on_evict` populates B1/B2 ghost lists, bounded by capacity
- Ghost hit (B1/B2) correctly adjusts adaptive `p` value
- Scan resistance: sequential access pattern doesn't evict the T2 frequent set
- `replace()` evicts correct victim when over capacity

### SinkWindowPolicy
- Sink blocks (first `S`) are never returned by `select_victims`
- Window blocks (last `W`) are never returned by `select_victims`
- Middle-region blocks are sorted by LRU (oldest access first)
- `register_block` correctly associates block with sequence position
- Edge case: sequence with ≤ `S + W` blocks has zero eligible victims

### SequencePriority
- `select_victim` evicts `Evictable` before `Low` before `Normal` before `High`
- Within same priority tier, falls back to epoch-based LRU with block-count tiebreaker
- `kcmm_hint` FFI sets correct priority class per hint type

### LatencyHistogram
- `record()` correctly buckets samples into [0,100), [100,250), ... buckets
- `avg_us()` computes correct mean
- `min_us` / `max_us` track extrema across multiple records
- Zero samples → `avg_us() == 0.0`, all buckets empty

### PolicyStats
- `avg_evict_batch_size` running average is numerically correct

## Acceptance criteria

- [ ] `cargo test --features kcmm -- 'kcmm::tiering::tests'` passes (new test module)
- [ ] `cargo test --features kcmm -- 'kcmm::metrics::tests'` passes (extended with histogram tests)
- [ ] `cargo test --features kcmm -- 'kcmm::pool::tests'` passes (extended with priority tests)
- [ ] All new tests are `#[cfg(test)]` only, no CUDA dependency
- [ ] ArcPolicy scan-resistance test demonstrates correct T2 retention

## Blocked by

None — can start immediately. These are pure Rust unit tests, no GPU needed.
