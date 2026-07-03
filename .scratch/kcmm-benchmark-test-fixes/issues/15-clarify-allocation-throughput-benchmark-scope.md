# Clarify scope of allocation throughput benchmark

Status: done
Type: AFK

## What to build

`kcmm_bench_alloc_throughput` reports alloc/free latencies around 140 ns. That timing corresponds to the CPU-side free-list metadata path inside `KcmmPool`, not to actual CUDA physical allocation such as `cuMemMap` or superblock creation. The benchmark name and output can be misread as measuring GPU allocation latency.

Rename the benchmark title and column headers to make it explicit that this is "pool allocator metadata path" latency. Consider whether an additional mode should measure the slow path (triggering a new superblock allocation) so there is data for real physical allocation cost; if that would expand scope too much, document it as a future slice instead.

## Acceptance criteria

- [ ] Benchmark title and printed output clearly state that the measured latency is the pool metadata/free-list path, not CUDA physical allocation.
- [ ] Assertions and existing behavior remain unchanged.
- [ ] A decision is recorded on whether to add a separate superblock-triggering allocation mode (implement or defer).
- [ ] `kcmm_bench_alloc_throughput`, `_pool_size_sweep`, and `_concurrent_sequences` all still pass.

## Verification

- `cargo test --features kcmm --release --test kcmm_bench_alloc -- --nocapture`
- Output now labels the tests as pool allocator metadata / free-list path latency.
- A separate superblock-triggering slow-path benchmark is explicitly deferred as future work.

## Blocked by

None - can start immediately.
