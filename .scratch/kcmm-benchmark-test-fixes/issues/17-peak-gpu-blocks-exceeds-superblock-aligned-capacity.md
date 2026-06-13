# Investigate Peak GPU blocks above superblock-aligned capacity

Status: done
Type: AFK

## What to build

Issue 09 documented that `total_physical_blocks()` and `free_physical_blocks()`
intentionally read one representative pool because K/V layer pools allocate and
free in lockstep. It also clarified that `max_blocks_total` is not the exact
physical capacity ceiling: each per-layer VA region is rounded up to the CUDA
2 MiB superblock granularity, so the expected physical ceiling is:

```text
ceil(max_blocks_total / blocks_per_superblock) * blocks_per_superblock
```

However, the existing integration result still showed `Peak GPU blocks` values
above that aligned ceiling for at least one config. Be careful with terminology:
the integration log's `block_bytes=65536` is the workload-wide, all-layer
footprint used for human-readable budget reporting. `KcmmPool`'s physical
allocator works per layer, so the same config uses 8192-byte per-layer blocks
and `blocks_per_superblock = 256`. With `max_blocks_total = 640`, the aligned
per-layer physical ceiling is therefore 768. A reported peak above 768 still
needs investigation.

Investigate whether the `Peak GPU blocks` metric is:

- counting the wrong object,
- comparing against the wrong capacity denominator,
- including temporary over-allocation during eviction/admission retry,
- observing a real lockstep/capacity bug, or
- affected by stale/free-list accounting.

## Acceptance criteria

- [x] The benchmark prints both `max_blocks_total` and the exact aligned
      per-layer physical ceiling used for comparison.
- [x] `blocks_per_superblock` is included in the diagnostic output.
- [x] `Peak GPU blocks` is asserted or warned against the aligned ceiling with
      a clear message.
- [x] If the metric is correct, the report explains what it counts and why it
      can exceed the apparent logical limit.
- [x] If the metric is wrong, fix the metric or rename it so it no longer
      suggests a physical capacity count.

## Blocked by

None - can start immediately.

## Resolution

The metric was correct: `Peak GPU blocks` is the representative per-layer
physical block count in use (`total_physical_blocks - free_physical_blocks`),
not a logical request count. It may exceed `max_blocks_total` because each
per-layer VA reservation is rounded up to CUDA's 2 MiB superblock granularity,
but it must not exceed:

```text
ceil(max_blocks_total / blocks_per_superblock) * blocks_per_superblock
```

The bug was real over-allocation. `KcmmPool::ensure_capacity()` created another
physical superblock whenever the free list was empty, without checking the
aligned physical ceiling. That allowed integration workloads to grow past the
per-layer physical capacity implied by the reserved VA range.

Fixes:

- Added `KcmmPool::max_physical_blocks_per_layer()` and made
  `ensure_capacity()` fail once `total_physical_blocks()` reaches that aligned
  ceiling.
- Added benchmark diagnostics for `max_blocks_total`,
  `blocks_per_superblock`, and `aligned_physical_ceiling`.
- Kept the integration benchmark warning when `Peak GPU blocks` exceeds the
  aligned ceiling.
- Made batched restore return an error if it restores only a subset of the
  requested blocks, instead of silently reporting success.

## Verification

- `cargo test --features kcmm test_alloc_stops_at_aligned_physical_capacity -- --nocapture --test-threads=1`
  passed.
- `cargo test --features kcmm --test kcmm_bench_engine_integration --no-run`
  passed.
- `cargo test --features kcmm --test kcmm_bench_tiering --no-run`
  passed.
- `scripts/run_kcmm_integration_bench.sh --single` passed:
  `results/kcmm_engine_integration_20260613_032231`.
  The single config prints `max_blocks_total=640`,
  `blocks_per_superblock=256`, `aligned_physical_ceiling=768`, and
  `Peak GPU blocks` is `766` OFF / `768` ON with no aligned-capacity warning.
- `scripts/run_kcmm_integration_bench.sh --sweep` passed:
  `results/kcmm_engine_integration_20260613_032650`.
  No `exceeded aligned` or `Peak GPU blocks` warning appears. The former stress
  config `bs16_mb10_msl384_pl[64,128,256]_mnt128_reqs40_ari4` now reports
  `max_blocks_total=240`, `blocks_per_superblock=256`,
  `aligned_physical_ceiling=256`, and still passes with `40/40` Tiering ON full
  completions.
