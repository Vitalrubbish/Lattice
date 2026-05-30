# Runtime-Level Fragmentation Rate Calculator

**2026-05-30** written by **Vitalrubbish**

## Overview

Replaced the static, single-request fragmentation rate calculation with a
runtime-level tracker that captures fragmentation dynamics during multi-request
inference.  The old `step3_fragmentation_rate` test allocated all sequences at
a fixed length, freed 50%, then re-allocated — giving only a single snapshot.
The new `step3_runtime_fragmentation` test simulates realistic multi-request
scheduling and records fragmentation at every step, averaging over the entire run.

## What was implemented

### `RuntimeFragmentationTracker` (in `src/cache/paged_kv.rs`)

A sampling tracker embedded in the fragmentation test that records metrics at
each scheduler step:

- **`memory_allocated_not_free`** — physical GPU memory consumed by blocks
  that are NOT in the free pool.  Computed as
  `(total_blocks_allocated - free_blocks_in_pool) × block_bytes`.
  Includes CUDA VMM alignment overhead because each superblock is allocated
  at 2 MiB granularity via `cuMemCreate`.

- **`memory_active_tokens`** — memory occupied by active token KV data.
  Since all layers share the same physical memory (one superblock is mapped
  into every layer's K/V VA region), we count one layer's worth of K data:
  `total_tokens × kv_heads × head_dim × sizeof(f16)`.

- **`ratio`** = `memory_allocated_not_free / memory_active_tokens`.
  - 1.0 = perfect efficiency (every byte stores token data)
  - \>1.0 = overhead from block granularity (partial last blocks)

- **`average_ratio()`** — time-averaged fragmentation ratio across all samples.
- **`ratio_stddev()`**, **`peak_ratio()`**, **`min_ratio()`** — distribution statistics.

### Bug fix: `unregister_sequence`

Fixed `PagedKvCache::unregister_sequence` to clear `block_table` and set
`seq_len = 0` after freeing.  Previously the metadata was left intact,
causing `stats()` and `internal_fragmentation()` to count tokens from
already-freed sequences.  The clone-then-clear pattern is safe because
`self.free_sequence` only needs the block indices, not ongoing metadata.

### New GPU test: `step3_runtime_fragmentation`

Simulates 80 requests with prompt lengths sampled from the sonnet distribution
(145 samples: median 42, range 8–289, 5 size categories).  The simulation loop:

1. Admit pending requests until `max_batch` (32) is reached or OOM
2. Record fragmentation snapshot
3. Advance each running sequence by 4 decode tokens
4. Grow block tables as needed (capped when VRAM exhausted)
5. Record another snapshot
6. Remove and clean up completed sequences
7. Repeat until all 80 requests are processed

Reports: average ratio, stddev, peak/min, bucketed by concurrent load,
sample snapshots, and final cache state.

### Updated `scripts/step3_test_wsl2.sh`

- Added `step3_runtime_fragmentation` to the test filter
- Added `collect_gpu_test_metrics` parsing for runtime fragmentation metrics:
  `runtime_frag_avg_ratio`, `runtime_frag_peak_ratio`, `runtime_frag_stddev`

## Results (RTX 5070, WSL2, tiny_llama config)

```
avg runtime fragmentation ratio:  1.09
stddev:                            0.04
peak (worst):                      1.22
min (best):                        1.03
```

The ~9% average overhead is due to block-level granularity: the last block of
each sequence is partially empty on average.  With block_size=16 tokens,
sequences whose length is not a multiple of 16 waste `15 − (len%16)` slots.

## Files changed

| File | Change |
|------|--------|
| `src/cache/paged_kv.rs` | Added `RuntimeFragmentationTracker`, `FragmentationSample`; fixed `unregister_sequence`; added `step3_runtime_fragmentation` test |
| `scripts/step3_test_wsl2.sh` | Added test filter entry; added metric parsing |
