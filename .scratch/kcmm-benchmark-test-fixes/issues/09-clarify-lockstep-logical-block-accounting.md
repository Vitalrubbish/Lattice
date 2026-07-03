# Clarify lockstep logical block accounting in pool metrics

Status: done
Type: AFK

## What to build

`KcmmPool` and `PagedKvCache` implement `total_physical_blocks()` and `free_physical_blocks()` by reading only `k_pools[0]`. Per the project glossary this is intentional: per-layer K and V pools allocate and free in lockstep, so the logical physical block count is identical across all pools and can be queried from any one of them. However, this invariant is not obvious from the code and has already been misread as a bug.

Add inline documentation/comments to these functions explaining the lockstep invariant. Add a regression test that exercises allocation, eviction, and free paths and asserts that every K and V layer pool reports the same `total_blocks_allocated()` and `free_count()`. Also investigate and document why the integration benchmark currently reports `Peak GPU blocks` values that appear to exceed `max_blocks_total` (e.g., alignment slack in VA reservation, a separate capacity bug, or a metric definition mismatch).

## Acceptance criteria

- [ ] `KcmmPool::total_physical_blocks()` and `free_physical_blocks()` have comments referencing the lockstep allocation glossary entry.
- [ ] `PagedKvCache::total_physical_blocks()` and `free_physical_blocks()` have the same clarification.
- [ ] A regression test verifies that after allocations, evictions, and frees, all K/V layer pools report identical `total_blocks_allocated()` and `free_count()`.
- [ ] The integration benchmark `Peak GPU blocks > max_blocks_total` observation is explained in a code comment or benchmark output note, and any real capacity bug is filed separately.
- [ ] No behavior of `total_physical_blocks()` or `free_physical_blocks()` is changed; only documentation/tests are added.

## Notes / follow-up

The `Peak GPU blocks > max_blocks_total` observation is now documented in the
benchmark code. For the integration config the per-layer VA region is rounded up
to the next 2 MiB superblock, so peaks up to `ceil(max_blocks_total /
blocks_per_superblock) * blocks_per_superblock` (768 blocks for the current
config) are expected VA alignment slack. Reported values above that aligned
capacity (e.g. ON=796) are tracked as a potential real capacity/lockstep bug in
`17-peak-gpu-blocks-exceeds-superblock-aligned-capacity.md`.

## Verification

- `cargo test --features kcmm test_lockstep_invariant_after_alloc_evict_free -- --nocapture`
- `scripts/run_kcmm_integration_bench.sh --single`
- `scripts/run_kcmm_integration_bench.sh --sweep`

## Blocked by

None - can start immediately.
