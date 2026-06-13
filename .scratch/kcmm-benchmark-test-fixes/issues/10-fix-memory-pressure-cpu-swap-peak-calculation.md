# Fix CPU swap peak calculation in memory pressure benchmark

Status: done
Type: AFK

## What to build

The memory pressure benchmark reports `cpu_swap_peak_bytes` to show the maximum CPU swap space KCMM holds at any point. The current implementation increments an `eviction_count` per eviction round and computes peak swap as `eviction_count × block_bytes`. This is wrong in two ways:

1. It counts rounds, not blocks: `evict_coldest_blocks` evicts at least `MIN_BATCH = 8` blocks per round, so the reported peak can under-estimate true usage by up to 8×.
2. It is cumulative: restored or freed blocks are not subtracted, so the metric grows monotonically and does not represent the *current* CPU-resident footprint.

Change the benchmark to track the live CPU-resident byte count. After each successful eviction add the evicted bytes, after each restore or free subtract the bytes, and keep the maximum value observed. Use the tiering engine's own accounting if it already tracks current CPU swap usage; otherwise maintain the counter inside the workload runner.

## Acceptance criteria

- [ ] `peak_cpu_swap_bytes` represents the maximum observed *live* CPU-resident bytes, not cumulative migrated bytes.
- [ ] Bytes are added on eviction and subtracted on restore/free.
- [ ] The single-config and sweep log lines for `cpu_swap_peak` are consistent with the number of currently CpuResident blocks × `block_bytes` at the moment of peak.
- [ ] Existing memory pressure assertions continue to pass.
- [ ] `kcmm_bench_memory_pressure_single` and `_sweep` are re-run and the new numbers are reviewed for sanity.

## Verification

- `cargo test --features kcmm --release --test kcmm_bench_memory_pressure kcmm_bench_memory_pressure_single -- --nocapture`
- `cargo test --features kcmm --release --test kcmm_bench_memory_pressure kcmm_bench_memory_pressure_sweep -- --nocapture`
- New single-config `cpu_swap_peak=43253760 B`, consistent with 30 eviction rounds × 8 evicted blocks × 180224 B/block at peak live residency.

## Blocked by

None - can start immediately.
