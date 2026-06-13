# Run integration benchmarks to compare eviction policies

Status: todo
Type: AFK

## What to build

Commit `9755b31` added `"arc"` and `"sink_window"` as new eviction policy options, plus background eviction and block-granularity eviction. These need integration-level validation against LRU baseline.

### Part A: Per-policy sweep

Run `scripts/run_kcmm_integration_bench.sh --sweep` with each policy:

- `--policy lru` (baseline)
- `--policy lfu`
- `--policy arc`
- `--policy sink_window`

For each run, collect:
- `eviction_count`, `evicted_blocks_total`, `eviction_failures`
- `avg_evict_batch_size` per policy
- `Peak GPU blocks` vs `aligned_physical_ceiling`
- Throughput (req/s), completion rate

Expected signals:
- ARC should show fewer evictions than LRU in mixed (scan + hot set) workloads
- SinkWindow should show stable decode on long sequences even with small `max_blocks_total`
- LFU vs LRU should show measurable difference on skewed access patterns

### Part B: Background eviction validation

Run a config with `low_watermark_threshold=0.3` and verify:

- `background_eviction_count > 0` in metrics output
- Admission tail latency (p99) is not worse than without background eviction
- No `exceeded aligned physical capacity` warnings

### Part C: Block-granularity eviction validation

Verify that with `max_blocks_total` tight enough to trigger eviction:

- Some sequences are **partially** evicted (remain in running with reduced GPU blocks)
- These sequences continue decoding correctly after partial eviction
- `evicted_blocks_total` per eviction operation < sequence block count (proving partial eviction is happening)

### Integration benchmark changes needed

The benchmark harness may need small updates:

- [ ] Sweep script accepts `--policy` flag to override config
- [ ] Benchmark output CSV includes eviction metrics columns
- [ ] Status column flags "thrashing" when eviction-per-completion exceeds threshold

## Acceptance criteria

- [ ] LRU/ARC/LFU/SinkWindow sweep results exist in `results/` with comparable configs
- [ ] ARC shows statistically significant reduction in eviction count vs LRU on at least one workload
- [ ] Background eviction produces `background_eviction_count > 0`
- [ ] Partial eviction is observed (block count per eviction < sequence block count)
- [ ] No regression: LRU baseline throughput/completion rate matches pre-change levels
- [ ] Benchmark compile check passes

## Blocked by

- `#07-unit-tests-for-new-policies-and-metrics` — unit tests should pass before running integration benchmarks
