# Fix cuMemMap/cuMemUnmap overhead benchmark latency reporting

Status: done
Type: AFK

## What to build

The Step 3 cuMemMap/cuMemUnmap overhead benchmark prints two columns, `map (µs)` and `unmap (µs)`, but both values are the same combined average computed over all map and unmap operations. This gives the false impression that map and unmap were measured independently. In addition, because the GPU map granularity is 2 MiB, the loop skips every size below 2 MiB, leaving the "latency vs. mapping size" table with only one valid data point.

Refactor the benchmark so it either measures map and unmap separately or reports a single honest combined latency. Add a clear note explaining why sub-granularity sizes are skipped, and apply the same fix to the standalone `kcmm_bench_cumemmap_latency` test if it shares the issue.

## Acceptance criteria

- [ ] `step3_cumemmap_overhead` no longer prints identical values for map and unmap unless they were actually measured separately.
- [ ] The benchmark output explains that only the 2 MiB size is shown because of GPU map granularity.
- [ ] `kcmm_bench_cumemmap_latency` is checked and fixed if it has the same map/unmap reporting problem.
- [ ] Both tests still pass and produce readable output after the change.

## Verification

- `cargo test --features kcmm --release --test step3_benchmarks step3_cumemmap_overhead -- --nocapture`
- `cargo test --features kcmm --release --test kcmm_bench_tiering kcmm_bench_cumemmap_latency -- --nocapture`
- Step 3 output now prints separate `map`, `unmap`, and `combined` columns plus the GPU map granularity note.

## Blocked by

None - can start immediately.
