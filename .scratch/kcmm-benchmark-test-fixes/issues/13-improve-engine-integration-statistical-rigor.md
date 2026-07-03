# Improve statistical rigor of engine integration benchmark

Status: done
Type: AFK

## What to build

The engine integration benchmark currently runs each OFF/ON comparison only twice and averages integer counters with integer division. Two runs are insufficient to account for CUDA context creation, kernel JIT compilation, GPU thermal state, and other sources of variance. The reported throughput ratio, capacity ratio, and latency percentiles therefore have unknown confidence.

Increase the repeat count to at least five, switch aggregate calculations to floating-point averaging, and report variance (standard deviation or confidence interval) alongside the mean for the key metrics: throughput, capacity ratio, and P50/P99 step latency. Keep the total runtime reasonable so the benchmark remains usable in CI.

## Acceptance criteria

- [ ] The integration benchmark repeats each OFF/ON comparison at least 5 times.
- [ ] Aggregate functions use floating-point averaging instead of integer truncation.
- [ ] Output reports mean ± standard deviation or confidence interval for throughput ratio, capacity ratio, and step latency percentiles.
- [ ] The alternating OFF-first/ON-first order is preserved or improved to cancel first-run bias.
- [ ] `kcmm_engine_integration_single` and `_sweep` are re-run and the new output format is verified.

## Verification

- `scripts/run_kcmm_integration_bench.sh --single`
- `scripts/run_kcmm_integration_bench.sh --sweep`
- Single result: `results/kcmm_engine_integration_20260613_025519`, 1/1 passed, 5 alternating repeats, throughput/capacity and P50/P99 mean ± std printed.
- Sweep result: `results/kcmm_engine_integration_20260613_025945`, 1/1 passed, 4 configs × 5 alternating repeats, per-config ratio and latency mean ± std printed.

## Blocked by

None - can start immediately. If a fairness-related blocker is desired, this can follow `.scratch/kcmm-benchmark-test-fixes/issues/14-make-ballast-allocation-failure-visible.md` so that OFF/ON resource parity is stable before measuring variance.
