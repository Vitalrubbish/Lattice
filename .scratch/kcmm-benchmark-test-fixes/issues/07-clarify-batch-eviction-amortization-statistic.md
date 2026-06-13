# Clarify batch eviction amortization statistic

Status: ready-for-agent
Type: AFK

## What to build

Make the batch eviction amortization benchmark use consistent statistical language. If amortization factors are based on P50 per-block latency, the output and variable names should say P50. If factors are intended to be mean-based, the benchmark should compute them from the mean.

The result should make it clear whether batch eviction is regressing in mean, P50, or both.

## Acceptance criteria

- [ ] Batch eviction factor computation and printed label use the same statistic.
- [ ] Variable names and comments no longer call a P50 value an average.
- [ ] The benchmark still prints distribution details with mean, P50, P99, and max.
- [ ] Results can clearly distinguish P50 amortization from mean amortization.
- [ ] Existing tiering benchmark compile checks pass.

## Blocked by

None - can start immediately.

## Comments

