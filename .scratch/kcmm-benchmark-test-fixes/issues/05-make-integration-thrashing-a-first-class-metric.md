# Make integration thrashing a first-class metric

Status: done
Type: AFK

## What to build

Promote engine integration thrashing detection from an extra warning line into a reported metric that uses the corrected full-completion counter. The sweep should make high eviction-per-completion configurations obvious in the same table that reports throughput and capacity.

The status logic should distinguish capacity success, marginal throughput-only improvement, and thrashing cases.

## Acceptance criteria

- [ ] The benchmark reports evictions per full completion or an equivalent named thrashing metric.
- [ ] The metric is included in sweep output, not only printed as a separate warning line.
- [ ] The metric uses full completions, not capped requests.
- [ ] Configurations above the threshold are visibly marked as thrashing or marginal in status.
- [ ] The threshold is documented in benchmark output or comments.
- [ ] Existing integration benchmark compile checks pass.

## Blocked by

- `.scratch/kcmm-benchmark-test-fixes/issues/02-split-integration-request-outcome-counters.md`

## Comments

