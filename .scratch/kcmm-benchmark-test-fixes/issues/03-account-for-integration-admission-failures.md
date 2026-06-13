# Account for engine integration admission failures

Status: ready-for-agent
Type: AFK

## What to build

Make dynamic request admission in the engine integration benchmark observable and fair under memory pressure. When admitting a new request fails, the benchmark should record the rejection. For Tiering ON, admission should either try eviction and retry allocation or explicitly report that admission-time eviction is not part of the policy being measured.

The result should explain whether lower completed counts come from admission rejection, decode-time capping, or requests left active at the end.

## Acceptance criteria

- [ ] Dynamic admission failures increment a rejection counter.
- [ ] Rejections are printed in both single-config and sweep results.
- [ ] Tiering ON either performs a bounded admission-time eviction retry or prints/reports that admission-time eviction is intentionally disabled.
- [ ] The benchmark can distinguish rejected arrivals from capped active sequences.
- [ ] The updated output makes the high-churn config interpretable without reading progress bars.
- [ ] Existing integration benchmark compile checks pass.

## Blocked by

- `.scratch/kcmm-benchmark-test-fixes/issues/02-split-integration-request-outcome-counters.md`

## Comments

