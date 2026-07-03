# Split engine integration request outcome counters

Status: done
Type: AFK

## What to build

Update the engine integration benchmark so request outcomes are counted separately instead of collapsing capped requests into completed requests. The benchmark should report full completions, capped requests, admission rejections, and requests left unfinished at the end of the fixed simulation window.

Capacity-at-Workload should be derived from full completions only, so the benchmark cannot claim capacity improvement from requests that were truncated early.

## Acceptance criteria

- [ ] The integration result model records `completed_full`, `capped`, `rejected`, and `leftover_at_end` or equivalent clearly named counters.
- [ ] A request increments the full-completion counter only after reaching its target length.
- [ ] A request that cannot grow its Block Table increments the capped counter, not the full-completion counter.
- [ ] End-of-run active requests are counted separately as leftover, not silently discarded.
- [ ] The single-config and sweep outputs show the new counters.
- [ ] Capacity ratio uses full completions only.
- [ ] Existing integration benchmark compile checks pass.

## Blocked by

- `.scratch/kcmm-benchmark-test-fixes/issues/01-make-free-sequence-safe-for-cpu-resident-blocks.md`

## Comments

