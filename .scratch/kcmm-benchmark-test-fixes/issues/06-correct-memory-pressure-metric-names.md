# Correct memory pressure metric names and throughput reporting

Status: ready-for-agent
Type: AFK

## What to build

Update the memory pressure benchmark so completed-request ratio is not labeled as throughput. The benchmark should report completion or capacity ratio separately from elapsed-time throughput, and keep capped/rejected counts visible in single and sweep outputs.

This preserves the useful memory-pressure result while preventing reports from interpreting completion count as elapsed throughput improvement.

## Acceptance criteria

- [ ] `throughput_ratio` output is renamed to `completion_ratio`, `capacity_ratio`, or another accurate term.
- [ ] The benchmark also reports real elapsed throughput or explicitly states that elapsed throughput is not the primary metric.
- [ ] Single-config output includes completed, capped, rejected, elapsed time, and the corrected ratio name.
- [ ] Sweep output keeps completed, rejected, capped, and eviction counts visible.
- [ ] Existing unused-assignment warnings in the memory pressure benchmark are removed.
- [ ] Existing memory pressure benchmark compile checks pass.

## Blocked by

None - can start immediately.

## Comments

