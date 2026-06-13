# Use deterministic identical weights for integration comparisons

Status: done
Type: AFK

## What to build

Make Tiering OFF and Tiering ON runs in the engine integration benchmark use deterministic, identical model weights. The comparison should not depend on separate `thread_rng()` streams or run order.

This keeps throughput and step-latency comparisons focused on the KV cache backend and tiering behavior, not random model initialization differences.

## Acceptance criteria

- [ ] The integration benchmark uses a fixed seed or a shared host-side weight fixture for each compared OFF/ON pair.
- [ ] OFF-first and ON-first runs use equivalent weights for the two modes.
- [ ] Re-running the benchmark with the same code produces deterministic model initialization.
- [ ] The benchmark output or comments state that OFF/ON weights are identical for each comparison.
- [ ] Existing integration benchmark compile checks pass.

## Blocked by

None - can start immediately.

## Comments

