# Expand tiering roundtrip integrity coverage

Status: ready-for-agent
Type: AFK

## What to build

Expand the tiering roundtrip data integrity benchmark so it validates all layers and both K and V cache paths, not only one layer and one cache direction. The test should prove that batch eviction and restore preserve the full KV cache contents that the transformer reads.

This gives stronger confidence before relying on integration benchmark performance numbers.

## Acceptance criteria

- [ ] The roundtrip integrity benchmark writes distinguishable patterns across every configured layer.
- [ ] The benchmark covers both K-cache and V-cache data.
- [ ] Verification fails if any layer or K/V path restores the wrong bytes.
- [ ] The output reports full coverage, not only block count.
- [ ] Existing tiering benchmark compile checks pass.

## Blocked by

None - can start immediately.

## Comments

