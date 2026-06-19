# Mirror vLLM allocations into KCMM shadow allocator

Status: ready-for-agent
Type: AFK

## What to build

Add a shadow allocator mode where vLLM remains behaviorally unchanged, but every
native block allocation and free is mirrored into KCMM. This validates lifetime
alignment between vLLM block IDs and KCMM block handles before KCMM is allowed
to influence allocation results.

The completed slice should maintain an internal mapping between vLLM logical
blocks and KCMM blocks, update it on allocation/free, surface mismatches as hard
errors, and expose summary metrics after the smoke request finishes.

## Acceptance criteria

- [ ] Shadow mode mirrors each vLLM allocation into `kcmm_alloc_blocks` and each free into `kcmm_free_blocks`.
- [ ] vLLM's returned block IDs and KV storage behavior are unchanged in shadow mode.
- [ ] The shadow mapping detects double-free, missing-free, and allocation-count mismatch cases.
- [ ] A completion request succeeds through the V2 block manager while shadow mode is enabled.
- [ ] The final report includes native allocation counts, KCMM allocation counts, outstanding shadow mappings, and KCMM pool stats.
- [ ] Any KCMM allocation failure aborts startup or request handling with a clear error instead of falling back silently.
- [ ] The mode can be disabled cleanly for stock and observer-only A/B runs.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/04-size-kcmm-pool-from-vllm-runtime-config.md`
