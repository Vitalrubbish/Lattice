# Enable KCMM-backed V2 allocator behind a flag

Status: ready-for-agent
Type: AFK

## What to build

Introduce the first actual Phase II.A allocator replacement mode behind an
explicit opt-in flag. In this mode, vLLM's V2 allocation/free decisions should
be delegated to KCMM with tiering disabled, while preserving the storage-of-record
decision made for this phase.

This is not yet the KV write/read replacement. The slice should either prove a
minimal completion path works under the chosen storage model or fail closed with
a precise report explaining which vLLM invariant prevents allocator-only
replacement from being correct.

## Acceptance criteria

- [ ] A launcher flag selects KCMM-backed allocator mode; the default remains observer-only or stock behavior.
- [ ] KCMM owns allocation/free decisions for the targeted V2 allocator seam when the flag is enabled.
- [ ] The implementation respects the storage-of-record decision from issue 03.
- [ ] A smoke completion either succeeds end-to-end or exits with a documented stop condition that points to the required Phase II.B/II.C work.
- [ ] Allocation/free metrics show no leaked KCMM blocks after the smoke request completes and the server shuts down.
- [ ] The mode is incompatible with unsupported vLLM versions or flags and fails before serving traffic.
- [ ] Existing observer-only and shadow modes continue to pass.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/03-decide-phase-ii-a-target-and-storage-of-record.md`
- `.scratch/kcmm-vllm-phase-ii-a/issues/05-mirror-vllm-allocations-into-kcmm-shadow-allocator.md`
