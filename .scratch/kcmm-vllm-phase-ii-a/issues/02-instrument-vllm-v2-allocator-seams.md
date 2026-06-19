# Instrument vLLM V2 allocator seams

Status: ready-for-agent
Type: AFK

## What to build

Add an observer-only instrumentation mode that wraps the vLLM V2 block manager
and allocator seams targeted by Phase II.A, records when allocation/free paths
are exercised, and leaves vLLM behavior unchanged.

This slice should prove that the current controlled environment really executes
the same Python objects that later monkey-patches will replace. It should run
through the same vLLM server smoke path and produce a compact structured trace
showing allocator construction, allocation, free, and relevant capacity values.

## Acceptance criteria

- [ ] A launcher flag enables allocator seam instrumentation without changing allocation decisions or block IDs.
- [ ] The trace records vLLM version, whether the V2 block manager is enabled, allocator class names, constructor arguments summarized safely, and allocation/free call counts.
- [ ] The instrumentation works with prefix caching disabled and does not require prefix-caching code paths to be active.
- [ ] The smoke-runner from issue 01 can run with instrumentation enabled and still returns a successful completion response.
- [ ] The trace is deterministic enough for review: it should avoid raw object reprs with memory addresses where possible.
- [ ] If a seam is not exercised, the run fails with a clear message rather than silently passing.
- [ ] Stock vLLM smoke mode remains available and unaffected.

## Blocked by

- None - can start immediately, but reuse `.scratch/kcmm-vllm-phase-ii-a/issues/01-make-vllm-smoke-runs-self-terminating.md` when available for automated verification.
