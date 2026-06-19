# Size KCMM pool from vLLM runtime config

Status: ready-for-agent
Type: AFK

## What to build

Replace the Phase I.C tiny fixed observer pool shape with a Phase II.A pool
configuration derived from vLLM's runtime model and cache configuration, while
still leaving vLLM behavior unchanged.

The completed slice should create a KCMM pool with tiering disabled and capacity
aligned with the vLLM GPU block budget for the served model. The observer report
should show both vLLM's relevant cache sizing inputs and KCMM's resulting pool
stats so reviewers can verify the two sides match before any allocator
replacement happens.

## Acceptance criteria

- [ ] KCMM pool creation can use vLLM runtime values for block size, model layer count, KV heads, head dimension, max sequence length, and effective GPU block capacity.
- [ ] The launcher retains the existing fixed-shape observer mode for quick CUDA probes.
- [ ] The vLLM server smoke test can run with runtime-derived KCMM sizing and still complete one request.
- [ ] The observer report includes enough vLLM and KCMM sizing fields to verify capacity alignment.
- [ ] Tiering remains disabled for this path.
- [ ] Invalid or unavailable vLLM sizing data fails closed with a clear error message.
- [ ] Unit or smoke coverage verifies that the runtime-derived config path does not regress the existing observer-only probe.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/03-decide-phase-ii-a-target-and-storage-of-record.md`
