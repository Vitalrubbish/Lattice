# Decide Phase II.A target and storage of record

Status: ready-for-human
Type: HITL

## What to build

Make the Phase II.A architecture decision explicit before replacing vLLM's
allocator: define the exact vLLM target for this branch and decide what is the
storage of record for KV data during allocator replacement.

The current documentation names vLLM 0.6.3 as the target, while the local
working CUDA 11.8 environment uses vLLM 0.6.1.post1. Phase II.A should not
proceed on an implicit version assumption. The decision must also say whether
Phase II.A keeps vLLM native KV tensors as storage while KCMM shadows logical
allocation, or whether KCMM VA becomes the storage of record in this phase.

## Acceptance criteria

- [ ] The exact vLLM version and required runtime flags for Phase II.A are recorded in the ADR or environment documentation.
- [ ] The storage-of-record model is recorded: native vLLM KV tensors, KCMM VA, or an explicitly staged transition between them.
- [ ] Stop criteria are documented for the case where allocator replacement cannot be made correct without also replacing write/read paths.
- [ ] The decision names which Phase II.A tests are required before moving to Phase II.B.
- [ ] The decision accounts for the observed local constraint that CUDA 11.8 wheels are required on the current driver.
- [ ] The decision accounts for transformer/tokenizer dependency pins required by the selected vLLM version.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/02-instrument-vllm-v2-allocator-seams.md`
