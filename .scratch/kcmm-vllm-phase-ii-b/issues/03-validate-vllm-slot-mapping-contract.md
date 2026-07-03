# Validate vLLM slot_mapping contract for KCMM writes

Status: done
Type: AFK

## What to build

Extend the Phase II.B KV write instrumentation so every observed
`reshape_and_cache` call decodes its bounded `slot_mapping` sample into the
physical KV slot contract needed by KCMM.

The validation should prove, for the pinned vLLM `0.6.1.post1+cu118` stack, that
`slot_mapping` values can be interpreted as `slot = block_id * block_size +
offset_in_block` at the `reshape_and_cache` seam. This is the missing bridge
between vLLM's write seam and KCMM's block-addressed memory layout.

## Acceptance criteria

- [x] KV write traces include an inferred cache layout, block size, and number
  of physical KV blocks.
- [x] KV write traces decode bounded `slot_mapping` samples into `block_id` and
  `offset_in_block`.
- [x] The smoke runner fails when an observed non-padding slot maps outside the
  inferred KV cache block range.
- [x] The trace preserves the observer-only guarantee and does not dump K/V
  payload contents.
- [x] Documentation records that `reshape_and_cache` exposes physical slots, so
  the next replacement slice must either add a direct-slot KCMM write API or
  patch metadata construction to recover sequence/position context.

## Implementation

- Extended KV write instrumentation with slot-mapping contract inference.
- The trace now records inferred cache layout, block size, physical block count,
  decoded slot samples, and invalid slot samples.
- The smoke runner fails if any observed write event has an invalid
  non-padding slot sample.
- Documented the direct-slot vs metadata-builder decision point in
  `docs/dev/kcmm-vllm-cu118-env.md` and
  `docs/adr/0001-vllm-integration-architecture.md`.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_smoke --instrument-kv-writes`

The local Phase II.B slot-mapping contract smoke passed on 2026-06-19:

- Completion succeeded.
- Observed seam: `vllm._custom_ops.reshape_and_cache`
- Write calls observed: `8`
- Inferred layout: `paged_kv_cache`
- Inferred block size: `16`
- Inferred physical block count: `134685`
- First decoded slots: `(slot=0, block_id=0, offset=0)` and
  `(slot=1, block_id=0, offset=1)`
- Invalid slots: `[]`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-b/issues/02-instrument-vllm-reshape-and-cache-contract.md`
