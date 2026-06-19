# Bind and gate KCMM KV write FFI

Status: done
Type: AFK

## What to build

Add a repeatable Phase II.B preflight gate for `kcmm_append_kv_step` before
patching vLLM's `reshape_and_cache` write path.

The gate should create a tiny KCMM pool, allocate/register a sequence, write
known FP16 K/V rows from CUDA tensors through the KCMM C ABI, then read the
destination KCMM VA bytes back to host and compare them with the source tensors.
This proves the existing KCMM D2D write API is callable from the Python launcher
environment and gives Phase II.B a byte-level correctness check independent of
vLLM scheduling.

## Acceptance criteria

- [x] Python bindings expose KCMM sequence-management, VA accessor, and
  `kcmm_append_kv_step` C ABI calls.
- [x] A local smoke command writes at least two K/V rows into different logical
  positions in a registered KCMM sequence.
- [x] The smoke command performs byte-level D2H comparison for K and V data.
- [x] The smoke command fails on mismatched bytes, missing VA accessors, FFI
  errors, or leaked KCMM blocks.
- [x] The gate runs in the existing `vllm-cu118` conda environment without
  downloading a model or starting vLLM.
- [x] The Phase II.B docs name this gate as the prerequisite for patching
  `vllm._custom_ops.reshape_and_cache`.

## Implementation

- Extended `scripts/kcmm/bindings.py` with sequence-management, VA accessor, and
  `kcmm_append_kv_step` bindings.
- Added `scripts/kcmm/kv_write_ffi_smoke.py`.
- Documented the Phase II.B preflight gate in
  `docs/dev/kcmm-vllm-cu118-env.md` and
  `docs/adr/0001-vllm-integration-architecture.md`.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.kv_write_ffi_smoke`

The local Phase II.B preflight gate passed on 2026-06-19:

- `passed=true`
- Compared K/V rows at positions `0` and `5`.
- Used two KCMM blocks through one registered sequence.
- Final KCMM stats recorded `blocks_in_use=0`.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/07-add-phase-ii-a-stock-vs-kcmm-ab-gate.md`
