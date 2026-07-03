# Instrument vLLM reshape_and_cache write contract

Status: done
Type: AFK

## What to build

Add an observer-only Phase II.B instrumentation mode for vLLM's KV write seam.
The mode should patch `vllm._custom_ops.reshape_and_cache` and
`vllm._custom_ops.reshape_and_cache_flash` without changing behavior, then
record the exact tensor contract that the next Phase II.B slice must replace
with `kcmm_append_kv_step`.

The report should capture which function is called, tensor shapes/dtypes/devices
for `key`, `value`, `key_cache`, and `value_cache`, and a bounded sample of
`slot_mapping` values. This turns the write-path integration from an assumption
into a version-pinned contract for vLLM `0.6.1.post1+cu118`.

## Acceptance criteria

- [x] A launcher flag enables observer-only KV write instrumentation.
- [x] The automated vLLM smoke runner can enable the instrumentation and still
  produce a completion.
- [x] The trace records calls to `reshape_and_cache` or
  `reshape_and_cache_flash`.
- [x] The trace records tensor shape, dtype, device, stride, numel, element size,
  and data pointer for `key`, `value`, `key_cache`, and `value_cache`.
- [x] The trace records a bounded `slot_mapping` sample without dumping K/V
  payload contents.
- [x] The smoke runner fails when the required KV write seam is not observed.
- [x] The documentation states that this contract trace is the prerequisite for
  replacing the write path with `kcmm_append_kv_step`.

## Implementation

- Added launcher flags:
  - `--kcmm-instrument-kv-writes`
  - `--kcmm-kv-write-trace-path`
  - `--kcmm-require-kv-write-seams`
- Added smoke runner flags:
  - `--instrument-kv-writes`
  - `--kv-write-trace-path`
  - `--require-kv-write-seams`
- Instrumented `vllm._custom_ops.reshape_and_cache` and
  `vllm._custom_ops.reshape_and_cache_flash` without changing return behavior.
- The trace captures tensor metadata and bounded `slot_mapping` samples, but not
  K/V payload contents.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_smoke --instrument-kv-writes`

The local Phase II.B write contract smoke passed on 2026-06-19:

- Completion succeeded.
- Observed seam: `vllm._custom_ops.reshape_and_cache`
- Write calls observed: `8`
- Required KV write seam groups missing: `{}`
- First `slot_mapping` sample: `[0, 1]`
- First `key`/`value` shape: `[2, 2, 64]`
- First `key_cache` shape: `[134685, 2, 8, 16, 8]`
- First `value_cache` shape: `[134685, 2, 64, 16]`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-b/issues/01-bind-and-gate-kcmm-kv-write-ffi.md`
