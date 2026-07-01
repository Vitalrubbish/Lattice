# Cache write slot block ensure checks

Status: done
Type: AFK

## What to build

Cache write-side slot block IDs after they have been confirmed present in the
KCMM pool so the write replacement tracker does not repeatedly call
`pool.block_location()` for the same physical block on every write seam.

## Why

After Issue 27, write-side pool shape lookup is no longer a meaningful
host-profile section. `write_ensure_slot_blocks` still runs for every write
replacement call and repeatedly validates the same block IDs. The validation is
needed for tensor-parallel workers, where slot mappings can reference
driver-scheduler block IDs before the worker-local KCMM pool has seen them, but
once a block ID has been confirmed or lazily allocated in the worker-local pool,
rechecking it on every subsequent write is redundant.

## Acceptance criteria

- [x] Track physical slot block IDs confirmed by `_ensure_slot_blocks`.
- [x] Skip `pool.block_location()` for block IDs already confirmed by this
  tracker.
- [x] Keep the existing lazy allocation path for first-seen/missing block IDs.
- [x] Report known block count and ensure-cache hit/miss counters.
- [x] Run the KV write FFI smoke gate.
- [x] Run the host-profile gate and record write-side host section changes.
- [x] Run the performance-clean gate and record stock-vs-KCMM result.

## Boundaries

- This does not change CUDA kernels or the KCMM C ABI.
- This does not change slot mapping semantics or padding-slot behavior.
- This does not remove the first-seen block validation/allocation behavior
  needed by tensor-parallel workers.
- This does not eliminate the CUDA-to-CPU copy for vLLM's `slot_mapping` tensor.

## Rejected approach

A host-pointer Python binding path was prototyped first: the tracker copied
`slot_mapping` to a CPU tensor and passed the tensor's host pointer directly to
the existing FFI instead of constructing a ctypes array from the Python list.
Correctness passed, but the host-profile gate did not improve:
`write_ctypes_launch` increased from about `7.18ms` to `7.52ms` and
`write_mirror_call_total` increased from about `45.32ms` to `46.57ms`.
That path was not kept.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm --output /tmp/kcmm-vllm-phase-ii-c-kv-write-ensure-cache-smoke-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-write-ensure-cache-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-write-ensure-cache-latest.json`

## Latest local result

KV write FFI smoke:

- Date: 2026-07-01
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-kv-write-ensure-cache-smoke-latest.json`
- Direct-slot writes still passed for slots `2` and `7`.
- Direct-slot padding slot `-1` was skipped.
- Invalid direct slot `16` still failed with `block_idx 4 from slot 16 not in use`.

Host-profile gate:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-write-ensure-cache-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Known write slot blocks: `3`
- Write slot block ensure cache hits/misses: `381/3`
- Request latency seconds: stock `1.798`, KCMM `1.845`, ratio `1.026`
- Tokens per second: stock `17.798`, KCMM `17.344`, ratio `0.974`
- Top write host sections:
  `write_mirror_call_total=42.697ms` total, `111.190us` avg;
  `write_slot_mapping_to_host=10.300ms` total, `26.824us` avg;
  `write_ctypes_launch=7.550ms` total, `19.662us` avg;
  `write_select_stream=4.522ms` total, `11.776us` avg;
  `write_prepare_rows=3.068ms` total, `7.989us` avg;
  `write_ensure_slot_blocks=1.208ms` total, `3.145us` avg.
- Compared with Issue 27 host-profile result,
  `write_ensure_slot_blocks` dropped from `3.535ms` to `1.208ms`, and
  `write_mirror_call_total` dropped from `45.316ms` to `42.697ms`.

Performance-clean gate:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-write-ensure-cache-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `97.600ms`
- Known write slot blocks: `3`
- Write slot block ensure cache hits/misses: `381/3`
- Request latency seconds: stock `1.815`, KCMM `1.805`, ratio `0.994`
- Tokens per second: stock `17.631`, KCMM `17.729`, ratio `1.006`
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`

## Follow-up

The repeated block-location validation is no longer the main write-side cost.
The remaining write-side hot sections are `slot_mapping` device-to-host copy and
Python/ctypes slot array construction around `append_kv_slots`.
