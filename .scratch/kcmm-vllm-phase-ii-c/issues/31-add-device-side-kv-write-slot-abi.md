# Add device-side KV write slot ABI

Status: done
Type: AFK

## What to build

Add a low-level KCMM write ABI that can consume vLLM's device-resident
`slot_mapping` tensor directly instead of requiring Python to materialize a
CPU-side `int64` slot list before every KV write.

## Why

The current vLLM-integrated write path calls
`kcmm_append_kv_slots_on_stream`, which expects a CPU-side slot array. That is
correct and stable, but it forces a host read-back of vLLM's CUDA
`slot_mapping` tensor before KCMM can launch the write. The performance-clean
gates have already removed tracing, D2H verification, per-update report writes,
and read-side validation from the hot path; the next write-side cleanup needs a
device-slot ABI so a later tracker slice can keep the slot contract on GPU.

## Acceptance criteria

- [x] Add a CUDA kernel that decodes `slot_mapping` on device.
- [x] Add a C ABI entrypoint that accepts a device `slot_mapping` pointer, a
  device f16-offset table pointer, source K/V pointers, optional device status,
  and caller stream.
- [x] Add Python ctypes bindings for the new ABI.
- [x] Extend `kv_write_ffi_smoke` to verify device-slot writes by D2H byte
  comparison.
- [x] Cover padding slot skip behavior.
- [x] Cover invalid device slot reporting through a device status tensor.
- [x] Document the ABI boundary and latest local result.

## Boundaries

- This issue introduced the low-level ABI. Issue 32 wires it into the
  performance-clean vLLM write tracker.
- The ABI now accepts both a caller-provided offset table and a caller-provided
  valid-block table. The optional device status tensor reports status `1` for a
  block id outside the offset-table length and status `2` for an in-range block
  marked inactive.
- The old host-slot ABI remains the default correctness path and low-level
  fallback.

## Implementation

- Added `src/cuda/kernels/kcmm_vllm_kv_write.cu`.
- Added Rust FFI export
  `kcmm_append_kv_device_slots_on_stream(...)`.
- Added `KcmmPool.append_kv_device_slots_on_stream(...)` in
  `scripts/kcmm/bindings.py`.
- Extended `scripts/kcmm/kv_write_ffi_smoke.py` with a device-slot section.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `cargo check --features kcmm`
- [x] `cargo build --features kcmm`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm --output /tmp/kcmm-kv-write-device-slot-smoke.json`

## Latest local result

- Date: 2026-07-02
- Result: `passed=true`
- Report: `/tmp/kcmm-kv-write-device-slot-epoch-smoke.json`
- Host direct-slot ABI still passed for slots `2`, `7`, and padding `-1`.
- Device direct-slot ABI passed for slots `1`, `4`, and padding `-1`.
- Device slot mapping pointer was CUDA memory.
- Device block-offset table entries: `3`.
- Device valid-block table: `[1, 1, 0]`.
- Invalid host slot `16` still failed through the old synchronous error path.
- Invalid device slot `16` set device status to `1`.
- Inactive device slot `8` set device status to `2`.
- Final KCMM stats recorded `blocks_in_use=0`.

## Follow-up

Completed by Issue 32: the performance-clean tracker can now use
`kcmm_append_kv_device_slots_on_stream(...)` behind an explicit
`--kcmm-kv-write-device-slots` flag, while correctness paths keep the host-slot
writer by default.
