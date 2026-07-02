# Use device-slot KV writes in performance-clean gate

Status: done
Type: AFK

## What to build

Wire the low-level device-slot KV write ABI into the vLLM write replacement
tracker behind an explicit performance-clean flag, so KCMM can consume vLLM's
CUDA `slot_mapping` tensor without materializing a CPU slot list on the hot
write path.

## Why

Issue 31 added `kcmm_append_kv_device_slots_on_stream(...)`, but the integrated
vLLM tracker still used the host-slot path. That left a `slot_mapping` D2H copy
in the performance-clean replacement path after read-side tracing, write-row
verification, report-on-update, read block-table validation, and first-call
kernel compile overhead had already been removed.

## Acceptance criteria

- [x] Add an opt-in `--kcmm-kv-write-device-slots` /
  `KCMM_KV_WRITE_DEVICE_SLOTS` flag.
- [x] Require device-slot writes to run only with KV write mirror or replacement
  mode and with write-row verification disabled.
- [x] Extend the device-slot ABI with an in-use block validity table.
- [x] Report device-slot status errors through a device status word.
- [x] Avoid `_slot_mapping_to_list()` when the device-slot path is active.
- [x] Keep correctness gates on the stable host-slot path by default.
- [x] Make performance-clean and performance-clean stress gates require the
  device-slot path.
- [x] Validate the low-level FFI smoke, end-to-end vLLM smoke,
  performance-clean gate, and performance-clean stress gate.
- [x] Document the updated contract and local results.

## Boundaries

- This is an opt-in performance path, not the default correctness path.
- Device-slot mode requires `--no-kcmm-kv-write-verify`; bounded D2H row
  verification still depends on a host-side slot list.
- The tracker still completes the selected stream before returning. This issue
  removes host slot materialization, not every write-path host interaction.
- Request-level latency remains measured through full vLLM server startup and
  scheduling overhead. The latest local run passed the contract but was not a
  performance win over stock.

## Implementation

- Added `KcmmPool::get_all_block_valid_flags()` and FFI
  `kcmm_get_all_block_valid(...)`.
- Extended `kcmm_append_kv_device_slots_on_stream(...)` with a
  `valid_blocks_ptr` CUDA u8 table.
- Added device status code `2` for slots whose block id is in range but inactive;
  status `1` remains the out-of-range block id code.
- Added `KcmmPool::block_state_epoch()` and FFI `kcmm_block_state_epoch(...)` so
  Python can cache device offset/valid tables safely across block reuse.
- Added Python bindings for the new FFI calls.
- Added `use_device_slot_write` to `KcmmKvWriteMirrorTracker`.
- In device-slot mode, the tracker validates CUDA tensor/device contracts,
  keeps `slot_mapping` on GPU, caches offset/valid tables by block-state epoch,
  launches `kcmm_append_kv_device_slots_on_stream(...)`, and reports status
  checks/errors.
- Added `kv_write_device_slots` to smoke/gate config and CLI plumbing.
- Updated the performance-clean and performance-clean stress gates to require
  device-slot write calls, zero host-slot write calls, status checks, and zero
  device status errors.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `cargo check --features kcmm`
- [x] `cargo build --features kcmm`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm --output /tmp/kcmm-kv-write-device-slot-final-smoke.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_smoke --mode kcmm --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-kv-write-verify --kv-write-device-slots --kv-read-gpu-kernel-candidate --kv-read-fast-current-context-launch --kv-read-precompile-gpu-kernel --no-kv-read-validate-block-tables --no-tracker-report-on-update --no-build-kcmm --no-print-seams`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-device-slots-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-device-slots-latest.json`

## Latest local results

- Date: 2026-07-02
- Low-level FFI smoke: `passed=true`
- Low-level device-slot valid writes: slots `1`, `4`, padding `-1`
- Low-level inactive slot: slot `8`, status `2`
- Low-level out-of-range slot: slot `16`, status `1`
- End-to-end vLLM smoke: `device_slot_write_calls=8`,
  `host_slot_write_calls=0`, `device_slot_status_codes={"0": 8}`
- Performance-clean gate: `passed=true`
- Performance-clean device writes: `384`
- Performance-clean host-slot writes: `0`
- Performance-clean device status checks/errors: `384/0`
- Performance-clean device table cache hits/rebuilds: offset `381/3`, valid
  `381/3`
- Performance-clean request latency: stock `1.808s`, KCMM `1.912s`, ratio
  `1.058`
- Performance-clean stress gate: `passed=true`
- Stress device writes: `300`
- Stress host-slot writes: `0`
- Stress device status checks/errors: `300/0`
- Stress request latency: stock `1.824s`, KCMM `1.887s`, ratio `1.035`

## Follow-up

The next useful issue should investigate the remaining KCMM-vs-stock request
latency gap with device-slot writes active. Likely targets are write-stream
completion behavior, per-call ctypes overhead, and read/write tracker host
bookkeeping that still remains in the replacement path.
