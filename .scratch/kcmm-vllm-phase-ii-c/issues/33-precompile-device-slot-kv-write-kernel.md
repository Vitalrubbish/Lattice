# Precompile device-slot KV write kernel

Status: done
Type: AFK

## What to build

Move the device-slot KV write kernel NVRTC/module-load cost out of the first
request by precompiling `kcmm_vllm_kv_write_slots_f16` when the runtime KCMM pool
attaches to the vLLM write tracker.

## Why

After Issue 32, the host-profile gate showed the write path still had a large
first-call cost:

- `write_ctypes_launch=84.111ms` total
- `write_ctypes_launch` max `80.366ms`
- `write_mirror_call_total=119.996ms` total

That matched the known pattern already fixed for the read kernel: the first FFI
launch compiled/loaded the CUDA kernel in the request path. The right fix is to
make the write kernel follow the read kernel's attach-time precompile model.

## Acceptance criteria

- [x] Add an idempotent Rust FFI precompile function for the vLLM KV write
  kernel.
- [x] Add Python ctypes binding and `KcmmPool` wrapper for the precompile call.
- [x] Call write-kernel precompile when the write tracker attaches a pool and
  device-slot write mode is active.
- [x] Report requested/calls/succeeded/elapsed precompile fields in the write
  tracker report.
- [x] Include write precompile fields in the GPU read A/B contract.
- [x] Make the performance-clean gate require write-kernel precompile.
- [x] Extend the low-level KV write FFI smoke to call the precompile ABI.
- [x] Validate short vLLM smoke, performance-clean, performance-clean stress,
  and host-profile gates.

## Boundaries

- This only moves compile/load overhead out of request execution. It does not
  reduce steady-state per-write launch overhead.
- The precompile path is only required for the device-slot performance-clean
  path. Correctness paths still default to host-slot writes and row
  verification.
- Host-profile section totals are nested; the attach-time precompile section is
  diagnostic and should not be summed into request-time hot-path sections.

## Implementation

- Added FFI `kcmm_precompile_vllm_kv_write_f16(...)`, backed by the same
  cached `compile_vllm_kv_write_kernel(...)` used by device-slot launches.
- Added `KcmmPool.precompile_vllm_kv_write_f16()`.
- Added attach-time precompile in `KcmmKvWriteMirrorTracker.attach_pool(...)`
  when `_should_use_device_slot_write()` is true.
- Added report fields:
  `device_slot_kernel_precompile_requested`,
  `device_slot_kernel_precompile_calls`,
  `device_slot_kernel_precompile_succeeded`, and
  `device_slot_kernel_precompile_elapsed_ms`.
- Added performance-clean failures for missing/failed/unexpected write
  precompile.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `cargo check --features kcmm`
- [x] `cargo build --features kcmm`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm --output /tmp/kcmm-kv-write-precompile-ffi-smoke.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_smoke --mode kcmm --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-kv-write-verify --kv-write-device-slots --kv-read-gpu-kernel-candidate --kv-read-fast-current-context-launch --kv-read-precompile-gpu-kernel --no-kv-read-validate-block-tables --no-tracker-report-on-update --no-build-kcmm --no-print-seams`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-write-precompile-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-write-precompile-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-write-precompile-latest.json`

## Latest local results

- Date: 2026-07-02
- Low-level FFI smoke: `passed=true`
- Low-level write precompile elapsed: `78.506ms`
- Short vLLM smoke: `device_slot_kernel_precompile_calls=1`,
  `device_slot_kernel_precompile_succeeded=true`, elapsed `76.153ms`
- Performance-clean gate: `passed=true`
- Performance-clean write precompile calls/succeeded/elapsed: `1/true/77.661ms`
- Performance-clean request latency: stock `1.828s`, KCMM `1.811s`, ratio
  `0.991`
- Performance-clean tokens/s: stock `17.505`, KCMM `17.670`, ratio `1.009`
- Host-profile gate: `passed=true`
- Host-profile `write_ctypes_launch`: `3.819ms` total, `9.944us` avg,
  max `23.124us`
- Previous host-profile before this issue: `write_ctypes_launch=84.111ms` total,
  max `80.366ms`
- Host-profile write precompile section: `77.389ms` one-time attach cost
- Performance-clean stress gate: `passed=true`
- Stress write precompile calls/succeeded/elapsed: `1/true/78.358ms`
- Stress request latency: stock `1.805s`, KCMM `1.805s`, ratio `1.000`

## Follow-up

Completed by Issue 34: the performance-clean read planner now uses compact
metadata, reducing `read_replace_build_plan` from `14.894ms` total to
`11.654ms` total in the local host-profile gate. Remaining follow-up should
focus on read GPU-kernel launch/stream selection and write device-slot table
lookup, stream selection, and ctypes launch.
