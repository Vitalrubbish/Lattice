# Cache default stream pointer

Status: done
Type: AFK

## What to build

Reduce steady-state host overhead in KCMM read/write stream selection by caching
the per-device CUDA default stream pointer inside `KcmmStreamProvider`.

## Why

After Issue 35, the latest host-profile run still showed stream-selection
overhead in the request path:

- `write_select_stream=4.667ms` total
- `read_gpu_kernel_select_stream=3.606ms` total

The stream provider queried both the current stream and the default stream on
every read/write seam. The current stream must still be queried to preserve
PyTorch stream semantics, but the default stream pointer is stable per device
and can be cached after the first lookup.

## Acceptance criteria

- [x] Cache default stream pointers by CUDA device in `KcmmStreamProvider`.
- [x] Preserve current stream querying on every `select(...)` call.
- [x] Preserve forced non-default stream behavior and validation.
- [x] Report select calls, current-stream queries, default-stream cache hits,
  and default-stream cache misses.
- [x] Include stream cache fields in the GPU read A/B contract.
- [x] Make the performance-clean gate require default-stream cache reuse for
  read and write stream selection.
- [x] Validate short vLLM smoke, performance-clean, performance-clean stress,
  and host-profile gates.

## Boundaries

- This does not assume the current stream is always the default stream.
- This does not change the raw stream pointer passed to read/write FFI launches.
- This does not change forced non-default stream synchronization semantics.

## Verification

- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_smoke --mode kcmm --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-kv-write-verify --kv-write-device-slots --kv-read-gpu-kernel-candidate --kv-read-fast-current-context-launch --kv-read-precompile-gpu-kernel --no-kv-read-validate-block-tables --no-tracker-report-on-update --no-build-kcmm --no-print-seams`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_smoke --mode kcmm --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-kv-write-verify --kv-write-device-slots --kv-read-gpu-kernel-candidate --kv-read-fast-current-context-launch --kv-read-precompile-gpu-kernel --no-kv-read-validate-block-tables --no-tracker-report-on-update --kv-force-non-default-stream --no-build-kcmm --no-print-seams`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stream-cache-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-stream-cache-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-stream-cache-latest.json`

## Latest local results

- Date: 2026-07-02
- Short vLLM smoke: read stream select/current/cache hits/cache misses
  `6/6/5/1`; write stream select/current/cache hits/cache misses `8/8/7/1`.
- Forced non-default stream smoke: read forced calls `6`, write forced calls
  `8`, read/write stream pointers non-zero, default stream pointer `0`, and
  default stream cache still reused as `5/1` and `7/1`.
- Performance-clean gate: `passed=true`
- Performance-clean stream select/current/cache hits/cache misses:
  read `372/372/371/1`, write `384/384/383/1`
- Performance-clean request latency: stock `1.818s`, KCMM `1.840s`, ratio
  `1.012`
- Performance-clean tokens/s: stock `17.602`, KCMM `17.391`, ratio `0.988`
- Performance-clean device-slot writes/total-block refreshes/epoch queries:
  `384/3/387`
- Performance-clean stress gate: `passed=true`
- Stress stream select/current/cache hits/cache misses: read `276/276/275/1`,
  write `300/300/299/1`
- Stress observed max read/write batch: `2/9`
- Stress request latency: stock `1.810s`, KCMM `1.773s`, ratio `0.980`
- Stress tokens/s: stock `26.519`, KCMM `27.073`, ratio `1.021`
- Host-profile gate: `passed=true`
- Host-profile `read_gpu_kernel_select_stream`: `3.305ms` total, `8.884us`
  avg
- Host-profile `write_select_stream`: `4.529ms` total, `11.794us` avg
- Previous host-profile before this issue:
  `read_gpu_kernel_select_stream=3.606ms` total and
  `write_select_stream=4.667ms` total

## Follow-up

The remaining steady-state host costs are mostly launch-side Python/ctypes work:
`write_ctypes_launch=3.900ms`, `read_gpu_kernel_ctypes_launch=6.066ms`, and
planner/prepare sections. The next useful issue should target launch-side
batching/fusion or reducing ctypes overhead without weakening stream semantics.
