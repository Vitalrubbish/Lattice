# Cache KV write pool shape for performance-clean gate

Status: done
Type: AFK

## What to build

Cache stable KCMM pool shape metadata in the Phase II.B/II.C KV write
replacement tracker and reuse it on the hot write path.

## Why

After Issue 26, the host-profile gate moved the read cold-start cost out of the
measured request. The remaining write-side profile still showed per-call host
overhead under `write_mirror_call_total`, including repeated pool shape lookups
in `write_pool_stats_shape_check` and `_ensure_slot_blocks`. Block size, block
bytes, step elements, and layer count are stable for a KCMM pool, so the write
tracker should cache them at attach time instead of asking the pool for stats on
every write.

## Acceptance criteria

- [x] Cache write-side pool `block_size`, `block_bytes`, `step_elements`, and
  `num_layers` when attaching the KCMM pool.
- [x] Reuse cached shape metadata in layer discovery, row-width validation, and
  slot block ensure logic.
- [x] Keep a lazy refresh fallback if the tracker is used before attach-time
  shape metadata is available.
- [x] Report whether pool shape metadata is cached and how many refreshes were
  performed.
- [x] Run the host-profile gate and record write-side host section changes.
- [x] Run the performance-clean gate and record stock-vs-KCMM result.

## Boundaries

- This does not change CUDA kernels or the KCMM C ABI.
- This does not change slot mapping semantics.
- This does not remove dynamic block allocation for missing external block IDs.
- This does not optimize the `slot_mapping` device-to-host copy or ctypes slot
  array conversion.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-write-shape-cache-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-write-shape-cache-latest.json`

## Latest local result

Host-profile gate:

- Date: 2026-06-30
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-write-shape-cache-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Write pool shape cached: `true`
- Write pool shape refreshes: `1`
- Cached write pool shape: `block_size=16`, `block_bytes=24576`,
  `step_elements=768`, `num_layers=12`
- Request latency seconds: stock `1.840`, KCMM `1.857`, ratio `1.009`
- Tokens per second: stock `17.391`, KCMM `17.232`, ratio `0.991`
- Top write host sections:
  `write_mirror_call_total=45.316ms` total, `118.010us` avg;
  `write_slot_mapping_to_host=11.156ms` total, `29.050us` avg;
  `write_ctypes_launch=7.181ms` total, `18.700us` avg;
  `write_select_stream=4.602ms` total, `11.983us` avg;
  `write_ensure_slot_blocks=3.535ms` total, `9.204us` avg;
  `write_pool_stats_shape_check=0.436ms` total, `1.134us` avg.
- Compared with Issue 26 host-profile result,
  `write_pool_stats_shape_check` dropped from `3.831ms` to `0.436ms`,
  `write_ensure_slot_blocks` dropped from `4.328ms` to `3.535ms`, and
  `write_mirror_call_total` dropped from `49.715ms` to `45.316ms`.

Performance-clean gate:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-write-shape-cache-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `96.446ms`
- Write pool shape cached: `true`
- Write pool shape refreshes: `1`
- Request latency seconds: stock `1.824`, KCMM `1.827`, ratio `1.002`
- Tokens per second: stock `17.544`, KCMM `17.515`, ratio `0.998`
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`

## Follow-up

The write-side per-call shape lookup is no longer a meaningful host-profile
section. The remaining write-side targets are `slot_mapping` device-to-host
copy and Python/ctypes slot array conversion around `append_kv_slots`.
