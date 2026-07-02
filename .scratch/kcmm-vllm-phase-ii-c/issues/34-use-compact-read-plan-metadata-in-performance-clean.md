# Use compact read plan metadata in performance-clean path

Status: done
Type: AFK

## What to build

Reduce steady-state Python overhead in the KCMM GPU read replacement path by
using compact read-plan metadata when performance-clean mode has already
disabled block-table validation and per-update report writes.

## Why

After Issue 33 moved read/write kernel compile costs to attach time, the latest
host-profile run still showed request-path read planner overhead:

- `read_replace_build_plan=14.894ms` total
- `read_build_plan_total=13.931ms` total
- `read_offset_table_lookup=2.694ms` total
- `read_tensor_shape_capture=1.268ms` total

In performance-clean mode the planner no longer validates block tables and only
writes the final report. Building full per-call diagnostic metadata still costs
Python time without contributing to the gate contract.

## Acceptance criteria

- [x] Enable compact read-plan metadata only for the performance-clean GPU
  kernel replacement path.
- [x] Keep correctness/diagnostic paths on full `ReadPlanCall` metadata.
- [x] Preserve a non-empty `recent_calls` report for smoke compatibility.
- [x] Report compact/detailed plan metadata call counts.
- [x] Include compact metadata fields in the GPU read A/B contract.
- [x] Make the performance-clean gate require compact metadata.
- [x] Validate short vLLM smoke, performance-clean, performance-clean stress,
  and host-profile gates.

## Boundaries

- This does not change the GPU read kernel ABI, stream selection, or offset table
  cache semantics.
- This does not disable detailed metadata for correctness gates, block-table
  validation, report-on-update diagnostics, or CUDA event kernel profiling.

## Implementation notes

- Compact metadata is active when native read replacement uses the GPU kernel,
  block-table validation is disabled, per-update reporting is disabled, and CUDA
  event kernel profiling is disabled.
- The compact path still builds/uses the offset table and records the final
  `ReadPlanCall`, but it skips stride/contiguity/sample diagnostic fields.

## Verification

- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_smoke --mode kcmm --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-kv-write-verify --kv-write-device-slots --kv-read-gpu-kernel-candidate --kv-read-fast-current-context-launch --kv-read-precompile-gpu-kernel --no-kv-read-validate-block-tables --no-tracker-report-on-update --no-build-kcmm --no-print-seams`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-compact-plan-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-compact-plan-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-compact-plan-latest.json`

## Latest local results

- Date: 2026-07-02
- Short vLLM smoke: `compact_plan_metadata=true`,
  `compact_plan_metadata_calls=6`, `detailed_plan_metadata_calls=0`
- Performance-clean gate: `passed=true`
- Performance-clean compact/detailed plan calls: `372/0`
- Performance-clean request latency: stock `1.826s`, KCMM `1.813s`, ratio
  `0.993`
- Performance-clean tokens/s: stock `17.525`, KCMM `17.650`, ratio `1.007`
- Host-profile gate: `passed=true`
- Host-profile `read_replace_build_plan`: `11.654ms` total, `31.329us` avg
- Host-profile `read_build_plan_total`: `10.695ms` total, `28.750us` avg
- Previous host-profile before this issue: `read_replace_build_plan=14.894ms`,
  `read_build_plan_total=13.931ms`
- Performance-clean stress gate: `passed=true`
- Stress compact/detailed plan calls: `276/0`
- Stress request latency: stock `1.818s`, KCMM `1.791s`, ratio `0.985`
- Stress tokens/s: stock `26.403`, KCMM `26.801`, ratio `1.015`

## Follow-up

The remaining request-path host hotspots are now mostly outside read metadata:
read GPU-kernel host launch/stream selection and write device-slot table lookup,
stream selection, and ctypes launch.
