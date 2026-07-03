# Add host-side performance attribution gate

Status: done
Type: AFK

## What to build

Add a diagnostic Phase II.C gate that keeps the performance-clean stock-vs-KCMM
comparison but enables low-overhead host-side section timing inside the KCMM
read and write trackers.

## Why

After Issue 24, the performance-clean gate still showed about `5-7%` KCMM
request latency overhead against stock vLLM. The GPU kernel profiler was
disabled for the clean baseline, so the next optimization target needed
host-side attribution without reintroducing per-update report writes or
correctness-only synchronizations.

## Acceptance criteria

- [x] Add optional host-side section timing to KCMM read/write trackers.
- [x] Keep host timing disabled by default.
- [x] Add CLI plumbing through `vllm_smoke`, the launcher, and the GPU read
  A/B gate.
- [x] Add a host-profile gate that wraps the performance-clean gate and forces
  host timing on.
- [x] Include read/write top host sections in the gate report.
- [x] Run the host-profile gate locally and record the report path and top
  sections.

## Boundaries

- This records host wall-clock timing only.
- This does not enable CUDA event profiling or add CUDA synchronizations.
- This is a diagnostic gate, not the default performance-clean baseline.
- Section totals are nested, so totals should not be summed as independent
  request-level costs.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --help`
- [x] `python -m scripts.kcmm.vllm_gpu_read_ab_gate --help`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-latest.json`
  passed.

## Latest local result

- Date: 2026-06-29
- Result: `passed=true`
- Report: `/tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Coverage case: `long_decode`, `32` generated tokens.
- Stock/KCMM completion text matched.
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- Offset table cache hits/rebuilds: `369/3`
- Request latency seconds: stock `1.822`, KCMM `2.179`, ratio `1.196`
- Tokens per second: stock `17.563`, KCMM `14.686`, ratio `0.836`
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the gate.

Top read-side host sections:

- `read_replace_call_total`: `143.137ms` total, `384.777us` avg, `372` calls.
- `read_replace_gpu_kernel_host`: `120.486ms` total, `323.887us` avg,
  `372` calls.
- `read_gpu_kernel_host_total`: `119.364ms` total, `320.871us` avg,
  `372` calls.
- `read_gpu_kernel_ctypes_launch`: `106.949ms` total, `287.497us` avg,
  `372` calls.
- `read_replace_build_plan`: `19.093ms` total, `51.326us` avg, `372` calls.
- `read_build_plan_total`: `17.991ms` total, `48.363us` avg, `372` calls.

Top write-side host sections:

- `write_mirror_call_total`: `53.675ms` total, `139.777us` avg, `384` calls.
- `write_slot_mapping_to_host`: `11.554ms` total, `30.088us` avg, `384` calls.
- `write_ctypes_launch`: `8.029ms` total, `20.907us` avg, `384` calls.
- `write_select_stream`: `5.004ms` total, `13.031us` avg, `384` calls.
- `write_ensure_slot_blocks`: `4.597ms` total, `11.972us` avg, `384` calls.

## Follow-up

The dominant measured host-side sections are nested under the read replacement
path, especially `read_gpu_kernel_ctypes_launch`. The next optimization should
reduce or bypass per-read Python/ctypes launch overhead before changing CUDA
kernel math.
