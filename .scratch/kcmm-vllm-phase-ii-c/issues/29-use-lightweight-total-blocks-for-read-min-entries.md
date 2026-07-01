# Use lightweight total blocks for read min entries

Status: done
Type: AFK

## What to build

Use the existing `kcmm_total_blocks()` C ABI from Python bindings and the read
replacement planner when block-table validation is disabled, instead of calling
full `pool.stats()` for every read seam just to compute offset-table
`min_entries`.

## Why

The performance-clean read path disables host-side block-table validation, so it
cannot derive `min_entries` from sampled block IDs without copying
`block_tables` to host. The previous safe fallback called `pool.stats()` on
every read seam and used the pool's total block count. Host profiling showed
`read_pool_stats_for_min_entries` at about `3.34ms` total across `372` read
calls. The Rust ABI already exposes `kcmm_total_blocks()`, which preserves the
same total-block coverage semantics with a narrower ctypes call and no stats
dict construction.

## Acceptance criteria

- [x] Bind `kcmm_total_blocks()` in Python.
- [x] Add `KcmmPool.total_blocks()`.
- [x] Use `pool.total_blocks()` for read `min_entries` when block-table
  validation is disabled.
- [x] Report the number of lightweight total-block calls in read/gate reports.
- [x] Run the KV write FFI smoke gate as a quick ABI smoke.
- [x] Run the host-profile gate and record read-side host section changes.
- [x] Run the performance-clean gate and record stock-vs-KCMM result.

## Boundaries

- This does not change CUDA kernels or the Rust ABI.
- This does not change offset-table semantics: min entries still cover the
  KCMM pool's total block count when block-table validation is disabled.
- This does not remove the full `pool.stats()` path from report generation or
  other diagnostics.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm --output /tmp/kcmm-vllm-phase-ii-c-total-blocks-smoke-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-total-blocks-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-total-blocks-latest.json`

## Latest local result

KV write FFI smoke:

- Date: 2026-07-01
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-total-blocks-smoke-latest.json`
- Direct-slot writes still passed for slots `2` and `7`.
- Invalid direct slot `16` still failed with `block_idx 4 from slot 16 not in use`.

Host-profile gate:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-total-blocks-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Lightweight total-block calls for read min entries: `372`
- `read_pool_stats_for_min_entries`: `0.959ms` total, `2.577us` avg,
  `372` calls.
- Compared with Issue 28 host-profile result,
  `read_pool_stats_for_min_entries` dropped from `3.342ms` to `0.959ms`.
- Request latency seconds: stock `1.831`, KCMM `1.909`, ratio `1.043`.
- Tokens per second: stock `17.477`, KCMM `16.763`, ratio `0.959`.
- The diagnostic run included one `read_gpu_kernel_ctypes_launch` outlier
  around `28ms`; use the performance-clean gate for request-level judgment.

Performance-clean gate:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-total-blocks-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `96.324ms`
- Lightweight total-block calls for read min entries: `372`
- Request latency seconds: stock `1.825`, KCMM `1.824`, ratio `0.999`
- Tokens per second: stock `17.534`, KCMM `17.544`, ratio `1.001`
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`

## Follow-up

The full-stats min-entry lookup is no longer a meaningful read-side section.
The remaining request-time read-side host overhead is dominated by normal kernel
enqueue variance and plan construction. The next meaningful step is either a
device-side write slot path or broader stress coverage for performance-clean
settings before deeper ABI changes.
