# Cache read offset table in performance-clean gate

Status: done
Type: AFK

## What to build

Reduce remaining performance-clean overhead by avoiding read-side host
validation and repeated GPU offset-table construction after correctness gates
have already passed.

## Why

After Issue 23, per-update JSON report writes were no longer the primary cost.
The performance-clean path still copied `block_tables` to CPU, checked block
locations, and rebuilt the GPU `offset_table[block_id] = kcmm_f16_va_offset`
tensor on every read seam call. These checks are useful in correctness gates,
but they are host-side test overhead in the request-level performance baseline.

## Acceptance criteria

- [x] Add an opt-out switch for host-side read block-table validation.
- [x] Keep host-side read block-table validation enabled by default for
  correctness gates.
- [x] Cache the read offset table and rebuild it only when the KCMM block count
  grows or the target device changes.
- [x] Include offset-table cache hit/rebuild counts in the KCMM GPU read
  contract.
- [x] Make `vllm_gpu_read_perf_clean_gate` disable host-side read block-table
  validation.
- [x] Run the performance-clean gate and compare request latency against Issue
  23's baseline.

## Boundaries

- This does not change the CUDA attention kernel.
- This does not remove correctness validation from the correctness/profile
  gates.
- This assumes the KCMM block offset for an allocated block id is stable for the
  lifetime of the pool.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-1782743265566.json`.
- Result: `passed=true`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- Coverage case: `long_decode`, `32` generated tokens.
- Stock/KCMM completion text matched.
- KCMM GPU read kernel calls: `372`.
- KCMM stream-aware read kernel calls: `372`.
- KCMM reference read bytes: `0`.
- Read block-table validation enabled: `false`.
- Offset table builds/read calls: `372`.
- Offset table cache hits: `369`.
- Offset table cache rebuilds: `3`.
- Read tracker `report_on_update=false`, `report_write_count=1`.
- Write tracker `report_on_update=false`, `report_write_count=1`.
- Request latency seconds: stock `1.814`, KCMM `1.934`, ratio `1.066`.
- Tokens per second: stock `17.641`, KCMM `16.546`, ratio `0.938`.
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`.
- Compared with Issue 23 baseline: KCMM request latency changed from `1.951s`
  to `1.934s`; KCMM/stock ratio changed from `1.072x` to `1.066x`.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the gate.
