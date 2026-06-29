# Defer tracker reports for performance-clean gate

Status: done
Type: AFK

## What to build

Remove per-seam JSON report writes from the performance-clean gate while
keeping final reports and existing correctness-gate defaults.

## Why

The first performance-clean run disabled KV read trace instrumentation, KV write
D2H verification, and read-kernel profiling, but still ran slower than stock.
The KCMM mode still wrote full JSON tracker reports on every read and write
seam call. In the 32-token `facebook/opt-125m` run that meant roughly hundreds
of read and write report writes during one request.

## Acceptance criteria

- [x] Add a launcher/smoke/A-B option to disable per-update tracker report
  writes without removing final reports.
- [x] Keep per-update tracker report writes enabled by default for existing
  correctness gates.
- [x] Make `vllm_gpu_read_perf_clean_gate` disable per-update tracker report
  writes.
- [x] Include read/write tracker report write counts in the KCMM GPU read
  contract.
- [x] Run the performance-clean gate and compare request latency against Issue
  22's baseline.

## Boundaries

- This optimizes test harness/reporting overhead, not the GPU read kernel.
- Final JSON reports must still exist for pass/fail validation.
- Correctness/profile gates should retain their current default reporting
  behavior.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-1782723060120.json`.
- Result: `passed=true`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- Coverage case: `long_decode`, `32` generated tokens.
- Stock/KCMM completion text matched.
- KCMM GPU read kernel calls: `372`.
- KCMM stream-aware read kernel calls: `372`.
- KCMM reference read bytes: `0`.
- KCMM write verified rows: `0`.
- Read tracker `report_on_update=false`, `report_write_count=1`.
- Write tracker `report_on_update=false`, `report_write_count=1`.
- Request latency seconds: stock `1.820`, KCMM `1.951`, ratio `1.072`.
- Tokens per second: stock `17.582`, KCMM `16.402`, ratio `0.933`.
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`.
- Compared with Issue 22 baseline: KCMM request latency improved from
  `3.285s` to `1.951s`; KCMM/stock ratio improved from `1.771x` to `1.072x`.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the gate.
