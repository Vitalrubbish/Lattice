# Add performance-clean stress gate

Status: done
Type: AFK

## What to build

Add a stress wrapper for the Phase II.C performance-clean real-model gate that
runs multiple concurrent completion cases while keeping the same fast-path
performance-clean settings.

## Why

The single-request performance-clean gate is now effectively equal to stock on
request latency after the recent read/write host-side cleanups. Before deeper
ABI changes such as device-side write slots, the fast path should be checked
under concurrent real-model scheduling pressure. The existing correctness batch
gate covers concurrency with test-only instrumentation enabled; this issue adds
the equivalent performance-clean coverage with read tracing, write D2H
verification, per-update reports, and block-table validation disabled.

## Acceptance criteria

- [x] Add a `vllm_gpu_read_perf_clean_stress_gate` wrapper.
- [x] Default to two real-model coverage cases with `completion_concurrency=2`
  and `max_num_seqs=2`.
- [x] Keep performance-clean flags enabled: no read tracing, no write
  verification, no per-update reports, no block-table validation,
  current-context read launch, and read-kernel precompile.
- [x] Fail if the KCMM read report does not observe a configured minimum read
  batch.
- [x] Run the stress gate and record stock-vs-KCMM result.
- [x] Update developer docs with the new command and latest local result.

## Boundaries

- This does not change CUDA kernels or KCMM FFI.
- This does not replace the existing single-request performance-clean gate.
- This does not cover tensor parallelism; the existing TP correctness gate
  remains separate.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --help`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-latest.json`

## Latest local result

- Date: 2026-07-01
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Real model: `facebook/opt-125m`
- Coverage cases: `stress_history`, `stress_memory`
- Completion concurrency: `2`
- Max num seqs: `2`
- Max num batched tokens: `192`
- Observed max read batch: `2`
- Observed max write batch: `9`
- GPU read kernel calls: `276`
- Stream-aware read kernel calls: `276`
- Reference KCMM read bytes: `0`
- Read block-table validation enabled: `false`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `111.443ms`
- Read min-entry total-block calls: `276`
- Offset table cache hits/rebuilds: `273/3`
- Write verification enabled: `false`
- KCMM write verified rows: `0`
- Write stream verification synchronizations: `0`
- Request latency seconds: stock `2.116`, KCMM `1.964`, ratio `0.928`
- Tokens per second: stock `22.684`, KCMM `24.440`, ratio `1.077`
- Peak GPU memory delta MiB: stock `5443`, KCMM `5593`, ratio `1.028`

## Follow-up

The current fast path now has both single-request and concurrent real-model
performance-clean coverage. The next meaningful risk-reduction step is to add a
performance-clean tensor-parallel wrapper, or move to the device-side write slot
path if TP performance-clean is deferred.
