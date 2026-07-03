# Add performance-clean GPU read gate

Status: done
Type: AFK

## What to build

Add a Phase II.C gate that compares stock vLLM with the KCMM GPU read-kernel
path after disabling correctness-only overhead in the KCMM mode.

## Acceptance criteria

- [x] Add an opt-out switch for KCMM KV write D2H verification while keeping
  verification enabled by default for existing correctness gates.
- [x] Add an opt-out switch for A/B gate KV read trace instrumentation while
  keeping it enabled by default for existing correctness gates.
- [x] Add a real-model performance-clean gate that disables read tracing, write
  D2H verification, and read-kernel profiling.
- [x] Verify the gate report still proves the KCMM mode used the GPU read
  kernel and did not fall back to CPU-staged reference reads.
- [x] Verify the gate report records zero write verified rows and zero
  verification synchronizations.
- [x] Run the performance-clean gate locally and record the report path.

## Boundaries

- This gate gives a cleaner request-level baseline; it is not a microbenchmark.
- This does not remove startup, Python monkey-patch, HTTP server, or vLLM
  scheduling overhead.
- This does not claim KCMM is faster than stock yet.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-1782722236025.json`.
- Result: `passed=true`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- Coverage case: `long_decode`, `32` generated tokens.
- Stock/KCMM completion text matched.
- KCMM GPU read kernel calls: `372`.
- KCMM stream-aware read kernel calls: `372`.
- KCMM reference read bytes: `0`.
- KCMM write verified rows: `0`.
- KCMM write verification synchronizations: `0`.
- KCMM write verification enabled: `false`.
- Request latency seconds: stock `1.855`, KCMM `3.285`, ratio `1.771`.
- Tokens per second: stock `17.251`, KCMM `9.741`, ratio `0.565`.
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the gate.
