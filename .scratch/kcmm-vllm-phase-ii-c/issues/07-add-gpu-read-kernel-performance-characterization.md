# Add GPU read-kernel performance characterization

Status: done
Type: AFK

## What to build

Extend the Phase II.C GPU read-kernel A/B gate so it records basic performance
characterization for stock vLLM versus KCMM GPU read replacement.

This should not turn performance into a hard pass/fail gate yet. Correctness
failures should still control the command exit code, while performance
regressions should be reported as warnings.

## Acceptance criteria

- [x] The GPU read-kernel A/B gate reports startup latency for stock and KCMM.
- [x] The gate reports completion request latency for stock and KCMM.
- [x] The gate reports generated-token throughput for stock and KCMM.
- [x] The gate reports peak GPU memory delta for stock and KCMM.
- [x] The report includes `performance_comparison` with KCMM-to-stock ratios.
- [x] The report includes `performance_warnings`.
- [x] Performance warnings do not fail the command.
- [x] Warning thresholds are configurable from the CLI.
- [x] Documentation records the latest local characterization.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/06-make-gpu-read-kernel-stream-aware.md`

## Implementation

- Added warning-threshold CLI options to `scripts/kcmm/vllm_gpu_read_ab_gate.py`:
  - `--latency-warning-ratio`
  - `--throughput-warning-ratio`
  - `--memory-warning-ratio`
  - `--memory-warning-min-delta-mib`
- Added `performance_comparison` to the JSON report.
- Added `performance_warnings` to the JSON report.
- Kept `passed` tied to correctness failures only.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `git diff --check`
- `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`

Latest local performance result on 2026-06-20:

- Result: `passed=true`
- Performance warnings: `[]`
- Stock completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM completion text: `" pioneer pioneer pioneer pioneer"`
- Startup seconds: stock `13.543`, KCMM `10.528`, ratio `0.777`
- Request latency seconds: stock `1.731`, KCMM `1.908`, ratio `1.102`
- Tokens per second: stock `2.311`, KCMM `2.096`, ratio `0.907`
- Peak GPU memory delta MiB: stock `3417`, KCMM `3425`, ratio `1.002`
- GPU kernel calls: `6`
- Stream-aware kernel calls: `6`
- Reference KCMM read bytes: `0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.

## Next step

Broaden correctness coverage beyond the single tiny local OPT prompt and shape.
