# Add GPU read-kernel per-call profiling

Status: done
Type: AFK

## What to build

Add an opt-in profiler for the Phase II.C KCMM GPU read-kernel path so future
kernel optimization has per-call CUDA timing data instead of only coarse
request-level latency and throughput.

## Acceptance criteria

- [x] Add a launcher/smoke flag that enables profiling only for the KCMM GPU
  read-kernel replacement path.
- [x] Time `kcmm_paged_attn_decode_f16_on_stream` with CUDA events recorded on
  the same stream used for the raw-pointer FFI launch.
- [x] Preserve the existing stream ordering and tensor lifetime handling.
- [x] Report per-call `gpu_kernel_elapsed_ms` in recent read calls.
- [x] Report `gpu_kernel_profile` with count, min, avg, p50, p95, p99, max, and
  raw samples in milliseconds.
- [x] Add a repeatable gate that fails if profiling is enabled but no samples
  are recorded.
- [x] Keep profiling disabled by default for normal correctness gates.

## Boundaries

- This does not optimize the scalar GPU read kernel.
- This does not add alibi, block-sparse mode, FP8 cache scales, or broader
  model coverage.
- Profiling synchronizes CUDA events and is diagnostic-only; it should not be
  treated as the default steady-state latency path.

## Implementation notes

- The launcher flag is `--kcmm-kv-read-profile`.
- The smoke/A-B flag is `--kv-read-profile`.
- The dedicated wrapper is
  `python -m scripts.kcmm.vllm_gpu_read_profile_gate`.
- The profiler records CUDA events around the KCMM GPU read-kernel FFI call on
  `stream_selection.stream`, then synchronizes the end event only when profiling
  is enabled.

## Verification

- `python -m py_compile scripts/kcmm/config.py scripts/kcmm/launcher.py scripts/kcmm/kv_read_plan.py scripts/kcmm/vllm_smoke.py scripts/kcmm/vllm_gpu_read_ab_gate.py scripts/kcmm/vllm_gpu_read_profile_gate.py scripts/kcmm/vllm_gpu_read_batch_gate.py scripts/kcmm/vllm_gpu_read_shape_gate.py scripts/kcmm/vllm_ab_gate.py`
- `python -m scripts.kcmm.vllm_gpu_read_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 240 --shutdown-timeout-seconds 45`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-profile-1782717377628.json`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- GPU read kernel calls: `16`.
- Profile sample count: `16`.
- Profile summary: min `0.029696 ms`, avg `6.36448 ms`, p50 `0.05632 ms`,
  p95 `100.549629 ms`, p99 `100.549629 ms`, max `100.549629 ms`.
- Raw samples:
  `[100.549629, 0.029696, 0.032768, 0.031744, 0.03584, 0.034816, 0.05632, 0.057344, 0.0512, 0.052224, 0.146432, 0.145408, 0.150528, 0.149504, 0.154624, 0.1536]`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the gate.

## Notes

- The first profiled call recorded `100.549629 ms`, while later calls were in
  the `0.03-0.15 ms` range. Treat the first sample as a cold-start/module
  warm-up outlier until a follow-up profiling pass proves otherwise.
