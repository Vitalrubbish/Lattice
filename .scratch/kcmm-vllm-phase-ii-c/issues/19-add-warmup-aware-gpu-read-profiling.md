# Add warm-up-aware GPU read profiling

Status: done
Type: AFK

## What to build

Refine the Phase II.C GPU read-kernel profiler so the first profiled call can
be separated from steady-state kernel timing. Issue 18 showed a large first-call
sample that should not drive scalar kernel optimization decisions.

## Acceptance criteria

- [x] Preserve the existing raw `samples_ms` and aggregate profile fields.
- [x] Add `first_call_ms` to make cold-start behavior explicit.
- [x] Add `warmup_excluded_count` and `steady_state` stats that exclude the
  first sample when multiple samples exist.
- [x] Make the profile gate fail if steady-state fields are missing when the
  profile has more than one sample.
- [x] Keep profiling disabled by default for non-profile gates.
- [x] Re-run the profile gate locally and document the steady-state result.

## Boundaries

- This does not change the CUDA kernel implementation.
- This does not remove the first-call cost; it only separates it in reports.
- This does not add model-scale performance coverage.

## Verification

- `python -m py_compile scripts/kcmm/kv_read_plan.py scripts/kcmm/vllm_gpu_read_profile_gate.py`
- `python -m scripts.kcmm.vllm_gpu_read_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 240 --shutdown-timeout-seconds 45`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-profile-1782718561901.json`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- GPU read kernel calls: `16`.
- Profile sample count: `16`.
- First call: `100.57933 ms`.
- Warm-up excluded count: `1`.
- Overall profile summary: min `0.029696 ms`, avg `6.367104 ms`,
  p50 `0.05632 ms`, p95 `100.57933 ms`, p99 `100.57933 ms`,
  max `100.57933 ms`.
- Steady-state summary after excluding the first sample: count `15`,
  min `0.029696 ms`, avg `0.086289 ms`, p50 `0.05632 ms`,
  p95 `0.1536 ms`, p99 `0.1536 ms`, max `0.1536 ms`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the gate.

## Notes

- The first-call outlier reproduced almost exactly across two profile runs:
  `100.549629 ms` in issue 18 and `100.57933 ms` here.
- The current steady-state scalar GPU read kernel cost on the tiny local OPT
  gate is sub-millisecond. The next optimization issue should therefore focus
  on repeated steady-state samples across shape/batch coverage, not on the
  cold-start sample alone.
