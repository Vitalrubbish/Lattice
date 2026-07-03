# Add current-context read launch fast path

Status: done
Type: AFK

## What to build

Add opt-in Phase II.C read-kernel launch optimizations for vLLM/PyTorch seams:

- a current-context stream launch ABI that skips the Rust-side context bind
  when the caller already has the correct CUDA context current,
- a read-kernel precompile hook that moves NVRTC/module load out of the first
  request.

## Why

Issue 25 showed the dominant measured host-side read section under
`read_gpu_kernel_ctypes_launch`. That section includes the Python ctypes call,
the Rust FFI wrapper, current-context binding, parameter packing, the first-call
NVRTC/module load, and the CUDA kernel enqueue. The least invasive optimizations
are to avoid repeated context binding in the stream-aware vLLM path and to
precompile/load the read kernel before the measured request starts, while
retaining the original C ABI as the safe default.

## Acceptance criteria

- [x] Add a C ABI function for stream-aware paged-attention launch that assumes
  the caller's CUDA context is already current.
- [x] Add a C ABI function to precompile/load the KCMM paged-attention kernel.
- [x] Keep the existing context-binding launch ABI unchanged.
- [x] Add optional Python binding/config plumbing for the fast launch and
  precompile paths.
- [x] Enable both optimizations in the performance-clean and host-profile gates.
- [x] Report whether the read tracker used the fast current-context launch path.
- [x] Report whether the read tracker precompiled the GPU read kernel.
- [x] Run correctness/performance gates and record results.

## Boundaries

- This does not change CUDA kernel math.
- This does not remove the safe context-binding path.
- This is valid only for vLLM/PyTorch seams that already have an active current
  CUDA context for the stream being passed.
- Precompile moves startup work earlier; it does not remove compile cost from
  the process lifetime.

## Verification

- [x] `python -m py_compile scripts/kcmm/*.py`
- [x] `cargo check --features kcmm`
- [x] `cargo build --features kcmm`
- [x] `cargo build --release --features kcmm`
- [x] `git diff --check`
- [x] `nm -D target/debug/libbaseline_llm_os.so | rg "kcmm_precompile_paged_attn_decode_f16|kcmm_paged_attn_decode_f16_on_current_context_stream"`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-fast-precompile-latest.json`
  passed.
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-fast-precompile-latest.json`
  passed.

## Latest local result

Performance-clean gate:

- Date: 2026-06-29
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-fast-precompile-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `99.706ms`
- Request latency seconds: stock `1.849`, KCMM `1.825`, ratio `0.987`
- Tokens per second: stock `17.307`, KCMM `17.534`, ratio `1.013`
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`

Host-profile gate:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-fast-precompile-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Read GPU kernel precompile elapsed: `97.574ms`
- Request latency seconds: stock `1.836`, KCMM `1.874`, ratio `1.021`
- Tokens per second: stock `17.429`, KCMM `17.076`, ratio `0.980`
- Top read host sections:
  `read_gpu_kernel_precompile=97.587ms` total, one attach-time call;
  `read_replace_call_total=40.176ms` total, `107.999us` avg;
  `read_replace_gpu_kernel_host=19.129ms` total, `51.420us` avg;
  `read_gpu_kernel_host_total=18.076ms` total, `48.592us` avg;
  `read_gpu_kernel_ctypes_launch=6.337ms` total, `17.034us` avg.
- Compared with Issue 25 host-profile result, `read_gpu_kernel_ctypes_launch`
  dropped from `106.949ms` total / `287.497us` avg to `6.337ms` total /
  `17.034us` avg because the first-call compile/load is now precompiled at
  pool attach.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the gates.

## Follow-up

The dominant request-time read overhead is no longer FFI cold start. Remaining
read-side host sections are mostly plan construction and normal kernel enqueue
cost. The next optimization should focus on reducing write replacement overhead
or lowering per-read plan construction further.
