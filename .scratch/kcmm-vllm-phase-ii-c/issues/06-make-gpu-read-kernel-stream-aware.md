# Make GPU read-kernel launch stream-aware

Status: done
Type: AFK

## What to build

Remove the per-call full CUDA context synchronization from the Phase II.C GPU
read-kernel path. The vLLM-integrated path should enqueue
`kcmm_paged_attn_decode_f16` on the caller's current PyTorch CUDA stream and
return without synchronizing the whole device.

The old synchronous C ABI should remain available as a compatibility wrapper.

## Acceptance criteria

- [x] Add a stream-aware C ABI entrypoint for the GPU read kernel.
- [x] Preserve the existing synchronous `kcmm_paged_attn_decode_f16` ABI.
- [x] Python bindings can call the stream-aware entrypoint.
- [x] The vLLM read replacement path passes `torch.cuda.current_stream(...).cuda_stream`.
- [x] The read report records stream-aware GPU kernel launches.
- [x] Smoke validation fails if the GPU read-kernel path uses no stream-aware
  launches.
- [x] Pool teardown waits before unloading the raw kernel module.
- [x] The deterministic GPU read-kernel A/B gate still passes.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/05-add-gpu-read-kernel-ab-gate.md`

## Implementation

- Added `kcmm_paged_attn_decode_f16_on_stream`.
- Changed the vLLM GPU read path to pass PyTorch's current CUDA stream pointer.
- Replaced the vLLM kernel cache with a raw CUDA Driver module/function cache
  for this kernel so `cuLaunchKernel` can receive the caller stream.
- Kept `kcmm_paged_attn_decode_f16` as a synchronous compatibility wrapper.
- Added `stream_aware_kernel_calls`, `stream_ptr`, and `stream_aware_launch` to
  the read report.
- Added smoke validation for `stream_aware_kernel_calls > 0`.
- Added device synchronization during pool destroy so the raw module is not
  unloaded while caller-stream work may still be in flight.

## Current boundaries

- The local vLLM eager seam reports PyTorch's current stream pointer as `0`,
  the legacy default stream. This is still passed through the stream-aware ABI,
  and the hot read path no longer synchronizes the entire CUDA context.
- The KV write replacement path still synchronizes in Python before/after direct
  slot writes. Making KV writes stream-aware remains future work.
- This does not add broader prompt/shape coverage or performance pass/fail
  thresholds.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `git diff --check`
- `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`

Latest local A/B result on 2026-06-20:

- Result: `passed=true`
- Stock completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `6`
- Stream-aware kernel calls: `6`
- Stream pointer sample: `[0, 0, 0, 0, 0, 0]`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `8`
- KCMM write verified rows: `10`
- Final KCMM pool stats: `blocks_in_use=0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.

## Next step

Add basic performance characterization for the GPU read-kernel path and start
broadening correctness coverage beyond the tiny local OPT shape.
