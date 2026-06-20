# Make KV write replacement stream-aware

Status: done
Type: AFK

## What to build

Remove the full-device synchronization from the vLLM KV write
mirror/replacement path. KCMM direct-slot writes should enqueue D2D copies on
the current PyTorch/vLLM CUDA stream, matching the Phase II.C GPU read kernel
launch model.

## Acceptance criteria

- [x] KCMM exports a stream-aware direct-slot C ABI for vLLM-style physical
  slot writes.
- [x] Python bindings can call the stream-aware direct-slot writer.
- [x] `kv_write_ffi_smoke` covers the stream-aware direct-slot writer.
- [x] vLLM KV write mirror/replacement uses
  `torch.cuda.current_stream(device).cuda_stream`.
- [x] The write path no longer calls `torch.cuda.synchronize(device)` or
  `pool.synchronize()` around every write.
- [x] D2H verification still waits for write completion with stream-level
  synchronization.
- [x] vLLM smoke reports stream-aware write counts.
- [x] The full GPU read-kernel A/B gate still passes.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/08-broaden-gpu-read-kernel-correctness-coverage.md`

## Implementation

- Added `kcmm_append_kv_slots_on_stream` to `src/kcmm/ffi.rs`.
- Declared `kcmm_append_kv_slots` and `kcmm_append_kv_slots_on_stream` in
  `include/kcmm.h`.
- Extended `scripts/kcmm/bindings.py` so `KcmmPool.append_kv_slots` accepts an
  optional `stream_ptr`.
- Updated `scripts/kcmm/kv_write_mirror.py` to enqueue KCMM writes on the
  current PyTorch CUDA stream.
- Replaced full-device write-path synchronization with stream synchronization
  only before D2H verification.
- Added write stream-awareness fields to the vLLM smoke and GPU read A/B
  reports.

## Validation

- `cargo check --features kcmm`
- `cargo build --features kcmm`
- `python -m py_compile scripts/kcmm/*.py`
- `git diff --check`
- `python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-build-kcmm`
- `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`

Latest local stream-aware write result on 2026-06-20:

- Low-level FFI gate: `passed=true`
- Direct-slot stream-aware write: `true`
- Direct-slot stream pointer: `0`
- vLLM write replacement smoke: `passed=true`
- Write seam: `vllm._custom_ops.reshape_and_cache`
- Native skipped calls: `8`
- Stream-aware write calls: `8`
- Stream-level verification synchronizations: `8`
- D2H verified rows: `10`
- Verification bytes: `5120`
- Cache layers mapped: `2`
- Full GPU read-kernel A/B gate: `passed=true`
- A/B correctness failures: `[]`
- A/B performance warnings: `[]`
- A/B stream-aware write calls: `22`
- A/B stream-level verification synchronizations: `22`
- A/B GPU read kernel calls: `16`
- A/B stream-aware read kernel calls: `16`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after runs.
- The temporary tiny OPT model directory was removed after the gate.

## Boundaries

- The current local vLLM eager seam still reports stream pointer `0`, the legacy
  default stream.
- Future non-default-stream vLLM scheduling still needs explicit validation
  that write and read seams are ordered by the framework stream graph or CUDA
  events.
- D2H verification remains intentionally synchronous at stream scope.
- Broader model shape coverage is tracked separately by issue 10.
- This does not broaden batching, concurrency, tensor parallelism, prefix
  cache, alibi, block-sparse mode, or FP8 cache scale coverage.

## Next step

Broaden Phase II.C from shape coverage to batch and concurrency coverage under
the supported `head_dim=64` envelope.
