# Add KCMM GPU read kernel candidate

Status: done
Type: AFK

## What to build

Replace the Phase II.C CPU-staged read replacement path with an opt-in GPU
kernel candidate. The mode should skip native vLLM `paged_attention_v1/v2`,
consume KCMM-managed K/V memory through the A2 offset table, and fill vLLM's
provided `out` tensor on GPU.

This is still a candidate path, not the final Phase II.C gate. It should prove
that the read path no longer needs CPU staging while keeping all constraints
explicit.

## Acceptance criteria

- [x] KCMM exposes a C ABI entrypoint for FP16 decode attention.
- [x] Python bindings expose the new C ABI entrypoint.
- [x] Launcher exposes `--kcmm-kv-read-gpu-kernel-candidate`.
- [x] Smoke exposes `--kv-read-gpu-kernel-candidate`.
- [x] The mode requires runtime-derived pool sizing, the KCMM-backed allocator,
  and KCMM KV writes.
- [x] The read seam skips native `paged_attention_v1/v2` when the GPU kernel
  candidate is enabled.
- [x] The GPU kernel consumes vLLM `query`, `out`, `block_tables`, `seq_lens`,
  KCMM K/V base addresses, and the A2 f16 offset table.
- [x] The report records `read_path=kcmm_paged_attn_decode_f16`,
  `replacement_backend=gpu_kernel`, `gpu_kernel_calls > 0`, and
  `reference_read_bytes=0`.
- [x] The combined write-replacement plus GPU read-kernel smoke completes
  successfully.
- [x] Documentation records current kernel boundaries and remaining gates.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/03-add-kcmm-read-replacement-candidate.md`

## Implementation

- Added `src/cuda/kernels/kcmm_vllm_paged_attn.cu` with an FP16 decode
  attention kernel for the tiny vLLM smoke contract.
- Added `kcmm_paged_attn_decode_f16` to the Rust FFI. It compiles the CUDA
  kernel with NVRTC, caches the loaded function per KCMM pool, launches it, and
  synchronizes before returning to vLLM.
- Added the C header declaration and Python ctypes binding.
- Extended `KcmmKvReadOffsetTableTracker` with `replacement_backend`, keeping
  the previous CPU-staged reference path available while adding the GPU kernel
  backend.
- Added launcher and smoke flags for the GPU kernel candidate.
- Added smoke validation for `replacement_backend=gpu_kernel`,
  `gpu_kernel_calls > 0`, and zero CPU-staged reference reads.

## Current boundaries

- FP16 query/output and FP16 KCMM K/V storage only.
- `head_dim <= 64`.
- Decode attention only for the current vLLM smoke shape.
- No alibi slopes, block-sparse attention, or FP8 cache scales.
- The FFI synchronizes before returning; stream-aware launch remains future
  work.
- The latest smoke proves runnable replacement, not a broad deterministic
  stock-vs-KCMM correctness/performance gate.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-replace-candidate --instrument-kv-reads --kv-read-gpu-kernel-candidate --no-build-kcmm`

Latest local GPU read-kernel smoke result on 2026-06-20:

- Result: `passed=true`
- Completion text: `" behaviour behaviour behaviour behaviour"`
- Native KV write calls skipped: `8`
- Native paged-attention calls replaced: `6`
- Read path: `kcmm_paged_attn_decode_f16`
- Kernel replaced: `true`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `6`
- Reference KCMM read bytes: `0`
- Offset table builds: `6`
- Observed read seam: `vllm._custom_ops.paged_attention_v1`
- KCMM write verified rows: `10`
- Storage of record: `kcmm_kv_storage_candidate`
- Final KCMM pool stats: `blocks_in_use=0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Next step

Add a deterministic stock-vs-KCMM correctness gate for the GPU read-kernel path,
then make the FFI launch stream-aware and start performance characterization.
