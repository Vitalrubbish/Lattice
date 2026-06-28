# Broaden GPU read head-dim coverage

Status: done
Type: AFK

## What to build

Broaden the Phase II.C GPU read-kernel path beyond the original `head_dim=64`
shape envelope. The current CUDA 11.8 vLLM/XFormers stack supports several
paged-attention head sizes above 64, but the KCMM correctness kernel and shape
gate were capped at 64.

## Acceptance criteria

- [x] Raise the KCMM GPU read-kernel head-dim envelope from `<=64` to `<=128`.
- [x] Keep the FFI guard aligned with the CUDA kernel local buffer size.
- [x] Update C ABI comments and Python gate validation to reflect the new
  supported envelope.
- [x] Add shape-gate coverage for at least one non-64 head dimension.
- [x] Run the shape gate locally across the default variants.
- [x] Keep the existing batch/concurrency gate passing or document why it was
  not rerun.

## Boundaries

- This remains the scalar correctness kernel; it is not a performance
  optimization.
- This does not cover `head_dim > 128`.
- This does not add tensor parallelism, prefix cache, alibi, block-sparse mode,
  or FP8 cache scale support.

## Implementation notes

- The CUDA correctness kernel now sizes local `acc` and `q_val` buffers with
  `KCMM_VLLM_PAGED_ATTN_MAX_HEAD_DIM=128`.
- The Rust FFI guard and C ABI docs now reject only `head_dim > 128`.
- The shape gate default variants now cover `head_dim=64`, `80`, `96`, and
  `128`; custom variants may use the CUDA 11.8 vLLM/XFormers-supported head
  dimensions up to `128`.
- The superblock allocator now supports logical block sizes that do not evenly
  divide 2 MiB. It hands out only full blocks and leaves the superblock tail as
  padding, which is required for shapes such as `head_dim=80` and `96`.

## Verification

- `cargo test --features kcmm superblock -- --nocapture`
- `cargo build --features kcmm`
- `python -m scripts.kcmm.vllm_gpu_read_shape_gate --no-build-kcmm --no-print-seams`
  passed with aggregate report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-shape-gate-1782621289455.json`.
- `python -m scripts.kcmm.vllm_gpu_read_batch_gate --no-build-kcmm --no-print-seams`
  passed with aggregate report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782621793642.json`.
- `nvidia-smi` showed both RTX 3080 GPUs back at `0 MiB` used after the local
  gates.
