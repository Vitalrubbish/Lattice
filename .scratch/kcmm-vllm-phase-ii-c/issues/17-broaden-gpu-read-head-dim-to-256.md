# Broaden GPU read head-dim coverage to 256

Status: done
Type: AFK

## What to build

Broaden the Phase II.C GPU read-kernel path from the previous
`head_dim <= 128` envelope to the full set of paged-attention head dimensions
supported by the local CUDA 11.8 vLLM/XFormers stack, including `192` and
`256`.

## Acceptance criteria

- [x] Raise the KCMM vLLM paged-attention CUDA kernel head-dimension limit to
  `256`.
- [x] Align the Rust FFI guard and C ABI documentation with the new `256`
  limit.
- [x] Update the shape coverage gate so custom variants can use
  `head_dim=192` and `256`.
- [x] Add default shape-gate coverage for `head192_layers2` and
  `head256_layers2`.
- [x] Keep existing `64`, `80`, `96`, and `128` default shape coverage.
- [x] Run the updated shape gate locally or document the blocking behavior
  precisely.
- [x] Keep the existing stream-aware and no-reference-read assertions intact.

## Boundaries

- This is correctness and coverage, not a performance optimization.
- This does not add alibi, block-sparse mode, FP8 cache scale, or multi-node
  tensor-parallel support.
- This does not replace the scalar one-thread-per-head GPU read kernel design.

## Implementation notes

- The scalar kernel uses fixed local arrays for `acc` and `q_val`; broadening to
  `256` increases per-thread local storage and may reduce occupancy. That is
  acceptable for this coverage slice because the Phase II.C path is still a
  correctness candidate.

## Verification

- `cargo build --features kcmm`
- `python -m py_compile scripts/kcmm/vllm_gpu_read_shape_gate.py scripts/kcmm/non_default_stream_ffi_smoke.py`
- `python -m scripts.kcmm.vllm_gpu_read_shape_gate --no-build-kcmm --no-print-seams --timeout-seconds 240 --shutdown-timeout-seconds 45`
  passed with aggregate report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-shape-gate-1782637499399.json`.
- The updated default shape gate covered `head64_layers2`, `head80_layers2`,
  `head96_layers2`, `head128_layers2`, `head192_layers2`, and
  `head256_layers2`.
- Correctness failures: `[]`; failed variants: `[]`; performance warnings:
  `[]`.
- Every variant used the KCMM GPU read path with `gpu_kernel_calls=10`,
  `stream_aware_kernel_calls=10`, `reference_read_bytes=0`, and final
  `blocks_in_use=0`.
- `python -m scripts.kcmm.non_default_stream_ffi_smoke --no-build-kcmm --head-dim 256 --output /tmp/kcmm-non-default-stream-head256.json`
  passed, confirming the stream-aware FFI read/write path at the new maximum
  head dimension.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the gates.

## Notes

- An initial full shape-gate run kept the old `long_context` coverage at
  `max_tokens=4` and failed only `head192_layers2` on generated text after the
  first long-context token. The CPU-staged reference replacement reproduced the
  KCMM GPU result, and a direct vLLM paged-attention-vs-PyTorch check showed
  FP16-scale attention output differences. The shape gate now uses a
  single-token `long_context` case: it still spans multiple KV blocks but avoids
  recursively amplifying normal FP16 decode differences into a later-token text
  mismatch.
