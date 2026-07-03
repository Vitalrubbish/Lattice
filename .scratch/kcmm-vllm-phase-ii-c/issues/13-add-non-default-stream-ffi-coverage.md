# Add non-default stream FFI coverage

Status: done
Type: AFK

## What to build

Add a Phase II.C low-level FFI gate that validates the stream-aware KCMM write
and read entrypoints on a real non-default CUDA stream.

The current vLLM eager seam reports stream pointer `0`, so integrated vLLM
smokes prove the `_on_stream` ABI is used but do not prove behavior when
`stream_ptr != 0`. This issue covers that gap without depending on vLLM's
scheduler selecting a non-default stream.

## Acceptance criteria

- [x] Create a CUDA stream with `torch.cuda.Stream()` and require its raw handle
  to differ from the default stream handle.
- [x] Call `kcmm_append_kv_slots_on_stream` with that non-default stream.
- [x] Call `kcmm_paged_attn_decode_f16_on_stream` with that same non-default
  stream.
- [x] Enqueue write then read on the same stream and validate the read output
  after synchronizing only that stream for verification.
- [x] Verify the direct-slot write bytes in KCMM VA memory after stream
  synchronization.
- [x] Emit a JSON report that records `stream_ptr`, default stream pointer,
  stream-aware write/read flags, and final KCMM stats.
- [x] Keep the existing Python KCMM scripts compiling.

## Implementation

Added `scripts/kcmm/non_default_stream_ffi_smoke.py`.

The smoke creates a tiny KCMM pool, allocates one block, creates a real
`torch.cuda.Stream()`, and requires its raw stream handle to be non-zero and
different from the default stream handle. It then enqueues:

1. `kcmm_append_kv_slots_on_stream` to write one K/V row into the KCMM physical
   slot.
2. `kcmm_paged_attn_decode_f16_on_stream` on the same stream to read that row
   through the KCMM GPU read kernel.

The query is zero and `seq_len=1`, so the expected decode output is exactly the
written V row. The smoke synchronizes only the non-default stream before
checking the output and D2H byte-level K/V contents. Cleanup synchronizes the
pool after the verification boundary.

## Verification

```bash
python -m py_compile scripts/kcmm/*.py
python -m scripts.kcmm.non_default_stream_ffi_smoke \
  --no-build-kcmm \
  --output /tmp/kcmm-vllm-phase-ii-c-non-default-stream-1782615543.json
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
```

Latest local result on 2026-06-28:

- Result: `passed=true`
- Report: `/tmp/kcmm-vllm-phase-ii-c-non-default-stream-1782615543.json`
- Device: `NVIDIA GeForce RTX 3080`
- PyTorch/CUDA: `2.4.0+cu118` / `11.8`
- Non-default stream pointer: `94207523571936`
- Default stream pointer: `0`
- Direct-slot write path: `kcmm_append_kv_slots_on_stream`
- GPU read path: `kcmm_paged_attn_decode_f16_on_stream`
- Verified direct-slot K/V rows: `1`
- Expected read output matched actual output:
  `[1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0]`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the run.

## Boundaries

- This does not force vLLM itself to schedule work on a non-default stream.
- This does not broaden tensor parallelism, prefix cache, non-64 head
  dimensions, alibi, block-sparse mode, or FP8 cache scale coverage.
- This is a low-level FFI causality gate, not an end-to-end vLLM A/B gate.
