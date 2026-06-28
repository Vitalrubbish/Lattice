# Fix long concurrent GPU read divergence

Status: done
Type: AFK

## Root cause

The KCMM GPU kernel indexes `query`, `block_tables`, and `seq_lens` by raw
pointer offsets and assumes a contiguous `[batch, heads, head_dim]` layout.
vLLM passes `query` as a strided slice from a fused QKV projection (e.g.
stride `[384, 64, 1]` for shape `[2, 2, 64]` instead of the expected
`[128, 64, 1]`). The reference backend used PyTorch indexing, which respects
those strides, so it matched stock vLLM. The standalone debug script used
contiguous tensors, so it also matched. The kernel read the wrong query vector
for every batch entry after the first, which only became visible in a
multi-sequence decode batch.

## Fix

In `scripts/kcmm/kv_read_plan.py`, `_run_gpu_kernel_attention` now materializes
contiguous copies of `query`, `block_tables`, and `seq_lens` before launching the
kernel. Those copies are enqueued on PyTorch's current CUDA stream, and the
KCMM kernel is launched through the stream-aware ABI with the same `stream_ptr`.
If vLLM passes a non-contiguous output tensor, the replacement writes into a
contiguous output tensor and copies back to `out` on the same stream.

The read report now records the original read-seam tensor layout diagnostics,
including shape, stride, and contiguity for `query`, `out`, `block_tables`, and
`seq_lens`.

## Verification

```bash
python -m scripts.kcmm.vllm_gpu_read_batch_gate --no-build-kcmm --no-print-seams
```

Passes: both `parallel_alpha` and `parallel_math` now match stock vLLM.

Latest local stream-aware result on 2026-06-28:

- Result: `passed=true`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Aggregate report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782614236702.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782614236702`
- `parallel_alpha` completion: `" Vol Vol Vol Vol Vol Vol Vol Vol"`
- `parallel_math` completion: `"gallgallgallgallgallgall cord cord"`
- Observed max read batch: `2`
- Observed max write batch: `14`
- GPU read kernel calls: `14`
- Stream-aware read kernel calls: `14`
- Native KV write calls skipped: `18`
- KCMM write verified rows: `44`
- Stream-aware KV write calls: `18`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.
- Read-seam diagnostic sample: `query_shape=[2, 2, 64]`,
  `query_stride=[384, 64, 1]`, and `query_is_contiguous=false`.

## Problem

The Phase II.C batch/concurrency gate passes for two concurrent requests with
`max_tokens=4`, but a longer concurrent run with the same prompts and
`max_tokens=8` produces a deterministic stock-vs-KCMM completion mismatch.

The failing run still proves the request scheduler exercised a multi-sequence
decode batch:

- Observed max read batch: `2`
- Observed max write batch: `14`
- GPU read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- Correctness failure case: `parallel_math`
- Stock completion: `"gallgallgallgallgallgall cord cord"`
- KCMM completion: `"gallgallgallgallgallgallgallgall"`
- Failed report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782006695770.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782006695770`

## Reproduction

```bash
python -m scripts.kcmm.vllm_gpu_read_batch_gate \
  --no-build-kcmm \
  --no-print-seams \
  --coverage-case 'parallel_alpha:8:alpha beta gamma delta epsilon zeta eta theta' \
  --coverage-case 'parallel_math:8:Question: 2 + 2 ='
```

## Acceptance criteria

- [x] Reproduce the mismatch with the command above.
- [x] Determine whether the divergence is caused by a kernel indexing bug,
  batch-order assumption, block table/sequence length handling, write ordering,
  or expected numeric sensitivity from the narrow scalar kernel.
- [x] Add focused diagnostics that record enough read-seam state to explain the
  failing token boundary without relying only on completion text.
- [x] Add a lower-level kernel or vLLM gate regression test for the failing
  multi-sequence decode scenario.
- [x] Either fix the GPU read kernel so the 8-token concurrent gate passes, or
  document a defensible non-bit-exact boundary and make the gate reflect that
  boundary explicitly.

## Boundaries

- Do not broaden to non-64 head dimensions in this issue.
- Do not include tensor parallelism, prefix cache, alibi, block-sparse mode, or
  FP8 cache scale support in this issue.
- Keep the existing short batch/concurrency gate passing throughout the work.
