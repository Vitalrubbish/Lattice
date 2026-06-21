# Fix long concurrent GPU read divergence

Status: ready-for-agent
Type: AFK

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

- [ ] Reproduce the mismatch with the command above.
- [ ] Determine whether the divergence is caused by a kernel indexing bug,
  batch-order assumption, block table/sequence length handling, write ordering,
  or expected numeric sensitivity from the narrow scalar kernel.
- [ ] Add focused diagnostics that record enough read-seam state to explain the
  failing token boundary without relying only on completion text.
- [ ] Add a lower-level kernel or vLLM gate regression test for the failing
  multi-sequence decode scenario.
- [ ] Either fix the GPU read kernel so the 8-token concurrent gate passes, or
  document a defensible non-bit-exact boundary and make the gate reflect that
  boundary explicitly.

## Boundaries

- Do not broaden to non-64 head dimensions in this issue.
- Do not include tensor parallelism, prefix cache, alibi, block-sparse mode, or
  FP8 cache scale support in this issue.
- Keep the existing short batch/concurrency gate passing throughout the work.
