# Add GPU read tensor-parallel gate

Status: done
Type: AFK

## What to build

Add Phase II.C coverage for the KCMM GPU read-kernel path under vLLM tensor
parallelism. The existing stock-vs-KCMM gates run with
`tensor_parallel_size=1`; this issue adds a dedicated gate that runs the same
GPU read replacement path with `tensor_parallel_size=2` on the local dual RTX
3080 machine.

## Acceptance criteria

- [x] Add a smoke/A-B gate option that passes `--tensor-parallel-size` through
  to vLLM.
- [x] Add a dedicated tensor-parallel GPU read gate with default
  `tensor_parallel_size=2`.
- [x] The gate compares deterministic stock-vs-KCMM completion text, finish
  reason, completion tokens, and total tokens.
- [x] The gate verifies the KCMM GPU read path launches the stream-aware GPU
  kernel and does not fall back to CPU-staged reference reads.
- [x] Run the tensor-parallel gate locally or document the blocking behavior
  precisely.
- [x] Keep existing single-process gates at default `tensor_parallel_size=1`.

## Boundaries

- This is coverage, not a performance optimization.
- This does not add multi-node tensor parallelism.
- This does not broaden prefix cache, alibi, block-sparse mode, FP8 cache
  scale, or `head_dim > 128` support.

## Implementation notes

- `scripts.kcmm.vllm_smoke` and the GPU read A/B gate now pass
  `--tensor-parallel-size` through to vLLM and include it in JSON reports.
- `scripts.kcmm.vllm_gpu_read_tensor_parallel_gate` wraps the GPU read A/B gate
  with default `tensor_parallel_size=2`.
- vLLM TP worker subprocesses inherit the KCMM monkey patches but do not run the
  driver `LLMEngine.__init__` runtime-pool callback. A worker-level
  `Worker.initialize_cache` hook now creates and attaches a local KCMM pool per
  worker before model execution.
- TP workers receive slot mappings for blocks allocated by the driver-side
  scheduler. The KV write replacement now lazily ensures local pool block IDs
  from `slot_mapping` before appending KV rows.

## Verification

- `python -m scripts.kcmm.vllm_gpu_read_tensor_parallel_gate --no-build-kcmm --no-print-seams --timeout-seconds 240 --shutdown-timeout-seconds 45`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-tensor-parallel-1782634782121.json`.
- Tensor-parallel requirements: `tensor_parallel_size=2`, GPU read kernel calls
  `16`, stream-aware kernel calls `16`, and reference read bytes `0`.
- Stock and KCMM completions matched for `hello`, `math`, and `long_context`.
- A single-GPU regression run also passed:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782635020234.json`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the gates.
