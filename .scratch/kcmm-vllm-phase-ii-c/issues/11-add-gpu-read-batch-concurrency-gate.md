# Add GPU read batch and concurrency gate

Status: done
Type: AFK

## What to build

Add a Phase II.C GPU read-kernel gate that runs stock vLLM and the KCMM
write-replacement plus GPU read-kernel path with multiple completion requests
in flight. The gate should prove that the harness can exercise a multi-sequence
decode batch and still compare deterministic stock-vs-KCMM outputs.

## Acceptance criteria

- [x] `vllm_smoke` can run multiple completion requests concurrently against
  one vLLM server process.
- [x] `vllm_smoke` exposes `max_model_len`, `max_num_seqs`, and
  `max_num_batched_tokens` instead of hard-coding the single-sequence settings.
- [x] KV read reports record the observed decode `batch` and aggregate
  `max_batch_seen`.
- [x] KV write reports record aggregate `max_batch_seen`.
- [x] The GPU read A/B report exposes `max_read_batch_seen` and
  `max_write_batch_seen` in the KCMM contract summary.
- [x] A dedicated batch/concurrency gate fails if the KCMM read seam does not
  observe at least the required decode batch size.
- [x] The default batch/concurrency gate passes locally on the CUDA 11.8 vLLM
  environment.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/10-add-gpu-read-shape-coverage-gate.md`

## Implementation

- Added `scripts/kcmm/vllm_gpu_read_batch_gate.py`.
- Extended `scripts/kcmm/vllm_smoke.py` with:
  - `--max-model-len`
  - `--max-num-seqs`
  - `--max-num-batched-tokens`
  - `--completion-concurrency`
- Extended `scripts/kcmm/vllm_gpu_read_ab_gate.py` so those settings flow into
  both stock and KCMM modes and are recorded in the JSON report.
- Added observed batch reporting to `scripts/kcmm/kv_read_plan.py` and
  `scripts/kcmm/kv_write_mirror.py`.
- Updated `scripts/kcmm/vllm_gpu_read_shape_gate.py` to pass the new A/B gate
  sizing fields explicitly.

The initial batch gate slice ran:

- `max_model_len=128`
- `max_num_seqs=2`
- `max_num_batched_tokens=128`
- `completion_concurrency=2`
- `require_min_read_batch=2`
- `parallel_alpha`: prompt `alpha beta gamma delta epsilon zeta eta theta`,
  `max_tokens=4`
- `parallel_math`: prompt `Question: 2 + 2 =`, `max_tokens=4`

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `git diff --check`
- `python -m scripts.kcmm.vllm_gpu_read_batch_gate --no-build-kcmm --no-print-seams`

Latest local batch/concurrency result on 2026-06-21:

- Result: `passed=true`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Aggregate report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782007014826.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782007014826`
- `parallel_alpha` completion: `" Vol Vol Vol Vol"`
- `parallel_math` completion: `"gallgallgallgall"`
- Completion concurrency: `2`
- Observed max read batch: `2`
- Observed max write batch: `14`
- GPU read kernel calls: `6`
- Stream-aware read kernel calls: `6`
- Native KV write calls skipped: `10`
- KCMM write verified rows: `28`
- Stream-aware KV write calls: `10`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.
- Startup seconds: stock `13.536`, KCMM `10.528`, ratio `0.778`
- Request latency seconds: stock `1.785`, KCMM `1.922`, ratio `1.077`
- Tokens per second: stock `4.482`, KCMM `4.162`, ratio `0.929`
- Peak GPU memory delta MiB: stock `3415`, KCMM `3423`, ratio `1.002`

An attempted longer concurrent decode with the same two prompts and
`max_tokens=8` observed `max_read_batch_seen=2` but failed exact completion
comparison:

- Failed report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782006695770.json`
- Failing case: `parallel_math`
- Stock completion: `"gallgallgallgallgallgall cord cord"`
- KCMM completion: `"gallgallgallgallgallgallgallgall"`

That longer-decode divergence was fixed by issue 12. The default gate now uses
the 8-token concurrent workload.

## Boundaries

- This issue established the initial short batch/concurrency regression gate.
- Issue 12 later promoted the default batch/concurrency gate to the longer
  8-token workload after fixing the strided-query divergence.
- This does not cover tensor parallelism, non-default streams, non-64 head
  dimensions, prefix cache, alibi, block-sparse mode, or FP8 cache scale
  coverage.

## Next step

Broaden Phase II.C beyond the current batch/concurrency gate toward
non-default-stream, tensor-parallel, and non-64 head-dimension coverage.
