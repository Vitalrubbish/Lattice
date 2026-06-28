# Add vLLM non-default stream gate

Status: done
Type: AFK

## What to build

Add a vLLM-integrated Phase II.C gate that validates KCMM KV write replacement
and GPU read replacement can run on non-default CUDA streams while preserving
the stream graph ordering around vLLM/PyTorch producers and consumers.

Issue 13 covered the low-level `_on_stream` FFI ABI. This issue covers the
integrated Python monkey-patch path.

## Acceptance criteria

- [x] Add an opt-in launcher/smoke flag that forces KCMM KV replacement launches
  onto non-default CUDA streams.
- [x] Insert stream waits so the forced KCMM stream waits for the original
  PyTorch/vLLM current stream before launch.
- [x] Insert a reverse wait so the original stream waits for KCMM work before
  downstream vLLM consumers continue.
- [x] Record PyTorch tensor lifetimes with `record_stream` when raw KCMM
  pointers are used on the forced stream.
- [x] Record forced stream counts and stream pointers in write and read reports.
- [x] Add a dedicated vLLM gate that fails unless write and read reports contain
  non-zero, non-default stream pointers.
- [x] Keep the default stream batch/concurrency gate passing.

## Implementation

- Added `scripts/kcmm/streaming.py` with `KcmmStreamProvider`.
- Added launcher/config flag `--kcmm-kv-force-non-default-stream` and smoke flag
  `--kv-force-non-default-stream`.
- Updated `KcmmKvWriteMirrorTracker` to route `kcmm_append_kv_slots_on_stream`
  through a dedicated non-default stream when forced mode is enabled.
- Updated `KcmmKvReadOffsetTableTracker` to route
  `kcmm_paged_attn_decode_f16_on_stream` through a dedicated non-default stream
  when forced mode is enabled.
- Added `scripts/kcmm/vllm_gpu_read_non_default_stream_gate.py`.
- Extended the GPU read A/B contract summary with read/write stream pointers and
  forced non-default stream call counts.
- Made vLLM smoke shutdown wait briefly after port close so Python atexit and
  CUDA teardown do not look like leaked process groups.

## Verification

```bash
python -m py_compile scripts/kcmm/*.py
python -m scripts.kcmm.vllm_gpu_read_non_default_stream_gate \
  --no-build-kcmm \
  --no-print-seams
python -m scripts.kcmm.vllm_gpu_read_batch_gate \
  --no-build-kcmm \
  --no-print-seams
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
```

Latest local non-default-stream vLLM result on 2026-06-28:

- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-non-default-stream-1782619860722.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782619860722`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Stock/KCMM completion text matched for `hello`, `math`, and `long_context`.
- Read forced non-default stream calls: `16`
- Read stream pointer: `139999223911008`
- Read default stream pointer: `0`
- Write forced non-default stream calls: `22`
- Write stream pointer: `139999223908544`
- Write default stream pointer: `0`
- GPU read kernel calls: `16`
- Stream-aware read kernel calls: `16`
- Native KV write calls skipped: `22`
- KCMM write verified rows: `36`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the run.

Latest default-stream batch/concurrency regression result on 2026-06-28:

- Result: `passed=true`
- Report: `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782619996234.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Observed max read batch: `2`
- Observed max write batch: `14`
- Read/write forced non-default stream calls: `0`
- Read/write stream pointers remained `0`.
- GPU memory returned to `0 MiB` on both RTX 3080 GPUs after the run.

## Boundaries

- This gate forces the KCMM replacement path onto non-default streams; it does
  not prove that the current vLLM eager scheduler naturally chooses non-default
  streams.
- The forced write and read streams are separate dedicated KCMM streams, ordered
  through the original PyTorch stream with `wait_stream`.
- This does not broaden tensor parallelism, prefix cache, non-64 head
  dimensions, alibi, block-sparse mode, or FP8 cache scale coverage.
