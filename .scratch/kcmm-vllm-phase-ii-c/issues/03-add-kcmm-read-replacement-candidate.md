# Add KCMM read replacement candidate

Status: done
Type: AFK

## What to build

Add an opt-in Phase II.C read replacement candidate for vLLM's paged-attention
read seam. This candidate should skip the native vLLM paged-attention kernel and
fill the provided `out` tensor from KCMM-managed K/V data.

This is a correctness-oriented reference path, not the final performance path.
It may use CPU staging and PyTorch tensor math as long as it proves that vLLM
can complete with both native KV writes and native paged-attention reads
disabled.

## Acceptance criteria

- [x] Launcher exposes `--kcmm-kv-read-replace-candidate`.
- [x] Smoke exposes `--kv-read-replace-candidate`.
- [x] The mode requires runtime-derived pool sizing, the KCMM-backed allocator,
  and KCMM KV writes.
- [x] The read seam skips native `paged_attention_v1/v2` when replacement is
  enabled.
- [x] The replacement consumes KCMM K/V memory via `block_tables`,
  `seq_lens`, KCMM K/V base addresses, and A2 f16 VA offsets.
- [x] The report records `kernel_replaced=true` and
  `read_path=kcmm_reference_attention`.
- [x] Smoke fails if no replacement calls or no KCMM read bytes are observed.
- [x] The combined write+read replacement smoke completes successfully.
- [x] A same-model stock-vs-KCMM A/B run produces identical completion text.
- [x] Documentation records that this is a reference path and not a performance
  kernel.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/02-prototype-kcmm-read-offset-table.md`

## Implementation

- Extended `KcmmKvReadOffsetTableTracker` with `replace_native=True`.
- Added a reference attention implementation that:
  - maps vLLM cache tensor pairs to KCMM layer indices,
  - reads KCMM K/V rows through CUDA D2H copies,
  - reconstructs the per-token K/V sequence from `block_tables` and `seq_lens`,
  - computes scaled dot-product attention with PyTorch,
  - writes the result into vLLM's `out` tensor,
  - returns without calling the native vLLM paged-attention kernel.
- Added launcher and smoke flags for the replacement candidate.
- Updated smoke report validation for replacement-specific fields.
- Renamed the write-replacement storage label to `kcmm_kv_storage_candidate`.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-replace-candidate --instrument-kv-reads --kv-read-replace-candidate`

Latest local replacement smoke result on 2026-06-19:

- Result: `passed=true`
- Native KV write calls skipped: `8`
- Native paged-attention calls replaced: `6`
- Read path: `kcmm_reference_attention`
- Kernel replaced: `true`
- Reference KCMM read bytes: `12288`
- Offset table builds: `6`
- Observed read seam: `vllm._custom_ops.paged_attention_v1`
- KCMM write verified rows: `10`
- Final KCMM pool stats: `blocks_in_use=0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

Same-model A/B result:

- Stock command:
  `python -m scripts.kcmm.vllm_smoke --mode stock --keep-model --no-build-kcmm`
- KCMM command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-replace-candidate --instrument-kv-reads --kv-read-replace-candidate --keep-model --no-build-kcmm`
- Stock completion text: `" 80 80 80 80"`
- KCMM replacement completion text: `" 80 80 80 80"`

## Next step

Move the reference read replacement to a GPU implementation. The next issue
should replace the CPU-staged PyTorch reference path with a CUDA kernel or
compiled extension that consumes KCMM K/V bases and the A2 offset table.
