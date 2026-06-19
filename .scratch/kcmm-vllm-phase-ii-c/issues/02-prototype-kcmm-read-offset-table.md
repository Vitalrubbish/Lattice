# Prototype KCMM read offset-table plan

Status: done
Type: AFK

## What to build

Add a Phase II.C A2 prototype that materializes a KCMM read-side offset table at
vLLM's `paged_attention_v1` and `paged_attention_v2` seams.

This does not replace the vLLM attention kernel yet. The purpose is to prove
that, under the KCMM-backed allocator, the Python read seam can construct a
side table indexed by the native `block_tables` block id:

```text
offset_table[block_id] = kcmm_f16_va_offset
```

The prototype should preserve vLLM's native `block_tables` semantics and record
that the native vLLM paged-attention kernel is still the read path.

## Acceptance criteria

- [x] Python bindings expose `kcmm_get_all_block_offsets_f16`.
- [x] Launcher exposes an opt-in KV read offset-table planning mode.
- [x] vLLM smoke exposes an opt-in `--kv-read-offset-table` gate.
- [x] The mode requires runtime-derived KCMM pool sizing and the
  KCMM-backed allocator.
- [x] The read seam builds a CUDA `torch.int64` offset table indexed by
  `block_id`.
- [x] The report validates that sampled `block_tables` block ids exist in KCMM.
- [x] The report explicitly records `kernel_replaced=false`.
- [x] Smoke validation fails if no read calls or no offset tables are observed.
- [x] Documentation records the A2 prototype boundary and the latest result.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/01-instrument-vllm-paged-attention-read-contract.md`

## Implementation

- Added `KcmmPool.all_block_offsets_f16()` to the Python KCMM bindings.
- Added `KcmmKvReadOffsetTableTracker` to build and validate the A2 side table.
- Added `apply_kv_read_offset_table` to patch
  `vllm._custom_ops.paged_attention_v1` and
  `vllm._custom_ops.paged_attention_v2`.
- Added launcher flags `--kcmm-kv-read-offset-table` and
  `--kcmm-kv-read-offset-table-report-path`.
- Added smoke flags `--kv-read-offset-table` and
  `--kv-read-offset-table-report-path`.
- Kept the native vLLM paged-attention kernel as the read path.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-mirror --instrument-kv-reads --kv-read-offset-table`

Latest local smoke result on 2026-06-19:

- Result: `passed=true`
- Read seam: `vllm._custom_ops.paged_attention_v1`
- Read calls observed: `6`
- Offset table builds: `6`
- Offset table dtype: `torch.int64`
- Offset table device: `cuda:0`
- Last offset table shape: `[1]`
- Max block id seen: `0`
- Offset f16 sample: `{ "0": 1046528 }`
- Kernel replaced: `false`
- Read path: `native_vllm_paged_attention`
- KCMM-backed allocator allocations/frees: `1/1`
- KCMM KV write mirror verified rows: `10`
- Final KCMM pool stats: `blocks_in_use=0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Next step

Replace the native paged-attention read path with a custom attention backend
that consumes the KCMM offset table and KCMM K/V bases instead of vLLM's native
KV cache tensor storage.
