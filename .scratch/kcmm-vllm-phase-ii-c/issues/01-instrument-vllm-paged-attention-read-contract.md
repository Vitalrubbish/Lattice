# Instrument vLLM paged-attention read contract

Status: done
Type: AFK

## What to build

Add an observer-only Phase II.C instrumentation mode for vLLM's paged-attention
read seam. The mode should patch `vllm._custom_ops.paged_attention_v1` and
`vllm._custom_ops.paged_attention_v2` without changing behavior, then record the
tensor contract for `query`, `key_cache`, `value_cache`, `block_tables`, and
`seq_lens`.

This is the first read-path slice after Phase II.B write replacement candidate
work. It should answer whether the current Python custom-op seam can support
A1 (`block_tables` as KCMM VA offsets) or whether Phase II.C needs A2/custom
attention.

## Acceptance criteria

- [x] Launcher exposes opt-in KV read instrumentation flags and trace path.
- [x] vLLM smoke exposes opt-in `--instrument-kv-reads`.
- [x] The trace records calls to `paged_attention_v1` or `paged_attention_v2`.
- [x] The trace records bounded `block_tables` and `seq_lens` samples.
- [x] The trace validates sampled `block_tables` entries as physical KV block
  ids within the observed KV cache block count.
- [x] The trace records an A1 assessment for the Python custom-op seam.
- [x] The smoke fails when required read seams are not observed.
- [x] Documentation records the observed read contract and Phase II.C decision.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-b/issues/06-add-kcmm-kv-write-replace-candidate.md`

## Implementation

- Added `apply_kv_read_instrumentation` for
  `vllm._custom_ops.paged_attention_v1` and
  `vllm._custom_ops.paged_attention_v2`.
- Added launcher flags `--kcmm-instrument-kv-reads`,
  `--kcmm-kv-read-trace-path`, and `--kcmm-require-kv-read-seams`.
- Added smoke flags `--instrument-kv-reads`, `--kv-read-trace-path`, and
  `--require-kv-read-seams`.
- Added `block_tables` contract validation and A1 assessment to the trace.
- Documented that A1 is not valid at the current vLLM Python custom-op seam.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `git diff --check`
- `python -m scripts.kcmm.vllm_smoke --instrument-kv-reads`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --instrument-kv-reads --kv-write-replace-candidate`

Read contract smoke result:

- Observed read seam: `vllm._custom_ops.paged_attention_v1`
- Read calls observed: `6`
- First `block_tables` dtype: `torch.int32`
- First `block_tables` shape: `[1, 1]`
- First `block_tables` sample: `[0]`
- First `seq_lens` sample: `[3]`
- First `key_cache` shape: `[134685, 2, 8, 16, 8]`
- First `value_cache` shape: `[134685, 2, 64, 16]`
- A1 assessment:
  `safe_to_replace_block_tables_with_va_offsets=false`

Combined write-candidate/read-trace smoke result:

- Write calls observed: `8`
- Read calls observed: `6`
- Native write passthrough calls: `0`
- Native write skipped calls: `8`
- KCMM write verified rows: `10`
- Final pool stats: `blocks_in_use=0`
