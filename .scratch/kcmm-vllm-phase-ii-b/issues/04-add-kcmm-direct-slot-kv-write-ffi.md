# Add KCMM direct-slot KV write FFI

Status: done
Type: AFK

## What to build

Add a KCMM KV write C ABI that consumes vLLM-style physical slot ids directly:
`slot = block_id * block_size + offset_in_block`.

The existing `kcmm_append_kv_step` API consumes sequence indices and token
positions, but vLLM's `reshape_and_cache` seam only exposes physical
`slot_mapping` values. This slice should add a direct-slot write path that can
copy source K/V rows into KCMM-managed KV memory without requiring sequence
metadata reconstruction.

## Acceptance criteria

- [x] Rust FFI exports a direct-slot KV write function.
- [x] Python bindings expose the new function.
- [x] The direct-slot API interprets non-negative slots as
  `block_id = slot // block_size` and `offset_in_block = slot % block_size`.
- [x] The direct-slot API skips negative padding slots.
- [x] The direct-slot API fails when a non-padding slot maps to an unallocated
  KCMM block.
- [x] The existing KV write smoke verifies direct-slot K/V writes with D2H
  byte-level comparison.
- [x] Documentation records direct-slot writing as the next implementation
  path for replacing vLLM `reshape_and_cache`.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-b/issues/03-validate-vllm-slot-mapping-contract.md`

## Implementation

- Added Rust C ABI `kcmm_append_kv_slots(layer_idx, slot_mapping, batch, k_src, v_src)`.
- Added Python binding `KcmmPool.append_kv_slots(...)`.
- Extended `scripts.kcmm.kv_write_ffi_smoke` to verify sequence-position writes
  and physical-slot writes in the same tiny KCMM pool.
- Documented direct-slot writing as the selected `reshape_and_cache`
  replacement path.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `python -m scripts.kcmm.kv_write_ffi_smoke`
- `git diff --check`

Smoke result:

- `passed=true`
- Direct physical slots written: `2`, `7`
- Padding slot skipped: `-1`
- Invalid slot rejected: `16`
- Final KCMM stats: `blocks_in_use=0`
