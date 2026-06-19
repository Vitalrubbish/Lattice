# Shadow mirror vLLM KV writes into KCMM

Status: done
Type: AFK

## What to build

Add a Phase II.B shadow mirror mode for the vLLM KV write seam. The mode should
leave native vLLM `reshape_and_cache` writes unchanged, then mirror the same
K/V rows into KCMM with `kcmm_append_kv_slots`.

This slice intentionally does not replace vLLM KV storage or attention reads.
It proves that the real vLLM write seam can feed KCMM physical-slot writes
without sequence metadata reconstruction.

## Scope constraints

- Require `--backed-allocations` for this first mirror slice so vLLM physical
  block ids and KCMM block ids are the same ids.
- Skip and report KV write calls that happen before the runtime KCMM pool is
  attached.
- Fail the smoke if any post-attach mirror write or D2H verification fails.

## Acceptance criteria

- [x] Launcher exposes an opt-in KV write mirror flag and report path.
- [x] vLLM smoke exposes an opt-in `--kv-write-mirror` mode.
- [x] KV write mirror mode requires KCMM-backed allocation mode.
- [x] The mirror wrapper calls native `reshape_and_cache` first and leaves its
  return behavior unchanged.
- [x] Post-attach write calls mirror K/V rows through `kcmm_append_kv_slots`.
- [x] The mirror report records calls, rows, padding slots, cache-layer mapping,
  and pool stats.
- [x] The smoke validates at least one D2H byte-level KCMM mirror comparison.
- [x] Documentation records the mirror gate and its current storage-of-record
  limit.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-b/issues/04-add-kcmm-direct-slot-kv-write-ffi.md`

## Implementation

- Added `scripts.kcmm.kv_write_mirror.KcmmKvWriteMirrorTracker`.
- Added launcher flags `--kcmm-kv-write-mirror` and
  `--kcmm-kv-write-mirror-report-path`.
- Added smoke flag `--kv-write-mirror` and report validation.
- Added `apply_kv_write_mirror` to wrap vLLM KV custom ops after native writes.
- Required mirror mode to run with `--backed-allocations` so vLLM slot block ids
  match KCMM block ids.
- Added D2H byte verification for mirrored KCMM K/V rows.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `git diff --check`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-mirror`

Smoke result:

- `vllm._custom_ops.reshape_and_cache` calls observed: `8`
- KV mirror calls: `8`
- Mirrored rows: `10`
- D2H verified rows: `10`
- Verification bytes: `5120`
- Cache layers mapped: `2`
- KCMM-backed allocator: `kcmm_allocations=1`, `kcmm_frees=1`,
  `outstanding_mappings=0`, `error_count=0`
- Final pool stats: `blocks_in_use=0`
