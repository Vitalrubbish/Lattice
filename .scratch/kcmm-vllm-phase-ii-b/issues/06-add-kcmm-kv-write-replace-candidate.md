# Add KCMM KV write replacement candidate

Status: done
Type: AFK

## What to build

Add an opt-in Phase II.B replacement-candidate mode for vLLM
`reshape_and_cache`. The mode should skip native vLLM KV writes and write the
same K/V rows only into KCMM using `kcmm_append_kv_slots`.

This is not an accepted end-to-end correctness mode because Phase II.C has not
replaced vLLM attention reads. The goal is to validate that the write seam can
be driven entirely by KCMM under a controlled flag.

## Scope constraints

- Require `--backed-allocations` so vLLM physical block ids match KCMM block ids.
- Do not combine replacement-candidate mode with mirror mode.
- Keep `--instrument-kv-writes` usable by installing instrumentation outside the
  replacement wrapper.
- Fail if a replacement write happens before the runtime KCMM pool is attached.

## Acceptance criteria

- [x] Launcher exposes an opt-in KCMM KV write replacement-candidate flag.
- [x] vLLM smoke exposes an opt-in `--kv-write-replace-candidate` flag.
- [x] Replacement-candidate mode requires KCMM-backed allocation mode.
- [x] Replacement-candidate mode is mutually exclusive with mirror mode.
- [x] The wrapper skips native `reshape_and_cache` and returns the custom-op
  compatible value.
- [x] The wrapper calls `kcmm_append_kv_slots` for post-attach write calls.
- [x] The report records native skipped calls, KCMM write calls, verified rows,
  and the storage-of-record limitation.
- [x] Documentation records that this is a Phase II.B write-path candidate only,
  not an end-to-end correctness mode before Phase II.C.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-b/issues/05-shadow-mirror-vllm-kv-writes-into-kcmm.md`

## Implementation

- Extended `KcmmKvWriteMirrorTracker` with `replace_native=True`.
- Added launcher flag `--kcmm-kv-write-replace-candidate`.
- Added smoke flag `--kv-write-replace-candidate`.
- Reordered launcher patch installation so KV instrumentation wraps the
  mirror/replacement wrapper and still observes calls when native writes are
  skipped.
- Replacement-candidate mode skips native `reshape_and_cache`, writes through
  `kcmm_append_kv_slots`, verifies KCMM bytes by D2H read-back, and returns
  `None` to match the custom-op return contract.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `git diff --check`
- `python -m scripts.kcmm.vllm_smoke --kv-write-replace-candidate --no-build-kcmm`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-mirror --kv-write-replace-candidate --no-build-kcmm`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-replace-candidate`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-mirror`

Replacement-candidate smoke result:

- `vllm._custom_ops.reshape_and_cache` calls observed: `8`
- Native passthrough calls: `0`
- Native skipped calls: `8`
- KCMM write calls: `8`
- Mirrored/written rows: `10`
- D2H verified rows: `10`
- Verification bytes: `5120`
- Cache layers mapped: `2`
- KCMM-backed allocator: `kcmm_allocations=1`, `kcmm_frees=1`,
  `outstanding_mappings=0`, `error_count=0`
- Final pool stats: `blocks_in_use=0`

Mirror regression result:

- Native passthrough calls: `8`
- Native skipped calls: `0`
- KCMM mirror calls: `8`
- D2H verified rows: `10`
