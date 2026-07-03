# Enable KCMM-backed V2 allocator behind a flag

Status: done
Type: AFK

## What to build

Introduce the first actual Phase II.A allocator replacement mode behind an
explicit opt-in flag. In this mode, vLLM's V2 allocation/free decisions should
be delegated to KCMM with tiering disabled, while preserving the storage-of-record
decision made for this phase.

This is not yet the KV write/read replacement. The slice should either prove a
minimal completion path works under the chosen storage model or fail closed with
a precise report explaining which vLLM invariant prevents allocator-only
replacement from being correct.

## Acceptance criteria

- [x] A launcher flag selects KCMM-backed allocator mode; the default remains observer-only or stock behavior.
- [x] KCMM owns allocation/free decisions for the targeted V2 allocator seam when the flag is enabled.
- [x] The implementation respects the storage-of-record decision from issue 03.
- [x] A smoke completion either succeeds end-to-end or exits with a documented stop condition that points to the required Phase II.B/II.C work.
- [x] Allocation/free metrics show no leaked KCMM blocks after the smoke request completes and the server shuts down.
- [x] The mode is incompatible with unsupported vLLM versions or flags and fails before serving traffic.
- [x] Existing observer-only and shadow modes continue to pass.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_smoke --backed-allocations`
- `python -m scripts.kcmm.vllm_smoke --shadow-allocations`
- `python -m scripts.kcmm --kcmm-observer-only --kcmm-lib-path target/debug/libbaseline_llm_os.so --kcmm-print-seams`
- `python -m scripts.kcmm --kcmm-backed-allocations --kcmm-lib-path target/debug/libbaseline_llm_os.so --kcmm-observer-only` exits with configuration error
- `python -m scripts.kcmm.vllm_smoke --mode stock --backed-allocations` exits with configuration error
- `python -m scripts.kcmm.vllm_smoke --shadow-allocations --backed-allocations` exits with configuration error

The local KCMM-backed smoke report recorded `decision_source=kcmm_alloc_blocks`,
`storage_of_record=native_vllm_kv_tensors`, `native_gpu_allocations=1`,
`native_gpu_frees=1`, `kcmm_allocations=1`, `kcmm_frees=1`,
`outstanding_mappings=0`, `error_count=0`, `stop_condition=null`, and KCMM pool
`blocks_in_use=0` after the request and shutdown path completed.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/03-decide-phase-ii-a-target-and-storage-of-record.md`
- `.scratch/kcmm-vllm-phase-ii-a/issues/05-mirror-vllm-allocations-into-kcmm-shadow-allocator.md`
