# Mirror vLLM allocations into KCMM shadow allocator

Status: done
Type: AFK

## What to build

Add a shadow allocator mode where vLLM remains behaviorally unchanged, but every
native block allocation and free is mirrored into KCMM. This validates lifetime
alignment between vLLM block IDs and KCMM block handles before KCMM is allowed
to influence allocation results.

The completed slice should maintain an internal mapping between vLLM logical
blocks and KCMM blocks, update it on allocation/free, surface mismatches as hard
errors, and expose summary metrics after the smoke request finishes.

## Acceptance criteria

- [x] Shadow mode mirrors each vLLM allocation into `kcmm_alloc_blocks` and each free into `kcmm_free_blocks`.
- [x] vLLM's returned block IDs and KV storage behavior are unchanged in shadow mode.
- [x] The shadow mapping detects double-free, missing-free, and allocation-count mismatch cases.
- [x] A completion request succeeds through the V2 block manager while shadow mode is enabled.
- [x] The final report includes native allocation counts, KCMM allocation counts, outstanding shadow mappings, and KCMM pool stats.
- [x] Any KCMM allocation failure aborts startup or request handling with a clear error instead of falling back silently.
- [x] The mode can be disabled cleanly for stock and observer-only A/B runs.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_smoke --shadow-allocations`
- `python -m scripts.kcmm --kcmm-observer-only --kcmm-lib-path target/debug/libbaseline_llm_os.so --kcmm-print-seams`
- `python -m scripts.kcmm --kcmm-shadow-allocations --kcmm-lib-path target/debug/libbaseline_llm_os.so --kcmm-observer-only` exits with configuration error
- `python -m scripts.kcmm.vllm_smoke --mode stock`
- `python -m scripts.kcmm.vllm_smoke`

The local shadow smoke report recorded `native_gpu_allocations=1`,
`native_gpu_frees=1`, `kcmm_allocations=1`, `kcmm_frees=1`,
`outstanding_mappings=0`, `error_count=0`, and KCMM pool `blocks_in_use=0` after
the request and shutdown path completed.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/04-size-kcmm-pool-from-vllm-runtime-config.md`
