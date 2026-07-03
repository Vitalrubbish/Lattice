# Size KCMM pool from vLLM runtime config

Status: done
Type: AFK

## What to build

Replace the Phase I.C tiny fixed observer pool shape with a Phase II.A pool
configuration derived from vLLM's runtime model and cache configuration, while
still leaving vLLM behavior unchanged.

The completed slice should create a KCMM pool with tiering disabled and capacity
aligned with the vLLM GPU block budget for the served model. The observer report
should show both vLLM's relevant cache sizing inputs and KCMM's resulting pool
stats so reviewers can verify the two sides match before any allocator
replacement happens.

## Acceptance criteria

- [x] KCMM pool creation can use vLLM runtime values for block size, model layer count, KV heads, head dimension, max sequence length, and effective GPU block capacity.
- [x] The launcher retains the existing fixed-shape observer mode for quick CUDA probes.
- [x] The vLLM server smoke test can run with runtime-derived KCMM sizing and still complete one request.
- [x] The observer report includes enough vLLM and KCMM sizing fields to verify capacity alignment.
- [x] Tiering remains disabled for this path.
- [x] Invalid or unavailable vLLM sizing data fails closed with a clear error message.
- [x] Unit or smoke coverage verifies that the runtime-derived config path does not regress the existing observer-only probe.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `cargo build --features kcmm`
- `python -m scripts.kcmm --kcmm-observer-only --kcmm-lib-path target/debug/libbaseline_llm_os.so --kcmm-print-seams`
- `python -m scripts.kcmm.vllm_smoke --runtime-derived-pool`
- `python -m scripts.kcmm.vllm_smoke`
- `python -m scripts.kcmm --kcmm-pool-mode runtime --kcmm-observer-only --kcmm-lib-path target/debug/libbaseline_llm_os.so` exits with configuration error
- `python -m scripts.kcmm --kcmm-pool-mode runtime --kcmm-lib-path target/debug/libbaseline_llm_os.so serve .scratch/nonexistent-model` exits before serving because `--disable-frontend-multiprocessing` is missing

The runtime-derived smoke report recorded `pool_source=runtime`,
`max_blocks_match=true`, `tiering_disabled=true`, vLLM
`effective_num_gpu_blocks=134685`, and KCMM `max_blocks=134685` for the local
tiny OPT smoke model.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/03-decide-phase-ii-a-target-and-storage-of-record.md`
