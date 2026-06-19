# Decide Phase II.A target and storage of record

Status: done
Type: HITL

## What to build

Make the Phase II.A architecture decision explicit before replacing vLLM's
allocator: define the exact vLLM target for this branch and decide what is the
storage of record for KV data during allocator replacement.

Earlier documentation named vLLM 0.6.3 as the target, while the local working
CUDA 11.8 environment uses vLLM 0.6.1.post1. Phase II.A should not proceed on an
implicit version assumption. The decision must also say whether Phase II.A keeps
vLLM native KV tensors as storage while KCMM shadows logical allocation, or
whether KCMM VA becomes the storage of record in this phase.

## Decision

Accepted 2026-06-19.

Phase II.A targets the locally verified CUDA 11.8 stack:

- vLLM `0.6.1.post1+cu118`
- PyTorch `2.4.0+cu118`
- xFormers `0.0.27.post2+cu118`
- transformers `4.45.2`
- tokenizers `0.20.3`
- huggingface-hub `0.36.2`

Required runtime flags:

- `--use-v2-block-manager`
- `--enforce-eager`
- `--disable-frontend-multiprocessing` for allocator instrumentation or
  replacement modes that must patch engine objects in-process

Phase II.A keeps native vLLM KV tensors as the storage of record. KCMM may size a
pool from vLLM runtime cache configuration, mirror allocation/free lifetimes, and
try allocator-backed ownership only behind an explicit opt-in flag. KCMM VA does
not become canonical KV storage until Phase II.B/II.C replace the write and read
paths.

Allocator-only replacement must fail closed if it requires KCMM VA to become the
true KV storage or violates a vLLM invariant such as contiguous native KV tensor
layout, block-id-to-offset arithmetic inside compiled kernels, or write/read
paths that cannot address KCMM-managed memory without Phase II.B/II.C.

Before Phase II.B starts, Phase II.A must pass stock, observer,
allocator-instrumented, runtime-sized pool, shadow allocator, and A/B smoke
checks. The KCMM-backed allocator mode must either pass completion with no leaked
KCMM blocks or produce a documented stop condition explaining why Phase II.B/II.C
is required first.

## Acceptance criteria

- [x] The exact vLLM version and required runtime flags for Phase II.A are recorded in the ADR or environment documentation.
- [x] The storage-of-record model is recorded: native vLLM KV tensors, KCMM VA, or an explicitly staged transition between them.
- [x] Stop criteria are documented for the case where allocator replacement cannot be made correct without also replacing write/read paths.
- [x] The decision names which Phase II.A tests are required before moving to Phase II.B.
- [x] The decision accounts for the observed local constraint that CUDA 11.8 wheels are required on the current driver.
- [x] The decision accounts for transformer/tokenizer dependency pins required by the selected vLLM version.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/02-instrument-vllm-v2-allocator-seams.md`
