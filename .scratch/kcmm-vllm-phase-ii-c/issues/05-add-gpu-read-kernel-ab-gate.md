# Add GPU read-kernel A/B gate

Status: done
Type: AFK

## What to build

Add a deterministic Phase II.C stock-vs-KCMM A/B gate for the GPU read-kernel
path. The gate should run stock vLLM and the KCMM GPU read-kernel candidate on
the same tiny local model, then fail if the generated completion differs.

This gate validates correctness for the current tiny-model decode shape. It is
not a performance gate and does not yet broaden coverage across prompts, batch
sizes, longer contexts, or different head dimensions.

## Acceptance criteria

- [x] A command exists for the GPU read-kernel A/B gate.
- [x] The gate reuses the same model directory for stock and KCMM runs.
- [x] Newly generated tiny models use a fixed default seed.
- [x] The stock mode runs without KCMM behavior changes.
- [x] The KCMM mode enables the backed allocator, KV write replacement, KV read
  instrumentation, and GPU read-kernel candidate.
- [x] The gate compares completion text, finish reason, completion tokens, and
  total tokens.
- [x] The gate records the GPU read-kernel contract summary.
- [x] The existing Phase II.A A/B gate still constructs `SmokeConfig`
  correctly after the KV read fields were added.
- [x] Documentation records the command, latest local result, and remaining
  boundaries.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/04-add-kcmm-gpu-read-kernel-candidate.md`

## Implementation

- Added `scripts/kcmm/vllm_gpu_read_ab_gate.py`.
- Added `--seed` to `scripts/kcmm/create_tiny_opt_model.py`, defaulting to `0`.
- Updated `scripts/kcmm/vllm_ab_gate.py` to pass the newer KV read fields when
  constructing `SmokeConfig`, preserving the Phase II.A gate.
- The new gate runs two modes:
  - `stock`
  - `kcmm_gpu_read`
- `kcmm_gpu_read` enables:
  - `--kcmm-pool-mode runtime`
  - KCMM-backed allocation
  - KV write replacement
  - KV read instrumentation
  - GPU read-kernel replacement

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- Constructor check for `scripts.kcmm.vllm_ab_gate`
- Constructor check for `scripts.kcmm.vllm_gpu_read_ab_gate`
- `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`

Latest local A/B result on 2026-06-20:

- Result: `passed=true`
- Model existed before gate: `false`
- Generated tiny model seed: `0`
- Prompt: `"Hello"`
- Max tokens: `4`
- Stock completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM completion text: `" pioneer pioneer pioneer pioneer"`
- Finish reason: `length`
- Completion tokens: `4`
- Total tokens: `6`
- KCMM read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `6`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `8`
- KCMM write verified rows: `10`
- Final KCMM pool stats: `blocks_in_use=0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.

## Next step

Make `kcmm_paged_attn_decode_f16` stream-aware instead of synchronizing the
whole KCMM CUDA context before returning, then add basic performance
characterization for the GPU read-kernel path.
