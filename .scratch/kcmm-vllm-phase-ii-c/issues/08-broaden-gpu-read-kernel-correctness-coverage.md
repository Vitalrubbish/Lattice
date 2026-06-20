# Broaden GPU read-kernel correctness coverage

Status: done
Type: AFK

## What to build

Extend the Phase II.C GPU read-kernel A/B gate beyond a single `Hello` prompt.
The gate should run multiple deterministic completion cases against the same
stock and KCMM server processes, then compare every case independently.

## Acceptance criteria

- [x] `vllm_smoke` supports multiple completion cases while preserving the
  existing single-completion result shape for compatibility.
- [x] The GPU read-kernel A/B gate defaults to multiple coverage cases.
- [x] The gate supports repeated `--coverage-case NAME:MAX_TOKENS:PROMPT`
  overrides for targeted local debugging.
- [x] The gate compares per-case completion text, finish reason, completion
  token count, and total token count.
- [x] The JSON report records `coverage_cases` and per-mode
  `completion_cases`.
- [x] Performance metrics use aggregate request latency and generated tokens
  across all completion cases.
- [x] The latest local gate run passes with no correctness failures.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/07-add-gpu-read-kernel-performance-characterization.md`

## Implementation

- Added `CompletionCase` to `scripts/kcmm/vllm_smoke.py`.
- Extended `SmokeConfig` with optional `completion_cases`.
- Updated `run_smoke` to run every configured case after one server startup and
  preserve the first completion under the existing `completion` field.
- Added default GPU read A/B coverage cases:
  - `hello`: `Hello`, `max_tokens=4`
  - `math`: `Question: 2 + 2 =`, `max_tokens=3`
  - `long_context`: a longer synthetic prompt, `max_tokens=4`
- Added `--coverage-case NAME:MAX_TOKENS:PROMPT` to
  `scripts/kcmm/vllm_gpu_read_ab_gate.py`.
- Added per-case correctness comparison and aggregate performance accounting.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `git diff --check`
- `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`

Latest local multi-case result on 2026-06-20:

- Result: `passed=true`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Coverage cases: `hello`, `math`, `long_context`
- `hello` completion: `" pioneer pioneer pioneer pioneer"`
- `math` completion: `"gallgallgall"`
- `long_context` completion: `" radar radar radar radar"`
- Aggregate completion tokens: `11`
- Aggregate total tokens: `53`
- Read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `16`
- Stream-aware kernel calls: `16`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `22`
- KCMM write verified rows: `36`
- Startup seconds: stock `13.545`, KCMM `10.526`, ratio `0.777`
- Request latency seconds: stock `1.752`, KCMM `1.958`, ratio `1.118`
- Tokens per second: stock `6.279`, KCMM `5.618`, ratio `0.895`
- Peak GPU memory delta MiB: stock `3417`, KCMM `3425`, ratio `1.002`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.
- The temporary tiny OPT model directory was removed after the run.

## Boundaries

- Still tiny local OPT only.
- Still batch size 1 and one request at a time.
- Still `head_dim=64`, `max_model_len=64`, FP16 decode attention.
- Does not cover larger head dimensions, tensor parallelism, concurrency,
  batching, prefix cache, alibi, block-sparse mode, or FP8 cache scales.

## Next step

Continue with
`.scratch/kcmm-vllm-phase-ii-c/issues/09-make-kv-write-replacement-stream-aware.md`.
