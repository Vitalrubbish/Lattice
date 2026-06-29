# Add real-model GPU read gate

Status: done
Type: AFK

## What to build

Add the first Phase II.C GPU read-kernel gate that runs against a real
Hugging Face/vLLM model instead of a generated tiny OPT model.

## Acceptance criteria

- [x] Allow smoke/gate runs to use an externally supplied model directory
  without auto-generating the tiny OPT model over it.
- [x] Make vLLM `--gpu-memory-utilization` configurable from the smoke and A/B
  gate harnesses.
- [x] Add a real-model gate wrapper with optional Hugging Face download.
- [x] Default the first real-model gate to `facebook/opt-125m` with short
  deterministic coverage cases.
- [x] Verify stock-vs-KCMM completion text, finish reason, and token counts
  match.
- [x] Verify the KCMM path uses the GPU read kernel and does not fall back to
  CPU-staged reference reads.
- [x] Keep existing tiny-model gates using tiny model generation by default.

## Boundaries

- This is the first real-model coverage slice, not broad model compatibility.
- This does not add alibi, block-sparse mode, FP8 cache scales, or model
  architectures outside the current supported kernel envelope.
- This does not claim production performance.

## Verification

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_gpu_read_real_model_gate --download-model --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-1782719715998.json`.
- Downloaded model:
  `.scratch/kcmm-vllm/real-models/facebook--opt-125m`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- Coverage cases: `hello`, `math`.
- Stock/KCMM `hello` completion: `", I"`.
- Stock/KCMM `math` completion: `" -2"`.
- Aggregate completion tokens: `4`.
- Aggregate total tokens: `13`.
- Read path: `kcmm_paged_attn_decode_f16`.
- Replacement backend: `gpu_kernel`.
- GPU read kernel calls: `24`.
- Stream-aware read kernel calls: `24`.
- Reference KCMM read bytes: `0`.
- Native KV write calls skipped: `48`.
- KCMM write verified rows: `96`.
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the gate.
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`.
- Request latency seconds: stock `1.729`, KCMM `2.233`, ratio `1.291`.
- Tokens per second: stock `2.313`, KCMM `1.791`, ratio `0.774`.
- Tiny-model default-generation regression:
  `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm --no-print-seams --coverage-case tiny_smoke:2:Hello --timeout-seconds 240 --shutdown-timeout-seconds 45`
  passed with report `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782719978073.json`.
- Tiny regression GPU read kernel calls: `2`; reference KCMM read bytes: `0`.

## Notes

- This issue also added `--no-generate-tiny-model` plumbing so real model
  directories are never overwritten by the tiny model generator.
- A one-token tiny smoke is too short for this gate because it may not trigger
  the paged-attention read seam; use at least `max_tokens=2` for GPU read
  replacement smoke checks.
- The next real-model coverage issue should add a second architecture or a
  longer-context real-model case, not just repeat OPT-125m.
