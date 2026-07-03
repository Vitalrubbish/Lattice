# Add real-model matrix gate

Status: done
Type: AFK

## What to build

Broaden Phase II.C real-model coverage from a single OPT-125m gate to a small
real-model matrix that includes a second architecture family and a longer prompt
case.

## Acceptance criteria

- [x] Add a matrix wrapper that runs the existing stock-vs-KCMM GPU read A/B
  gate for multiple real model ids.
- [x] Default the matrix to `facebook/opt-125m` and `distilgpt2`.
- [x] Add a longer prompt coverage case that spans multiple KV blocks.
- [x] Download missing models into `.scratch/kcmm-vllm/real-models/` behind an
  explicit `--download-model` flag.
- [x] Avoid downloading unrelated CoreML/TF/Flax artifacts when fetching
  Hugging Face models.
- [x] Treat a model directory as usable only when it has both `config.json` and
  PyTorch/safetensors weights.
- [x] Verify every model has matching stock-vs-KCMM completion text, finish
  reason, and token counts.
- [x] Verify every KCMM model run uses the GPU read kernel and does not fall
  back to CPU-staged reference reads.

## Boundaries

- This is still local single-GPU real-model coverage.
- This does not add models outside the current FP16/head-dim/no-alibi envelope.
- This does not claim broad production model compatibility.

## Verification

- `python -m py_compile scripts/kcmm/*.py`
- `git diff --check`
- `python -m scripts.kcmm.vllm_gpu_read_real_model_matrix_gate --download-model --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
  passed with report
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-1782721112360.json`.
- Model order: `facebook/opt-125m`, `distilgpt2`.
- Failed models: `[]`.
- Correctness failures: `[]`.
- Performance warnings: `[]`.
- Coverage cases: `hello`, `math`, `long_context`.
- `facebook/opt-125m` report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-1782721112360-reports/real-facebook-opt-125m.json`.
- `facebook/opt-125m` completions matched:
  `hello=", I"`, `math=" -2"`, `long_context=" rho"`.
- `facebook/opt-125m` GPU read kernel calls: `36`.
- `facebook/opt-125m` stream-aware read kernel calls: `36`.
- `facebook/opt-125m` reference KCMM read bytes: `0`.
- `facebook/opt-125m` final `blocks_in_use=0`.
- `distilgpt2` report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-1782721112360-reports/real-distilgpt2.json`.
- `distilgpt2` completions matched:
  `hello=" The first"`, `math=" 1 +"`, `long_context=" pia"`.
- `distilgpt2` GPU read kernel calls: `18`.
- `distilgpt2` stream-aware read kernel calls: `18`.
- `distilgpt2` reference KCMM read bytes: `0`.
- `distilgpt2` final `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the gate.

## Notes

- The first attempted `distilgpt2` download used a broad `*.bin` allow pattern
  and started pulling CoreML artifacts. The download filter now names only the
  model/tokenizer files needed by vLLM.
- The matrix gate keeps downloaded Hugging Face models under `.scratch` for
  follow-up runs; these artifacts are not tracked by git.
- The next coverage issue should either add a third real model with different
  dimensions or combine real-model coverage with batch/concurrency.
