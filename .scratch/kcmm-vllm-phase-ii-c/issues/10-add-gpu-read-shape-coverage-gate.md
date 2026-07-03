# Add GPU read shape coverage gate

Status: done
Type: AFK

## What to build

Add a Phase II.C GPU read-kernel shape coverage gate that runs the existing
stock-vs-KCMM GPU read A/B check across multiple tiny OPT model shapes. The
gate should keep the current deterministic completion comparison while
separating per-variant reports from the aggregate shape gate result.

## Acceptance criteria

- [x] A shape gate can run multiple named tiny OPT variants.
- [x] Each variant generates a local model with explicit hidden size, head
  count, layer count, FFN dimension, max positions, and seed.
- [x] Unsupported variants fail at CLI parse time before launching vLLM.
- [x] Every variant runs the same stock-vs-KCMM GPU read A/B coverage cases.
- [x] The aggregate report records variant order, per-variant report paths,
  failed variants, correctness failures, and performance warnings.
- [x] Generated model directories are cleaned after the gate unless
  `--keep-model` is passed.
- [x] The gate passes locally on the CUDA 11.8 vLLM environment.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-c/issues/09-make-kv-write-replacement-stream-aware.md`

## Implementation

- Added `scripts/kcmm/vllm_gpu_read_shape_gate.py`.
- Reused `scripts.kcmm.vllm_gpu_read_ab_gate.run_gate` so the shape gate uses
  the same stock-vs-KCMM correctness and performance checks.
- Added default variants:
  - `head64_layers2`: hidden size `128`, heads `2`, layers `2`, FFN dimension
    `256`.
  - `head64_heads4_layers3`: hidden size `256`, heads `4`, layers `3`, FFN
    dimension `512`.
- Added repeated `--variant NAME:HIDDEN_SIZE:NUM_HEADS:NUM_LAYERS:FFN_DIM`
  and `--coverage-case NAME:MAX_TOKENS:PROMPT` overrides.
- Restricted accepted variants to `head_dim=64`.

The head-dimension restriction is intentional for this local stack: current
vLLM/XFormers accepts paged-attention head sizes starting at `64`, while the
current KCMM GPU read kernel is limited to `head_dim <= 64`. The overlap is
therefore exactly `head_dim=64` until either the backend or KCMM kernel is
broadened.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `git diff --check`
- `python -m scripts.kcmm.vllm_gpu_read_shape_gate --no-build-kcmm --no-print-seams`

Latest local shape gate result on 2026-06-20:

- Result: `passed=true`
- Failed variants: `[]`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Aggregate report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-shape-gate-1781964629065.json`
- Per-variant reports:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-shape-gate-1781964629065-reports/`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.
- The temporary shape model directories were removed after the gate.

`head64_layers2` result:

- Completion cases: `hello`, `math`, `long_context`
- `hello` completion: `" pioneer pioneer pioneer pioneer"`
- `math` completion: `"gallgallgall"`
- `long_context` completion: `" radar radar radar radar"`
- GPU read kernel calls: `16`
- Stream-aware read kernel calls: `16`
- Native KV write calls skipped: `22`
- KCMM write verified rows: `36`
- Stream-aware KV write calls: `22`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- Startup seconds: stock `13.537`, KCMM `10.532`, ratio `0.778`
- Request latency seconds: stock `1.784`, KCMM `1.951`, ratio `1.094`
- Tokens per second: stock `6.166`, KCMM `5.638`, ratio `0.914`
- Peak GPU memory delta MiB: stock `3417`, KCMM `3425`, ratio `1.002`

`head64_heads4_layers3` result:

- Completion cases: `hello`, `math`, `long_context`
- `hello` completion: `" playoff playoff playoff playoff"`
- `math` completion: `" MORE MORE MORE"`
- `long_context` completion: `" belts belts belts belts"`
- GPU read kernel calls: `24`
- Stream-aware read kernel calls: `24`
- Native KV write calls skipped: `33`
- KCMM write verified rows: `54`
- Stream-aware KV write calls: `33`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- Startup seconds: stock `11.546`, KCMM `8.525`, ratio `0.738`
- Request latency seconds: stock `1.787`, KCMM `1.808`, ratio `1.012`
- Tokens per second: stock `6.156`, KCMM `6.084`, ratio `0.988`
- Peak GPU memory delta MiB: stock `3443`, KCMM `3455`, ratio `1.003`

The first attempted default variant with `head_dim=32` failed in stock vLLM
before reaching KCMM:

```text
ValueError: Head size 32 is not supported by PagedAttention. Supported head
sizes are: [64, 80, 96, 112, 120, 128, 192, 256].
```

The shape gate now rejects that class of variant during argument parsing.

## Boundaries

- This does not cover non-64 attention head dimensions on the current local
  CUDA 11.8 stack.
- This does not cover batching, concurrency, tensor parallelism, prefix cache,
  alibi, block-sparse mode, or FP8 cache scale coverage.
- This does not optimize the narrow Phase II.C GPU read kernel.

## Next step

Broaden Phase II.C beyond the current shape and batch/concurrency gates toward
non-default-stream, tensor-parallel, and non-64 head-dimension coverage.
