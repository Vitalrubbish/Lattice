# KCMM vLLM CUDA 11.8 Conda Environment

This is the known-good environment for the Phase I.C vLLM observer smoke test
and the current Phase II.A allocator work on the local RTX 3080 host.

## Create the environment

```bash
conda create -n vllm-cu118 python=3.10 -y
conda activate vllm-cu118

pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.4.0+cu118 \
  torchvision==0.19.0+cu118

pip install \
  https://github.com/vllm-project/vllm/releases/download/v0.6.1.post1/vllm-0.6.1.post1%2Bcu118-cp310-cp310-manylinux1_x86_64.whl

pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  xformers==0.0.27.post2+cu118

# vLLM 0.6.1 has no upper bound on transformers, but transformers 5.x breaks
# its tokenizer API usage. Keep these pins until vLLM is upgraded.
pip install \
  transformers==4.45.2 \
  tokenizers==0.20.3 \
  huggingface-hub==0.36.2
```

Verify:

```bash
pip check
python - <<'PY'
import torch, vllm, transformers
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
print(vllm.__version__)
print(transformers.__version__)
PY
```

Expected key versions:

```text
torch==2.4.0+cu118
torchvision==0.19.0+cu118
vllm==0.6.1.post1+cu118
xformers==0.0.27.post2+cu118
transformers==4.45.2
tokenizers==0.20.3
huggingface-hub==0.36.2
```

## Phase II.A target

The current Phase II.A branch targets this exact local stack:

- vLLM `0.6.1.post1+cu118`
- PyTorch `2.4.0+cu118`
- xFormers `0.0.27.post2+cu118`
- transformers `4.45.2`
- tokenizers `0.20.3`
- huggingface-hub `0.36.2`

This replaces the earlier ADR target of vLLM `0.6.3.post1` for the current
branch because the local NVIDIA 515.48.07 host driver needs CUDA 11.8 wheels.
Revisit the vLLM target after a host-driver upgrade.

Required runtime flags for Phase II.A:

- `--use-v2-block-manager`
- `--enforce-eager`
- `--disable-frontend-multiprocessing` for allocator instrumentation or
  replacement modes that must patch engine objects in-process

Phase II.A keeps native vLLM KV tensors as the storage of record. KCMM may size a
pool from vLLM runtime cache configuration, mirror allocation/free events, and
try allocator-backed ownership behind an opt-in flag. KCMM VA does not become
canonical KV storage until the Phase II.B write path and Phase II.C read path are
implemented.

## KCMM observer smoke test

```bash
cargo build --features kcmm
python -m scripts.kcmm \
  --kcmm-observer-only \
  --kcmm-lib-path target/debug/libbaseline_llm_os.so \
  --kcmm-print-seams
```

## vLLM server smoke test

Run the automated self-terminating smoke test:

```bash
python -m scripts.kcmm.vllm_smoke
```

Use stock vLLM behavior through the same process harness:

```bash
python -m scripts.kcmm.vllm_smoke --mode stock
```

Run with observer-only V2 allocator seam instrumentation:

```bash
python -m scripts.kcmm.vllm_smoke --instrument-allocators
```

Instrumentation mode keeps the vLLM engine in the launcher process with
`--disable-frontend-multiprocessing` so the Python monkey-patches apply to the
actual block manager and allocator objects.

The manual steps below are the expanded form of the same check.

Generate a tiny local OPT model with a vLLM-supported attention head size. This
avoids downloading `facebook/opt-125m` during environment validation.

```bash
python scripts/kcmm/create_tiny_opt_model.py
```

Start vLLM through the KCMM observer launcher:

```bash
python -m scripts.kcmm \
  --kcmm-lib-path target/debug/libbaseline_llm_os.so \
  --kcmm-print-seams \
  serve .scratch/kcmm-vllm/tiny-opt-head64 \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype float16 \
  --max-model-len 64 \
  --gpu-memory-utilization 0.25 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 64 \
  --enforce-eager \
  --max-seq-len-to-capture 64 \
  --guided-decoding-backend lm-format-enforcer \
  --disable-log-requests \
  --served-model-name tiny-opt-kcmm \
  --use-v2-block-manager
```

Probe the OpenAI-compatible API:

```bash
curl -sS http://127.0.0.1:8001/v1/models
curl -sS http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-opt-kcmm","prompt":"Hello","max_tokens":4,"temperature":0}'
```
