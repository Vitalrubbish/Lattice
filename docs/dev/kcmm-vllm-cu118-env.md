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

Run with a Phase II.A runtime-derived KCMM pool:

```bash
python -m scripts.kcmm.vllm_smoke --runtime-derived-pool
```

This passes `--kcmm-pool-mode runtime` to the launcher. The KCMM pool is created
after vLLM has profiled and recorded cache capacity, using vLLM runtime values
for block size, GPU block budget, attention layer count, KV heads, head
dimension, max model length, and max sequences. The launcher still leaves vLLM
allocation behavior unchanged and keeps tiering disabled.

Run with the Phase II.A KCMM shadow allocator:

```bash
python -m scripts.kcmm.vllm_smoke --shadow-allocations
```

This enables runtime-derived pool sizing and mirrors vLLM GPU block
allocation/free events into KCMM without changing vLLM block IDs or KV tensor
storage. The smoke runner fails if the shadow report records errors, leaked
shadow mappings, zero observed GPU allocations, or mismatched KCMM alloc/free
counts.

Run with the Phase II.A KCMM-backed allocator:

```bash
python -m scripts.kcmm.vllm_smoke --backed-allocations
```

This enables runtime-derived pool sizing and lets KCMM choose vLLM GPU block IDs
through `kcmm_alloc_blocks`. vLLM native KV tensors remain the storage of record:
the KCMM-selected block ID is accepted only if it is also a free native vLLM GPU
block ID. The smoke runner fails if the backed report records a stop condition,
errors, leaked mappings, zero observed GPU allocations, mismatched KCMM
alloc/free counts, or KCMM blocks still in use after shutdown.

## Phase II.A A/B gate

Run the stock-vs-KCMM gate before starting Phase II.B work:

```bash
python -m scripts.kcmm.vllm_ab_gate
```

The gate runs stock vLLM, KCMM observer, KCMM shadow allocator, and KCMM-backed
allocator modes sequentially on the same tiny local OPT model, prompt, and
generation parameters. The JSON report records startup time, request latency,
generated tokens, token throughput, GPU memory footprint, and KCMM allocation
stats where applicable. Correctness failures fail the command; performance
regressions are reported as warnings and do not fail the gate.

Phase II.B must not start until this gate produces `passed: true` for the branch
and local environment being promoted.

Latest local Phase II.A gate result on 2026-06-19:

- Command: `python -m scripts.kcmm.vllm_ab_gate`
- Result: `passed=true`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Modes completed: `stock`, `observer`, `shadow`, `backed`
- Each mode generated 4 completion tokens on the tiny local model.
- Shadow and backed reports recorded `kcmm_allocations=1`, `kcmm_frees=1`,
  `outstanding_mappings=0`, and `error_count=0`.
- Backed mode recorded `blocks_in_use=0` after shutdown.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs and port `8001` was free
  after the run.

## Phase II.B KV write preflight gate

Run the KCMM KV write FFI gate before patching vLLM's
`vllm._custom_ops.reshape_and_cache` write path:

```bash
python -m scripts.kcmm.kv_write_ffi_smoke
```

The gate creates a tiny KCMM pool, registers a sequence backed by two KCMM
blocks, writes known FP16 K/V rows through `kcmm_append_kv_step`, then verifies
the vLLM-style physical-slot writer `kcmm_append_kv_slots` with
`slot = block_id * block_size + offset_in_block`. It reads the destination KCMM
VA bytes back to host and compares them with the source CUDA tensors. This
verifies the C ABI, VA accessors, D2D write paths, direct-slot decoding, padding
skip behavior, unallocated-slot failure, and D2H byte-level comparison without
downloading a model or starting vLLM.

Latest local Phase II.B preflight result on 2026-06-19:

- Command: `python -m scripts.kcmm.kv_write_ffi_smoke`
- Result: `passed=true`
- Compared two K rows and two V rows at positions `0` and `5`.
- The smoke wrote into two KCMM blocks through a registered sequence.
- Direct-slot writes passed for slots `2` and `7` using
  `slot = block_id * block_size + offset_in_block`.
- Direct-slot padding slot `-1` was skipped.
- Invalid direct slot `16` failed with `block_idx 4 from slot 16 not in use`.
- Final KCMM stats recorded `blocks_in_use=0`.

## Phase II.B vLLM write contract trace

Run the observer-only vLLM KV write instrumentation before replacing
`vllm._custom_ops.reshape_and_cache`:

```bash
python -m scripts.kcmm.vllm_smoke --instrument-kv-writes
```

The smoke patches `reshape_and_cache` and `reshape_and_cache_flash` without
changing their behavior, then records which function is called and the tensor
contract for `key`, `value`, `key_cache`, `value_cache`, and `slot_mapping`.
The trace includes shape, dtype, device, stride, element size, data pointer, and
a bounded `slot_mapping` sample. It intentionally does not dump K/V payload
contents.

The trace also decodes the bounded `slot_mapping` sample using the vLLM contract
`slot = block_id * block_size + offset_in_block`. This proves the write seam
exposes physical KV slots. The replacement path for this seam is the KCMM
direct-slot writer `kcmm_append_kv_slots`; `kcmm_append_kv_step` remains the
lower-level sequence/position writer.

Latest local Phase II.B write contract result on 2026-06-19:

- Command: `python -m scripts.kcmm.vllm_smoke --instrument-kv-writes`
- Result: completion succeeded.
- Observed write seam: `vllm._custom_ops.reshape_and_cache`
- Write calls observed: `8`
- Required KV write seam groups missing: `{}`
- First `slot_mapping` sample: `[0, 1]`
- First decoded slots: `(block_id=0, offset=0)`, `(block_id=0, offset=1)`
- First `key`/`value` shape: `[2, 2, 64]`
- First `key_cache` shape: `[134685, 2, 8, 16, 8]`
- First `value_cache` shape: `[134685, 2, 64, 16]`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Phase II.B vLLM KV write mirror gate

Run the KCMM-backed allocator with KV write mirroring after the direct-slot FFI
gate and write contract trace pass:

```bash
python -m scripts.kcmm.vllm_smoke \
  --backed-allocations \
  --instrument-kv-writes \
  --kv-write-mirror
```

This mode keeps native vLLM KV tensors as the storage of record. It calls native
`reshape_and_cache` first, then mirrors post-attach writes into KCMM through
`kcmm_append_kv_slots`. It requires `--backed-allocations` so vLLM physical block
ids in `slot_mapping` are also valid KCMM block ids. The smoke fails if the
mirror report records errors, mirrors no rows, or verifies no D2H KCMM rows.

Latest local Phase II.B KV write mirror result on 2026-06-19:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-mirror`
- Result: `passed=true`
- Observed write seam: `vllm._custom_ops.reshape_and_cache`
- Write calls observed: `8`
- KV mirror calls: `8`
- Mirrored rows: `10`
- D2H verified rows: `10`
- Verification bytes: `5120`
- Cache layers mapped: `2`
- KCMM-backed allocator recorded `kcmm_allocations=1`, `kcmm_frees=1`,
  `outstanding_mappings=0`, and `error_count=0`.
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Phase II.B vLLM KV write replacement candidate

Run the replacement-candidate mode only after the mirror gate passes:

```bash
python -m scripts.kcmm.vllm_smoke \
  --backed-allocations \
  --instrument-kv-writes \
  --kv-write-replace-candidate
```

This mode skips native vLLM `reshape_and_cache` writes and writes only to KCMM
through `kcmm_append_kv_slots`. It is a Phase II.B write-path candidate, not an
end-to-end correctness mode: native vLLM attention still reads native KV tensors
until Phase II.C replaces the read path. The report must therefore be interpreted
as write-path validation only.

Latest local Phase II.B KV write replacement-candidate result on 2026-06-19:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-replace-candidate`
- Result: `passed=true`
- Observed write seam: `vllm._custom_ops.reshape_and_cache`
- Write calls observed: `8`
- Native passthrough calls: `0`
- Native skipped calls: `8`
- KCMM write calls: `8`
- D2H verified rows: `10`
- Verification bytes: `5120`
- Cache layers mapped: `2`
- KCMM-backed allocator recorded `kcmm_allocations=1`, `kcmm_frees=1`,
  `outstanding_mappings=0`, and `error_count=0`.
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

The mirror gate was also rerun after the patch-order change:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-mirror`
- Result: `passed=true`
- Native passthrough calls: `8`
- Native skipped calls: `0`
- KCMM mirror calls: `8`
- D2H verified rows: `10`

## Phase II.C vLLM KV read contract trace

Run the observer-only paged-attention read instrumentation before attempting to
replace vLLM attention reads:

```bash
python -m scripts.kcmm.vllm_smoke --instrument-kv-reads
```

The smoke patches `vllm._custom_ops.paged_attention_v1` and
`vllm._custom_ops.paged_attention_v2` without changing behavior. It records the
tensor contract for `query`, `key_cache`, `value_cache`, `block_tables`, and
`seq_lens`, validates sampled block table entries against the observed KV cache
block count, and records whether A1 can safely replace `block_tables` entries
with KCMM VA offsets.

Latest local Phase II.C read contract result on 2026-06-19:

- Command: `python -m scripts.kcmm.vllm_smoke --instrument-kv-reads`
- Result: `passed=true`
- Observed read seam: `vllm._custom_ops.paged_attention_v1`
- Read calls observed: `6`
- Required KV read seam groups missing: `{}`
- First `block_tables` dtype: `torch.int32`
- First `block_tables` shape: `[1, 1]`
- First `block_tables` sample: `[0]`
- First `seq_lens` sample: `[3]`
- First `key_cache` shape: `[134685, 2, 8, 16, 8]`
- First `value_cache` shape: `[134685, 2, 64, 16]`
- A1 assessment at the Python custom-op seam:
  `safe_to_replace_block_tables_with_va_offsets=false`
- Reason: this seam passes native `key_cache`/`value_cache` tensor bases plus
  integer block ids. Replacing `block_tables` with KCMM VA offsets would exceed
  the KV cache block-id range unless the attention kernel address calculation is
  also changed.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

The read trace was also run with the Phase II.B replacement-candidate write
path:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --instrument-kv-reads --kv-write-replace-candidate`
- Result: `passed=true`
- Write calls observed: `8`
- Read calls observed: `6`
- Native write passthrough calls: `0`
- Native write skipped calls: `8`
- KCMM write verified rows: `10`
- Final KCMM pool stats recorded `blocks_in_use=0`.

## Phase II.C A2 read offset-table plan

After the read contract rejected A1 at the Python custom-op seam, the next
Phase II.C slice is an A2 side-table prototype:

```bash
python -m scripts.kcmm.vllm_smoke \
  --backed-allocations \
  --kv-write-mirror \
  --instrument-kv-reads \
  --kv-read-offset-table
```

This keeps `block_tables` as native vLLM block ids, requires the KCMM-backed
allocator so vLLM block ids and KCMM block ids are identical, and builds a CUDA
side table at every paged-attention read seam:

```text
offset_table[block_id] = kcmm_f16_va_offset
```

This is still a planning/prototype mode. The native vLLM paged-attention kernel
continues to execute, and the report records `kernel_replaced=false`.

Latest local Phase II.C A2 offset-table result on 2026-06-19:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-mirror --instrument-kv-reads --kv-read-offset-table`
- Result: `passed=true`
- Read seam: `vllm._custom_ops.paged_attention_v1`
- Read calls observed: `6`
- Offset table builds: `6`
- Offset table dtype: `torch.int64`
- Offset table device: `cuda:0`
- Last offset table shape: `[1]`
- Max block id seen: `0`
- Offset f16 sample: `{ "0": 1046528 }`
- Kernel replaced: `false`
- Read path: `native_vllm_paged_attention`
- KCMM-backed allocator allocations/frees: `1/1`
- KCMM KV write mirror verified rows: `10`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

The next Phase II.C step is to replace the native read kernel with a custom
attention backend or kernel entrypoint that consumes the KCMM K/V base addresses
and the A2 offset table.

## Phase II.C read replacement candidate

The first read replacement mode is a correctness/reference path, not the final
performance implementation:

```bash
python -m scripts.kcmm.vllm_smoke \
  --backed-allocations \
  --kv-write-replace-candidate \
  --instrument-kv-reads \
  --kv-read-replace-candidate
```

This mode skips native `reshape_and_cache` writes, skips native
`paged_attention_v1/v2` reads, reads K/V rows from KCMM-managed memory via CUDA
D2H copies, computes scaled dot-product attention with PyTorch, writes the
result into vLLM's `out` tensor, and returns without calling the native vLLM
paged-attention kernel.

Latest local Phase II.C read replacement result on 2026-06-19:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-replace-candidate --instrument-kv-reads --kv-read-replace-candidate`
- Result: `passed=true`
- Native KV write calls skipped: `8`
- Native paged-attention calls replaced: `6`
- Read path: `kcmm_reference_attention`
- Kernel replaced: `true`
- Reference KCMM read bytes: `12288`
- Offset table builds: `6`
- Observed read seam: `vllm._custom_ops.paged_attention_v1`
- KCMM write verified rows: `10`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

Same-model A/B check:

- Stock command:
  `python -m scripts.kcmm.vllm_smoke --mode stock --keep-model --no-build-kcmm`
- KCMM command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-replace-candidate --instrument-kv-reads --kv-read-replace-candidate --keep-model --no-build-kcmm`
- Stock completion text: `" 80 80 80 80"`
- KCMM replacement completion text: `" 80 80 80 80"`

The CPU-staged path remains useful as a debugging fallback. The GPU candidate
below is the next implementation slice for removing CPU staging from the read
replacement path.

## Phase II.C GPU read kernel candidate

Run the GPU read-kernel candidate after the read replacement path has proved the
storage-of-record transition:

```bash
python -m scripts.kcmm.vllm_smoke \
  --backed-allocations \
  --kv-write-replace-candidate \
  --instrument-kv-reads \
  --kv-read-gpu-kernel-candidate \
  --no-build-kcmm
```

This mode skips native `reshape_and_cache` writes, skips native
`paged_attention_v1/v2` reads, and launches `kcmm_paged_attn_decode_f16` to fill
vLLM's `out` tensor from KCMM K/V memory and the A2 offset table. The current
candidate is intentionally narrow: FP16 only, `head_dim <= 64`, no alibi,
no block-sparse mode, no FP8 cache scales, and a synchronous FFI return path.

Latest local Phase II.C GPU read-kernel result on 2026-06-20:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --kv-write-replace-candidate --instrument-kv-reads --kv-read-gpu-kernel-candidate --no-build-kcmm`
- Result: `passed=true`
- Completion text: `" behaviour behaviour behaviour behaviour"`
- Native KV write calls skipped: `8`
- Native paged-attention calls replaced: `6`
- Read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `6`
- Reference KCMM read bytes: `0`
- Offset table builds: `6`
- Observed read seam: `vllm._custom_ops.paged_attention_v1`
- KCMM write verified rows: `10`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Phase II.C GPU read-kernel A/B gate

Run the deterministic stock-vs-KCMM gate for the GPU read-kernel path:

```bash
python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm
```

The gate uses one tiny local OPT model directory for both modes. If the model is
absent, `create_tiny_opt_model.py` generates it with the default `seed=0`.
The gate then runs:

- stock vLLM through the same smoke harness,
- KCMM-backed allocator plus KV write replacement plus
  `kcmm_paged_attn_decode_f16`,
- exact comparison of completion text, finish reason, completion tokens, and
  total tokens.

Latest local Phase II.C GPU read-kernel A/B result on 2026-06-20:

- Command: `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`
- Result: `passed=true`
- Model existed before gate: `false`
- Prompt: `"Hello"`
- Max tokens: `4`
- Stock completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM completion text: `" pioneer pioneer pioneer pioneer"`
- Finish reason: `length`
- Completion tokens: `4`
- Total tokens: `6`
- Read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `6`
- Stream-aware kernel calls: `6`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `8`
- KCMM write verified rows: `10`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.

## Phase II.C stream-aware GPU read launch

The vLLM-integrated GPU read path uses:

```text
kcmm_paged_attn_decode_f16_on_stream(..., stream_ptr)
```

The Python read replacement passes:

```python
torch.cuda.current_stream(device_index).cuda_stream
```

This enqueues the read kernel on the caller's current CUDA stream and returns
without a per-call full context synchronize. The older
`kcmm_paged_attn_decode_f16` ABI remains as a synchronous compatibility wrapper.
Pool teardown still synchronizes before unloading the raw CUDA module.

Latest local stream-aware validation on 2026-06-20:

- Command: `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`
- Result: `passed=true`
- Stock completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM completion text: `" pioneer pioneer pioneer pioneer"`
- Read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `6`
- Stream-aware kernel calls: `6`
- Stream pointer sample: `[0, 0, 0, 0, 0, 0]`
- Reference KCMM read bytes: `0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.

The current vLLM eager seam reports stream pointer `0`, the legacy default
stream. The next Phase II.C work is performance characterization plus broader
prompt/shape coverage beyond the tiny local OPT gate. The KV write replacement
path still contains Python-side synchronizations and should be made
stream-aware separately.

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
