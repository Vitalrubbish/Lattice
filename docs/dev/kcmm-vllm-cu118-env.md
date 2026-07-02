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
the vLLM-style physical-slot writer `kcmm_append_kv_slots_on_stream` with
`slot = block_id * block_size + offset_in_block`. It also verifies the
device-slot writer `kcmm_append_kv_device_slots_on_stream`, which consumes a
CUDA int64 `slot_mapping` tensor pointer, a CUDA int64 f16-offset table indexed
by block id, and a CUDA u8 valid-block table. It reads the destination KCMM VA
bytes back to host and compares them with the source CUDA tensors. This verifies
the C ABI, VA accessors, D2D write paths, host-slot and device-slot decoding,
caller stream enqueue, padding skip behavior, invalid-slot handling, inactive
block handling, and D2H byte-level comparison without downloading a model or
starting vLLM.

The performance-clean vLLM write tracker can use the device-slot writer behind
`--kcmm-kv-write-device-slots` when row verification is disabled. Correctness
gates still use the stable host-slot path by default because bounded D2H row
verification needs the host slot list.

Latest local Phase II.B preflight result on 2026-07-02:

- Command: `python -m scripts.kcmm.kv_write_ffi_smoke --no-build-kcmm`
- Result: `passed=true`
- Compared two K rows and two V rows at positions `0` and `5`.
- The smoke wrote into two KCMM blocks through a registered sequence.
- Direct-slot writes passed for slots `2` and `7` using
  `slot = block_id * block_size + offset_in_block`.
- Direct-slot stream-aware write: `true`
- Direct-slot stream pointer: `0`
- Direct-slot padding slot `-1` was skipped.
- Invalid direct slot `16` failed with `block_idx 4 from slot 16 not in use`.
- Device direct-slot writes passed for slots `1` and `4` using the same slot
  formula.
- Device direct-slot padding slot `-1` was skipped.
- Device direct-slot invalid slot `16` set the device status tensor to `1`.
- Device direct-slot inactive slot `8` set the device status tensor to `2`.
- Device direct-slot offset table entries: `3`.
- Device direct-slot valid-block table: `[1, 1, 0]`.
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
stream-aware direct-slot writer `kcmm_append_kv_slots_on_stream`;
`kcmm_append_kv_step` remains the lower-level sequence/position writer.

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
`kcmm_append_kv_slots_on_stream`. It requires `--backed-allocations` so vLLM
physical block ids in `slot_mapping` are also valid KCMM block ids. The smoke
fails if the mirror report records errors, mirrors no rows, records no
stream-aware writes, or verifies no D2H KCMM rows.

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
through `kcmm_append_kv_slots_on_stream`. It is a Phase II.B write-path
candidate, not an end-to-end correctness mode: native vLLM attention still reads
native KV tensors until Phase II.C replaces the read path. The report must
therefore be interpreted as write-path validation only.

Latest local Phase II.B KV write replacement-candidate result on 2026-06-20:

- Command:
  `python -m scripts.kcmm.vllm_smoke --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-build-kcmm`
- Result: `passed=true`
- Observed write seam: `vllm._custom_ops.reshape_and_cache`
- Write calls observed: `8`
- Native passthrough calls: `0`
- Native skipped calls: `8`
- KCMM write calls: `8`
- Stream-aware write calls: `8`
- Stream-level verification synchronizations: `8`
- Last stream pointer: `0`
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

The vLLM-integrated write path no longer performs a full-device synchronize
around every KCMM write. D2H verification still synchronizes the current stream
before reading KCMM bytes back to host.

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
candidate is intentionally narrow: FP16 only, `head_dim <= 256`, no alibi,
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
- exact per-case comparison of completion text, finish reason, completion
  tokens, and total tokens.

By default the gate runs three coverage cases in each server process:
`hello`, `math`, and `long_context`. Replace the default cases with targeted
cases by passing repeated `--coverage-case NAME:MAX_TOKENS:PROMPT` flags.
Passing `--prompt` or `--max-tokens` without `--coverage-case` keeps the older
single-case behavior.

Latest local Phase II.C GPU read-kernel A/B result on 2026-06-20:

- Command: `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`
- Result: `passed=true`
- Model existed before gate: `false`
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
- Stream-aware KV write calls: `22`
- Stream-level write verification synchronizations: `22`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.
- Performance warnings: `[]`

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
without a per-call full context synchronize. The write replacement path uses the
same stream handoff model through `kcmm_append_kv_slots_on_stream`. The older
`kcmm_paged_attn_decode_f16` ABI remains as a synchronous compatibility wrapper.
Pool teardown still synchronizes before unloading the raw CUDA module.

Latest local stream-aware validation on 2026-06-20:

- Command: `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`
- Result: `passed=true`
- Stock completion text: `" pioneer pioneer pioneer pioneer"`
- KCMM completion text: `" pioneer pioneer pioneer pioneer"`
- Read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU kernel calls: `16`
- Stream-aware kernel calls: `16`
- Stream pointer sample: all `16` recent calls reported `0`
- Stream-aware KV write calls: `22`
- KV write stream pointer sample: all recent calls reported `0`
- Reference KCMM read bytes: `0`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after both modes.

The current vLLM eager seam reports stream pointer `0`, the legacy default
stream. D2H verification still synchronizes that stream, but the integrated
write/read replacement paths no longer require full-device synchronization per
call. The low-level FFI gate below validates non-default stream behavior for the
KCMM `_on_stream` entrypoints. The vLLM gate after it validates the integrated
replacement path with forced non-default KCMM streams and explicit stream waits.

## Phase II.C non-default-stream FFI gate

Run the low-level non-default-stream gate:

```bash
python -m scripts.kcmm.non_default_stream_ffi_smoke --no-build-kcmm
```

The gate creates a real `torch.cuda.Stream()` and requires its raw CUDA handle
to differ from the default stream handle. It then enqueues a KCMM direct-slot
write through `kcmm_append_kv_slots_on_stream` and a KCMM GPU read through
`kcmm_paged_attn_decode_f16_on_stream` on that same stream. The test uses a
single-token sequence with zero query, so the expected attention output is
exactly the V row written immediately before the read. Verification synchronizes
only the non-default stream before comparing GPU output and KCMM D2H bytes.

This gate covers the `_on_stream` FFI behavior when `stream_ptr != 0`. It does
not force the current vLLM eager seam to schedule on a non-default stream.

Latest local non-default-stream FFI result on 2026-06-28:

- Command:
  `python -m scripts.kcmm.non_default_stream_ffi_smoke --no-build-kcmm --output /tmp/kcmm-vllm-phase-ii-c-non-default-stream-1782615543.json`
- Result: `passed=true`
- Report: `/tmp/kcmm-vllm-phase-ii-c-non-default-stream-1782615543.json`
- Device: `NVIDIA GeForce RTX 3080`
- PyTorch/CUDA: `2.4.0+cu118` / `11.8`
- Non-default stream pointer: `94207523571936`
- Default stream pointer: `0`
- Direct-slot write path: `kcmm_append_kv_slots_on_stream`
- GPU read path: `kcmm_paged_attn_decode_f16_on_stream`
- Verified direct-slot K/V rows: `1`
- Read output matched the expected V row:
  `[1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0]`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Phase II.C vLLM non-default-stream gate

Run the vLLM-integrated non-default-stream gate:

```bash
python -m scripts.kcmm.vllm_gpu_read_non_default_stream_gate --no-build-kcmm
```

This gate reuses the stock-vs-KCMM GPU read A/B harness, but enables:

```text
--kcmm-kv-force-non-default-stream
```

The forced mode routes KCMM KV write replacement and GPU read replacement
through dedicated non-default CUDA streams. For each raw-pointer KCMM launch,
the KCMM stream waits for PyTorch's original current stream before launch, the
original stream waits for the KCMM stream before downstream vLLM consumers
continue, and PyTorch tensors passed by raw pointer are recorded on the KCMM
stream with `record_stream`.

This proves the integrated monkey-patch path can preserve stream graph ordering
when KCMM work is not launched on the legacy default stream. It does not prove
that the current vLLM eager scheduler naturally chooses non-default streams.

Latest local vLLM non-default-stream result on 2026-06-28:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_non_default_stream_gate --no-build-kcmm --no-print-seams`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-non-default-stream-1782619860722.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782619860722`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Stock/KCMM completion text matched for `hello`, `math`, and `long_context`.
- Read forced non-default stream calls: `16`
- Read stream pointer: `139999223911008`
- Read default stream pointer: `0`
- Write forced non-default stream calls: `22`
- Write stream pointer: `139999223908544`
- Write default stream pointer: `0`
- GPU read kernel calls: `16`
- Stream-aware read kernel calls: `16`
- Native KV write calls skipped: `22`
- KCMM write verified rows: `36`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

## Phase II.C GPU read-kernel performance characterization

The GPU read-kernel A/B gate records performance metrics in addition to the
correctness check. These metrics are warnings only; correctness failures still
control the command exit code.

The report includes:

- `startup_seconds`
- `request_latency_seconds`
- `tokens_per_second`
- `gpu_memory_peak_delta_mib`
- `performance_comparison`
- `performance_warnings`

Default warning thresholds:

- startup or request latency above `2.0x` stock,
- token throughput below `0.5x` stock,
- peak GPU memory delta above `1.5x` stock and at least `256 MiB` higher.

For performance work, use the performance-clean real-model gate:

```bash
python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate \
  --no-build-kcmm \
  --no-print-seams
```

This wraps the stock-vs-KCMM GPU read-kernel A/B gate with
`facebook/opt-125m` by default, but disables correctness-only overhead in the
KCMM mode:

- KV read trace instrumentation is disabled.
- KV write D2H row verification is disabled.
- GPU read-kernel profiling is disabled.

The gate still fails if stock-vs-KCMM completions differ, if the KCMM mode does
not launch the GPU read kernel, if CPU-staged reference read bytes are observed,
or if the write report shows any verified rows or verification synchronizations.
Use the regular correctness/profile gates when debugging contracts; use this
gate as the cleaner request-level baseline before kernel optimization.
The performance-clean gate also disables per-update tracker report writes,
disables host-side read block-table validation, caches the GPU read offset table
across read seams, uses the current-context stream launch ABI, precompiles the
GPU read kernel when the runtime pool attaches, caches stable write-side KCMM
pool shape metadata at attach time, uses the lightweight `kcmm_total_blocks()`
ABI for read offset-table minimum entry sizing, and enables device-slot KV
writes. Device-slot writes keep vLLM's CUDA `slot_mapping` tensor on device and
cache the KCMM write offset/valid-block tables by `kcmm_block_state_epoch()`.
The device-slot write kernel is also precompiled at pool attach so first-call
NVRTC/module-load cost is not charged to the first request. The read planner
uses compact plan metadata in performance-clean mode so it does not collect
stride/contiguity/sample diagnostics on every request-time read seam.
Correctness gates keep per-update reports, block-table validation, row
verification, and the host-slot writer enabled by default for better failure
diagnostics.

Latest local performance-clean result on 2026-07-02 after enabling compact read
plan metadata:

- Command:
  `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-compact-plan-latest.json`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-compact-plan-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Model: `facebook/opt-125m`
- Coverage case: `long_decode`, `32` generated tokens
- Stock/KCMM completion text matched.
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- KCMM write verified rows: `0`
- Read block-table validation enabled: `false`
- Read compact plan metadata enabled/calls: `true/372`
- Read detailed plan metadata calls: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `93.363ms`
- Lightweight read total-block calls: `372`
- Write pool shape cached/refreshes: `true/1`
- Cached write pool shape: `block_size=16`, `block_bytes=24576`,
  `step_elements=768`, `num_layers=12`
- Device-slot write enabled/active: `true/true`
- Device-slot write calls: `384`
- Device-slot write-kernel precompile requested/succeeded/calls: `true/true/1`
- Device-slot write-kernel precompile elapsed: `77.829ms`
- Host-slot write calls: `0`
- Device-slot status checks/errors: `384/0`
- Device-slot status codes: `{"0": 384}`
- Device-slot offset table cache hits/rebuilds: `381/3`
- Device-slot valid table cache hits/rebuilds: `381/3`
- Offset table cache hits/rebuilds: `369/3`
- Read tracker report writes: `1`
- Write tracker report writes: `1`
- Stream-level write verification synchronizations: `0`
- KCMM write verification enabled: `false`
- Request latency seconds: stock `1.826`, KCMM `1.813`, ratio `0.993`
- Tokens per second: stock `17.525`, KCMM `17.650`, ratio `1.007`
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`
- Compared with the previous write-precompile run, compact read-plan metadata
  keeps the same clean contract while removing per-call diagnostic field
  collection from the read planner.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

For concurrent real-model performance-clean coverage, use the stress wrapper:

```bash
python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate \
  --no-build-kcmm \
  --no-print-seams \
  --timeout-seconds 420 \
  --shutdown-timeout-seconds 60
```

This wrapper keeps the same performance-clean settings as the single-request
gate but runs two real-model completion cases with `completion_concurrency=2`,
`max_num_seqs=2`, and `max_num_batched_tokens=192`. It fails if the KCMM read
report does not observe a decode batch of at least `2`, so it validates that the
fast path is exercised under concurrent vLLM scheduling. It also inherits the
device-slot KV write requirement from the performance-clean gate.

Latest local performance-clean stress result on 2026-07-02:

- Command:
  `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-compact-plan-latest.json`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-compact-plan-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Coverage cases: `stress_history`, `stress_memory`
- Completion concurrency: `2`
- Observed max read batch: `2`
- Observed max write batch: `17`
- GPU read kernel calls: `276`
- Stream-aware read kernel calls: `276`
- Reference KCMM read bytes: `0`
- Read compact plan metadata enabled/calls: `true/276`
- Read detailed plan metadata calls: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `94.898ms`
- Offset table cache hits/rebuilds: `273/3`
- Device-slot write enabled/active: `true/true`
- Device-slot write calls: `288`
- Device-slot write-kernel precompile requested/succeeded/calls: `true/true/1`
- Device-slot write-kernel precompile elapsed: `76.366ms`
- Host-slot write calls: `0`
- Device-slot status checks/errors: `288/0`
- Device-slot status codes: `{"0": 288}`
- Device-slot offset table cache hits/rebuilds: `285/3`
- Device-slot valid table cache hits/rebuilds: `285/3`
- Write verification enabled: `false`
- KCMM write verified rows: `0`
- Request latency seconds: stock `1.818`, KCMM `1.791`, ratio `0.985`
- Tokens per second: stock `26.403`, KCMM `26.801`, ratio `1.015`
- Peak GPU memory delta MiB: stock `5443`, KCMM `5593`, ratio `1.028`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

For host-side attribution, use the diagnostic host-profile wrapper:

```bash
python -m scripts.kcmm.vllm_gpu_read_host_profile_gate \
  --no-build-kcmm \
  --no-print-seams \
  --timeout-seconds 420 \
  --shutdown-timeout-seconds 60
```

This wrapper keeps the performance-clean settings, including current-context
read launch and read-kernel precompile, and forces `--kcmm-tracker-host-profile`
in the KCMM mode. It records section-level wall-clock timing in the read/write
tracker final reports without enabling CUDA event profiling or per-update report
writes. The timings are nested diagnostic sections; do not sum them as
independent request-level costs.

Latest local host-profile result on 2026-07-02 after enabling compact read plan
metadata:

- Command:
  `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-compact-plan-latest.json`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-compact-plan-latest.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- GPU read kernel calls: `372`
- Stream-aware read kernel calls: `372`
- Reference KCMM read bytes: `0`
- Read fast current-context launch: `true`
- Read GPU kernel precompile requested/succeeded/calls: `true/true/1`
- Read GPU kernel precompile elapsed: `94.536ms`
- Read compact plan metadata enabled/calls: `true/372`
- Read detailed plan metadata calls: `0`
- Lightweight read total-block calls: `372`
- Write pool shape cached/refreshes: `true/1`
- Cached write pool shape: `block_size=16`, `block_bytes=24576`,
  `step_elements=768`, `num_layers=12`
- Device-slot write-kernel precompile requested/succeeded/calls: `true/true/1`
- Device-slot write-kernel precompile elapsed: `78.703ms`
- Device-slot write calls: `384`
- Host-slot write calls: `0`
- Offset table cache hits/rebuilds: `369/3`
- Request latency seconds: stock `1.814`, KCMM `1.823`, ratio `1.005`
- Tokens per second: stock `17.641`, KCMM `17.553`, ratio `0.995`
- Top read host sections: `read_gpu_kernel_precompile=94.542ms`,
  `read_replace_call_total=33.429ms`,
  `read_replace_gpu_kernel_host=18.687ms`,
  `read_replace_build_plan=11.654ms`,
  `read_build_plan_total=10.695ms`,
  `read_gpu_kernel_ctypes_launch=6.001ms`.
- Top write host sections: `write_device_slot_kernel_precompile=78.715ms`,
  `write_mirror_call_total=41.051ms`,
  `write_device_slot_table_lookup=5.424ms`,
  `write_select_stream=4.655ms`, `write_ctypes_launch=3.844ms`.
- Compared with the host-profile run immediately before Issue 34,
  `read_replace_build_plan` dropped from `14.894ms` total to `11.654ms`, and
  `read_build_plan_total` dropped from `13.931ms` total to `10.695ms`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

Latest local performance characterization on 2026-06-20:

- Command: `python -m scripts.kcmm.vllm_gpu_read_ab_gate --no-build-kcmm`
- Result: `passed=true`
- Performance warnings: `[]`
- Coverage cases: `hello`, `math`, `long_context`
- Aggregate completion tokens: `11`
- Startup seconds: stock `13.54`, KCMM `10.529`, ratio `0.778`
- Request latency seconds: stock `1.763`, KCMM `1.972`, ratio `1.119`
- Tokens per second: stock `6.239`, KCMM `5.578`, ratio `0.894`
- Peak GPU memory delta MiB: stock `3417`, KCMM `3425`, ratio `1.002`

## Phase II.C GPU read-kernel per-call profiling

Run the opt-in per-call profiler for the GPU read-kernel path:

```bash
python -m scripts.kcmm.vllm_gpu_read_profile_gate --no-build-kcmm
```

This wraps the stock-vs-KCMM GPU read A/B gate, enables
`--kcmm-kv-read-profile` only in the KCMM mode, and fails if the KCMM read
report does not include timing samples. The lower-level smoke/A-B flag is
`--kv-read-profile`.

Profiling records CUDA events on the same stream passed to
`kcmm_paged_attn_decode_f16_on_stream`. The report includes per-call
`gpu_kernel_elapsed_ms` values in `recent_calls` and a `gpu_kernel_profile`
summary with count, min, avg, p50, p95, p99, max, `first_call_ms`,
`warmup_excluded_count`, a `steady_state` summary that excludes the first sample
when multiple samples exist, and raw `samples_ms`.

This is a diagnostic mode, not the default correctness path: reading CUDA event
timing synchronizes the event and adds overhead. Use it to guide kernel
optimization, then validate correctness/performance again with profiling
disabled.

Latest local GPU read-kernel profiling result on 2026-06-29:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 240 --shutdown-timeout-seconds 45`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-profile-1782718561901.json`
- Correctness failures: `[]`
- Performance warnings: `[]`
- GPU read kernel calls: `16`
- Profile sample count: `16`
- First call: `100.57933 ms`
- Warm-up excluded count: `1`
- Overall profile summary: min `0.029696 ms`, avg `6.367104 ms`,
  p50 `0.05632 ms`, p95 `100.57933 ms`, p99 `100.57933 ms`,
  max `100.57933 ms`
- Steady-state summary after excluding the first sample: count `15`,
  min `0.029696 ms`, avg `0.086289 ms`, p50 `0.05632 ms`,
  p95 `0.1536 ms`, p99 `0.1536 ms`, max `0.1536 ms`
- Raw samples:
  `[100.57933, 0.029696, 0.031744, 0.032768, 0.03584, 0.03584, 0.05632, 0.062464, 0.0512, 0.0512, 0.146432, 0.1536, 0.150528, 0.149504, 0.1536, 0.1536]`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

The first profiled call recorded about `100.5 ms` in two consecutive profile
runs, while steady-state calls stayed in the `0.03-0.15 ms` range on the tiny
local OPT gate. Treat the first sample as a cold-start/module warm-up outlier
when choosing steady-state kernel optimization work.

## Phase II.C GPU read-kernel shape coverage gate

Run the shape coverage gate for the GPU read-kernel path:

```bash
python -m scripts.kcmm.vllm_gpu_read_shape_gate --no-build-kcmm
```

The gate generates each tiny OPT variant under
`.scratch/kcmm-vllm/shape-gate/`, then runs the stock-vs-KCMM GPU read A/B gate
for every variant with the same completion coverage cases. By default it runs:

- `head64_layers2`: hidden size `128`, heads `2`, layers `2`, FFN dimension
  `256`.
- `head80_layers2`: hidden size `160`, heads `2`, layers `2`, FFN dimension
  `320`.
- `head96_layers2`: hidden size `192`, heads `2`, layers `2`, FFN dimension
  `384`.
- `head128_layers2`: hidden size `256`, heads `2`, layers `2`, FFN dimension
  `512`.
- `head192_layers2`: hidden size `384`, heads `2`, layers `2`, FFN dimension
  `768`.
- `head256_layers2`: hidden size `512`, heads `2`, layers `2`, FFN dimension
  `1024`.

The current CUDA 11.8 vLLM/XFormers stack supports paged-attention head sizes
`64`, `80`, `96`, `112`, `120`, `128`, `192`, and `256`. The current KCMM GPU
read kernel and FFI guard cover this full local vLLM-supported set. The default
shape coverage set exercises `64`, `80`, `96`, `128`, `192`, and `256`.

The default shape coverage cases keep `hello` at `4` generated tokens and
`math` at `3` generated tokens. The `long_context` prompt uses `1` generated
token: the prompt still spans three `16`-token KV blocks, so the decode read
exercises multi-block block-table lookup without recursively amplifying normal
FP16 paged-attention rounding differences across several generated tokens.

For `head_dim=80`, `96`, and `192`, the per-layer logical KCMM block sizes do
not evenly divide a 2 MiB superblock. The physical block allocator now hands
out only full logical blocks and leaves the superblock tail as padding. For the
current shape gate parameters, `head_dim=80` uses `5120` byte logical blocks and
leaves `3072` bytes of tail padding; `head_dim=96` uses `6144` byte logical
blocks and leaves `2048` bytes of tail padding; `head_dim=192` uses `12288`
byte logical blocks and leaves `8192` bytes of tail padding. `head_dim=64`,
`128`, and `256` divide the 2 MiB superblock exactly for the current parameters.

Latest local shape coverage result on 2026-06-28:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_shape_gate --no-build-kcmm --no-print-seams`
- Result: `passed=true`
- Failed variants: `[]`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Aggregate report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-shape-gate-1782637499399.json`
- Per-variant reports:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-shape-gate-1782637499399-reports/`
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.
- Temporary shape model directories were removed after the gate.

`head64_layers2` result:

- `hello` completion: `" pioneer pioneer pioneer pioneer"`
- `math` completion: `"gallgallgall"`
- `long_context` completion: `" radar"`
- GPU read kernel calls: `10`
- Stream-aware read kernel calls: `10`
- Native KV write calls skipped: `16`
- KCMM write verified rows: `30`
- Stream-aware KV write calls: `16`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.

`head80_layers2` result:

- `hello` completion: `"gunsguns valleys valleys"`
- `math` completion: `" Coverage Coverage Coverage"`
- `long_context` completion: `"620"`
- GPU read kernel calls: `10`
- Stream-aware read kernel calls: `10`
- Native KV write calls skipped: `16`
- KCMM write verified rows: `30`
- Stream-aware KV write calls: `16`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.

`head96_layers2` result:

- `hello` completion: `" manufacturing manufacturingarrayarray"`
- `math` completion: `" inject inject inject"`
- `long_context` completion: `" Puzzles"`
- GPU read kernel calls: `10`
- Stream-aware read kernel calls: `10`
- Native KV write calls skipped: `16`
- KCMM write verified rows: `30`
- Stream-aware KV write calls: `16`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.

`head128_layers2` result:

- `hello` completion: `" Bengal Bengal BengalComplete"`
- `math` completion: `" Objects Objects Jung"`
- `long_context` completion: `"lett"`
- GPU read kernel calls: `10`
- Stream-aware read kernel calls: `10`
- Native KV write calls skipped: `16`
- KCMM write verified rows: `30`
- Stream-aware KV write calls: `16`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.

`head192_layers2` result:

- `hello` completion: `" SY edited edited edited"`
- `math` completion: `" = = ="`
- `long_context` completion: `" acceleration"`
- GPU read kernel calls: `10`
- Stream-aware read kernel calls: `10`
- Native KV write calls skipped: `16`
- KCMM write verified rows: `30`
- Stream-aware KV write calls: `16`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.

`head256_layers2` result:

- `hello` completion: `"ivoivoivo charg"`
- `math` completion: `" Faw Faw Faw"`
- `long_context` completion: `"8000"`
- GPU read kernel calls: `10`
- Stream-aware read kernel calls: `10`
- Native KV write calls skipped: `16`
- KCMM write verified rows: `30`
- Stream-aware KV write calls: `16`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.

## Phase II.C GPU read-kernel batch/concurrency gate

Run the batch/concurrency gate for the GPU read-kernel path:

```bash
python -m scripts.kcmm.vllm_gpu_read_batch_gate --no-build-kcmm
```

The gate starts stock vLLM and the KCMM-backed GPU read-kernel mode with:

- `max_model_len=128`
- `max_num_seqs=2`
- `max_num_batched_tokens=128`
- `completion_concurrency=2`
- `require_min_read_batch=2`

The default coverage cases are two concurrent completion requests:

- `parallel_alpha`: `alpha beta gamma delta epsilon zeta eta theta`,
  `max_tokens=8`
- `parallel_math`: `Question: 2 + 2 =`, `max_tokens=8`

The gate fails if stock-vs-KCMM completion text, finish reason, completion
tokens, or total tokens differ. It also fails if the KCMM read report does not
observe a decode batch of at least `2`.

Latest local batch/concurrency result on 2026-06-28:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_batch_gate --no-build-kcmm --no-print-seams`
- Result: `passed=true`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Aggregate report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-batch-1782621793642.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782621793642`
- `parallel_alpha` completion: `" Vol Vol Vol Vol Vol Vol Vol Vol"`
- `parallel_math` completion: `"gallgallgallgallgallgall cord cord"`
- Observed max read batch: `2`
- Observed max write batch: `14`
- GPU read kernel calls: `14`
- Stream-aware read kernel calls: `14`
- Native KV write calls skipped: `18`
- KCMM write verified rows: `44`
- Stream-aware KV write calls: `18`
- Reference KCMM read bytes: `0`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.
- Startup seconds: stock `13.540`, KCMM `10.529`, ratio `0.778`
- Request latency seconds: stock `1.765`, KCMM `1.940`, ratio `1.099`
- Tokens per second: stock `9.065`, KCMM `8.247`, ratio `0.910`
- Peak GPU memory delta MiB: stock `3415`, KCMM `3423`, ratio `1.002`
- Read-seam diagnostic sample: `query_shape=[2, 2, 64]`,
  `query_stride=[384, 64, 1]`, and `query_is_contiguous=false`.

The batch/concurrency fix preserves the stream-aware read path: the replacement
materializes compact tensor views on PyTorch's current CUDA stream and launches
`kcmm_paged_attn_decode_f16_on_stream` with the same stream pointer. Future
framework-originated non-default stream scheduling should still be revalidated
if vLLM starts invoking the patched seams from non-default current streams
itself.

## Phase II.C GPU read-kernel tensor-parallel gate

Run the tensor-parallel gate for the GPU read-kernel path:

```bash
python -m scripts.kcmm.vllm_gpu_read_tensor_parallel_gate --no-build-kcmm
```

The gate wraps the stock-vs-KCMM GPU read A/B gate with
`tensor_parallel_size=2` and serves the same tiny local OPT model on both RTX
3080 GPUs. It compares the default `hello`, `math`, and `long_context`
completion cases, then verifies the KCMM report used the stream-aware GPU read
kernel and did not fall back to CPU-staged reference reads.

Tensor-parallel vLLM uses worker subprocesses. These workers inherit the KCMM
monkey patches but do not run the driver process's `LLMEngine.__init__`
runtime-pool callback. The KCMM launcher therefore also patches
`Worker.initialize_cache` to create and attach a worker-local KCMM pool before
model execution. Because TP workers receive scheduler-chosen slot mappings from
the driver process, the KV write replacement lazily ensures local KCMM block IDs
from `slot_mapping` before appending KV rows.

Latest local tensor-parallel result on 2026-06-28:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_tensor_parallel_gate --no-build-kcmm --no-print-seams --timeout-seconds 240 --shutdown-timeout-seconds 45`
- Result: `passed=true`
- Tensor parallel size: `2`
- Correctness failures: `[]`
- Performance warnings: startup warning only. KCMM startup `59.111s`, stock
  startup `17.042s`, warning threshold `34.084s`.
- Aggregate report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-tensor-parallel-1782634782121.json`
- Run directory:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782634782121`
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
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.

The worker-pool hook preserves the single-GPU path. A follow-up single-GPU A/B
regression run passed on 2026-06-28 with report
`/tmp/kcmm-vllm-phase-ii-c-gpu-read-ab-1782635020234.json`.

## Phase II.C GPU read-kernel real-model gate

Run the first real-model gate for the GPU read-kernel path:

```bash
python -m scripts.kcmm.vllm_gpu_read_real_model_gate \
  --download-model \
  --no-build-kcmm
```

The gate downloads `facebook/opt-125m` into
`.scratch/kcmm-vllm/real-models/facebook--opt-125m` unless an existing
`--model-path` is supplied. It then reuses the stock-vs-KCMM GPU read A/B
harness with tiny-model generation disabled, so real model directories are not
overwritten by `create_tiny_opt_model.py`.

The default real-model coverage cases are short by design:

- `hello`: `Hello`, `max_tokens=2`
- `math`: `Question: 2 + 2 =`, `max_tokens=2`

This is the first real-model coverage slice. It proves the replacement path can
serve a non-generated vLLM/Hugging Face model locally, but it does not yet claim
broad model compatibility.

Latest local real-model result on 2026-06-29:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_real_model_gate --download-model --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-1782719715998.json`
- Model: `facebook/opt-125m`
- Local model path:
  `.scratch/kcmm-vllm/real-models/facebook--opt-125m`
- Correctness failures: `[]`
- Performance warnings: `[]`
- `hello` completion: `", I"`
- `math` completion: `" -2"`
- Aggregate completion tokens: `4`
- Aggregate total tokens: `13`
- Read path: `kcmm_paged_attn_decode_f16`
- Replacement backend: `gpu_kernel`
- GPU read kernel calls: `24`
- Stream-aware read kernel calls: `24`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `48`
- KCMM write verified rows: `96`
- Final KCMM pool stats recorded `blocks_in_use=0`.
- GPU memory returned to 0 MiB on both RTX 3080 GPUs after the run.
- Peak GPU memory delta MiB: stock `5441`, KCMM `5591`, ratio `1.028`
- Request latency seconds: stock `1.729`, KCMM `2.233`, ratio `1.291`
- Tokens per second: stock `2.313`, KCMM `1.791`, ratio `0.774`

## Phase II.C GPU read-kernel real-model matrix gate

Run the multi-model real-model gate:

```bash
python -m scripts.kcmm.vllm_gpu_read_real_model_matrix_gate \
  --download-model \
  --no-build-kcmm
```

The default matrix currently covers:

- `facebook/opt-125m`
- `distilgpt2`

The default coverage cases are:

- `hello`: `Hello`, `max_tokens=2`
- `math`: `Question: 2 + 2 =`, `max_tokens=2`
- `long_context`: a 20-token Greek-letter prompt, `max_tokens=2`

The longer prompt forces multi-block read behavior on real Hugging Face models
without making the local gate expensive. Missing models are downloaded into
`.scratch/kcmm-vllm/real-models/` only when `--download-model` is passed. The
download filter is restricted to vLLM-required model/tokenizer files so it does
not pull unrelated CoreML, TensorFlow, or Flax artifacts.

Latest local real-model matrix result on 2026-06-29:

- Command:
  `python -m scripts.kcmm.vllm_gpu_read_real_model_matrix_gate --download-model --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60`
- Result: `passed=true`
- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-1782721112360.json`
- Failed models: `[]`
- Correctness failures: `[]`
- Performance warnings: `[]`
- Model order: `facebook/opt-125m`, `distilgpt2`

`facebook/opt-125m` matrix result:

- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-1782721112360-reports/real-facebook-opt-125m.json`
- `hello` completion: `", I"`
- `math` completion: `" -2"`
- `long_context` completion: `" rho"`
- GPU read kernel calls: `36`
- Stream-aware read kernel calls: `36`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `72`
- KCMM write verified rows: `156`
- Final KCMM pool stats recorded `blocks_in_use=0`.

`distilgpt2` matrix result:

- Report:
  `/tmp/kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-1782721112360-reports/real-distilgpt2.json`
- `hello` completion: `" The first"`
- `math` completion: `" 1 +"`
- `long_context` completion: `" pia"`
- GPU read kernel calls: `18`
- Stream-aware read kernel calls: `18`
- Reference KCMM read bytes: `0`
- Native KV write calls skipped: `36`
- KCMM write verified rows: `72`
- Final KCMM pool stats recorded `blocks_in_use=0`.

GPU memory returned to 0 MiB on both RTX 3080 GPUs after the matrix gate.

The manual steps below are the expanded form of the single-model GPU
read-kernel check.

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
