# KCMM memory model and vLLM integration code map

This document is a code-oriented guide to two topics:

1. KCMM's own KV-cache memory model.
2. The key code points where KCMM integrates with vLLM.

It is meant for reading the implementation, not for defining a new architecture.
The higher-level architecture decision remains
[`docs/adr/0001-vllm-integration-architecture.md`](../adr/0001-vllm-integration-architecture.md).

## 1. KCMM memory model

KCMM models KV cache as a paged, per-layer, per-K/V GPU memory pool.

```text
KcmmPool
  ├─ layer 0
  │   ├─ K GPU VA region: va_k[0]
  │   └─ V GPU VA region: va_v[0]
  ├─ layer 1
  │   ├─ K GPU VA region: va_k[1]
  │   └─ V GPU VA region: va_v[1]
  ├─ ...
  └─ layer N
      ├─ K GPU VA region: va_k[N]
      └─ V GPU VA region: va_v[N]
```

### 1.1 Core object

The central data structure is
[`KcmmPool`](../../src/kcmm/pool.rs#L132).

| Code point | Responsibility |
|---|---|
| [`src/kcmm/pool.rs::KcmmPool`](../../src/kcmm/pool.rs#L132) | Central KCMM pool state |
| [`src/kcmm/pool.rs::KcmmPool::new`](../../src/kcmm/pool.rs#L199) | Creates per-layer K/V VA regions and physical pools |
| [`src/kcmm/ffi.rs::kcmm_pool_create`](../../src/kcmm/ffi.rs#L470) | C ABI entrypoint for Python/vLLM |
| [`scripts/kcmm/bindings.py::KcmmLibrary.create_pool`](../../scripts/kcmm/bindings.py#L431) | Python ctypes wrapper around `kcmm_pool_create` |
| [`scripts/kcmm/bindings.py::KcmmPool`](../../scripts/kcmm/bindings.py#L438) | Python handle wrapper used by launcher and trackers |

Key `KcmmPool` fields:

| Field | Meaning |
|---|---|
| `config` | KCMM runtime configuration |
| `ctx` | CUDA device context |
| `block_size` | Tokens per KV block, default `16` |
| `elem_per_block` | FP16 elements in one K or V block |
| `block_bytes` | Bytes in one K or V block |
| `va_k` | Per-layer K-cache GPU virtual base addresses |
| `va_v` | Per-layer V-cache GPU virtual base addresses |
| `k_pools` | Per-layer K physical pools |
| `v_pools` | Per-layer V physical pools |
| `block_info` | `block_id -> BlockInfo` metadata |
| `free_block_indices` | Recyclable block ids |
| `sequences` | Per-sequence logical block tables |
| `block_state_epoch` | Monotonic version used to invalidate cached offset tables |
| `tiering` | Optional GPU/CPU/NVMe migration engine |

### 1.2 Layer count

The number of layers comes from the model, not from KCMM.

[`KcmmPool::new`](../../src/kcmm/pool.rs#L199) receives:

```text
model_num_layers
model_kv_heads
model_head_dim
max_batch
max_seq_len
```

For every transformer layer, KCMM reserves one K VA region and one V VA region.

```text
model_num_layers = 12

KCMM creates:
  va_k[0..11]
  va_v[0..11]
  k_pools[0..11]
  v_pools[0..11]
```

Different transformer layers cannot share KV storage because every layer has
different hidden states and therefore different K/V values.

### 1.3 GPU virtual address regions

KCMM uses CUDA VMM to reserve stable GPU virtual address ranges.

```text
reserve GPU VA
  ↓
map physical GPU memory later
  ↓
CUDA kernels read/write through stable VA
```

In code:

```text
va_k[layer_idx] = K cache GPU VA base for this layer
va_v[layer_idx] = V cache GPU VA base for this layer
```

The VA is represented as `u64` in Rust and passed as integer CUDA addresses
through the C ABI.

### 1.4 Blocks

KV cache is split into fixed-size blocks.

```text
block_size = tokens per block
block_bytes = block_size * kv_heads * head_dim * sizeof(fp16)
step_elements = block_bytes / block_size / sizeof(fp16)
```

For the local OPT-125m-shaped run documented in
[`docs/dev/kcmm-vllm-cu118-env.md`](kcmm-vllm-cu118-env.md):

```text
block_size = 16
block_bytes = 24576
step_elements = 768
num_layers = 12
```

Each block id has metadata in `block_info`. Sequence metadata stores logical
block tables:

```text
sequence logical block table:
  logical block 0 -> block_id 7
  logical block 1 -> block_id 3
  logical block 2 -> block_id 12
```

This is the same paged-KV idea that vLLM exposes through `block_tables`.

### 1.5 Physical memory and superblocks

KCMM reserves virtual address space separately from physical GPU memory.

Physical GPU memory is allocated through CUDA VMM in superblock-sized chunks.
The project vocabulary uses:

```text
Superblock = 2 MiB CUDA VMM physical allocation
Block      = fixed-token KV-cache unit carved from a superblock
```

When the free list is empty, KCMM provisions more physical memory and adds
blocks back to the allocator. This is the basis for grow-on-demand behavior.

### 1.6 Lockstep allocation

KCMM allocates K and V blocks across layers in lockstep.

The important invariant is:

```text
block_id = X

layer 0 K: va_k[0] + offset(X)
layer 0 V: va_v[0] + offset(X)
layer 1 K: va_k[1] + offset(X)
layer 1 V: va_v[1] + offset(X)
...
```

This lets one `block_id` from vLLM's `block_tables` identify the corresponding
K/V block for every layer. The layer selects the VA base; the block id selects
the offset inside that layer's K/V region.

### 1.7 Address formulas

KCMM often stores offsets in FP16-element units, not byte units. This is why
many APIs use names like `block_offsets_f16`.

For a write using vLLM `slot_mapping`:

```text
slot = slot_mapping[row]

block_id        = slot / block_size
offset_in_block = slot % block_size

dst_idx =
  block_offsets_f16[block_id]
  + offset_in_block * step_elements
  + col

K destination = va_k[layer_idx] + dst_idx
V destination = va_v[layer_idx] + dst_idx
```

For a read using vLLM `block_tables`:

```text
block_id =
  block_tables[batch_idx][logical_block]

block_offset =
  offset_table[block_id]

src_idx =
  block_offset
  + offset_in_block * kv_heads * head_dim
  + kv_head * head_dim
  + dim

K source = va_k[layer_idx] + src_idx
V source = va_v[layer_idx] + src_idx
```

## 2. vLLM integration code map

KCMM integrates with vLLM through a launcher-time monkey patch. The vLLM package
is not modified on disk.

```text
scripts/kcmm/launcher.py
  ├─ creates KCMM runtime pool
  ├─ attaches allocator tracker
  ├─ attaches KV write tracker
  ├─ attaches KV read planner
  └─ patches vLLM custom ops
```

### 2.1 Launcher and runtime pool

| Code point | Responsibility |
|---|---|
| [`scripts/kcmm/launcher.py::_create_observer_pool`](../../scripts/kcmm/launcher.py#L101) | Creates a KCMM pool via Python binding |
| [`scripts/kcmm/launcher.py::attach_runtime_pool`](../../scripts/kcmm/launcher.py#L445) | Creates runtime-sized pool and attaches it to trackers |
| [`scripts/kcmm/launcher.py::create_runtime_pool`](../../scripts/kcmm/launcher.py#L479) | Builds sizing from `LLMEngine` |
| [`scripts/kcmm/launcher.py::create_worker_runtime_pool`](../../scripts/kcmm/launcher.py#L486) | Builds worker-local pool for tensor parallel workers |

The runtime pool is attached to:

| Tracker | Purpose |
|---|---|
| [`KcmmShadowAllocatorTracker`](../../scripts/kcmm/shadow_allocator.py#L15) | Mirrors vLLM allocation/free lifetimes |
| [`KcmmBackedAllocatorTracker`](../../scripts/kcmm/backed_allocator.py#L17) | Keeps vLLM and KCMM block ids aligned |
| [`KcmmKvWriteMirrorTracker`](../../scripts/kcmm/kv_write_mirror.py#L99) | Handles KV write mirror/replace |
| [`KcmmKvReadOffsetTableTracker`](../../scripts/kcmm/kv_read_plan.py#L154) | Handles A2 offset table and read replacement |

### 2.2 Monkey-patched vLLM seams

The patch target list is in
[`scripts/kcmm/patch_vllm.py`](../../scripts/kcmm/patch_vllm.py#L127).

| vLLM seam | Patched functions | KCMM purpose |
|---|---|---|
| KV write | `vllm._custom_ops.reshape_and_cache` | Intercept K/V writes |
| KV write | `vllm._custom_ops.reshape_and_cache_flash` | Intercept flash variant |
| KV read | `vllm._custom_ops.paged_attention_v1` | Intercept decode attention read |
| KV read | `vllm._custom_ops.paged_attention_v2` | Intercept v2 read variant |

Write patching:

| Code point | Responsibility |
|---|---|
| [`patch_vllm.py::KV_WRITE_FUNCTIONS`](../../scripts/kcmm/patch_vllm.py#L127) | Declares write functions to patch |
| [`patch_vllm.py::_wrap_kv_write_mirror_function`](../../scripts/kcmm/patch_vllm.py#L900) | Replaces write function with wrapper |
| [`patch_vllm.py::apply_kv_write_mirror`](../../scripts/kcmm/patch_vllm.py#L1441) | Applies write monkey patch |

Read patching:

| Code point | Responsibility |
|---|---|
| [`patch_vllm.py::KV_READ_FUNCTIONS`](../../scripts/kcmm/patch_vllm.py#L141) | Declares read functions to patch |
| [`patch_vllm.py::_wrap_kv_read_offset_table_function`](../../scripts/kcmm/patch_vllm.py#L876) | Replaces read function with wrapper |
| [`patch_vllm.py::apply_kv_read_offset_table`](../../scripts/kcmm/patch_vllm.py#L1523) | Applies read monkey patch |

## 3. KV write integration

### 3.1 Modes

KCMM write replacement has two modes.

| Mode | Behavior | Storage of record |
|---|---|---|
| Mirror | Call native `reshape_and_cache`, then copy K/V into KCMM | Native vLLM KV tensors |
| Replace | Skip native `reshape_and_cache`, write only KCMM | KCMM KV storage candidate |

The mode is selected through `KcmmKvWriteMirrorTracker(replace_native=...)`,
created in [`scripts/kcmm/launcher.py`](../../scripts/kcmm/launcher.py#L385).

### 3.2 Python write orchestration

The main function is
[`KcmmKvWriteMirrorTracker.mirror_call`](../../scripts/kcmm/kv_write_mirror.py#L631).

It receives the arguments from the patched vLLM custom op:

```text
key
value
key_cache
value_cache
slot_mapping
native_written
```

Its responsibilities:

| Step | Code point | Purpose |
|---|---|---|
| Validate dtype | [`_validate_dtype`](../../scripts/kcmm/kv_write_mirror.py#L521) | Require FP16 K/V |
| Determine batch | [`_slot_mapping_numel`](../../scripts/kcmm/kv_write_mirror.py#L363) or [`_slot_mapping_to_list`](../../scripts/kcmm/kv_write_mirror.py#L356) | Count rows to write |
| Determine layer | [`_layer_for_cache`](../../scripts/kcmm/kv_write_mirror.py#L325) | Map vLLM cache tensor pair to `layer_idx` |
| Prepare rows | [`_prepare_rows`](../../scripts/kcmm/kv_write_mirror.py#L527) | Convert K/V to `[batch, row_width]` |
| Check shape | [`_pool_shape`](../../scripts/kcmm/kv_write_mirror.py#L248) | Verify row width matches KCMM pool shape |
| Prepare device slot tensor | [`_prepare_device_slot_tensor`](../../scripts/kcmm/kv_write_mirror.py#L378) | Reuse CUDA `slot_mapping` when canonical |
| Build write tables | [`_device_slot_tables_for_device`](../../scripts/kcmm/kv_write_mirror.py#L422) | Build/cache offset and valid-block tables |
| Select stream | [`KcmmStreamProvider.select`](../../scripts/kcmm/streaming.py#L71) | Use current PyTorch/vLLM CUDA stream |
| Launch write | [`pool.append_kv_device_slots_on_stream`](../../scripts/kcmm/bindings.py#L690) | Call KCMM C ABI |

### 3.3 Device-slot write

Device-slot write keeps vLLM's `slot_mapping` on GPU.

```text
vLLM CUDA slot_mapping
  ↓ data_ptr()
KCMM C ABI
  ↓
CUDA kernel decodes slots on device
  ↓
KCMM VA write
```

Key code points:

| Code point | Responsibility |
|---|---|
| [`scripts/kcmm/kv_write_mirror.py::mirror_call`](../../scripts/kcmm/kv_write_mirror.py#L631) | Calls `pool.append_kv_device_slots_on_stream` |
| [`scripts/kcmm/bindings.py::KcmmPool.append_kv_device_slots_on_stream`](../../scripts/kcmm/bindings.py#L690) | Python ctypes wrapper |
| [`src/kcmm/ffi.rs::kcmm_append_kv_device_slots_on_stream`](../../src/kcmm/ffi.rs#L1399) | Rust C ABI and CUDA launch |
| [`src/cuda/kernels/kcmm_vllm_kv_write.cu::kcmm_vllm_kv_write_slots_f16`](../../src/cuda/kernels/kcmm_vllm_kv_write.cu#L8) | CUDA kernel that writes K/V into KCMM VA |

The CUDA kernel does:

```text
row = idx / step_elements
col = idx % step_elements

slot = slot_mapping[row]
if slot < 0:
  return  # padding slot

block_id        = slot / block_size
offset_in_block = slot % block_size

if block_id is out of range:
  status = 1
  return

if block is inactive:
  status = 2
  return

dst_idx =
  block_offsets_f16[block_id]
  + offset_in_block * step_elements
  + col

va_k[dst_idx] = k_src[row, col]
va_v[dst_idx] = v_src[row, col]
```

### 3.4 Host-slot fallback

The older path is still present:

| Code point | Responsibility |
|---|---|
| [`kv_write_mirror.py::_slot_mapping_to_list`](../../scripts/kcmm/kv_write_mirror.py#L356) | Copies CUDA `slot_mapping` to CPU list |
| [`bindings.py::KcmmPool.append_kv_slots`](../../scripts/kcmm/bindings.py#L660) | Python wrapper |
| [`src/kcmm/ffi.rs::kcmm_append_kv_slots_on_stream`](../../src/kcmm/ffi.rs#L1249) | C ABI using host-side slot list |

This path is useful for correctness and verification because Python can inspect
the slot list and run bounded D2H row verification. Performance-clean runs use
device-slot write instead.

## 4. KV read integration

### 4.1 Why read replacement is separate

Replacing write alone is insufficient.

If KCMM skips native `reshape_and_cache`, vLLM's native `key_cache/value_cache`
no longer hold the canonical K/V bytes. The attention read path must therefore
also be replaced.

```text
KV write replacement:
  key/value -> KCMM

KV read replacement:
  query + block_tables + seq_lens -> read KCMM K/V -> out
```

### 4.2 A2 offset table

KCMM uses the A2 plan:

```text
block_tables keep native vLLM block-id semantics
offset_table[block_id] = KCMM f16 VA offset
```

Key code points:

| Code point | Responsibility |
|---|---|
| [`scripts/kcmm/kv_read_plan.py::_offset_table_for_device`](../../scripts/kcmm/kv_read_plan.py#L320) | Builds/caches GPU offset table |
| [`scripts/kcmm/bindings.py::KcmmPool.all_block_offsets_f16`](../../scripts/kcmm/bindings.py#L563) | Gets `block_id -> f16 offset` from KCMM |
| [`scripts/kcmm/kv_read_plan.py::_build_plan`](../../scripts/kcmm/kv_read_plan.py#L355) | Parses vLLM read call and validates block ids |

### 4.3 Python read replacement

The main replacement entry is
[`KcmmKvReadOffsetTableTracker.replace_call`](../../scripts/kcmm/kv_read_plan.py#L586).

It:

1. Builds or reuses the A2 offset table.
2. Validates the read call shape and dtype.
3. Prepares compact `query`, `block_tables`, and `seq_lens` tensors.
4. Selects the current CUDA stream.
5. Calls the KCMM GPU read kernel.
6. Writes the result into vLLM's `out` tensor.

The CUDA-launch helper is
[`_run_gpu_kernel_attention`](../../scripts/kcmm/kv_read_plan.py#L783).

### 4.4 GPU read kernel

Key code points:

| Code point | Responsibility |
|---|---|
| [`scripts/kcmm/bindings.py::KcmmPool.paged_attn_decode_f16`](../../scripts/kcmm/bindings.py#L740) | Python wrapper for stream-aware read launch |
| [`scripts/kcmm/bindings.py::KcmmPool.paged_attn_decode_f16_on_current_context_stream`](../../scripts/kcmm/bindings.py#L784) | Fast current-context launch wrapper |
| [`src/kcmm/ffi.rs::kcmm_paged_attn_decode_f16_on_stream`](../../src/kcmm/ffi.rs#L1618) | C ABI for caller-owned CUDA stream |
| [`src/kcmm/ffi.rs::kcmm_paged_attn_decode_f16_on_current_context_stream`](../../src/kcmm/ffi.rs#L1670) | Current-context stream ABI |
| [`src/kcmm/ffi.rs::kcmm_paged_attn_decode_f16_impl`](../../src/kcmm/ffi.rs#L1710) | Shared Rust launch implementation |
| [`src/cuda/kernels/kcmm_vllm_paged_attn.cu::kcmm_vllm_paged_attn_decode_f16`](../../src/cuda/kernels/kcmm_vllm_paged_attn.cu#L10) | CUDA read/attention kernel |

The CUDA kernel does:

```text
idx -> (batch_idx, query_head)

for pos in 0..seq_len:
  logical_block   = pos / block_size
  offset_in_block = pos % block_size
  block_id        = block_tables[batch_idx, logical_block]
  block_offset    = block_offsets_f16[block_id]

  k = va_k[layer_idx] + block_offset + token_offset + kv_head_offset
  v = va_v[layer_idx] + block_offset + token_offset + kv_head_offset

  score = dot(query, k) * scale
  accumulate online softmax(score, v)

out[batch_idx, query_head, :] = attention result
```

The current kernel envelope is intentionally narrow and is documented in the
ADR: FP16 decode attention, supported head dimensions up to `256`, no alibi,
no block-sparse mode, and no FP8 cache scales.

## 5. End-to-end vLLM flow

The useful mental model is:

```text
ModelRunner
  ├─ builds input_ids
  ├─ builds positions
  ├─ builds attention_metadata
  │    ├─ slot_mapping  -> write location
  │    ├─ block_tables  -> read block ids
  │    └─ seq_lens      -> read lengths
  └─ builds sampling_metadata

Model forward
  └─ computes query/key/value

KV write
  ├─ native: reshape_and_cache(...)
  └─ KCMM: kcmm_append_kv_device_slots_on_stream(...)

KV read / attention
  ├─ native: paged_attention_v1/v2(...)
  └─ KCMM: kcmm_vllm_paged_attn_decode_f16(...)

Sampler
  └─ unchanged
```

KCMM does not replace:

| vLLM component | Status |
|---|---|
| Scheduler | unchanged |
| ModelRunner input construction | unchanged |
| Q/K/V projection math | unchanged |
| MLP layers | unchanged |
| Sampler | unchanged |
| Tokenizer/API server | unchanged |

KCMM replaces or mirrors:

| vLLM component | KCMM code path |
|---|---|
| KV allocation/block lifetime | [`shadow_allocator.py`](../../scripts/kcmm/shadow_allocator.py), [`backed_allocator.py`](../../scripts/kcmm/backed_allocator.py) |
| KV write | [`kv_write_mirror.py`](../../scripts/kcmm/kv_write_mirror.py), [`kcmm_vllm_kv_write.cu`](../../src/cuda/kernels/kcmm_vllm_kv_write.cu) |
| KV read | [`kv_read_plan.py`](../../scripts/kcmm/kv_read_plan.py), [`kcmm_vllm_paged_attn.cu`](../../src/cuda/kernels/kcmm_vllm_paged_attn.cu) |

## 6. Runtime flags worth knowing

Defined in [`scripts/kcmm/config.py`](../../scripts/kcmm/config.py#L480) and
used by [`scripts/kcmm/vllm_smoke.py`](../../scripts/kcmm/vllm_smoke.py#L112)
and the gate wrappers.

| Flag | Purpose |
|---|---|
| `--kcmm-kv-write-mirror` | Native write still runs; KCMM mirrors K/V |
| `--kcmm-kv-write-replace-candidate` | Skip native `reshape_and_cache`; write only KCMM |
| `--kcmm-kv-write-device-slots` | Use CUDA `slot_mapping` directly |
| `--no-kcmm-kv-write-verify` | Disable D2H row verification for performance-clean path |
| `--kcmm-kv-read-offset-table` | Build A2 offset table without replacing native kernel |
| `--kcmm-kv-read-replace-candidate` | Replace read using correctness reference path |
| `--kcmm-kv-read-gpu-kernel-candidate` | Replace read using KCMM GPU kernel |
| `--kcmm-kv-read-fast-current-context-launch` | Use current-context read launch ABI |
| `--kcmm-kv-read-precompile-gpu-kernel` | Precompile read kernel at pool attach |
| `--kcmm-tracker-host-profile` | Enable host section timing diagnostics |

## 7. Verification entrypoints

| Gate | Purpose |
|---|---|
| [`python -m scripts.kcmm.kv_write_ffi_smoke`](../../scripts/kcmm/kv_write_ffi_smoke.py#L166) | Low-level KV write ABI validation |
| [`python -m scripts.kcmm.non_default_stream_ffi_smoke`](../../scripts/kcmm/non_default_stream_ffi_smoke.py#L201) | Low-level stream-aware FFI validation |
| [`python -m scripts.kcmm.vllm_smoke`](../../scripts/kcmm/vllm_smoke.py#L1480) | Single vLLM smoke with selected KCMM flags |
| [`python -m scripts.kcmm.vllm_gpu_read_ab_gate`](../../scripts/kcmm/vllm_gpu_read_ab_gate.py#L1051) | Stock-vs-KCMM A/B gate |
| [`python -m scripts.kcmm.vllm_gpu_read_real_model_matrix_gate`](../../scripts/kcmm/vllm_gpu_read_real_model_matrix_gate.py#L240) | Real-model matrix coverage |
| [`python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate`](../../scripts/kcmm/vllm_gpu_read_perf_clean_gate.py#L934) | Request-level clean performance baseline |
| [`python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate`](../../scripts/kcmm/vllm_gpu_read_perf_clean_stress_gate.py#L245) | Concurrent clean performance baseline |
| [`python -m scripts.kcmm.vllm_gpu_read_host_profile_gate`](../../scripts/kcmm/vllm_gpu_read_host_profile_gate.py#L141) | Host-side section timing diagnostics |

## 8. Quick code-reading order

For KCMM memory model:

1. [`src/kcmm/pool.rs::KcmmPool`](../../src/kcmm/pool.rs#L132)
2. [`src/kcmm/pool.rs::KcmmPool::new`](../../src/kcmm/pool.rs#L199)
3. [`src/kcmm/ffi.rs::kcmm_pool_create`](../../src/kcmm/ffi.rs#L470)
4. [`scripts/kcmm/bindings.py::KcmmPool`](../../scripts/kcmm/bindings.py#L438)

For vLLM write integration:

1. [`scripts/kcmm/patch_vllm.py::KV_WRITE_FUNCTIONS`](../../scripts/kcmm/patch_vllm.py#L127)
2. [`scripts/kcmm/patch_vllm.py::_wrap_kv_write_mirror_function`](../../scripts/kcmm/patch_vllm.py#L900)
3. [`scripts/kcmm/kv_write_mirror.py::KcmmKvWriteMirrorTracker.mirror_call`](../../scripts/kcmm/kv_write_mirror.py#L631)
4. [`scripts/kcmm/bindings.py::KcmmPool.append_kv_device_slots_on_stream`](../../scripts/kcmm/bindings.py#L690)
5. [`src/kcmm/ffi.rs::kcmm_append_kv_device_slots_on_stream`](../../src/kcmm/ffi.rs#L1399)
6. [`src/cuda/kernels/kcmm_vllm_kv_write.cu::kcmm_vllm_kv_write_slots_f16`](../../src/cuda/kernels/kcmm_vllm_kv_write.cu#L8)

For vLLM read integration:

1. [`scripts/kcmm/patch_vllm.py::KV_READ_FUNCTIONS`](../../scripts/kcmm/patch_vllm.py#L141)
2. [`scripts/kcmm/patch_vllm.py::_wrap_kv_read_offset_table_function`](../../scripts/kcmm/patch_vllm.py#L876)
3. [`scripts/kcmm/kv_read_plan.py::KcmmKvReadOffsetTableTracker.replace_call`](../../scripts/kcmm/kv_read_plan.py#L586)
4. [`scripts/kcmm/kv_read_plan.py::_run_gpu_kernel_attention`](../../scripts/kcmm/kv_read_plan.py#L783)
5. [`scripts/kcmm/bindings.py::KcmmPool.paged_attn_decode_f16`](../../scripts/kcmm/bindings.py#L740)
6. [`src/kcmm/ffi.rs::kcmm_paged_attn_decode_f16_impl`](../../src/kcmm/ffi.rs#L1710)
7. [`src/cuda/kernels/kcmm_vllm_paged_attn.cu::kcmm_vllm_paged_attn_decode_f16`](../../src/cuda/kernels/kcmm_vllm_paged_attn.cu#L10)
