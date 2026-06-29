# ADR 0001: vLLM Integration Architecture

Lattice's OS-layer vision requires demonstrating KCMM accelerates external inference
engines, not just our own Rust engine. vLLM is the primary target. We integrate via
a **monkey-patch launcher script** that intercepts vLLM's block allocator and
attention paths at well-defined points, without modifying vLLM source.

Status: **accepted for Phase I.C observer; provisional for Phase II/III**
(accepted 2026-06-13, narrowed 2026-06-18 after implementation review)

## Scope of acceptance

This ADR accepts the near-term goal of proving KCMM can coexist with a vLLM
process without changing vLLM behavior. It does **not** yet accept the full
allocator/write/read/tiering replacement as a stable architecture.

The current repository has a working KCMM path in the Rust engine plus a
Phase I.C vLLM observer launcher and allocator-seam instrumentation. The full
allocator/write/read/tiering replacement is still not accepted as stable
architecture. The vLLM integration phases below are therefore validation gates,
not implementation claims.

## Core decisions

1. **Target vLLM 0.6.1.post1+cu118 for the current Phase II.A branch** —
   the local host driver supports CUDA 11.8 wheels, and this version keeps the
   Python-side V2 block allocator path accessible for monkey-patching. It must
   run with `--use-v2-block-manager` because V2 is not the default in vLLM
   0.6.1. vLLM 0.6.3.post1 remains a future re-evaluation target after a host
   driver/toolkit upgrade, not the target for this branch.
2. **Wrapper script, not source patch** — `scripts/kcmm/launcher.py` applies
   patches at import time, keeping the vLLM installation pristine for A/B
   comparison.
3. **Python bindings in `scripts/kcmm/`** via `ctypes` — zero dependencies, simple
   C function signatures, no need for cffi or pyo3. This requires ABI layout
   tests for every exported struct before Phase I.C can be considered complete.
4. **Per-process lazy singleton** for KcmmPool — `get_or_create_pool()` called by
   the first interception point, reused across all patches within the process.
   Correct for TP=1 (current), correct per-worker for TP>1 (future).
5. **CLI passthrough mode** — launcher accepts all `vllm serve` flags transparently
   plus `--kcmm-*` extensions, forwards to `vllm.entrypoints.openai.api_server`.

## Interception points

| # | Intercepts | vLLM component | Replaced by |
|---|---|---|---|
| 1 | Block alloc/free | `NaiveBlockAllocator` | `kcmm_alloc_blocks` / `kcmm_free_blocks` |
| 2 | Block table management | `BlockSpaceManager` | `kcmm_register_sequence` / `kcmm_append_block_to_sequence` |
| 3 | block_id → GPU VA | `block_tables` tensor → attention kernel | Experimental: A1 (`block_tables` as offsets, base=0) or A2/custom attention backend |
| 4 | Swap / eviction | `BlockSpaceManager.swap_out/in` | `kcmm_cool`/`touch` + `kcmm_evict_blocks` / `kcmm_restore_evicted_blocks` |
| 5 | Prefix caching | `PrefixCachingBlockAllocator` | `kcmm_share_prefix` (Step 4) |
| 6 | Hint API | (none — new capability) | `kcmm_hint` / `kcmm_protect` |
| 7 | Metrics | `/metrics` endpoint | `kcmm_get_metrics` / `kcmm_get_pool_stats` |
| 8 | GPU memory init | `torch.cuda.malloc` pre-allocation | `kcmm_pool_create` (grow-on-demand) |

## Phased implementation

```
Phase I.C  — Observer (no behavior change)
  ├─ V1: pool create + destroy in vLLM process
  ├─ V2: allocate one real GPU block (cuMemCreate + cuMemMap)
  ├─ V3: idle pool during vLLM serving, periodic metrics sampling
  └─ Gate: all three pass → Phase II-A

Phase II-A — Allocator replacement (intercepts 1, 8)
  ├─ Target: vLLM 0.6.1.post1+cu118 with explicit V2 block manager
  ├─ KCMM pool with tiering=OFF, capacity equivalent to vLLM num_gpu_blocks
  ├─ Storage of record: native vLLM KV tensors remain canonical in Phase II.A
  ├─ Shadow allocator mirrors vLLM block lifetimes into KCMM without behavior change
  ├─ Optional KCMM-backed allocator is gated behind an explicit opt-in flag
  └─ Gate: stock/observer/shadow/KCMM-backed A/B smoke report before Phase II-B

Phase II-B — KV write path (intercept 2)
  ├─ Preflight: bind and gate kcmm_append_kv_step from Python with D2H read-back
  ├─ Bind and gate kcmm_append_kv_slots for reshape_and_cache slot_mapping writes
  ├─ Shadow mirror reshape_and_cache into KCMM behind KCMM-backed allocation mode
  ├─ Replacement-candidate mode skips native writes and writes only to KCMM
  ├─ Replace reshape_and_cache with stream-aware direct-slot KCMM writes
  ├─ D2D copy must run on the current PyTorch/vLLM CUDA stream or synchronize by event
  └─ Gate: D2H read-back byte-level K/V comparison vs reference computation

Phase II-C — KV read path (intercept 3)
  ├─ Trace paged_attention_v1/v2 block_tables and KV cache tensor contract
  ├─ A1 decision: Python custom-op seam expects block ids plus native KV tensor base
  ├─ Prototype A2 block_id → KCMM f16 VA offset side table
  ├─ Replace native read kernel with KCMM reference attention candidate
  ├─ Replace CPU-staged reference path with CUDA kernel/compiled extension candidate
  └─ Gate: token-exact match vs stock vLLM (same prompt → same completion)

Phase III  — Tiering (intercepts 4, 6, 7)
  ├─ Enable tiering, add hint API calls, expose UFS metrics
  └─ Gate: capacity ratio ≥ 1.2×, P99 overhead bounded, evictions/full-completion ≤ 3.0
```

## Key technical decisions

### Phase II.A target and storage of record

Phase II.A is fixed to the locally verified CUDA 11.8 stack:

- vLLM `0.6.1.post1+cu118`
- PyTorch `2.4.0+cu118`
- xFormers `0.0.27.post2+cu118`
- transformers `4.45.2`
- tokenizers `0.20.3`
- huggingface-hub `0.36.2`

The required vLLM runtime flags are:

- `--use-v2-block-manager`
- `--enforce-eager`
- `--disable-frontend-multiprocessing` when Python allocator instrumentation or
  replacement must run in the same process as the engine

The Phase II.A storage-of-record model is staged:

1. Native vLLM KV cache tensors remain the canonical storage for KV bytes.
2. KCMM may size a pool from vLLM runtime cache configuration.
3. KCMM may mirror allocation/free lifetimes as a shadow allocator.
4. KCMM may attempt allocator-backed block ownership only behind an explicit
   opt-in flag and only if vLLM's native write/read path still addresses valid
   native KV tensor storage.
5. KCMM VA does not become the canonical KV storage until Phase II.B/II.C replace
   the write and read paths.

Phase II.A must stop rather than silently continue if allocator-only replacement
requires KCMM VA to become the true KV storage. The stop report must identify the
vLLM invariant being violated, such as a required contiguous native KV tensor
layout, block-id-to-offset arithmetic inside compiled kernels, or a write/read
path that cannot address KCMM-managed memory without Phase II.B/II.C changes.

Phase II.A is complete only after these local checks pass:

- stock vLLM smoke completion
- KCMM observer smoke completion
- allocator-seam instrumentation smoke completion with required seams observed
- KCMM runtime-derived pool sizing smoke completion
- KCMM shadow allocator smoke completion with no leaked or mismatched blocks
- KCMM-backed allocator smoke completion, or a documented fail-closed stop
  condition showing why Phase II.B/II.C is required first
- a single A/B report comparing stock, observer, shadow, and enabled
  KCMM-backed modes on the same tiny local model and prompt

The A/B gate is operationalized by `python -m scripts.kcmm.vllm_ab_gate`.
Phase II.B must not start until this command produces `passed: true` for the
branch and local environment being promoted.

### Why `--enforce-eager` is the linchpin

vLLM with CUDA graphs can bake attention metadata and addresses into captured
execution. `--enforce-eager` disables CUDA graphs and keeps execution step-wise,
which is the safest mode for Phase I.C and early interception experiments.
Attention backend selection must still be pinned and verified per vLLM version.

### Why direct KCMM D2D writes over slot_mapping modification (intercept 2)

`reshape_and_cache` is a compiled CUDA kernel. Modifying `key_cache`/`value_cache`
tensors to point at KCMM VA regions requires PyTorch to accept `cuMemMap`-backed
memory as tensor storage. Modifying `slot_mapping` values depends on
kernel-internal address computation that varies across vLLM versions.
KCMM D2D write APIs bypass both by copying into KCMM-managed VA directly via the
CUDA driver API, but they must be stream-aware before they are safe inside a
PyTorch/vLLM forward pass.

Before patching vLLM, `python -m scripts.kcmm.kv_write_ffi_smoke` must pass. This
preflight gate proves the Python launcher can call the sequence-position writer
`kcmm_append_kv_step`, the physical-slot writer `kcmm_append_kv_slots`, and the
stream-aware physical-slot writer `kcmm_append_kv_slots_on_stream`; read back
KCMM VA bytes; and detect K/V mismatches independently of vLLM scheduling. Then
`python -m scripts.kcmm.vllm_smoke --instrument-kv-writes` must pass to record
the version-pinned `reshape_and_cache` tensor contract that the replacement must
preserve. The trace decodes `slot_mapping` as
`block_id = slot // block_size` and `offset_in_block = slot % block_size`,
which means the `reshape_and_cache` seam exposes physical KV slots but not
sequence ids.

The chosen replacement path at this seam is therefore the direct-slot writer
`kcmm_append_kv_slots_on_stream(layer_idx, slot_mapping, batch, k_src, v_src,
stream_ptr)`, where `stream_ptr` is PyTorch's current CUDA stream for the tensor
device. The older `kcmm_append_kv_slots` ABI remains available for low-level
tests and compatibility. `kcmm_append_kv_step` remains useful for lower-level
sequence/position tests and future metadata-aware integration points, but it is
not the direct replacement for vLLM `reshape_and_cache` unless a separate
metadata-builder patch restores sequence/position context.

The first vLLM-integrated slice is a shadow mirror, not a replacement: native
vLLM `reshape_and_cache` still writes the canonical KV tensors, then KCMM mirrors
the same write through `kcmm_append_kv_slots_on_stream` and verifies KCMM bytes
by D2H read-back. This mirror mode requires KCMM-backed allocation mode so vLLM
physical block ids and KCMM block ids are the same ids. Allocator shadow mode
is not sufficient for direct-slot mirroring because its KCMM block ids are
separate from vLLM's native block ids.

The next write-path slice is a replacement candidate behind a stronger opt-in
flag: native `reshape_and_cache` is skipped and KCMM is the only write target.
This validates the Phase II.B seam but still does not establish end-to-end
correctness, because vLLM attention reads continue to use native KV tensors
until Phase II.C replaces the read path.

The vLLM-integrated write replacement path now enqueues KCMM D2D writes on the
current PyTorch CUDA stream and returns without full-device synchronization.
D2H verification in smoke tests still synchronizes that stream before reading
KCMM bytes back to host.

### Why A1 is not valid at the vLLM Python custom-op seam (intercept 3)

A1: replace `block_tables` values with f16-unit VA offsets, set kernel `kv_cache_base=0`.
A2: maintain a separate GPU-side offset table indexed by block_id, replace all
Python-side `block_tables` reads with KCMM offset queries.

A1 was the simpler hypothesis because it reused the existing `block_tables`
tensor channel. Phase II.C read instrumentation rejects A1 at the Python
custom-op seam for vLLM `0.6.1.post1+cu118`: `paged_attention_v1` receives
native `key_cache`/`value_cache` tensors plus `torch.int32` `block_tables`
entries whose observed semantics are physical KV block ids. Replacing those
entries with KCMM VA offsets would exceed the KV cache block-id range while the
kernel still receives the native tensor base. Phase II.C must therefore continue
with A2 or a custom attention backend that explicitly resolves block ids to KCMM
addresses.

### A2 prototype boundary for the vLLM Python custom-op seam

A2 keeps `block_tables` as the native vLLM physical block-id table and adds a
separate side table:

```text
offset_table[block_id] = kcmm_f16_va_offset
```

The first A2 prototype builds this side table at the
`vllm._custom_ops.paged_attention_v1/v2` seam under the KCMM-backed allocator.
That allocator mode is required because it guarantees that vLLM block ids and
KCMM block ids are identical. The prototype validates that every observed
`block_tables` entry exists in KCMM and materializes a CUDA
`torch.int64[f16_va_offset_by_block_id]` tensor.

This is still not end-to-end read replacement: `kernel_replaced=false`, and the
native vLLM paged-attention kernel remains the read path. The next Phase II.C
step must introduce a custom attention backend or kernel entrypoint that
consumes the KCMM K/V base addresses plus this offset table instead of using
the native vLLM `key_cache`/`value_cache` tensor storage.

### Reference read replacement candidate

The first read replacement candidate is correctness-oriented rather than
performance-oriented. Under the KCMM-backed allocator and KCMM KV write path, it
patches `paged_attention_v1/v2`, reconstructs the K/V sequence from
`block_tables` and `seq_lens`, reads KCMM K/V rows via CUDA D2H copies, computes
scaled dot-product attention with PyTorch, writes the result into the provided
`out` tensor, and returns without calling the native vLLM paged-attention
kernel.

This establishes the storage-of-record transition for the tiny vLLM smoke:
native `reshape_and_cache` writes can be skipped, native paged-attention reads
can be skipped, and the same-model stock-vs-KCMM A/B run produced identical
completion text. It is not an acceptable performance implementation because it
uses CPU staging and Python loops. The next Phase II.C step is to replace this
reference path with a CUDA kernel or compiled extension that consumes the KCMM
K/V bases and the A2 offset table directly on GPU.

### GPU read kernel candidate

The GPU read candidate keeps the same vLLM Python custom-op seam and A2 offset
table contract, but replaces the CPU-staged reference attention loop with the
KCMM C ABI entrypoint `kcmm_paged_attn_decode_f16`. The launcher passes raw CUDA
VAs for vLLM-owned `query`, `out`, `block_tables`, and `seq_lens` tensors,
plus the KCMM K/V base VAs and the GPU `offset_table[block_id] =
kcmm_f16_va_offset` side table. Rust compiles the CUDA source with NVRTC,
caches the resulting function per pool, and returns without calling the native
vLLM paged-attention kernel.

The current kernel is intentionally narrow: FP16 decode attention only,
`head_dim <= 256`, no alibi, no block-sparse mode, and no FP8 cache scales. It
proves that vLLM can run with KCMM as the only KV write target and a GPU-side
KCMM read path on the tiny local OPT smoke, but it does not yet satisfy the
final Phase II.C acceptance gate.

The first deterministic stock-vs-KCMM GPU read-kernel A/B gate is
`python -m scripts.kcmm.vllm_gpu_read_ab_gate`. It generates the tiny local OPT
model with a fixed default seed when the model is absent, runs stock vLLM and
the KCMM-backed write-replacement plus GPU read-kernel path against the same
model directory, and compares completion text, finish reason, and token counts
for every configured coverage case. The local tiny-model gate has passed with
the default `hello`, `math`, and `long_context` cases. It also records startup
latency, request latency, generated-token throughput, peak GPU memory delta,
and warning classifications for KCMM-vs-stock regressions.

For kernel optimization work, the GPU read path also has an opt-in per-call
profiler. `--kcmm-kv-read-profile` records CUDA events on the same stream passed
to `kcmm_paged_attn_decode_f16_on_stream`, and
`python -m scripts.kcmm.vllm_gpu_read_profile_gate` wraps the A/B gate with that
profiling enabled only for the KCMM mode. The resulting read report includes
per-call `gpu_kernel_elapsed_ms` values and a `gpu_kernel_profile` summary with
count, min, avg, p50, p95, p99, max, `first_call_ms`, raw samples, and a
`steady_state` summary that excludes the first sample when multiple samples are
available. Profiling is intentionally disabled by default because event timing
synchronizes the end event and is diagnostic rather than the normal correctness
path.

`python -m scripts.kcmm.vllm_gpu_read_shape_gate` now broadens that gate across
six tiny OPT shape variants inside the currently supported local envelope:
`head64_layers2`, `head80_layers2`, `head96_layers2`, `head128_layers2`,
`head192_layers2`, and `head256_layers2`. This CUDA 11.8 vLLM/XFormers stack
supports paged-attention head sizes `64`, `80`, `96`, `112`, `120`, `128`,
`192`, and `256`; the current KCMM GPU read kernel and FFI guard cover this
full local vLLM-supported set. Non-divisible per-layer logical block sizes,
such as the `head_dim=80`, `96`, and `192` shape-gate variants, are handled by
allocating only full logical blocks from each 2 MiB superblock and leaving the
superblock tail as padding. The shape gate keeps a single-token long-context
case so multi-block decode reads are covered without recursively amplifying
normal FP16 paged-attention rounding differences across several generated
tokens.

`python -m scripts.kcmm.vllm_gpu_read_batch_gate` now adds a batch/concurrency
gate. It runs two concurrent completion requests with `max_num_seqs=2`,
requires the KCMM read seam to observe `max_read_batch_seen` of at least `2`,
and compares deterministic stock-vs-KCMM outputs. The local 8-token concurrent
gate passes. The first long-concurrency failure was caused by vLLM passing
`query` as a non-contiguous fused-QKV projection view; the replacement now
materializes compact inputs on the current PyTorch CUDA stream before launching
the stream-aware KCMM kernel on that same stream. The remaining work before
treating this as a stable read path is framework-originated non-default-stream
scheduling revalidation if vLLM starts invoking the patched seams from
non-default current streams, broader real-model coverage beyond the first
OPT-125m gate, broader TP coverage, and performance optimization.

`python -m scripts.kcmm.vllm_gpu_read_real_model_gate` added the first
non-generated model gate with `facebook/opt-125m`. The broader
`python -m scripts.kcmm.vllm_gpu_read_real_model_matrix_gate` now covers both
`facebook/opt-125m` and `distilgpt2`, including a longer prompt that spans
multiple KV blocks. The local matrix run compared deterministic stock-vs-KCMM
completions for every model and verified the KCMM path used the stream-aware GPU
read kernel with zero CPU-staged reference read bytes. This moves Phase II.C
beyond generated tiny OPT models, but the matrix is still local single-GPU
coverage inside the current kernel envelope.

`python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate` is the request-level
baseline for performance work after correctness has been proven. It runs the
same stock-vs-KCMM real-model comparison, but disables test-only KV read trace
instrumentation, disables KV write D2H row verification, and leaves read-kernel
profiling off. It also disables per-update tracker report writes and relies on
the final process-exit reports for validation. After correctness coverage has
passed, it disables host-side read block-table validation and caches the GPU
`offset_table[block_id] = kcmm_f16_va_offset` tensor across read seams, rebuilding
only when the KCMM block count grows or the target device changes. The gate still
requires token-exact stock-vs-KCMM output, GPU read-kernel calls, zero
CPU-staged reference read bytes, and zero write verification rows/synchronizations.
It is cleaner than the correctness gates, but it still includes vLLM server,
Python monkey-patch, and scheduling overhead.

`python -m scripts.kcmm.vllm_gpu_read_tensor_parallel_gate` now covers the
local tensor-parallel case with `tensor_parallel_size=2` on the dual RTX 3080
machine. vLLM TP worker subprocesses inherit the KCMM monkey patches but do not
run the driver process's `LLMEngine.__init__` runtime-pool callback, so the
launcher also patches `Worker.initialize_cache` to attach a worker-local KCMM
pool before model execution. TP workers receive driver-scheduler slot mappings;
the write replacement therefore lazily ensures local KCMM block IDs from
`slot_mapping` before appending KV rows. The local TP gate passes with matching
stock-vs-KCMM completions, stream-aware GPU read calls, and zero CPU-staged
reference read bytes.

The vLLM-integrated GPU read path now uses the stream-aware C ABI
`kcmm_paged_attn_decode_f16_on_stream`, passing PyTorch's current CUDA stream
handle from the patched read seam and returning without synchronizing the whole
CUDA context. The old `kcmm_paged_attn_decode_f16` remains as a synchronous
compatibility wrapper. The write replacement path likewise uses
`kcmm_append_kv_slots_on_stream`. On the current local eager vLLM seam both
patched write and read paths report stream handle `0`, the legacy default
stream.

The low-level FFI gate
`python -m scripts.kcmm.non_default_stream_ffi_smoke` now covers the
non-default-stream `_on_stream` behavior independently of vLLM scheduling. It
creates a real `torch.cuda.Stream()` with a non-zero handle, enqueues
`kcmm_append_kv_slots_on_stream` and `kcmm_paged_attn_decode_f16_on_stream` on
that same stream, synchronizes only that stream for verification, and confirms
the decoded output matches the just-written V row.

The vLLM-integrated gate
`python -m scripts.kcmm.vllm_gpu_read_non_default_stream_gate` now forces KCMM
write and read replacement launches onto dedicated non-default streams while
preserving ordering through the original PyTorch stream with `wait_stream` and
preserving temporary tensor lifetimes with `record_stream`. This covers the
monkey-patched integration path when KCMM work is not launched on stream `0`.
It does not claim the current vLLM eager scheduler naturally invokes the seams
from non-default current streams. The remaining Phase II.C work is tensor
parallel coverage beyond the local two-GPU gate, framework-originated
non-default stream revalidation if vLLM changes scheduling behavior, broader
real-model/workload coverage beyond the local OPT-125m/distilgpt2 matrix, and
performance optimization using the per-call profiling data beyond the local tiny
OPT and first real-model gates.

### CUDA context sharing risk

KCMM and PyTorch share the same primary CUDA context per device. `cuMemAddressReserve`
by KCMM could conflict with `cudaMalloc` by PyTorch in the same VA space. Phase I.C
tests this explicitly: if V2 fails (allocation after PyTorch init), the conflict
is real and we need separate VA reservation strategies.

## Considered options

**vLLM native plugin (rejected for now).** vLLM 0.8+ is working toward a stable
`BlockAllocator` plugin interface but it's not production-ready. We chose the
0.6.x Python monkey-patch path for immediate progress, with a clear upgrade path
when the plugin interface stabilizes. The current Phase II.A branch is pinned to
0.6.1.post1+cu118 because that is the locally verified CUDA 11.8 wheel.

**vLLM V1/plugin/KV-offload integration (deferred).** Current vLLM has moved
toward V1, plugin hooks, hybrid KV cache management, and native KV offloading.
Those seams may be better long-term targets than monkey-patching v0.6.x, but
they are not the fastest path to a controlled Phase I.C coexistence test.

**Modifying vLLM source directly (rejected).** Would complicate A/B comparison,
reproducibility, and version upgrades. Wrapper script keeps the patch surface
explicit and auditable.

**Single global pool across TP workers (rejected).** CUDA contexts are per-process;
a pool handle from `kcmm_pool_create` can't cross process boundaries. Per-process
lazy singleton is the correct granularity.

**Merging Phase I+II into one step (rejected).** Physical VA layout differences
between KCMM superblocks and vLLM's contiguous pool mean the allocator and attention
paths must change together for correctness. Separating allocator (II-A) from
attention (II-B/II-C) lets us isolate bugs — but observer-only Phase I.C comes
first to validate the CUDA context foundation without touching the forward pass.

## Constraints

- **Local RTX 3080 GPUs (Ampere, sm_86)** — current Phase II.A validation host.
- **CUDA 11.8 wheel target for the current host** — the local NVIDIA 515.48.07
  driver cannot run current CUDA 12.x vLLM wheels, so Phase II.A uses the
  verified cu118 stack until a host-driver upgrade changes the baseline.
- **No modification to vLLM installation** — reproducible A/B comparison
- **`--enforce-eager` always on** — non-negotiable for interception
- **Prefix sharing (intercept 5) deferred to Step 4** — KCMM SharingManager is
  placeholder, vLLM APC serves as comparison baseline in the meantime
