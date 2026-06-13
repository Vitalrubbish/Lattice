# ADR 0001: vLLM Integration Architecture

Lattice's OS-layer vision requires demonstrating KCMM accelerates external inference
engines, not just our own Rust engine. vLLM is the primary target. We integrate via
a **monkey-patch launcher script** that intercepts vLLM's block allocator and
attention paths at well-defined points, without modifying vLLM source.

Status: **accepted** (2026-06-13)

## Core decisions

1. **Target vLLM 0.6.3.post1** — `BlockSpaceManagerV2` is default (block-level prefix
   caching baseline), `--enforce-eager` is stable, Python-side block allocator path
   is still accessible for monkey-patching, FlashInfer has pre-built sm_80 wheels
   for A30.
2. **Wrapper script, not source patch** — `scripts/kcmm/launcher.py` applies patches
   at import time, keeps vLLM installation pristine for A/B comparison.
3. **Python bindings in `scripts/kcmm/`** via `ctypes` — zero dependencies, ~20 C
   function signatures all simple scalars/pointers, no need for cffi or pyo3.
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
| 3 | block_id → GPU VA | `block_tables` tensor → attention kernel | `kcmm_get_all_block_offsets_f16()` remap table (A1: base=0) |
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
  ├─ KCMM pool with tiering=OFF, pre-allocate equivalent to vLLM num_gpu_blocks
  ├─ Keep reshape_and_cache + block_tables native — only the allocator changes
  └─ Gate: throughput deviation < 5% vs stock vLLM on fragmentation benchmark

Phase II-B — KV write path (intercept 2)
  ├─ Replace reshape_and_cache with kcmm_append_kv_step (D2D copy per token)
  └─ Gate: D2H read-back byte-level K/V comparison vs reference computation

Phase II-C — KV read path (intercept 3, A1 approach)
  ├─ Replace block_tables values with KCMM VA offsets, set attention kernel base=0
  └─ Gate: token-exact match vs stock vLLM (same prompt → same completion)

Phase III  — Tiering (intercepts 4, 6, 7)
  ├─ Enable tiering, add hint API calls, expose UFS metrics
  └─ Gate: capacity ratio ≥ 1.2× vs stock vLLM under memory pressure
```

## Key technical decisions

### Why `--enforce-eager` is the linchpin

vLLM with CUDA graphs bakes block table addresses into compiled graphs — interception
is impossible. `--enforce-eager` disables graphs and FlasHinfer fused attention,
making every step go through the Python scheduler → model runner → attention kernel
path individually. This gives us a clean insertion point for intercept 3.

### Why `kcmm_append_kv_step` over slot_mapping modification (intercept 2)

`reshape_and_cache` is a compiled CUDA kernel. Modifying `key_cache`/`value_cache`
tensors to point at KCMM VA regions requires PyTorch to accept `cuMemMap`-backed
memory (it doesn't — PyTorch's allocator tracks `cudaMalloc` allocations).
Modifying `slot_mapping` values depends on kernel-internal address computation that
varies across vLLM versions. `kcmm_append_kv_step` bypasses both by doing D2D copies
directly via CUDA driver API on the default stream.

### Why A1 over A2 for VA remapping (intercept 3)

A1: replace `block_tables` values with f16-unit VA offsets, set kernel `kv_cache_base=0`.
A2: maintain a separate GPU-side offset table indexed by block_id, replace all
Python-side `block_tables` reads with KCMM offset queries.

A1 is simpler to prototype — it reuses the existing `block_tables` tensor channel
and only changes what values flow through it. If kernel-internal `base + block_id *
stride` assumptions break A1, we fall back to A2.

### CUDA context sharing risk

KCMM and PyTorch share the same primary CUDA context per device. `cuMemAddressReserve`
by KCMM could conflict with `cudaMalloc` by PyTorch in the same VA space. Phase I.C
tests this explicitly: if V2 fails (allocation after PyTorch init), the conflict
is real and we need separate VA reservation strategies.

## Considered options

**vLLM native plugin (rejected for now).** vLLM 0.8+ is working toward a stable
`BlockAllocator` plugin interface but it's not production-ready. We chose 0.6.3
monkey-patch for immediate progress, with a clear upgrade path when the plugin
interface stabilizes.

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

- **A30 GPU (Ampere, sm_80)** — FlashInfer 0.2.x compatible, 24 GB VRAM
- **CUDA 12.x** — required by both cudarc (Rust) and vLLM 0.6.3
- **No modification to vLLM installation** — reproducible A/B comparison
- **`--enforce-eager` always on** — non-negotiable for interception
- **Prefix sharing (intercept 5) deferred to Step 4** — KCMM SharingManager is
  placeholder, vLLM APC serves as comparison baseline in the meantime
