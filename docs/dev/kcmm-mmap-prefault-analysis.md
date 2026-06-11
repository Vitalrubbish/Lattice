# KCMM MAP_POPULATE and madvise Prefault Analysis

**Date:** 2026-06-10
**Status:** Complete
**Related:** `docs/dev/kcmm-eviction-latency-wsl2-vs-baremetal.md`, `docs/cr/kcmm-phase-e-batch-eviction-issue.md`
**Source commit:** `0604f5e` — `perf(kcmm): reduce eviction overhead with ptrs_dev reuse and mmap prefault`
**Files implicated:** `src/kcmm/tiering.rs` (L392–L481)

---

## 1. Background

KCMM implements a three-tier storage hierarchy: **GPU HBM → CPU DRAM → NVMe SSD**. The tiering engine (`TieringEngine`) moves KV-cache blocks between tiers. When GPU memory is exhausted, blocks are evicted to a **CPU swap buffer** — a file-backed `mmap` region at `config.cpu_cache_path`. Conversely, when evicted blocks are needed again, they are restored from the CPU swap buffer back to GPU memory.

During **batched eviction** and **batched restoration**, the CPU scatter/gather phases perform linear reads and writes across the mmap buffer:

- **CPU scatter (eviction):** contiguous GPU staging buffer → per-block CPU slots (write)
- **CPU gather (restoration):** per-block CPU slots → contiguous CPU staging buffer (read)

Without pre-faulting, each first access to a virtual page triggers a **demand page fault**, adding substantial overhead to these phases.

---

## 2. Mechanism Breakdown

The optimization consists of three cooperative Linux memory-management primitives applied at engine construction time (`TieringEngine::new()`, [tiering.rs:L400-L427](src/kcmm/tiering.rs#L400-L427)).

### 2.1 MAP_POPULATE

```rust
let ptr = unsafe {
    libc::mmap(
        std::ptr::null_mut(),
        cpu_buffer_size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED | libc::MAP_POPULATE,  // ← key flag
        file.as_raw_fd(),
        0,
    )
};
```

| Flag | Purpose |
|------|---------|
| `MAP_SHARED` | Writes propagate to the backing file; enables cross-process sharing and persistence across engine restarts |
| `MAP_POPULATE` | Instructs the kernel to **eagerly fault in** all page-table entries (PTEs) before `mmap()` returns, rather than lazily on first access |

**Kernel behavior:** With `MAP_POPULATE`, the kernel synchronously walks every virtual page during the `mmap` syscall, triggers disk I/O for file-backed pages (populating the page cache), and establishes PTEs up front. The mmap call itself becomes slower, but subsequent first access to the mapped region is zero-cost — no page faults occur.

### 2.2 madvise — MADV_WILLNEED

```rust
libc::madvise(ptr, cpu_buffer_size, libc::MADV_WILLNEED);
```

| Advice | Semantics |
|--------|-----------|
| `MADV_WILLNEED` | "I will need these pages soon" — triggers **asynchronous readahead**, pulling pages into the page cache in the background without blocking the caller |

This is a stronger signal than `MAP_POPULATE` alone: while `MAP_POPULATE` faults pages synchronously, `MADV_WILLNEED` initiates background readahead that may pull in additional pages beyond the immediate range, filling the page cache ahead of the actual access pattern.

### 2.3 madvise — MADV_SEQUENTIAL

```rust
libc::madvise(ptr, cpu_buffer_size, libc::MADV_SEQUENTIAL);
```

| Advice | Semantics |
|--------|-----------|
| `MADV_SEQUENTIAL` | "I will access these pages in order" — hints the kernel to use **aggressive readahead** and **lower the eviction priority** of already-accessed pages |

This optimizes the kernel's readahead strategy for a linear-scan pattern. Pages earlier in the range are reclaimed from the page cache sooner after they have been consumed, leaving more space for prefetching pages ahead of the scan cursor.

### 2.4 Combined Effect

The three primitives cooperate in a layered fashion:

```
                    ┌─────────────────────────────────┐
                    │         mmap(MAP_POPULATE)       │
                    │  Synchronous: fault all PTEs now │
                    │  Cost: paid once at setup time   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      madvise(MADV_WILLNEED)      │
                    │  Asynchronous: background pre-   │
                    │  read beyond the immediate range │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     madvise(MADV_SEQUENTIAL)     │
                    │  Policy: aggressive readahead +  │
                    │  early reclaim behind cursor     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   CPU scatter/gather at runtime  │
                    │  All PTEs valid → zero page      │
                    │  faults during the hot loop      │
                    └─────────────────────────────────┘
```

---

## 3. Performance Impact

### 3.1 Before Optimization

During the CPU scatter/gather phases, every access to a new page in the mmap buffer triggered a demand page fault:

```
CPU scatter phase: each store to a new page
  → page fault (minor, file-backed)
  → kernel allocates frame + updates PTE
  → write continues
  Per-page cost: ~2–5 µs (varies by CPU and kernel version)
  4 MiB buffer (64 × 64 KiB blocks): ~64 pages × 4 µs ≈ ~256 µs
  Larger buffers: ~8 MiB → ~128 pages × 4 µs ≈ ~512 µs
  Large-scale eviction batches: ~4 ms per batch (as noted in the commit message)
```

### 3.2 After Optimization

```
mmap() time: one-time cost of faulting all pages (offline, at setup)
CPU scatter/gather phases: all PTEs already valid → direct memory access, zero faults
Per-batch saving: ~4 ms
```

### 3.3 Trade-off

| Aspect | Before | After |
|--------|--------|-------|
| `TieringEngine::new()` | Fast (lazy faults) | Slower (all pages faulted now) |
| Each eviction batch | ~4 ms fault overhead | ~0 ms fault overhead |
| Each restoration batch | ~4 ms fault overhead | ~0 ms fault overhead |
| Frequency of `new()` | Once per engine lifetime | Once per engine lifetime |
| Frequency of evict/restore | Steady-state, high frequency | Steady-state, high frequency |

Since `new()` is called once at initialization while eviction and restoration are steady-state operations, moving the cost from the hot path to setup is a clear win.

---

## 4. Relationship to ptrs_dev Reuse

The same commit (`0604f5e`) also introduced a second optimization: reusing a single device-side pointer array (`ptrs_dev`) across all 44 layer×KV iterations in `evict_blocks_batched` and `restore_blocks_batched`, instead of allocating a new 64-byte buffer per iteration. This is safe because legacy default-stream ordering guarantees the previous layer's kernel has consumed its data before the next H2D write overwrites the buffer.

Together, these two optimizations address **different stages** in the batched eviction/restore pipeline:

| Stage | Before | After | Saving |
|-------|--------|-------|--------|
| GPU gather — pointer array allocation (evict) | 44 `cuMemAlloc_v2` calls | 1 `cuMemAlloc_v2` + reuse | ~43 driver calls |
| CPU scatter — write mmap buffer (evict) | ~4 ms from page faults | ~0 ms (pre-faulted) | ~4 ms |
| CPU gather — read mmap buffer (restore) | ~4 ms from page faults | ~0 ms (pre-faulted) | ~4 ms |
| GPU scatter — pointer array allocation (restore) | 44 `cuMemAlloc_v2` calls | 1 `cuMemAlloc_v2` + reuse | ~43 driver calls |

---

## 5. Dominant Remaining Cost

According to the commit message and supporting benchmark data, after these optimizations the **dominant remaining cost (~6.6 ms/batch) is `cuMemcpyDtoHAsync` DMA overhead intrinsic to WSL2 GPU-PV**. In WSL2's para-virtualized GPU path, DMA transfers must traverse the virtual-machine boundary, adding hypervisor round-trip latency. On bare-metal Linux, PCIe DMA bypasses the hypervisor, so this cost disappears automatically — no code changes are required.

This is consistent with the quantitative analysis in [`docs/dev/kcmm-eviction-latency-wsl2-vs-baremetal.md`](kcmm-eviction-latency-wsl2-vs-baremetal.md), which confirms that DMA overhead is the primary WSL2-specific tax.

---

## 6. Summary

| Mechanism | Linux Primitive | What It Does | Paid When | Saved Where |
|-----------|----------------|---------------|-----------|-------------|
| Pre-fault | `mmap(MAP_POPULATE)` | Establishes all PTEs up front in one pass | `TieringEngine::new()` (setup) | CPU scatter/gather phases at steady state |
| Readahead | `madvise(MADV_WILLNEED)` | Asynchronously triggers file page-cache readahead | `new()`, async | First-access latency, background fill |
| Sequential hint | `madvise(MADV_SEQUENTIAL)` | Optimizes readahead strategy for linear scan | `new()`, no-cost hint | Readahead efficiency, cache pressure |

The three primitives collectively eliminate ~4 ms of per-batch page-fault overhead from the CPU scatter/gather phases by moving page-table population off the critical path and into the initialization phase. This is a net win because initialization happens once while eviction/restoration happen repeatedly throughout the engine's lifetime.
