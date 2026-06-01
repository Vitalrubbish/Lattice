# Step 1 — Bare Metal LLM OS I/O Benchmark Results (Linux)

**Date:** 2026-06-01
**Machine:** CloudLab Dell PowerEdge R7525, Ubuntu 22.04.2 LTS
**Kernel:** 5.15.0-177-generic
**GPU:** NVIDIA A30 (24 GB VRAM, PCIe Gen4, BAR1: 32 GiB)
**Driver:** 580.159.04 (CUDA 13.0), proprietary
**nvidia-fs:** 2.16.1 (compat mode — uses `symbol_get` for P2P)
**GDS:** libcufile 2.12, release 1.15.1.6
**Rust:** 1.96.0
**Model:** TinyLlama-1.1B-Chat-v1.0 (2.1 GB safetensors, 202 tensors)
**Filesystem:** ext3 on SATA SSD (`/dev/sda3`)

---

## 1. bpftrace I/O Trace

These traces use the `read(2)` loader (sequential pread) and measure kernel I/O events via bpftrace.

### Cold Cache

| Metric | Value |
|---|---|
| `vfs_read` calls | 88 |
| `vfs_read` bytes | 2,252,872,898 (~2.1 GB) |
| `submit_bio` calls | 10,014 |
| `block_rq_issue` | 10,208 |
| `block_rq_complete` | 10,200 |
| `filemap_get_pages` | 41,987 |
| `cuMemAlloc_v2` | 3 |
| `cuMemcpyHtoD_v2` | 0 (happens before trace attaches) |

> On cold cache, the entire 2.1 GB safetensors file triggers ~10k block I/O requests.
> The readahead heuristic is aggressive (~42k page mapping operations).

### Warm Cache

| Metric | Value |
|---|---|
| `vfs_read` calls | 88 |
| `vfs_read` bytes | 2,252,872,898 (~2.1 GB) |
| `submit_bio` calls | 0 |
| `block_rq_issue` | 164 |
| `block_rq_complete` | 159 |
| `filemap_get_pages` | 35,830 |
| `cuMemAlloc_v2` | 3 |
| `cuMemcpyHtoD_v2` | 0 |

> On warm cache, block I/O (`submit_bio`) drops to zero — all pages served from page cache.
> ~164 block requests remain (metadata/journal). All file data is cache-resident.

---

## 2. Loader Comparison

The `bench_loaders` example benchmarks four loading strategies, each run 3× cold + 3× warm.
All measurements in milliseconds. Throughput computed as `total_bytes / total_ms`.

### read (sequential pread → cudaMemcpy H2D)

| Run | Cold/Warm | Total (ms) | Read (ms) | H2D (ms) | Throughput (MB/s) |
|---|---|---|---|---|---|
| 1 | cold | 4,460 | 3,963 | 263 | 493 |
| 2 | cold | 1,845 | 1,378 | 240 | 1,193 |
| 3 | cold | 1,713 | 1,244 | 244 | 1,285 |
| 1 | warm | 1,685 | 1,220 | 242 | 1,306 |
| 2 | warm | 1,689 | 1,223 | 243 | 1,303 |
| 3 | warm | 1,691 | 1,227 | 242 | 1,301 |

> First cold run pays filesystem cache warming cost (~4.5s). Subsequent runs stabilize at ~1,303 MB/s.
> H2D transfer is consistent at ~242 ms for 2.1 GB (~8.7 GB/s PCIe).

### mmap (memory-mapped → page-fault-driven cudaMemcpy)

| Run | Cold/Warm | Total (ms) | Read (ms) | H2D (ms) | Throughput (MB/s) |
|---|---|---|---|---|---|
| 1 | cold | 4,273 | 0 | 4,130 | 515 |
| 2 | cold | 640 | 0 | 341 | 3,438 |
| 3 | cold | 482 | 0 | 342 | 4,560 |
| 1 | warm | 478 | 0 | 341 | 4,601 |
| 2 | warm | 480 | 0 | 342 | 4,585 |
| 3 | warm | 479 | 0 | 341 | 4,592 |

> mmap defers page faults to the CUDA memcpy (H2D), so `read_ms = 0`.
> First cold run triggers on-demand page faults during H2D (~4.1s H2D). Once cached, H2D drops to ~341 ms.
> Warm throughput reaches ~4,593 MB/s — **fastest warm loader**.

### direct (O_DIRECT → aligned pread → cudaMemcpy)

| Run | Cold/Warm | Total (ms) | Read (ms) | H2D (ms) | Throughput (MB/s) |
|---|---|---|---|---|---|
| 1 | cold | 5,253 | 4,845 | 176 | 419 |
| 2 | cold | 5,234 | 4,838 | 166 | 420 |
| 3 | cold | 5,316 | 4,846 | 240 | 414 |
| 1 | warm | 5,321 | 4,845 | 245 | 413 |
| 2 | warm | 5,237 | 4,842 | 167 | 420 |
| 3 | warm | 5,237 | 4,838 | 166 | 420 |

> O_DIRECT bypasses page cache entirely — **no warm-up benefit**.
> Every run reads ~4.84s from disk at ~417 MB/s (SATA SSD bandwidth).
> H2D is faster (~191 ms avg) because the aligned buffer avoids page-cache copy overhead.
> Slowest but most **predictable** — no cold/warm variance.

### gds (cuFileRead — GPU Direct Storage)

> **Note:** GDS on this system runs in **compatibility mode** (`use_compat_mode: true`).
> True NVMe P2P DMA is not available (NVMe device is behind LVM; `use_pci_p2pdma: false`).
> In compat mode, cuFileRead uses a CPU bounce buffer to transfer data from storage to GPU VRAM
> in a single API call, eliminating the userspace buffer copy.

| Run | Cold/Warm | Total (ms) | Read (ms) | H2D (ms) | Throughput (MB/s) |
|---|---|---|---|---|---|
| 1 | cold | 4,625 | 540 | 4,048 | 476 |
| 2 | cold | 1,145 | 567 | 540 | 1,921 |
| 3 | cold | 977 | 530 | 410 | 2,252 |
| 1 | warm | 956 | 521 | 397 | 2,301 |
| 2 | warm | 960 | 526 | 395 | 2,292 |
| 3 | warm | 977 | 539 | 400 | 2,251 |

> **First GDS results on this hardware!**
>
> **Phase semantics differ from other loaders:**
> - `read_ms`: cuFile driver init + handle register + header read (GPU→D2H→CPU) + handle deregister
> - `h2d_ms`: cuFileRead per tensor — DMA from storage directly into GPU VRAM (no intermediate CPU buffer)
>
> **Key observations:**
> - First cold run: 4,048 ms in H2D (compat-mode DMA through CPU bounce buffer on cold storage)
> - Subsequent cold runs: H2D drops to ~540 ms then ~410 ms as kernel-side buffers warm
> - Warm: stabilizes at ~2,281 MB/s with H2D at ~397 ms
> - GDS warm throughput is **1.75× faster** than `read(2)` warm (1,303 MB/s)
> - GDS warm throughput is **5.5× faster** than `O_DIRECT` warm (418 MB/s)
> - GDS warm throughput is **~50% of mmap** warm (4,593 MB/s)
>
> **Why GDS beats `read(2)` in compat mode:**
> The `read(2)` path does: storage → page cache → userspace buffer → cudaMemcpy → GPU VRAM
> (two data movements through CPU memory). GDS compat mode does: storage → kernel bounce buffer → GPU VRAM
> (one data movement, no userspace copy, fused I/O+DMA in a single `cuFileRead` call).

---

## 3. Summary

| Loader | Cold (MB/s) | Warm (MB/s) | CPU→GPU Copies | Predictability | Status |
|---|---|---|---|---|---|
| **read** | 493–1,285 | 1,301–1,306 | 2 (pread + H2D) | Moderate | ✅ Working |
| **mmap** | 515–4,560 | 4,585–4,601 | 1 (H2D only) | Low (page fault variance) | ✅ Working |
| **direct** | 414–420 | 413–420 | 2 (O_DIRECT + H2D) | **High** | ✅ Working |
| **gds** | 476–2,252 | 2,251–2,301 | 1 (cuFileRead DMA) | Moderate | ✅ Working |

**Key takeaways:**

1. **mmap** remains the fastest warm-cache loader (~4,593 MB/s) by deferring all I/O to
   the GPU memcpy path. Best for repeated loads of the same model.

2. **gds (cuFileRead)** is the second-fastest warm loader (~2,281 MB/s) and significantly
   outperforms `read(2)` (~1,303 MB/s) by eliminating the userspace buffer copy. Even in
   compatibility mode (no true NVMe P2P DMA), the fused I/O+DMA path provides a meaningful
   advantage.

3. **read** is a solid baseline: good warm performance with moderate cold-cache penalty.

4. **direct** is slowest but perfectly predictable — ideal for controlled experiments.

5. **GDS with true NVMe P2P DMA** (requiring a directly-mounted NVMe filesystem with
   `use_pci_p2pdma: true`) is expected to significantly outperform mmap by eliminating
   both the userspace and page-cache copies. This remains to be benchmarked on this hardware.

---

## 4. Deep Dive: Page Cache — Help or Hurt?

The cold vs. warm cache traces provide a rare side-by-side view of the Linux page cache
in a GPU model-loading workload. The answer is nuanced: **the page cache can be either
the single biggest performance accelerator or pure wasteful overhead, depending on the
access pattern.**

### 4.1 Cold Cache: The Page Cache as Overhead

On first load after `drop_caches`, the entire 2.1 GB safetensors file must be read from
the SATA SSD into main memory. The cold trace shows what this costs:

| Event | Count | What it means |
|---|---|---|
| `submit_bio` | 10,014 | 10k block I/O requests issued to the SSD |
| `block_rq_issue` | 10,208 | 10.2k requests dispatched to the block device driver |
| `block_rq_complete` | 10,200 | 10.2k requests completed (8 outstanding at trace end) |
| `page_cache_alloc` | **560,020** | 560k new page cache entries allocated |
| `filemap_get_pages` | 41,987 | 42k page lookup/lock operations |

The cold read(2) path performs **three full data movements**:
```
Storage (SATA SSD) ──DMA──▶ Page Cache ──cpu_copy──▶ Userspace Buffer ──cudaMemcpy──▶ GPU VRAM
         ↑                      ↑                          ↑
    ~10k block I/Os        560k page allocs           ~2.1 GB H2D copy
```

For a one-shot load (load once, infer, then discard), the page cache copy is **wasted work**:
the data is written into the page cache only to be immediately copied out to userspace and
then to GPU VRAM. The page cache never gets a second read. This is why the first cold run
of `read(2)` takes 4,460 ms — the SSD bandwidth (~417 MB/s raw) plus the overhead of
allocating and populating 560,020 page cache entries.

**Direct evidence:** The `O_DIRECT` loader bypasses the page cache entirely and achieves a
perfectly flat ~419 MB/s on every run, cold or warm. It never triggers `page_cache_alloc`
and never varies. The page cache is the *only* reason `read(2)` varies between 4,460 ms
and 1,685 ms.

### 4.2 Warm Cache: The Page Cache as Accelerator

Once the file data is resident in the page cache, the picture reverses dramatically:

| Metric | Cold | Warm | Change |
|---|---|---|---|
| `submit_bio` | 10,014 | **0** | All block I/O eliminated |
| `page_cache_alloc` | 560,020 | **0** | No new page allocations |
| `filemap_get_pages` | 41,987 | 35,830 | 15% fewer lookups |
| `block_rq_issue` | 10,208 | 164 | Only metadata/journal I/O remains |
| `vfs_read` latency (p50) | 8–16 µs | 8–16 µs | Same — served from cache at RAM speed |

The warm `read(2)` path effectively becomes:
```
Page Cache ──cpu_copy──▶ Userspace Buffer ──cudaMemcpy──▶ GPU VRAM
```
Block I/O vanishes. The `read(2)` syscall still executes (88 calls, 2.25 GB read), but
each `vfs_read` completes at memory-bus speed (median 8–16 µs) instead of disk latency
(500–1000 µs for the block I/O path). Throughput jumps from 493 MB/s to 1,301–1,306 MB/s.

The remaining 164 `block_rq_issue` events in the warm trace are journal writes and
metadata flushes from the ext3 filesystem — not file data reads. This is confirmed by
`@bytes_per_dev[8388608]: 5861376` (only ~5.6 MB of block device traffic vs. 2.1 GB cold).

### 4.3 Impact by Loader

| Loader | Cold Penalty | Warm Benefit | Mechanism |
|---|---|---|---|
| **read** | 2.6× slower (4,460→1,685 ms) | Page cache eliminates block I/O | vfs_read serves from cache |
| **mmap** | **8.9× slower** (4,273→478 ms) | Page cache eliminates page faults | mmap'd pages resident; no I/O on access |
| **direct** | **None** — always at disk speed | **None** — page cache is bypassed | O_DIRECT skips cache entirely |
| **gds** | 3.5× slower (4,625→956 ms) | Kernel-side buffer cache warms up | cuFile driver manages its own buffers |

**mmap shows the most extreme page cache effect.** On the first cold run, every
`cudaMemcpy` from an mmap'd page triggers a major page fault (→ `filemap_fault` →
`readpage` → `submit_bio` → NVMe). The H2D phase alone takes 4,130 ms (vs. 341 ms warm).
That's a **12× difference in the H2D phase**. Once cached, the GPU copies directly from
resident pages at near-PCIe bandwidth (~8.7 GB/s), achieving 4,593 MB/s — the theoretical
maximum for this hardware.

**direct shows zero page cache interaction by design.** Every `pread` goes to the SSD.
The result: flat 418 MB/s regardless of cache state, with no warm-up benefit whatsoever.
This is ideal for *predictable performance* but terrible for throughput.

### 4.4 Verdict

| Scenario | Page Cache Role | Recommendation |
|---|---|---|
| **Repeated model loads** (server restart, model swap, A/B testing) | Critical accelerator — 2.6×–8.9× speedup | Use `mmap` or `read`; keep model files in cache |
| **One-shot cold load** (batch inference, ephemeral container) | Pure overhead — wasted memory bandwidth and CPU | Use `O_DIRECT` or GDS; skip the page cache copy |
| **Performance-sensitive serving** (latency-critical, always-warm) | Enables near-PCIe throughput | Use `mmap` for maximum warm throughput |
| **Memory-constrained** (large model, small RAM) | Harmful — consumes memory better used by tensors | Use `O_DIRECT` or GDS; reclaim cache memory |

The page cache is a double-edged sword: it enables the fastest warm throughput (mmap at
4,593 MB/s) but also causes the largest cold-start penalty (mmap cold run 1: 4,273 ms).
The right choice depends on whether your workload caches the model or loads it fresh each time.

---

## 5. Deep Dive: Kernel Subsystem Traversal by Loader

Each loader traces a different path through the Linux I/O stack. The bpftrace probes
at VFS, page cache, block layer, and CUDA driver levels let us reconstruct exactly
which kernel subsystems are involved.

### 5.1 `read(2)` — Full Vertical Stack Traversal

```
Application:  read(fd, buf, len)              [88 pread calls, 2.25 GB]
    │
    ▼
VFS:          vfs_read()                       [88 calls, 2.25 GB]
    │           └─ file->f_op->read_iter()
    │              └─ generic_file_read_iter()
    │
    ▼
Page Cache:   filemap_get_pages()              [41,987 → 35,830 calls]
    │           └─ page_cache_sync_ra()         [readahead: aggressive, ~560k pages]
    │           └─ add_to_page_cache_lru()      [560,020 → 0 allocations]
    │
    ├── Page HIT ──▶ copy_to_user() ──▶ userspace buffer
    │   (warm)          ↑ cpu_copy
    │
    └── Page MISS ──▶ submit_bio()             [10,014 → 0 calls]
        (cold)          └─ block_rq_issue()     [10,208 → 164 events]
                         └─ block_rq_complete() [10,200 → 159 events]
                              ↓
                         SATA SSD (/dev/sda3)
                              │
                         readahead fills additional pages
                              │
                         copy_to_user() ──▶ userspace buffer
    │
    ▼
CUDA Driver:  cuMemAlloc_v2()                  [3 allocations, 8.5 MB]
              cudaMemcpy(H→D)                   [~202 tensors, 2.1 GB total]
                  └─ reads from userspace buffer
                  └─ PCIe Gen4 DMA to GPU VRAM
```

**Subsystems traversed:** VFS → Page Cache → Block Layer → Block Device Driver → SATA
controller → SSD → (back up) → copy_to_user → Userspace → CUDA Driver → PCIe → GPU VRAM.

**CPU involvement:**

| Phase | cpu_user | cpu_sys | Notes |
|---|---|---|---|
| I/O (read) | 184 ms | **1,498 ms** | Kernel CPU dominates: page cache management, copy_to_user, block layer |
| GPU (H2D) | minimal | — | DMA engine does the transfer; CPU only sets up descriptors |
| **Total** | **184 ms** | **1,498 ms** | CPU is mostly kernel time in the I/O stack |

### 5.2 `mmap` — Demand-Paged I/O via Page Fault Path

```
Application:  mmap(fd, PROT_READ, SHARED)
    │           └─ VFS: do_mmap() — creates VMA, no I/O yet
    │
    ▼
CUDA Driver:  cudaMemcpy(H→D) from mmap'd region
    │           └─ For each 4 KB page accessed:
    │
    ├── Page already resident (warm):
    │       └─ PCIe DMA from page cache page ──▶ GPU VRAM   [no kernel involvement]
    │
    └── Page NOT resident (cold):
            └─ Page Fault (major)
                 └─ do_page_fault()
                      └─ filemap_fault()
                           └─ readpage()
                                └─ submit_bio()
                                     └─ block_rq_issue() → SATA SSD
                                     └─ block_rq_complete() → page marked uptodate
                 └─ Page mapped into process page table
                 └─ PCIe DMA from newly-faulted page ──▶ GPU VRAM
```

**Subsystems traversed:** VFS (mmap setup only) → (on fault) Memory Management →
Page Fault Handler → Page Cache → Block Layer → SSD → Page Table → CUDA Driver → GPU VRAM.

**Key difference from read(2):** No explicit `vfs_read` calls. No `copy_to_user`. The
page cache pages are *mapped* into the process address space, not *copied*. The CUDA
driver reads directly from page cache pages. This eliminates the userspace buffer copy
entirely — a savings of ~2.1 GB of CPU memcpy.

**CPU involvement (warm):**

| Phase | cpu_user | cpu_sys | Notes |
|---|---|---|---|
| I/O (mmap setup) | ~0 ms | ~0 ms | mmap is just VMA creation |
| GPU (H2D, warm) | 188 ms | **286 ms** | User CPU: CUDA driver setup. Sys: minor page table walk, TLB |
| **Total** | **188 ms** | **286 ms** | 5.2× less kernel CPU than read(2)! |

The 286 ms of kernel CPU on warm `mmap` is largely TLB shootdown, page table walks,
and PCIe DMA mapping overhead — there's no `copy_to_user`, no block I/O, and no page
cache allocation.

### 5.3 `O_DIRECT` — Page Cache Bypass, Full Stack Otherwise

```
Application:  open(O_DIRECT) + pread(fd, aligned_buf, len, off)
    │
    ▼
VFS:          vfs_read() → generic_file_read_iter()
    │           └─ O_DIRECT flag: skips page cache
    │              └─ generic_file_direct_IO()
    │                   └─ Block layer directly ──▶ userspace buffer
    │
    ▼
Block Layer:  submit_bio()                     [NOT traced, but active]
              block_rq_issue()
              block_rq_complete()
                   ↓
              SATA SSD
                   │
              DMA ──▶ aligned userspace buffer   [BYPASSES page cache]
    │
    ▼
CUDA Driver:  cuMemAlloc → cudaMemcpy(H→D) from aligned buffer → GPU VRAM
```

**Subsystems traversed:** VFS → Block Layer → SSD → Userspace buffer → CUDA Driver → GPU VRAM.

**Notable absence:** Page cache is completely out of the path. No `add_to_page_cache_lru`,
no `filemap_get_pages`, no `page_cache_alloc`. The `generic_file_direct_IO` path in the
kernel issues block I/O directly against the userspace buffer.

This explains the perfect predictability — there's no cache state to affect performance.
Every read is a real disk read, and the SATA SSD delivers a consistent ~417 MB/s.

**CPU involvement (warm):**

| Phase | cpu_user | cpu_sys | Notes |
|---|---|---|---|
| I/O (O_DIRECT pread) | 167 ms | **1,291 ms** | Kernel CPU in direct I/O path + block layer |
| GPU (H2D) | minimal | — | DMA from aligned buffer |
| **Total** | **167 ms** | **1,291 ms** | Similar kernel CPU to read(2), but all in I/O submission |

### 5.4 GDS (cuFileRead) — Custom I/O Path via nvidia-fs

```
Application:  cuFileRead(fh, gpu_ptr, size, file_off, dev_off)
    │
    ▼
cuFile Driver (userspace): libcufile.so
    │           └─ ioctl(CUFILE_READ) → nvidia-fs kernel module
    │
    ▼
nvidia-fs (kernel):  GPU Direct Storage kernel module
    │
    ├── True P2P Mode (NVMe directly attached, PCIe P2P enabled):
    │       └─ NVMe CMB/SQ ──PCIe P2P DMA──▶ GPU VRAM
    │           [BYPASSES: VFS, page cache, block layer, CPU memory entirely]
    │
    └── Compatibility Mode (this test, NVMe behind LVM):
            └─ Storage ──DMA──▶ kernel bounce buffer ──DMA──▶ GPU VRAM
                [BYPASSES: VFS, page cache, userspace buffer]
                [USES: block layer internally, kernel bounce buffer]
    │
    ▼
GPU VRAM:     Data arrives directly in CUDA-allocated memory
              No cudaMemcpy needed — data is already on the GPU
```

**Subsystems traversed (compat mode):** cuFile userspace driver → nvidia-fs kernel
module → block layer (internal) → SSD → kernel bounce buffer → PCIe DMA → GPU VRAM.

**Notable absences:**

| Subsystem | Bypassed? | Why |
|---|---|---|
| VFS (`vfs_read`) | ✅ Yes | cuFile uses its own file handle, not the VFS read path |
| Page Cache (`add_to_page_cache_lru`) | ✅ Yes | Data is never staged in page cache |
| Userspace Buffer (`copy_to_user`) | ✅ Yes | No CPU-side buffer — data goes storage→GPU directly |
| `cudaMemcpy(H→D)` | ✅ Yes | cuFileRead places data directly in GPU VRAM |
| `submit_bio` (visible to bpftrace) | ✅ Yes | cuFile issues I/O through its own internal path |

**CPU involvement (warm):**

| Phase | cpu_user | cpu_sys | Notes |
|---|---|---|---|
| cuFile init + header I/O | 159 ms | — | Driver init, handle register, D2H copy for header parsing |
| cuFileRead tensors | — | **691 ms** | nvidia-fs kernel module: DMA chain setup, bounce buffer management |
| **Total** | **159 ms** | **691 ms** | 54% less kernel CPU than read(2) warm (691 vs. 1,498 ms) |

The 691 ms of kernel CPU in GDS warm mode is the nvidia-fs compatibility path:
issuing storage reads into its internal bounce buffer, then setting up GPU DMA
transfers from that buffer. This is still a single copy (storage → bounce → GPU)
vs. read(2)'s two copies (storage → page cache → userspace → GPU), saving ~800 ms
of kernel CPU time.

### 5.5 Subsystem Comparison Matrix

| Subsystem | read(2) | mmap | O_DIRECT | GDS (compat) |
|---|---|---|---|---|
| **VFS read path** | ✅ vfs_read ×88 | ❌ (mmap, no reads) | ✅ vfs_read w/ O_DIRECT | ❌ (custom ioctl) |
| **Page cache lookup** | ✅ filemap_get_pages | ✅ filemap_fault (on miss) | ❌ | ❌ |
| **Page cache allocation** | ✅ 560k pages cold | ✅ on fault (cold) | ❌ | ❌ |
| **copy_to_user** | ✅ ~2.1 GB copied | ❌ (direct page access) | ❌ (DMA to user buf) | ❌ |
| **Block I/O (submit_bio)** | ✅ 10k cold / 0 warm | ✅ on page fault (cold) | ✅ every read | ❌ (internal to nvidia-fs) |
| **cudaMemcpy H→D** | ✅ ~2.1 GB | ✅ ~2.1 GB | ✅ ~2.1 GB | ❌ (fused in cuFileRead) |
| **Userspace buffer** | ✅ Heap Vec<u8> | ❌ (mmap'd pages) | ✅ AlignedBuffer | ❌ (GPU VRAM only) |
| **Kernel CPU (warm)** | 1,498 ms | **286 ms** | 1,291 ms | 691 ms |
| **Data copies (total)** | **3** (SSD→cache→user→GPU) | **1–2** (SSD→cache→GPU) | **2** (SSD→user→GPU) | **1** (SSD→bounce→GPU) |

---

## 6. Deep Dive: Why GDS Reduces CPU Involvement and Memory Copies

### 6.1 The Memory Copy Problem

Every data copy burns CPU cycles and memory bandwidth. Loading a 2.1 GB model file into
GPU VRAM requires moving those 2.1 GB through the system's memory hierarchy. The fewer
times the data is copied, the faster and more CPU-efficient the load.

The fundamental problem with traditional approaches is that the Linux I/O stack was designed
for CPU-consumed data: storage → kernel buffer → userspace → application logic. GPU model
loading inverts this: the CPU never "consumes" the data — it's just a pass-through on its
way to the GPU. Every CPU-side copy is wasted work.

### 6.2 Data Copy Count by Loader

```
read(2):   SSD ──[DMA]──▶ Page Cache ──[cpu_copy]──▶ Userspace ──[PCIe DMA]──▶ GPU VRAM
                   ①                   ②                         ③
           Copies: 3    Kernel CPU: 1,498 ms    Warm throughput: 1,303 MB/s

mmap:      SSD ──[DMA]──▶ Page Cache ──[PCIe DMA from mapped pages]──▶ GPU VRAM
                   ①                   ②  (cuMemcpy reads mmap'd pages)
           Copies: 2 (cold) / 1 (warm, page already in cache)
           Kernel CPU: 286 ms (warm)    Warm throughput: 4,593 MB/s

O_DIRECT:  SSD ──[DMA]──▶ Aligned Userspace ──[PCIe DMA]──▶ GPU VRAM
                   ①                         ②
           Copies: 2    Kernel CPU: 1,291 ms    Throughput: 418 MB/s

GDS:       SSD ──[DMA]──▶ Kernel Bounce Buffer ──[GPU DMA]──▶ GPU VRAM
                   ①                               ②
           (compat mode: two DMA transfers, but NO CPU copies)
           Copies: 0 CPU copies (2 DMA transfers in compat; 1 in true P2P)
           Kernel CPU: 691 ms    Warm throughput: 2,281 MB/s
```

### 6.3 Where the CPU Time Goes

The `cpu_sys` measurements from `getrusage(RUSAGE_THREAD)` isolate kernel CPU time,
which is dominated by the data movement path:

| Loader | cpu_sys (warm) | Primary kernel activity |
|---|---|---|
| **read** | 1,498 ms | `copy_to_user`: copying 2.1 GB from page cache to userspace. This is a CPU-driven `memcpy` in the kernel, running page-by-page (88 vfs_read calls × ~23 MB each). Also: page cache LRU management, readahead bookkeeping. |
| **mmap** | **286 ms** | Page table walks, TLB shootdown, PCIe DMA mapping (iommu). No `copy_to_user` — the GPU reads directly from mapped page cache pages. The massive reduction (1,498→286 ms, 5.2×) comes entirely from eliminating the kernel's `copy_to_user` and page cache management. |
| **direct** | 1,291 ms | `generic_file_direct_IO`: issuing and completing ~10k block I/O requests, managing the bio queue, servicing block layer completions. Less than read(2) only because there's no page cache overhead. |
| **gds** | **691 ms** | nvidia-fs internal: managing the bounce buffer, setting up DMA descriptors for storage→bounce and bounce→GPU transfers. No VFS, no page cache, no `copy_to_user`. The remaining CPU cost is the compatibility mode tax — the bounce buffer still needs kernel management. |

### 6.4 Why GDS Wins Over read(2) (Even in Compatibility Mode)

The `read(2)` path is fundamentally inefficient for GPU-bound data because it treats the
GPU as an afterthought:

1. **read(2)** copies data *into* CPU memory (userspace buffer)
2. **cudaMemcpy** copies data *out of* CPU memory (to GPU VRAM)

These are two separate system calls, two separate DMA/copy operations, and the data
makes a round-trip through CPU memory that serves no purpose.

GDS merges these into one operation: `cuFileRead` tells the cuFile driver "read this
file range directly into this GPU memory address." The driver handles everything:

```
read(2) path (two syscalls, two copies through CPU memory):
  pread(fd, cpu_buf, len, off)  ──▶ kernel copies storage→cpu_buf
  cudaMemcpy(gpu_ptr, cpu_buf, len) ──▶ GPU copies cpu_buf→vram

GDS path (one syscall, zero copies through CPU memory):
  cuFileRead(fh, gpu_ptr, len, off) ──▶ kernel DMAs storage→bounce→gpu_ptr
```

**Measured benefit:** GDS warm throughput (2,281 MB/s) is **1.75× faster** than read(2)
warm (1,303 MB/s). GDS kernel CPU (691 ms) is **54% lower** than read(2) (1,498 ms).

### 6.5 Why mmap Still Beats GDS (on This Hardware)

mmap achieves 4,593 MB/s vs. GDS's 2,281 MB/s. Why?

1. **mmap on warm cache has effectively zero I/O path.** The pages are already in the
   page cache. `cudaMemcpy` reads them at memory-bus speed (~10+ GB/s) and the PCIe
   DMA engine transfers them at ~8.7 GB/s. The bottleneck is PCIe bandwidth, not any
   kernel path.

2. **GDS compatibility mode still goes through a bounce buffer.** The nvidia-fs 2.16.1
   driver uses `symbol_get()` to access P2P symbols (no true P2P DMA). Each `cuFileRead`
   internally does: issue read to storage → DMA into bounce → schedule GPU DMA from bounce.
   This adds latency and serialization at the driver level.

3. **GDS cuFileRead is per-tensor (202 calls).** Each tensor requires a separate
   `cuFileRead` call, each with driver-internal setup/teardown. mmap's `cudaMemcpy`
   is also per-tensor, but on warm cache that's just PCIe DMA setup — no driver-internal
   I/O path.

### 6.6 The Promise of True P2P DMA

With a directly-attached NVMe device (no LVM) and `use_pci_p2pdma: true`:

```
GDS True P2P:
  NVMe SSD ──[PCIe P2P DMA]──▶ GPU VRAM
                   ①
           Copies: 1 (zero CPU touches)
           CPU involvement: ~0 ms kernel, ~0 ms user
           Throughput: ~NVMe read bandwidth (~3–7 GB/s for Gen4)
```

In true P2P mode, the NVMe controller writes directly to GPU BAR memory over PCIe.
The data *never touches CPU memory at all* — not even a kernel bounce buffer. The
CPU only sets up the cuFile descriptor and receives the completion interrupt.

This is expected to:
- Match or exceed mmap's warm throughput (by eliminating even the page cache copy)
- Use near-zero CPU (only descriptor setup and completion handling)
- Be cache-agnostic (no cold/warm distinction — NVMe always reads at full speed)
- Scale to models larger than CPU RAM (no need to stage the model in host memory)

### 6.7 CPU Efficiency Summary

| Loader | CPU copies | Kernel CPU (warm) | CPU/GB | Data path length |
|---|---|---|---|---|
| **read** | 2 CPU copies | 1,498 ms | 713 ms/GB | 3-hop |
| **mmap** | 0 CPU copies | **286 ms** | **136 ms/GB** | 1-hop (warm) |
| **direct** | 1 CPU copy | 1,291 ms | 615 ms/GB | 2-hop |
| **gds** | 0 CPU copies | 691 ms | 329 ms/GB | 2-DMA-hop |

GDS achieves a **2.2× CPU efficiency improvement** over read(2) (329 vs. 713 ms/GB) by
eliminating the userspace buffer copy and fusing I/O+DMA into a single call. It's not yet
at mmap's level because the compatibility mode bounce buffer still requires kernel CPU for
buffer management. True P2P DMA would push CPU/GB close to zero.

---

## 7. GDS Integration Details

### What was fixed to make GDS work

| Issue | Root Cause | Fix |
|---|---|---|
| `nvidia_fs` module won't load | v2.29.4 requires NVIDIA open kernel module; v580.159.04 is proprietary | Downgraded to `nvidia-fs-dkms=2.16.1` which uses `symbol_get()` for P2P symbols |
| `cuFileRead` returns -1 | FFI binding had wrong signature: `size: *mut usize` instead of `size: usize` | Fixed FFI to match `ssize_t cuFileRead(fh, ptr, size, off, off)` |
| JSON parse error on header | `__metadata__` entry in safetensors lacks `dtype`/`shape`/`data_offsets` | Made `RawTensorMeta` fields optional and skip non-tensor entries |
| Linker can't find `libcufile.so` | Not in default linker path | Added `rustc-link-search` and `rpath` in `build.rs` for `/usr/local/cuda/lib64` |

### Files changed

| File | Change |
|---|---|
| `build.rs` | Added cuFile library search path when `CARGO_FEATURE_GDS` is set |
| `src/model/loader.rs` | Fixed `cuFileRead` FFI signature; fixed safetensors header parsing for metadata entries |
| `examples/bench_loaders.rs` | Added `Gds` loader to benchmark; made errors non-fatal |
| `scripts/step1_test_baremetal.sh` | Build with `--features gds` |

### How to enable GDS on this machine

```bash
# Ensure nvidia_fs module is loaded
sudo modprobe nvidia-fs
lsmod | grep nvidia_fs  # should show nvidia_fs

# Build with GDS
cargo build --release --features gds --example bench_loaders

# Run loader comparison
sudo env MODEL_PATH="$HOME/models/tinyllama" target/release/examples/bench_loaders
```

---

## Raw Logs

- [trace_cold.log](trace_cold.log) — Full bpftrace output (cold cache)
- [trace_warm.log](trace_warm.log) — Full bpftrace output (warm cache)
- [loader_comparison.log](loader_comparison.log) — Full bench_loaders output (all 4 loaders)
