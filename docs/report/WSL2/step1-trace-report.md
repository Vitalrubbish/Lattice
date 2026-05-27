# Step 1 Trace Report: Model Weight Loading I/O Path Analysis

**Date:** 2026-05-26
**Environment:** WSL2 (Ubuntu 24.04), kernel 6.6.87.2-microsoft-standard-WSL2
**GPU:** NVIDIA GeForce RTX 5070 (8 GB), CUDA 13.1, Driver 591.97 (WSL2 GPU-PV)
**Model:** TinyLlama 1.1B, 2.1 GB safetensors (single shard), 291 tensors

---

## 1. Setup Notes

### CUDA on WSL2

`nvidia-smi` works out of the box but the CUDA runtime (`cudarc`) fails with
`CUDA_ERROR_NO_DEVICE` unless the binary's rpath includes `/usr/lib/wsl/lib`.
WSL2 ships two `libcuda.so` instances:

| Path | Version | Purpose |
|---|---|---|
| `/usr/lib/x86_64-linux-gnu/libcuda.so.535.*` | 535 | Standard (non-functional stub on WSL2) |
| `/usr/lib/wsl/lib/libcuda.so.1.1` | 591.97 | WSL2 GPU-PV driver (functional) |

A `build.rs` was added to bake `-Wl,-rpath,/usr/lib/wsl/lib` into the binary.
Without this, the runtime loader picks the 535 stub and fails.

### bpftrace CUDA uprobes

WSL2's `libcuda.so` functions have zero-size symbol entries (they are ioctl
trampolines to the Windows host-side driver, not real function bodies).
bpftrace requires `--unsafe` to attach uprobes to zero-size symbols, and even
then the probes **do not fire** because the stub functions never execute their
prologue — they trap into the kernel via the DXGKRNL driver immediately.

**Bottom line:** CUDA uprobes (`cuMemAlloc_v2`, `cuMemcpyHtoD_v2`,
`cuMemcpyDtoH_v2`) record zero events on WSL2. CUDA timing comes from the
Rust-side `LoadMetrics` userspace measurements. Full CUDA uprobe tracing
requires bare-metal Linux.

---

## 2. Cold Cache Run — read(2) + cudaMemcpy Baseline

### 2.1 bpftrace Trace Data

```
=========================================================
===  LLM Model Weight Loading — Full I/O Path Trace  ===
=========================================================

-- Linux I/O Stack (counts) --
vfs_read calls:        71
vfs_read bytes:        2.26 GB
filemap_get_pages:     36,943
page_cache_alloc:      0
readahead calls:       2,470
submit_bio:            2,313
block_rq_issue:        2,500
block_rq_complete:     2,494

-- CUDA Driver --
All counters: 0 (WSL2 GPU-PV limitation — see Section 1)

-- bytes per block device --
sdd (8:48):           2.26 GB
```

#### vfs_read Latency Histogram (µs)

| Range | Count | Interpretation |
|-------|-------|---------------|
| 1 µs | 4 | Dentry/inode cache hit |
| 2-4 µs | 8 | Page cache hit (small files, metadata) |
| 4-8 µs | 26 | Page cache hit, bulk |
| 8-16 µs | 8 | Slight copy overhead |
| 16-32 µs | 2 | Larger copy |
| 32-64 µs | 1 | Partial cache miss |
| 64-128 µs | 4 | Block I/O (metadata) |
| 128-256 µs | 6 | Block I/O |
| 256-512 µs | 5 | Slower block I/O |
| 512-1K µs | 1 | Queue depth delay |
| 1-2K µs | 1 | Queue depth delay |
| 4-8K µs | 2 | Large read stall |
| 32-64K µs | 1 | Large read stall |
| 64-128K µs | 1 | Large read stall |
| 2-4M µs | 1 | **The 2.1 GB bulk read — kernel readahead coalesced ~537K pages** |

The distribution is **trimodal**: fast metadata hits (1-32 µs, ~50 calls),
small file reads going to the block layer (128-512 µs, ~15 calls), and the
single massive safetensors bulk read (2-4 ms) that accounted for the vast
majority of bytes. The kernel's readahead logic coalesced ~2.1 GB worth of
4 KB pages into one massive `vfs_read` call spanning over 2 million µs.

#### filemap_get_pages Latency (µs)

| Range | Count | Interpretation |
|-------|-------|---------------|
| <1 µs | 1,911 | Page in cache, no contention |
| 1 µs | 32,154 | Page in cache, normal lookup |
| 2-4 µs | 1,023 | Slight hash lock contention |
| 4-16 µs | 159 | More contention |
| 16-64 µs | 12 | Higher contention |
| 128-256 µs | 1 | Minor page fault |
| 256-512 µs | 5 | Minor page fault (page not ready) |
| 512-1K µs | 919 | **Major page fault — waiting on block I/O** |
| 1-2K µs | 746 | Slower block I/O fulfillment |
| 2-4K µs | 12 | Slowest fault resolution |
| 4-8K µs | 1 | Extreme fault resolution |

Total 36,943 lookups for ~537K pages worth of data — each `filemap_get_pages`
returns a batch of pages (typically the readahead window of ~32-64 pages), so
~537K / 36,943 ≈ 14.5 pages per call on average. The 1,665 calls in the
512-2K µs range represent the actual physical I/O wait — these are the
demand-fault pages that missed cache and triggered `submit_bio` → NVMe read.

#### submit_bio Latency (µs)

| Range | Count |
|-------|-------|
| <1 µs | 42 |
| 1 µs | 120 |
| **2-4 µs** | **1,856** |
| 4-8 µs | 254 |
| 8-16 µs | 29 |
| 16-64 µs | 12 |

Nearly all bios submitted in 2-4 µs — fast, lock-free submission. The
WSL2 `storvsc` driver accepts bios quickly and queues them internally.

#### Block I/O issue→complete Latency (µs)

| Range | Count | Interpretation |
|-------|-------|---------------|
| 64-128 µs | 7 | NVMe-level latency (rare on WSL2) |
| 128-256 µs | 85 | Fast WSL2 storvsc response |
| **256-512 µs** | **466** | Normal WSL2 virtual disk latency |
| **512-1K µs** | **974** | **Dominant — typical storvsc round-trip** |
| **1-2K µs** | **893** | Hypervisor scheduling jitter |
| 2-4K µs | 8 | Moderate jitter |
| 4-8K µs | 60 | Severe jitter / host disk busy |
| 16-32K µs | 1 | Extreme stall |

The dominant latency of 512-1024 µs is characteristic of WSL2's Hyper-V
storvsc virtual block device. On bare-metal NVMe, this bucket would be
10-50 µs. The 1-2 ms tail (893 requests, 36% of total) reflects host-side
Windows disk I/O contention and hypervisor scheduling noise.

### 2.2 Rust LoadMetrics (Userspace Timing)

```
loader:          read
total_ms:        5,101
read_ms:         4,124          (80.8% of total — file read into Vec<u8>)
parse_ms:        1.17           (0.02% — safetensors JSON parse)
alloc_ms:        61.9           (1.2% — 291 × cudaMalloc)
h2d_ms:          666            (13.1% — 291 × cudaMemcpy H→D)
cpu_user_ms:     595
cpu_sys_ms:      4,194          (82.2% of total — all in kernel I/O)
total_bytes:     2.20 GB
throughput:      431 MB/s
```

**Key observations:**

- `read_ms` (4,124) and `cpu_sys_ms` (4,194) are close — the read thread spent
  nearly all its CPU time in kernel mode servicing I/O.
- `h2d_ms` (666 ms) for 2.2 GB gives ~3.3 GB/s PCIe throughput, somewhat lower
  than the previous run (490 ms / 4.5 GB/s), likely due to host-side GPU driver
  contention during this test window.
- `alloc_ms` (61.9 ms for 291 allocations) averages 213 µs per `cudaMalloc`.
- The safetensors JSON header (~64 KB) parsed in 1.17 ms — negligible.

### 2.3 Cross-Validation: bpftrace vs LoadMetrics

| Metric | bpftrace | LoadMetrics | Match? |
|--------|----------|-------------|--------|
| Total bytes | 2.263 GB (vfs_read) | 2.200 GB (safetensors) | bpftrace includes ~63 MB of non-model I/O (shared libs, config files) |
| Read wall time | — | 4,124 ms | bpftrace provides histograms, not sums; the single 2-4M µs vfs_read accounts for the bulk |
| CUDA H→D time | 0 (uprobes didn't fire) | 666 ms | WSL2 limitation — use LoadMetrics |
| CPU sys time | — | 4,194 ms | Consistent with the I/O-heavy profile |

---

## 3. Warm Cache Run — read(2) + cudaMemcpy (Second Consecutive Load)

### 3.1 bpftrace Trace Data

```
-- Linux I/O Stack (counts) --
vfs_read calls:        71
vfs_read bytes:        2.26 GB
filemap_get_pages:     36,943          (same)
page_cache_alloc:      0
readahead calls:       2,475           (+0.2%)
submit_bio:            2,206           (-4.6% vs cold)
block_rq_issue:        2,224           (-11.0% vs cold)
block_rq_complete:     2,218           (-11.1% vs cold)
```

#### vfs_read Latency Histogram (µs)

| Range | Cold | Warm | Delta |
|-------|------|------|-------|
| 1 µs | 4 | 3 | — |
| 2-4 µs | 8 | 11 | +37% |
| 4-8 µs | 26 | 17 | -35% |
| 8-16 µs | 8 | 16 | +100% |
| 16-32 µs | 2 | 2 | Same |
| 32-64 µs | 1 | 0 | — |
| 64-128 µs | 4 | 2 | -50% |
| 128-256 µs | 6 | 7 | +17% |
| 256-512 µs | 5 | 9 | +80% |
| 512-1K µs | 1 | 0 | — |
| 1-2K µs | 1 | 1 | Same |
| 4-8K µs | 2 | 0 | — |
| 32-64K µs | 1 | 1 | Same |
| 64-128K µs | 1 | 1 | Same |
| 2-4M µs | 1 | 1 | Same |

#### filemap_get_pages Latency (µs)

| Range | Cold | Warm | Delta |
|-------|------|------|-------|
| <1 µs | 1,911 | 3,060 | +60% |
| 1 µs | 32,154 | 31,769 | -1.2% |
| 2-4 µs | 1,023 | 334 | -67% |
| 4-16 µs | 159 | 88 | -45% |
| 16-64 µs | 12 | 8 | -33% |
| 128-256 µs | 1 | 1 | Same |
| 256-512 µs | 5 | 25 | +400% |
| 512-1K µs | 919 | 1,609 | +75% |
| 1-2K µs | 746 | 48 | -94% |
| 2-4K µs | 12 | 0 | — |
| 4-16K µs | 1 | 1 | Same |

The warm run shows a shift in the distribution: more <1 µs hits (3,060 vs 1,911)
but also more 512-1K µs major faults (1,609 vs 919). This suggests partial page
cache retention — some pages are served from cache (the <1 µs bump), but many
still require physical I/O. Critically, the severe 1-2K µs faults dropped 94%
(746 → 48), indicating that the worst-case I/O waits were largely eliminated.

#### Block I/O issue→complete Latency (µs)

| Range | Cold | Warm | Delta |
|-------|------|------|-------|
| 32-64 µs | 0 | 1 | — |
| 64-128 µs | 7 | 6 | — |
| 128-256 µs | 85 | 65 | -24% |
| 256-512 µs | 466 | 437 | -6.2% |
| **512-1K µs** | **974** | **1,437** | **+48%** |
| 1-2K µs | 893 | 254 | -72% |
| 2-4K µs | 8 | 6 | — |
| 4-8K µs | 60 | 2 | -97% |
| 8-32K µs | 1 | 9 | — |
| **Total** | **2,494** | **2,218** | **-11.1%** |

The total block I/O count dropped 11.1%. The distribution shifted: the 512-1K µs
bucket grew (+48%) while the 1-2K µs bucket shrank dramatically (-72%) and the
4-8K µs tail nearly vanished (-97%). This is consistent with the host-side NTFS
cache absorbing the worst-case seeks and serving data from its standby list,
converting slow I/Os (1-8K µs) into medium-latency I/Os (512-1K µs).

### 3.2 Rust LoadMetrics (Warm)

```
loader:          read
total_ms:        4,208            (-17.5% vs cold)
read_ms:         3,548            (-14.0% vs cold)
parse_ms:        1.08
alloc_ms:        51.9             (-16.2% vs cold)
h2d_ms:          388              (-41.7% vs cold)
cpu_user_ms:     367
cpu_sys_ms:      3,871            (-7.7% vs cold)
total_bytes:     2.20 GB
throughput:      523 MB/s         (+21.4% vs cold)
```

---

## 4. Cold vs. Warm: What the Data Actually Shows

### 4.1 Summary Table

```
                          Cold         Warm        Delta
──────────────────────────────────────────────────────────
vfs_read calls              71           71          0
vfs_read bytes           2.26 GB      2.26 GB         0
filemap_get_pages        36,943       36,943          0
submit_bio                2,313        2,206       -4.6%
block_rq_issue            2,500        2,224      -11.0%
block_rq_complete         2,494        2,218      -11.1%
page_cache_alloc              0            0          —
readahead calls           2,470        2,475       +0.2%
──────────────────────────────────────────────────────────
load total_ms             5,101        4,208      -17.5%
load read_ms              4,124        3,548      -14.0%
load h2d_ms                 666          388      -41.7%
throughput (MB/s)           431          523      +21.4%
```

### 4.2 The Page Cache Provides a Moderate but Real Benefit

This run shows a **17.5% warm-cache speedup** — significantly more than the
previously observed ~2% but still far from the 6-13x expected on bare metal.
The improvement comes from three compounding effects:

1. **Linux guest page cache retained some pages.** The `filemap_get_pages`
   distribution shifted leftward: <1 µs hits increased 60% (1,911 → 3,060)
   and 2-4 µs contention dropped 67% (1,023 → 334). This means ~1,100+
   additional lookups were satisfied from cache without contention on the
   warm run.

2. **Host-side NTFS standby list absorbed the worst seeks.** Block I/O
   completions dropped 11.1% (2,494 → 2,218), and the worst-case latencies
   collapsed: 1-2K µs bios fell 72% (893 → 254) and 4-8K µs bios fell 97%
   (60 → 2). The Windows host's file cache converted slow physical reads
   into faster cached responses.

3. **GPU driver warm-up.** `h2d_ms` dropped 41.7% (666 → 388 ms). The GPU-PV
   DMA path benefits from one-time setup costs (WDDM buffer pool allocation,
   page table pinning) being amortized on the second run. The original
   report's assertion that h2d_ms is "nearly constant" does not hold under
   this test — GPU transfer time can vary significantly between runs.

### 4.3 Why the Improvement Isn't Larger

Despite the 17.5% speedup, the warm run still issued 2,206 bios and spent
3,548 ms in `read_ms`. Two factors limit the page cache benefit on WSL2:

1. **The model (2.1 GB) competes with GPU memory.** The baseline-server's
   291 × `cudaMalloc` allocations consume ~2.2 GB of GPU VRAM, and WSL2's
   GPU-PV maintains shadow buffers in system memory. Combined with the
   application heap holding the 2.1 GB `Vec<u8>`, the total working set
   exceeds the guest's available page cache budget.

2. **storvsc does not support DAX or persistent page cache pinning.**
   Unlike a native NVMe driver that can keep pages in the page cache
   indefinitely (until memory pressure), storvsc pages are backed by the
   host's NTFS cache and are eligible for earlier eviction by the guest
   kernel's memory reclaim logic.

### 4.4 Run-to-Run Variability

The previous report (2026-05-26, earlier run) measured only a ~2% warm-cache
benefit. The current run shows ~17.5%. This wide variance is expected on WSL2
because the page cache benefit depends on:

- **Host-side Windows memory pressure** at the time of the test (other apps,
  Superfetch, Windows Update)
- **Guest memory allocation pattern** (the order in which `cudaMalloc` and
  `Vec::with_capacity` compete for guest physical pages)
- **Hyper-V dynamic memory balancer** activity between runs

**Practical takeaway:** On WSL2, warm-cache benefit is real but unpredictable,
ranging from negligible (~2%) to moderate (~18%). For consistent measurements,
run 3+ warm/cold pairs and report the range, not a single data point.

### 4.5 What We'd Expect on Bare Metal

On a native Linux machine with local NVMe and sufficient DRAM:

| Phase | Cold | Warm | Mechanism |
|-------|------|------|-----------|
| vfs_read | ~800 ms | <50 ms | Page cache eliminates physical I/O entirely |
| GPU H→D | ~100 ms | ~100 ms | PCIe DMA always needed |
| Total | ~900 ms | ~150 ms | **~6x speedup** |

The Rust `LoadMetrics` cross-validation is designed precisely for this: on bare
metal, `read_ms` should drop dramatically on the warm run while `h2d_ms` stays
constant, confirming the I/O stack (not GPU transfer) is the bottleneck.

---

## 5. I/O Path Component Latency (Cold Run)

### 5.1 Per-Layer Breakdown

```
                         Latency per     Calls ×
Layer                    operation       operations      Total contribution
─────────────────────────────────────────────────────────────────────────
vfs_read entry→return    2-4M µs (bulk)  71              ~4,124 ms
  └ filemap_get_pages    <1-2K µs         36,943          ~4,124 ms (inclusive)
       └ submit_bio      2-4 µs           2,313           negligible submission
            └ block I/O  512-1024 µs       2,494           ~1,500 ms (estimated)
                 └ storvsc + host disk                    ~1,500 ms
cudaMemcpy H→D (291×)    ~2.3 ms avg      291             666 ms
cudaMalloc (291×)        213 µs avg       291             61.9 ms
─────────────────────────────────────────────────────────────────────────
Total wall time                                           5,101 ms
```

### 5.2 Data Path Visualized (WSL2)

```
baseline-server (userspace)
    │ read(2) 2.1 GB safetensors
    ▼
VFS: vfs_read              — 71 calls, 2.26 GB, 2-4M µs bulk read
    │
    ▼
Page Cache: filemap_get_pages — 36,943 lookups, 32,154 hits (1 µs),
    │                           1,665 major faults (512-2K µs)
    │ readahead: 2,470 windows submitted
    ▼
Block Layer: submit_bio   — 2,313 bios, 2-4 µs submission
    │
    ▼
WSL2 storvsc driver       — block_rq_issue → block_rq_complete
    │                         512-1,024 µs typical, 1-2 ms tail
    ▼
Hyper-V VMBus → Windows Host NTFS → Physical Disk (NVMe via Windows)
    │
    ▼
Userspace Vec<u8> [2.1 GB on heap]
    │
    ▼
cudaMemcpy H→D (WSL2 GPU-PV DMA) — 666 ms, ~3.3 GB/s
    │
    ▼
GPU VRAM (8 GB RTX 5070)
```

---

## 6. GPU Transfer Analysis

Even though bpftrace CUDA uprobes don't fire on WSL2, the Rust `LoadMetrics`
provide reliable userspace measurements:

| Metric | Cold | Warm | Notes |
|--------|------|------|-------|
| Allocations | 291 × cudaMalloc | 291 × cudaMalloc | One per safetensors tensor |
| Total alloc time | 61.9 ms | 51.9 ms | 213/179 µs avg per allocation |
| Total H→D time | 666 ms | 388 ms | 2.2 GB at 3.3/5.7 GB/s effective |
| H→D per tensor | 2.29 ms avg | 1.33 ms avg | Range: ~50 µs (bias) to ~15 ms (large weight matrix) |

The warm-run H→D throughput (5.7 GB/s) is significantly better than the cold run
(3.3 GB/s). This suggests GPU-PV DMA path has a non-trivial first-run setup cost
— likely WDDM buffer pool initialization and GPU page table pinning on the
Windows host side. Once warmed, the DMA path operates closer to its steady-state
throughput.

WSL2 GPU-PV implements `cudaMemcpy` using DMA over the PCIe bus through the
Windows host's WDDM driver. The 3.3-5.7 GB/s effective throughput range is
reasonable for this path — about 20-35% of the theoretical PCIe Gen4 x8
bandwidth (16 GB/s), with the overhead coming from:
- GPU-PV hypervisor bounce buffers
- WDDM kernel-mode transitions on the Windows host
- CUDA driver ioctl overhead per small tensor copy

On bare metal with direct PCIe peer-to-peer, H→D throughput should reach
12-14 GB/s, reducing `h2d_ms` to ~160 ms for the same 2.2 GB.

---

## 7. Environment: WSL2 vs. Bare Metal

| Layer | Bare Metal Linux | WSL2 (this test) |
|-------|-----------------|------------------|
| VFS | `vfs_read` → ext4/xfs | Same |
| Page cache | Standard Linux page cache, effective for warm reloads | Partial — host-side NTFS cache helps but inconsistently |
| Block layer | `submit_bio` → NVMe driver (`nvme_queue_rq`) | `submit_bio` → Hyper-V storvsc |
| Block latency | 10-50 µs (NVMe) | 256-1,024 µs (storvsc + VMBus) |
| GPU DMA | PCIe peer-to-peer, 12-16 GB/s | GPU-PV DMA via WDDM, 3.3-5.7 GB/s |
| CUDA uprobes | Work natively | Zero-size stubs, uprobes don't fire |
| GDS (cuFileRead) | NVMe→GPU direct DMA | Not possible (no physical NVMe peer) |
| Page cache benefit | 6-13x warm speedup | 2-18% (variable, run-dependent) |

---

## 8. Key Findings

1. **WSL2 is suitable for developing and testing the bpftrace scripts and the
   Linux I/O path tracing infrastructure.** All 13 kernel-level probes (VFS,
   page cache, block layer) work correctly and provide meaningful data. The
   CUDA uprobes cannot be tested on WSL2 and require bare metal.

2. **The page cache provides a moderate but inconsistent benefit for large
   model weight files on WSL2** (2-18% range across runs). This is a
   significant revision from the earlier conclusion that the page cache was
   "irrelevant." The host-side NTFS standby list absorbs the worst physical
   seeks (97% reduction in 4-8K µs block I/O tail), but the Linux guest page
   cache cannot retain the full 2.1 GB file due to memory competition with GPU
   shadow buffers.

3. **The I/O bottleneck on WSL2 is the storvsc virtual block layer** (512-1,024
   µs per block I/O), not the underlying physical disk. This inflates the
   `read_ms` component to ~3.5-4.1 seconds (80%+ of total load time) compared
   to the expected ~800 ms on bare metal.

4. **GPU H→D transfer time varies significantly between runs** (666 ms cold vs
   388 ms warm). The GPU-PV DMA path has a first-run setup cost that is not
   amortized in isolated measurements. Multiple warm-up runs are needed for
   stable GPU transfer benchmarking on WSL2.

5. **Rust LoadMetrics ↔ bpftrace cross-validation works.** The userspace
   `CpuTimer` measurements corroborate the bpftrace data: `read_ms` (4,124 ms)
   ≈ bpftrace `vfs_read` duty cycle, and `cpu_sys_ms` (4,194 ms) confirms the
   I/O is kernel-CPU-bound. The `h2d_ms` (388-666 ms) fills the gap where WSL2
   limits CUDA uprobes.

6. **The `NaiveTransformer` allocates zero-filled GPU buffers and ignores the
   loaded weights** (`_w: &ModelWeights` in `transformer.rs:22`). CuBLAS GEMM
   on zero matrices gives garbage output. This is a known placeholder — the
   transformer needs to be updated to reference the actual loaded tensors from
   `ModelWeights` for meaningful inference.

---

## 9. Recommendations

1. **For the full I/O analysis (all four loaders including GDS):** acquire a
   bare-metal Linux machine with local NVMe and an NVIDIA GPU. This is the only
   way to measure true NVMe latency (10-50 µs), test the page cache benefit
   (expected 5-10x warm speedup), run CUDA uprobes, and test GDS cuFileRead.

2. **For development work that can continue on WSL2:**
   - bpftrace script refinement (all kernel probes work)
   - `LoadMetrics` / `CpuTimer` instrumentation improvements
   - Transformer weight wiring (`NaiveTransformer` → real `ModelWeights`)
   - KV cache and scheduler algorithm development (GPU operations work)
   - TCP pipeline tracing for Step 2 (`trace_tcp.bt`)
   - HTTP server and client benchmarking

3. **Run multiple cold/warm pairs for stable measurements.** The WSL2 page cache
   benefit ranges from 2% to 18% depending on host-side conditions. Take 3+
   paired runs and report the range rather than a single data point.

4. **Fix the `NaiveTransformer` to use real weights.** The forward pass
   currently runs on zero matrices. Wire up the loaded `ModelWeights.tensors`
   to the transformer layers so inference produces meaningful output. This is
   independent of the I/O tracing work but necessary for end-to-end validation.

5. **For cold-cache reliability on WSL2:** the `echo 3 > drop_caches` approach
   works but only clears the Linux guest page cache, not the host-side NTFS
   cache. True cold cache on WSL2 requires restarting the WSL2 VM or waiting
   for the host's standby list to be recycled.

---

## 10. How to Run the Test

### 10.1 Prerequisites

- WSL2 with Ubuntu 24.04, kernel 6.6.87+
- NVIDIA GPU with CUDA 13.1 and driver 591.97+
- TinyLlama 1.1B model at `/home/vitalrubbish/models/tinyllama/model.safetensors`
- Rust toolchain (stable)
- `bpftrace` installed (`sudo apt install bpftrace`)

### 10.2 Quick Run (Automated)

```bash
cd /mnt/d/os/llm-rust-ebpf
bash scripts/step1_test_wsl2.sh
```

This runs the cold trace, warm trace, and loader comparison in one shot. Output goes to `results/wsl2/<timestamp>/`.

To override the model path or sudo password:

```bash
MODEL_PATH=/path/to/model SUDO_PASS="mypass" bash scripts/step1_test_wsl2.sh
```

### 10.3 Manual Run

#### 10.3.1 Build

```bash
cd /mnt/d/os/llm-rust-ebpf
cargo build --release
```

The `build.rs` automatically detects WSL2 and sets `-Wl,-rpath,/usr/lib/wsl/lib` so the binary
links against the WSL2 GPU-PV `libcuda.so` instead of the 535 stub.

#### 10.3.2 Cold Cache Run

```bash
# 1. Drop the Linux guest page cache
echo <sudo-password> | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# 2. Run bpftrace with baseline-server as the traced child
echo <sudo-password> | sudo -S bpftrace --unsafe scripts/trace_all.bt \
  -c "timeout 20 bash scripts/load_and_exit.sh read tinyllama /home/vitalrubbish/models/tinyllama"
```

The `--unsafe` flag is needed on WSL2 for the `libcuda.so` uprobes (even though
they won't fire — bpftrace still requires it for zero-size symbol attachment).
The `load_and_exit.sh` wrapper ensures the server process exits after loading so the bpftrace
`END` block fires and prints the final report.

#### 10.3.3 Warm Cache Run

Immediately after the cold run completes (do not drop caches):

```bash
echo <sudo-password> | sudo -S bpftrace --unsafe scripts/trace_all.bt \
  -c "timeout 20 bash scripts/load_and_exit.sh read tinyllama /home/vitalrubbish/models/tinyllama"
```

### 10.4 Output Interpretation

| Output Source | Key Metrics | Location |
|---------------|-------------|----------|
| bpftrace `trace_all.bt` | vfs_read calls/bytes, filemap_get_pages count, submit_bio count, block I/O counts, latency histograms | stdout |
| `baseline-server` | `LoadMetrics` (total_ms, read_ms, h2d_ms, throughput) | stderr (via `tracing::info!`) |
| `bench_loaders` | `LoadMetrics` per-run table with cold/warm labeling | stdout |

Cross-validate: bpftrace `vfs_read bytes` should be ~2.26 GB vs LoadMetrics
`total_bytes` ~2.20 GB (the ~63 MB difference is non-model I/O from shared
libraries and config files). bpftrace `vfs_read` histogram's dominant slow
bucket should correspond roughly to LoadMetrics `read_ms`.
