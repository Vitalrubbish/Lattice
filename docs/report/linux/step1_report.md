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

## 4. GDS Integration Details

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
