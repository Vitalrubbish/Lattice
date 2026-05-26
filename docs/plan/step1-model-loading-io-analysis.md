# Step 1: Model Weight Loading and I/O Stack Analysis

**Weight:** 15% of total project  
**Status:** `read` loader implemented; `mmap`, `O_DIRECT`, `GDS` stubbed  
**Key files:** `src/model/loader.rs`, `src/model/weights.rs`, `src/cuda/mod.rs`

---

## 1. Objectives and Deliverables

This step has two complementary goals:

1. **I/O path tracing** — Use eBPF (bpftrace or custom BPF programs) to trace kernel functions along the full data path from VFS through to the NVMe device driver, producing a latency breakdown of the naive `read()` + `cudaMemcpy` baseline.
2. **Alternative loading methods** — Implement and benchmark `mmap`, `O_DIRECT`, and GDS `cuFileRead` loaders, then compare them against the baseline to answer: *does the page cache help or hurt for model-weight I/O, and which kernel subsystems does each method traverse?*

### Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| D1 | bpftrace scripts | Scripts tracing `vfs_read`, `filemap_get_pages`, `submit_bio`, NVMe completion, and `cudaMemcpy` |
| D2 | Latency flame graphs / tables | Per-function latency distribution across the I/O stack for the `read` path |
| D3 | `mmap` loader | Page-fault-driven on-demand weight loading via `mmap` + `cudaMemcpy` |
| D4 | `O_DIRECT` loader | Bypass page cache via `O_DIRECT` reads + `cudaMemcpy` |
| D5 | GDS loader | NVMe → GPU direct DMA via `cuFile` / `cuFileRead` |
| D6 | Comparative benchmark report | Throughput, latency, CPU utilization, and kernel-subsystem traversal for all four methods |

---

## 2. Current Baseline: `read()` + `cudaMemcpy`

### 2.1 Code Walkthrough

The existing `LoadKind::Read` path in `src/model/loader.rs:54-91` does the following for each `.safetensors` shard:

```
read_whole_file(path)
  → File::open(path)                 // open(2) — no O_DIRECT
  → f.read_to_end(&mut buf)          // read(2) in a loop until EOF
  → SafeTensors::deserialize(&buf)   // CPU-side parse of safetensors headers
  → ctx.alloc_bytes(view.data().len())  // cudaMalloc
  → ctx.h2d_sync(view.data(), &mut dev) // cudaMemcpy H→D
```

Key observations:

- **Synchronous, serialized:** Each shard is read in full to a CPU buffer, parsed, then copied to the GPU. The next shard is not touched until the current one finishes GPU transfer.
- **Page cache pollution:** `read()` pulls the entire model file through the Linux page cache. For a 7B-parameter model in fp16 (~14 GB of weights), this means 14 GB of page cache churn — evicting hot filesystem metadata and application pages, with zero reuse (weights are read once at startup).
- **Double-buffering:** Data resides in both a kernel page cache page and a userspace `Vec<u8>` before the final GPU copy, wasting memory bandwidth.
- **No CPU-GPU overlap:** The design is purely sequential — no CUDA streams, no pinned memory, no overlap of I/O and transfer.

### 2.2 I/O Stack Traversed

Under the naive `read()`, data flows through these kernel layers:

```
userspace:  read(fd, buf, len)
────────────────────────────────────────────
VFS:        vfs_read() → file->f_op->read_iter()
Page Cache: filemap_get_pages() / page_cache_sync_readahead()
            ─ may trigger readahead if sequential ─
Block I/O:  submit_bio() → blk_mq_submit_bio()
NVMe:       nvme_queue_rq() → nvme_map_data() → MMIO write to SQ doorbell
IRQ:        nvme_irq() → nvme_process_cq() → bio_endio()
DMA:        NVMe controller DMAs data from flash to host DRAM
────────────────────────────────────────────
userspace:  buffer filled
────────────────────────────────────────────
CUDA:       cuMemcpyHtoD() → GPU driver pins source pages → DMA from host DRAM to GPU VRAM
```

**Latency components:**

- `t_read`: VFS entry to buffer populated (dominated by NVMe flash read latency + PCIe transfer)
- `t_copy`: CPU buffer → GPU VRAM via `cudaMemcpy` (PCIe bandwidth-bound, ~24 GB/s on PCIe 4.0 x16)
- `t_alloc`: `cudaMalloc` overhead (kernel launch + device-side allocator)

Expected per-shard latency for a ~3.5 GB safetensors shard on a typical NVMe SSD (3 GB/s read, 24 GB/s PCIe H→D):

- Flash read: ~1.2 s
- cudaMemcpy H→D: ~0.15 s
- Total per shard (serial): ~1.35 s × 4 shards ≈ **5.4 s total**

For a model that fits GPU memory, this is a one-time startup cost. Its significance grows when reloading weights during pipeline reconfiguration or when models exceed GPU memory and require repeated offload/reload.

---

## 3. eBPF Tracing Instrumentation

### 3.1 Trace Points and Kernel Functions

The following kernel functions will be instrumented to capture the full I/O path:

| Probe point | Kernel function | What it measures |
|-------------|----------------|------------------|
| kprobe/kretprobe | `vfs_read` | Entry/exit of the VFS read path; total syscall duration |
| kprobe/kretprobe | `generic_file_read_iter` | File-type-specific read iteration |
| kprobe/kretprobe | `filemap_get_pages` | Page cache lookup; hit vs. miss |
| kprobe | `page_cache_sync_readahead` | Read-ahead page allocations |
| kprobe/kretprobe | `submit_bio` | Block I/O submission; bio size and sector |
| tracepoint | `block:block_rq_issue` | Request issued to NVMe driver |
| tracepoint | `block:block_rq_complete` | Request completed by NVMe driver |
| tracepoint | `nvme:nvme_complete_rq` | NVMe command completion |
| kprobe/kretprobe | `dma_direct_map_page` | DMA mapping for NVMe transfer |
| uprobe | `cuMemcpyHtoD_v2` (libcuda.so) | CUDA host-to-device transfer |

### 3.2 bpftrace Script Design

**Script 1: `trace_vfs_read.bt` — Per-process VFS read latency**

```bpftrace
#!/usr/bin/env bpftrace

BEGIN {
    printf("Tracing vfs_read for PID %d...\n", $1);
    @target = $1;
}

kprobe:vfs_read /pid == @target/ {
    @start[tid] = nsecs;
    @bytes[tid] = arg2;
}

kretprobe:vfs_read /pid == @target && @start[tid] != 0/ {
    $dur_us = (nsecs - @start[tid]) / 1000;
    @lat_us = hist($dur_us);
    @total_bytes = sum(@bytes[tid]);
    printf("vfs_read: %d bytes, %d us\n", @bytes[tid], $dur_us);
    delete(@start[tid]);
    delete(@bytes[tid]);
}

END {
    print(@lat_us);
    printf("total bytes: %d\n", @total_bytes);
}
```

**Script 2: `trace_block_io.bt` — Block layer and NVMe tracing**

```bpftrace
#!/usr/bin/env bpftrace

tracepoint:block:block_rq_issue {
    @issued[tid] = nsecs;
    @rq_sectors[tid] = args->sector;
    @rq_nr_sector[tid] = args->nr_sector;
}

tracepoint:block:block_rq_complete /@issued[tid] != 0/ {
    $dur_us = (nsecs - @issued[tid]) / 1000;
    @block_lat_us = hist($dur_us);
    @total_sectors = sum(@rq_nr_sector[tid]);
    delete(@issued[tid]);
}

tracepoint:nvme:nvme_complete_rq {
    printf("NVME complete: cmd_id=%d, status=%d\n", args->cid, args->status);
}

END {
    print(@block_lat_us);
    printf("total sectors read: %d\n", @total_sectors);
}
```

**Script 3: `trace_cuda_memcpy.bt` — CUDA H→D transfer tracing**

```bpftrace
#!/usr/bin/env bpftrace

uprobe:/usr/local/cuda/lib64/libcuda.so:cuMemcpyHtoD_v2 {
    @cuda_start[tid] = nsecs;
    @cuda_bytes[tid] = arg2;
}

uretprobe:/usr/local/cuda/lib64/libcuda.so:cuMemcpyHtoD_v2 /@cuda_start[tid] != 0/ {
    $dur_us = (nsecs - @cuda_start[tid]) / 1000;
    @cuda_lat_us = hist($dur_us);
    printf("cuMemcpyHtoD: %d bytes, %d us\n", @cuda_bytes[tid], $dur_us);
    delete(@cuda_start[tid]);
    delete(@cuda_bytes[tid]);
}
```

**Script 4: `trace_page_cache.bt` — Page cache hit/miss analysis**

```bpftrace
#!/usr/bin/env bpftrace

kprobe:filemap_get_pages /pid == $1/ {
    @fmgp_calls = count();
}

// filemap_get_pages returns a page vector; on first read, pages need allocation
kprobe:page_cache_alloc /pid == $1/ {
    @pc_alloc = count();
}

kprobe:add_to_page_cache_lru /pid == $1/ {
    @pc_add_lru = count();
}

kprobe:mark_page_accessed /pid == $1/ {
    @pc_accessed = count();
}

END {
    printf("filemap_get_pages calls: %d\n", @fmgp_calls);
    printf("page_cache_alloc: %d\n", @pc_alloc);
    printf("add_to_page_cache_lru: %d\n", @pc_add_lru);
    printf("mark_page_accessed: %d\n", @pc_accessed);
}
```

### 3.3 Custom BPF Program (Optional Enhancement)

For deeper introspection (e.g., associating specific bio completions with their originating VFS reads), a custom BPF program using `libbpf-rs` can attach to additional tracepoints and use BPF maps to correlate events. This requires:

- A BPF ring buffer to push per-request timing tuples to userspace
- A userspace aggregator that matches bio submissions to completions by `(pid, sector)` tuples
- Histogram output via `tracing` spans for integration with the Rust application's log output

This is a stretch goal — the bpftrace scripts above are sufficient for the required analysis.

---

## 4. Implementation: Alternative Loading Methods

### 4.1 `mmap` Loader (Page-Fault-Driven On-Demand Loading)

**Mechanism:** Instead of `read()` copying data into a userspace buffer, `mmap()` maps the file pages directly into the process address space. The kernel populates pages on demand via page faults — the data is read from disk only when the CPU first accesses a given virtual address.

**I/O path:** `mmap` + page fault → `filemap_fault` → `readpage` / `readahead` → `submit_bio` → NVMe.

**Implementation plan** (`src/model/loader.rs`):

```rust
fn load_with_mmap(&self, path: &Path) -> Result<ModelWeights> {
    let files = enumerate_safetensors(path)?;
    let mut weights = ModelWeights::empty(self.cfg);
    let t0 = Instant::now();

    for shard in files {
        let file = File::open(&shard)?;
        let len = file.metadata()?.len() as usize;

        // SAFETY: file is kept alive for the lifetime of the mapping
        let mmap = unsafe {
            memmap2::Mmap::map(&file)?
        };

        let st = SafeTensors::deserialize(&mmap)?;

        for (name, view) in st.tensors() {
            let mut dev = self.ctx.alloc_bytes(view.data().len())?;

            // cudaMemcpy will trigger page faults for each accessed page
            // as the CPU reads from the mmap'd region
            self.ctx.h2d_sync(view.data(), &mut dev)?;

            weights.insert(name.to_string(), RawTensor { ... });
        }
    }

    self.ctx.synchronize()?;
    // ...
}
```

**Key behaviors to trace:**

- First access to a page triggers a major page fault → `submit_bio` to NVMe
- Subsequent accesses (within the same boot) may hit warm page cache
- kernel CPU time includes page fault handling (`do_fault`, `filemap_fault`)
- No double-buffering: DMA from NVMe → page cache page ≡ userspace-visible page

**Hypothesis:** `mmap` should reduce CPU copies (no `copy_to_user`), but the page fault interrupt overhead on first access may offset the savings for a one-shot workload. If weights are reloaded multiple times (e.g., model swapping), warm page cache could make `mmap` significantly faster.

**Dependencies:** Add `memmap2` crate to `Cargo.toml`.

### 4.2 `O_DIRECT` Loader (Bypassing Page Cache)

**Mechanism:** `open()` with `O_DIRECT` flag. Reads go directly from the NVMe device to a userspace buffer (which must be aligned to the device logical block size, typically 512 or 4096 bytes). The page cache is entirely bypassed.

**I/O path:** `read()` with `O_DIRECT` → `generic_file_direct_read` → `submit_bio` → NVMe — no VFS page cache involvement.

**Implementation plan** (`src/model/loader.rs`):

```rust
fn load_with_direct(&self, path: &Path) -> Result<ModelWeights> {
    let files = enumerate_safetensors(path)?;
    let mut weights = ModelWeights::empty(self.cfg);
    let t0 = Instant::now();

    for shard in files {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&shard)?;

        let len = file.metadata()?.len() as usize;

        // O_DIRECT requires aligned buffers
        let align = 4096;
        let buf_len = (len + align - 1) & !(align - 1);
        let mut buf = Vec::<u8>::with_capacity(buf_len);
        // SAFETY: allocation with proper alignment
        // In practice, use aligned-vec or posix_memalign equivalent
        file.read_exact(&mut buf[..len])?;

        let st = SafeTensors::deserialize(&buf[..len])?;

        for (name, view) in st.tensors() {
            let mut dev = self.ctx.alloc_bytes(view.data().len())?;
            self.ctx.h2d_sync(view.data(), &mut dev)?;
            weights.insert(name.to_string(), RawTensor { ... });
        }
    }

    self.ctx.synchronize()?;
    // ...
}
```

**Key challenges:**

- **Alignment:** `O_DIRECT` requires the userspace buffer address, file offset, and transfer size to all be aligned to the logical block size of the underlying device. This means using `posix_memalign` (or `aligned` crate) rather than standard `Vec`.
- **No readahead:** Without the page cache, readahead is disabled. Reads smaller than the NVMe optimum (~128 KB) will suffer.
- **Read size tuning:** `read_exact` may issue many small reads. A production implementation should read in large chunks (e.g., 2 MB) and then parse safetensors headers from the accumulated buffer.

**Hypothesis:** `O_DIRECT` should have more predictable latency (no page cache cold/warm variance), and lower CPU utilization (no kernel time spent on page cache LRU management). However, it may be *slower* on lightly-loaded systems where the page cache would otherwise absorb the read from DRAM on warm reloads. For a one-shot model load, the performance difference from naive `read()` may be small since there is no reuse — the main win is avoiding pollution.

**Dependencies:** Add `libc` crate (already transitively available) and `aligned` crate.

### 4.3 GDS `cuFileRead` Loader (GPU Direct Storage)

**Mechanism:** NVIDIA GPU Direct Storage (GDS, aka `cuFile`) enables the NVMe device to DMA data directly to GPU VRAM over PCIe, completely bypassing both the CPU and host DRAM. The data path is:

```
NVMe SSD → PCIe switch → GPU VRAM
```

No CPU copies, no host memory staging buffer, no kernel I/O stack involvement (after setup).

**I/O path:** `cuFileRead` → `cuFile` driver → NVMe CMB (Controller Memory Buffer) → DMA to GPU BAR (Base Address Register) → GPU VRAM.

**Prerequisites:**

- NVIDIA GPU with GDS support (A100, H100, or later; some L40S/RTX 6000 Ada)
- `nvidia-fs` and `cufile` driver packages installed (`nvidia-fs-dkms`, `libcufile-dev`)
- NVMe device compatible with GDS (most enterprise NVMe drives)
- CUDA 12.x toolkit with `cufile.h` header

**Implementation plan** (`src/model/loader.rs`):

Since the Rust ecosystem has no mature `cuFile` binding, this will use FFI to `libcufile.so`:

```rust
// FFI bindings for cuFile (minimal subset)
#[link(name = "cufile")]
extern "C" {
    fn cuFileRead(
        fh: CUfileHandle_t,
        devPtr: CUdeviceptr,
        size: *mut size_t,
        file_offset: off_t,
        devPtr_offset: off_t,
    ) -> CUfileError_t;

    fn cuFileHandleRegister(
        fh: *mut CUfileHandle_t,
        desc: *mut CUfileDescr_t,
    ) -> CUfileError_t;

    fn cuFileDriverOpen() -> CUfileError_t;
    fn cuFileDriverClose() -> CUfileError_t;
}

fn load_with_gds(&self, path: &Path) -> Result<ModelWeights> {
    unsafe { cuFileDriverOpen() };

    let files = enumerate_safetensors(path)?;
    let mut weights = ModelWeights::empty(self.cfg);
    let t0 = Instant::now();

    for shard in files {
        let file = File::open(&shard)?;

        // Register file descriptor with cuFile
        let mut desc: CUfileDescr_t = std::mem::zeroed();
        desc.type_ = CU_FILE_OPEN_FD;
        desc.cookie = file.as_raw_fd() as *mut c_void;
        let mut fh: CUfileHandle_t = std::mem::zeroed();
        unsafe { cuFileHandleRegister(&mut fh, &mut desc) };

        // Read safetensors header first (first ~64 KB)
        let header_len = 65536;
        let header_dev = self.ctx.alloc_bytes(header_len)?;
        let mut bytes_read: size_t = 0;
        unsafe {
            cuFileRead(
                fh,
                header_dev.device_ptr() as CUdeviceptr,
                &mut bytes_read,
                0, // file offset
                0, // devPtr offset
            );
        }

        // Copy header back to CPU for parsing (GDS is NVMe→GPU only)
        let mut header_cpu = vec![0u8; bytes_read];
        self.ctx.d2h_sync(&header_dev[..bytes_read], &mut header_cpu)?;
        let st = SafeTensors::deserialize(&header_cpu)?;

        // For each tensor, read directly into GPU memory
        for (name, view) in st.tensors() {
            let tensor_dev = self.ctx.alloc_bytes(view.data().len())?;
            let mut bytes_read: size_t = 0;

            // Calculate byte offset from tensor metadata
            let data_offset = view.data_offsets().0 as off_t;

            unsafe {
                cuFileRead(
                    fh,
                    tensor_dev.device_ptr() as CUdeviceptr,
                    &mut bytes_read,
                    data_offset,
                    0,
                );
            }

            weights.insert(name.to_string(), RawTensor {
                shape: view.shape().to_vec(),
                dtype: format!("{:?}", view.dtype()),
                bytes: tensor_dev,
            });
        }
    }

    self.ctx.synchronize()?;
    unsafe { cuFileDriverClose() };
    // ...
}
```

**Key challenges:**

- **Safetensors headers are on CPU:** GDS can only DMA to GPU VRAM, not back to CPU. The safetensors header (JSON describing tensor shapes and offsets) must be read to GPU first, then copied back to CPU for parsing. This adds a round-trip.
- **Tensor granularity:** Each tensor can be transferred independently via `cuFileRead`, enabling fine-grained overlap if combined with CUDA streams.
- **cuFile handle lifecycle:** File descriptors must be registered with `cuFileHandleRegister` and deregistered after use.
- **Error handling:** cuFile errors (CUfileError_t) must be mapped to Rust `anyhow::Error`.

**Hypothesis:** GDS should provide the lowest CPU utilization (near-zero CPU involvement in the data path), and potentially the highest throughput if the NVMe-to-GPU PCIe path is not bottlenecked by the PCIe switch topology. The header round-trip adds fixed overhead (~100 µs) that amortizes to zero for large weight tensors.

---

## 5. I/O Path Comparison

| Aspect | `read()` | `mmap` | `O_DIRECT` | GDS `cuFileRead` |
|--------|----------|--------|------------|------------------|
| **Kernel VFS** | full path | page fault only | `generic_file_direct_read` | none (after fd reg) |
| **Page cache** | populated | populated (on fault) | bypassed | bypassed entirely |
| **Host DRAM** | staging buffer | page cache page | aligned staging buffer | not touched |
| **CPU copies** | DMA→DRAM + `copy_to_user` + `cudaMemcpy` | DMA→DRAM + `cudaMemcpy` | DMA→DRAM + `cudaMemcpy` | DMA→GPU (no CPU copy) |
| **CPU involvement** | syscall + page cache mgmt + cudaMemcpy launch | page fault handler + cudaMemcpy launch | syscall + cudaMemcpy launch | GPU driver setup only |
| **GPU VRAM target** | CPU→GPU DMA | CPU→GPU DMA | CPU→GPU DMA | NVMe→GPU DMA directly |
| **Warm-cache reuse** | yes (ramfs speed) | yes (page hit, no I/O) | no | no |
| **Double buffering** | yes (page cache + Vec<u8>) | yes (page cache page ≡ user page) | no (single user buffer) | no (single GPU buffer) |

---

## 6. Benchmarking and Comparison Methodology

### 6.1 Measurement Infrastructure

Extend `src/model/loader.rs` to expose structured timing data:

```rust
pub struct LoadMetrics {
    pub loader: String,
    pub total_ms: f64,
    pub read_ms: f64,        // time spent in I/O syscalls
    pub parse_ms: f64,       // time spent deserializing safetensors headers
    pub alloc_ms: f64,       // cumulative cudaMalloc time
    pub h2d_ms: f64,         // cumulative cudaMemcpy H→D time (N/A for GDS)
    pub total_bytes: usize,
    pub cpu_user_ms: f64,    // getrusage(RUSAGE_THREAD).ru_utime
    pub cpu_sys_ms: f64,     // getrusage(RUSAGE_THREAD).ru_stime
}
```

Use `Instant` for wall-clock measurements and `getrusage` (via the `libc` or `rusage` crate) for CPU utilization breakdown.

### 6.2 Test Matrix

| Parameter | Values |
|-----------|--------|
| Model size | 1.5B (3.5 GB), 7B (14 GB), 13B (26 GB) safetensors shards |
| Loader | `read`, `mmap`, `direct`, `gds` |
| Page cache state | cold (drop_caches before each run), warm (second consecutive load) |
| Runs per config | 5 |

### 6.3 Metrics Collected

1. **Wall-clock time:** end-to-end from first `open` to `cudaDeviceSynchronize`
2. **Throughput:** `total_bytes / total_time` in GB/s
3. **Per-phase breakdown:** I/O read time, safetensors parse time, cudaMalloc time, cudaMemcpy/GDS transfer time
4. **CPU utilization:** user CPU time, system CPU time (from `getrusage`)
5. **Kernel traces:** for the `read` and `mmap` paths, collect bpftrace histograms of:
   - `vfs_read` duration (µs)
   - `submit_bio` → `block_rq_complete` latency (µs)
   - Page cache allocation count
   - Page fault count (via `perf stat -e page-faults`)
6. **Memory usage:** peak RSS of the process (from `/proc/self/status` at key points)

### 6.4 Expected Results (Hypotheses)

Based on known Linux I/O behavior and published GDS benchmarks:

| Loader | Expected throughput (cold) | Expected throughput (warm) | CPU sys% |
|--------|---------------------------|---------------------------|----------|
| `read` | ~3 GB/s (NVMe limit) | ~12 GB/s (DRAM) | medium |
| `mmap` | ~3 GB/s | ~12 GB/s | low-med |
| `direct` | ~3 GB/s | ~3 GB/s (no reuse) | low |
| `gds` | ~3-6 GB/s (PCIe direct) | ~3-6 GB/s | near-zero |

The key insight is expected to be: **page cache helps only on reloads, and even then `mmap` avoids one copy.** For a one-shot startup load, page cache activity is pure overhead. GDS eliminates host involvement entirely, which becomes significant at scale (multi-GPU servers loading models concurrently).

---

## 7. Implementation Order and Milestones

### Milestone 1: Tracing Infrastructure (days 1-2)
- Write and test the four bpftrace scripts (D1)
- Add structured `LoadMetrics` to the `read` loader
- Produce latency histograms for the baseline `read` path (D2)

### Milestone 2: `O_DIRECT` Loader (days 2-3)
- Implement `load_with_direct` with aligned buffers
- Handle remainder reads (file size not multiple of alignment)
- Benchmark vs. baseline `read`, collect bpftrace data

### Milestone 3: `mmap` Loader (days 3-4)
- Add `memmap2` dependency
- Implement `load_with_mmap`
- Trace page fault behavior with `perf stat` and custom bpftrace
- Benchmark vs. `read` and `direct`

### Milestone 4: GDS Loader (days 4-6)
- Set up GDS environment (install `nvidia-fs-dkms`, `libcufile-dev`)
- Validate GDS compatibility of the test system
- Implement `cuFileRead` FFI bindings
- Implement `load_with_gds`
- Benchmark vs. all previous methods
- Measure CPU utilization reduction

### Milestone 5: Comparative Report (day 7)
- Aggregate all benchmark data
- Produce latency breakdown tables and charts
- Write analysis of page cache effect, kernel subsystem traversal, and GDS benefits

---

## 8. Risks and Mitigations

| Risk | Probability | Mitigation |
|------|------------|------------|
| GDS not available on test GPU | Medium | Implement `load_with_gds` under a Cargo feature flag (`gds`); fall back to `direct` if hardware is unsupported |
| safetensors header parsing for GDS requires D→H copy | Certain | Accept one small GPU→CPU transfer per shard for header parsing; this is ~64 KB per shard, negligible vs. multi-GB tensor data |
| `O_DIRECT` alignment bugs (EINVAL on read) | Medium | Use 4096-byte alignment for all buffers and offsets; read first partial block into aligned scratch, then bulk-align the rest |
| bpftrace symbols missing on kernel | Low | Use `tracepoint` alternatives where kprobes fail; fall back to `perf` for high-level PMU counts |
| Warm page cache measurements contaminated by kernel background activity | Medium | Use `echo 3 > /proc/sys/vm/drop_caches` before each cold run; run warm runs immediately after cold to minimize eviction |

---

## 9. Dependencies and Crate Changes

Add to `Cargo.toml`:

```toml
[dependencies]
# ... existing deps ...
memmap2 = "0.9"            # for mmap loader
aligned = "0.4"            # for O_DIRECT aligned buffers
libc = "0.2"               # for O_DIRECT flag, getrusage
```

Optional (feature-gated):
```toml
[features]
gds = []                   # enables cuFile FFI bindings
```

---

## 10. Success Criteria

1. All four loaders produce identical `ModelWeights` (verified by checksumming GPU memory after load, or by running inference and comparing logit outputs for a fixed input).
2. bpftrace scripts produce histograms clearly showing the I/O latency distribution at each kernel layer.
3. The comparative report answers the analytic questions:
   - Does the page cache help or hurt? (Quantified by cold-vs-warm throughput ratio.)
   - Which kernel subsystems does each loader traverse? (Enumerated with evidence from bpftrace.)
   - Why does GDS reduce CPU involvement and memory copies? (Demonstrated by CPU utilization metrics.)
4. GDS throughput is ≥ the best CPU-mediated loader, with CPU utilization ≤ 10% of the `read` baseline.
