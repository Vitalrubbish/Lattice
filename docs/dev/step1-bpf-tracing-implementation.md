# eBPF Tracing Implementation

**Date:** 2026-05-26

## Overview

The project uses bpftrace to instrument the complete data path that model weights travel from disk to GPU VRAM. Six scripts collectively attach probes at the VFS layer, page cache, block I/O layer, CUDA driver API, and TCP stack. This document explains how each script works and how to interpret its output.

All scripts run under `sudo bpftrace` and target the `baseline-server` process by filtering on `comm == "baseline-server"` (kernel probes) or tracking per-tid state.

---

## 1. bpftrace Mechanism

bpftrace compiles a high-level awk-like script into Linux BPF bytecode, loads it into the kernel, and attaches it to probe points. When the probed function executes, the BPF program runs, records timestamps or counters in BPF maps (kernel-resident hash tables), and the bpftrace userspace process reads and prints results.

**Probe types used in this project:**

| Probe type | Example | What it hooks |
|---|---|---|
| `kprobe` / `kretprobe` | `vfs_read` | Kernel function entry and return |
| `tracepoint` | `block:block_rq_issue` | Static kernel tracepoints (stable ABI) |
| `uprobe` / `uretprobe` | `cuMemcpyHtoD_v2` | Userspace function entry/return in a shared library |
| `interval` | `interval:s:1` | Timer-based periodic reporting |
| `BEGIN` / `END` | — | Script startup and shutdown |

**Core pattern** — per-thread timing via BPF maps:

```
kprobe:target_fn /comm == "baseline-server"/
{
    @start[tid] = nsecs;       // record start time, keyed by thread ID
}

kretprobe:target_fn /@start[tid] != 0/
{
    $dur_us = (nsecs - @start[tid]) / 1000;  // compute duration
    @lat_hist = hist($dur_us);                // accumulate histogram
    delete(@start[tid]);                      // clean up
}
```

The `@start[tid]` map is keyed by kernel thread ID (`tid`), so concurrent I/O from different threads doesn't interfere. The guard `/@start[tid] != 0/` on the kretprobe prevents processing returns from threads that weren't in the function when tracing started.

---

## 2. Script-by-Script Breakdown

### 2.1 `trace_vfs.bt` — VFS → Page Cache → Block I/O

**Purpose:** Trace the full Linux I/O stack for model weight reading.

**Probes attached (7):**

| Probe | Function | What it measures |
|---|---|---|
| `kprobe:vfs_read` | Kernel VFS read entry | Start timestamp, byte count (`arg2`) |
| `kretprobe:vfs_read` | Kernel VFS read return | Latency histogram, total bytes, emission of slow (>1ms) reads |
| `kprobe:filemap_get_pages` | Page cache lookup entry | Start timestamp |
| `kretprobe:filemap_get_pages` | Page cache lookup return | Latency histogram, call count |
| `kprobe:add_to_page_cache_lru` | New page added to cache | Counter (indicates cold read / cache miss) |
| `kprobe:mark_page_accessed` | Page accessed (LRU reorder) | Counter (indicates warm reuse) |
| `kprobe:submit_bio` | Block I/O submission | Start timestamp, bio counter |
| `kretprobe:submit_bio` | Block I/O submission return | Latency histogram |
| `tracepoint:block:block_rq_issue` | Request issued to device driver | Start timestamp keyed by `(dev, sector)`, per-device byte sum |
| `tracepoint:block:block_rq_complete` | Request completed by device driver | Issue→complete latency histogram |

**How the block tracepoint tracking works:**

Block requests are matched by `(args->dev, args->sector)` tuple — the device major:minor and starting sector uniquely identify a request from issue to completion:

```
tracepoint:block:block_rq_issue
{
    @issue_ts[args->dev, args->sector] = nsecs;
    @bytes_per_dev[args->dev] = sum(args->bytes);
}

tracepoint:block:block_rq_complete
/@issue_ts[args->dev, args->sector] != 0/
{
    $dur_us = (nsecs - @issue_ts[args->dev, args->sector]) / 1000;
    @block_io_lat = hist($dur_us);
    delete(@issue_ts[args->dev, args->sector]);
}
```

The guard `/@issue_ts[args->dev, args->sector] != 0/` ensures we only process completions for requests we saw being issued. Without this, stray completions from before tracing started would produce bogus negative deltas.

**Periodic status** (`interval:s:1`): prints live counters every second showing bios submitted/completed and vfs_read count, giving a real-time view of I/O progress during loading.

**Output:** The `END` block prints histograms for each layer and a breakdown of bytes per block device, answering "which disk did the data come from?"

### 2.2 `trace_page_cache.bt` — Page Cache Hit/Miss Analysis

**Purpose:** Determine whether the page cache helps or hurts during model loading.

**Probes attached (7):**

| Probe | What it tells us |
|---|---|
| `kprobe:do_fault` | Major page faults — when a page is not in cache and I/O is needed |
| `kretprobe:do_fault` | Fault handling latency |
| `kprobe:add_to_page_cache_lru` | New pages brought into cache from disk |
| `kprobe:find_get_page` | Page cache lookups (both hits and misses) |
| `kprobe:page_cache_sync_readahead` | Synchronous readahead — kernel prefetches pages during sequential read. `arg1` gives the number of pages requested |
| `kprobe:page_cache_async_readahead` | Async readahead — background prefetch after a readahead window is hit |
| `kprobe:mark_buffer_dirty` | WARNING probe — should be zero during pure reads |
| `kprobe:shrink_page_list` | Page eviction under memory pressure — should be zero |

**Key analysis:** The END block computes a readahead-to-I/O ratio:

```
$ratio = $total_ra_pages * 100 / $total_io_pages;
```

- Ratio > 150%: heavy readahead, kernel detected strong sequential pattern
- Ratio 80-150%: moderate, expected for sequential reads
- Ratio < 80%: possible random I/O or very small reads

This ratio tells you whether the kernel's readahead logic is helping (prefetching useful pages) or wasting I/O bandwidth (prefetching pages the application doesn't need).

**Cold vs. warm comparison:** To test page cache effect, run twice:
```
# cold
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo bpftrace scripts/trace_page_cache.bt -c "./baseline-server ..."

# warm (immediately after, no drop_caches)
sudo bpftrace scripts/trace_page_cache.bt -c "./baseline-server ..."
```

On cold: most `do_fault` events are major faults, `add_to_page_cache_lru` count ≈ pages in the model file. On warm: zero or near-zero `add_to_page_cache_lru`, `find_get_page` hits in cache.

### 2.3 `trace_cuda_memcpy.bt` — CUDA Driver API Tracing

**Purpose:** Measure GPU memory allocation and host↔device transfer latency.

**Probes attached (3 pairs of uprobe/uretprobe):**

| uprobe | Function | Arg tracked |
|---|---|---|
| `libcuda.so:cuMemAlloc_v2` | GPU memory allocation | `arg1` = byte size |
| `libcuda.so:cuMemcpyHtoD_v2` | Host → device copy | `arg2` = byte count |
| `libcuda.so:cuMemcpyDtoH_v2` | Device → host copy | `arg2` = byte count |

**How uprobes work here:** Unlike kernel probes, uprobes attach to a userspace shared library. bpftrace places a breakpoint instruction at the function entry point. When any thread in the target process calls that function, the CPU hits the breakpoint, the kernel runs the BPF program, then returns control to the original instruction.

The script needs the path to `libcuda.so`:

```
#ifndef CUDA_LIB
#define CUDA_LIB /usr/lib/wsl/lib/libcuda.so.1.1
#endif
```

On bare metal, this is typically `/usr/local/cuda/lib64/libcuda.so` or `/usr/lib/x86_64-linux-gnu/libcuda.so.1`. Pass it explicitly if auto-detection fails:

```
sudo bpftrace scripts/trace_cuda_memcpy.bt -c "..." \
  -b /usr/local/cuda/lib64/libcuda.so
```

**Output:** The END block prints per-function call counts, total bytes transferred, and latency histograms. For the `read` loader, H→D transfers should dominate (uploading weights to GPU); D→H transfers happen only during inference for logit retrieval.

**WSL2 caveat:** On WSL2, `libcuda.so` contains stub functions that forward to the Windows host-side driver. The symbols exist (verified with `nm -D`) but CUDA runtime may return `CUDA_ERROR_NO_DEVICE`. The uprobes will fire only if the function actually executes — if the CUDA init fails before any memcpy, these probes record zero events.

### 2.4 `trace_all.bt` — Combined Full-Path Trace

**Purpose:** Run all the above probes in a single bpftrace invocation for a complete picture.

This script merges all probes from `trace_vfs.bt` and `trace_cuda_memcpy.bt`, plus adds `kprobe:page_cache_ra_order` (readahead tracking). It omits the detailed page cache probes from `trace_page_cache.bt` (do_fault, find_get_page, etc.) to keep probe count manageable.

**Periodic status** runs every second and prints live values for all counters:

```
[1s] vfs=106 bio=121 blk-iss=120 blk-done=120 h2d=20 d2h=0 alloc=21 pg-new=350
```

This gives a real-time dashboard of I/O progress — you can watch the model load progress through each layer of the stack.

**END report** prints everything: per-layer call counts, total bytes, and latency histograms for all seven instrumented functions.

### 2.5 `trace_nvme.bt` — Block Layer Only (Lightweight)

Minimal script: attaches only to `block:block_rq_issue` and `block:block_rq_complete`. Use when you want block I/O latency data without the overhead of 15+ other probes.

### 2.6 `trace_tcp.bt` — TCP Send/Recv for Pipeline Parallel

**Purpose:** Measure TCP stack latency during pipeline-parallel activation transfers between GPUs (Step 2).

Attaches to `tcp_sendmsg`, `tcp_recvmsg`, and `skb_copy_datagram_iter` — the core kernel functions in the TCP send and receive paths. The `skb_copy_datagram_iter` probe specifically measures the time spent copying data from socket buffers to userspace, which is the primary source of per-packet CPU overhead.

Periodic status prints send and receive call rates per 5-second window.

---

## 3. Cross-Validation: Rust LoadMetrics vs. bpftrace

The Rust code in `src/model/loader.rs` independently measures the same I/O phases so that userspace and kernel-space measurements can be compared.

### 3.1 CpuTimer — Userspace Timing

```rust
struct CpuTimer {
    t0: Instant,          // wall-clock
    ru0: libc::rusage,    // CPU time (RUSAGE_THREAD)
}
```

- `start()` — captures `Instant::now()` and `getrusage(RUSAGE_THREAD)` 
- `elapsed()` — returns `(wall_ms, user_cpu_ms, sys_cpu_ms)` since start

This uses `RUSAGE_THREAD` (not `RUSAGE_SELF`) so measurements are per-thread, matching bpftrace's per-tid tracking.

### 3.2 LoadMetrics — Structured Phase Breakdown

The loader records wall-clock time per phase:

| Field | bpftrace equivalent |
|---|---|
| `read_ms` | `vfs_read` latency histogram |
| `parse_ms` | Not directly traced (CPU-only, no kernel involvement) |
| `alloc_ms` | `cuMemAlloc_v2` histogram |
| `h2d_ms` | `cuMemcpyHtoD_v2` histogram |
| `cpu_user_ms` / `cpu_sys_ms` | No direct bpftrace equivalent; validates that bpftrace-reported kernel time matches `RUSAGE_THREAD` sys time |

### 3.3 Validation Approach

For a correct implementation, `read_ms` (Rust) should approximately match the sum of `vfs_read` durations (bpftrace), and `h2d_ms` should match the sum of `cuMemcpyHtoD_v2` durations. Discrepancies indicate either a missing probe (some I/O happening through an untraced path) or measurement overhead.

---

## 4. Data Path Visualized

```
                      trace_vfs.bt                    trace_cuda_memcpy.bt
                    ═══════════════                  ══════════════════════
Disk (NVMe)
    │
    ▼
block_rq_issue ─── block:block_rq_issue
    │
    ▼
block_rq_complete  block:block_rq_complete
    │
    ▼
submit_bio ─────── kprobe:submit_bio
    │
    ▼
page cache ─────── kprobe:filemap_get_pages
                   kprobe:add_to_page_cache_lru
    │
    ▼
vfs_read ───────── kprobe:vfs_read
    │
    ▼
Userspace Vec<u8>
    │
    ▼
cuMemAlloc ──────                              uprobe:cuMemAlloc_v2
    │
    ▼
cuMemcpyHtoD ────                              uprobe:cuMemcpyHtoD_v2
    │
    ▼
GPU VRAM
```

The scripts `trace_vfs.bt` and `trace_cuda_memcpy.bt` split neatly at the userspace boundary. `trace_all.bt` covers the entire path in a single run.

---

## 5. Usage Patterns

### Cold cache measurement
```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo bpftrace scripts/trace_all.bt \
  -c "./target/release/baseline-server --model-path ./models/llama-7b --loader read"
```

### Warm cache measurement (run immediately after cold)
```bash
sudo bpftrace scripts/trace_all.bt \
  -c "./target/release/baseline-server --model-path ./models/llama-7b --loader read"
```

### Attach to running process
```bash
sudo bpftrace scripts/trace_all.bt -p $(pgrep baseline-server)
```

### Single-script targeted tracing
```bash
# Just block I/O latency
sudo bpftrace scripts/trace_nvme.bt -c "./baseline-server ..."

# Just page cache behavior
sudo bpftrace scripts/trace_page_cache.bt -c "./baseline-server ..."

# Just CUDA transfers
sudo bpftrace scripts/trace_cuda_memcpy.bt -c "./baseline-server ..."
```

### WSL2-specific: override CUDA library path
```bash
sudo bpftrace scripts/trace_all.bt \
  -b /usr/lib/wsl/lib/libcuda.so.1.1 \
  -c "./target/release/baseline-server ..."
```

---

## 6. Interpreting Key Metrics

| Observation | What it means |
|---|---|
| Bimodal `vfs_read` latency (fast cluster at ~5 µs, slow at ~300 µs) | Fast = page cache hits; slow = physical I/O. The ratio tells you the cache hit rate |
| `add_to_page_cache_lru` > 0 on warm run | Page cache was partially evicted — memory pressure or cache size limit |
| `page_cache_alloc` ≈ 0 but many bios | Readahead pages being allocated, not demand-fault pages |
| `submit_bio` count ≈ 0 on warm run | All data served from page cache, zero disk I/O |
| `block_rq_issue` latency > 1ms | Storage bottleneck — check NVMe queue depth or PCIe bandwidth |
| `cuMemAlloc` latency > 1ms | GPU memory fragmentation or driver overhead — consider pre-allocation |
| `cuMemcpyHtoD` throughput < 10 GB/s | PCIe bottleneck — check link width/speed with `nvidia-smi topo -m` |
| `shrink_page_list` > 0 | Memory pressure — model file too large for available page cache |
