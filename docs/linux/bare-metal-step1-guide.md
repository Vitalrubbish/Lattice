# Bare-Metal Linux — Step 1 Trace & Loader-Comparison Guide

**Date:** 2026-05-27

---

## 1. Prerequisites

Run `scripts/setup_cloudlab.sh` first. After reboot, verify:

```bash
nvidia-smi                  # should show A30 with driver 535
nvcc --version               # should show CUDA 12.2
bpftrace --version           # should show v0.16+
cargo --version              # should show 1.78.0
ls /root/models/tinyllama/   # model.safetensors should exist
```

---

## 2. Required Source Modification

One file needs a path change for bare metal. The bpftrace CUDA uprobe path differs between WSL2 and bare metal, and bpftrace's `#define` preprocessor does not expand inside uprobe declarations, so the path must be hardcoded.

### 2.1 Fix CUDA uprobe paths in bpftrace scripts

Two files need the WSL2 path replaced:

```bash
cd /root/llm-rust-ebpf
sed -i 's|/usr/lib/wsl/lib/libcuda.so.1.1|/usr/lib/x86_64-linux-gnu/libcuda.so.1|g' \
    scripts/trace_all.bt \
    scripts/trace_cuda_memcpy.bt
```

Verify (should show only bare-metal paths):
```bash
grep 'libcuda.so' scripts/trace_all.bt
```

### 2.2 No other changes needed

- `build.rs` auto-detects WSL2 and only emits rpath when the WSL2 path exists — no-op on bare metal
- `bench_loaders` reads `MODEL_PATH` env var (defaults to `./models/tinyllama`)
- `load_and_exit.sh` reads `MODEL_PATH` env var (defaults to `./models/tinyllama`)
- `baseline-server` takes `--model-path` as a CLI argument

---

## 3. Build

```bash
cd /root/llm-rust-ebpf
cargo build --release --bin baseline-server --example client --example bench_loaders
```

---

## 4. Trace Test (bpftrace + baseline-server)

### 4.1 Cold cache

```bash
sync && echo 3 > /proc/sys/vm/drop_caches

bpftrace scripts/trace_all.bt \
  -c "timeout 15 ./target/release/baseline-server \
      --model-path /root/models/tinyllama \
      --model-type tinyllama \
      --loader read"
```

### 4.2 Warm cache

Run immediately after cold (do NOT drop caches):

```bash
bpftrace scripts/trace_all.bt \
  -c "timeout 15 ./target/release/baseline-server \
      --model-path /root/models/tinyllama \
      --model-type tinyllama \
      --loader read"
```

### 4.3 VFS-only trace (no CUDA uprobes)

```bash
bpftrace scripts/trace_vfs.bt \
  -c "timeout 15 ./target/release/baseline-server \
      --model-path /root/models/tinyllama \
      --model-type tinyllama \
      --loader read"
```

### 4.4 Bare-metal vs WSL2 expectations

| Metric | WSL2 | Bare metal |
|---|---|---|
| vfs_read calls | 71 | ~71 |
| vfs_read bytes | 2.26 GB | ~2.2 GB |
| block I/O latency | 512-1024 µs | **10-50 µs** |
| CUDA uprobe counters | **0** (didn't fire) | **291** (one per tensor) |
| H→D throughput | 3-5 GB/s | **12-14 GB/s** |
| Total load time (cold) | ~4-5 s | **~0.9-1.5 s** |
| Warm speedup | 2-18% | **5-6x** |

CUDA uprobes firing (291 counts) is the key sign bare-metal tracing works.

---

## 5. Loader Comparison Test

```bash
MODEL_PATH=/root/models/tinyllama SUDO_PASS="<your-password>" \
  sudo -E ./target/release/examples/bench_loaders
```

If `sudo -E` fails with a terminal error, use:

```bash
echo "<password>" | sudo -S env MODEL_PATH=/root/models/tinyllama \
  SUDO_PASS="<password>" ./target/release/examples/bench_loaders
```

### 5.1 Expected output

```
=== read(2) ===
  [cold] total=~900ms read=~700ms ... h2d=~150ms
  [warm] total=~200ms read=~50ms ... h2d=~150ms   (page cache hit)

=== mmap(2) ===
  [cold] total=~900ms (first fault) → ~600ms (subsequent)
  [warm] total=~200ms                              (page cache hit)

=== O_DIRECT ===
  [cold] total=~900ms
  [warm] total=~900ms                              (no page cache by design)
```

Key validations:

- **Cold `read_ms`** similar across all three loaders (all NVMe-bound)
- **Warm `read_ms`** drops dramatically for `read(2)` and `mmap`, stays constant for O_DIRECT
- **`h2d_ms`** ~150-200 ms for 2.2 GB on A30 (PCIe Gen4, 12-14 GB/s)
- Cross-validate: bpftrace CUDA uprobe counts (291) = LoadMetrics tensor count

---

## 6. Standalone Server + Client (Optional)

```bash
MODEL=/root/models/tinyllama LOADER=read ./scripts/benchmark.sh
```
