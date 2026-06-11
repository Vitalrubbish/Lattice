# Step 1 Loader Comparison: read(2) vs O_DIRECT vs mmap(2)

**Date:** 2026-05-27
**Environment:** WSL2 (Ubuntu 24.04), kernel 6.6.87.2-microsoft-standard-WSL2
**GPU:** NVIDIA GeForce RTX 5070 (8 GB), CUDA 13.1, Driver 591.97 (WSL2 GPU-PV)
**Model:** TinyLlama 1.1B, 2.1 GB safetensors (single shard), 291 tensors

---

## 1. Summary

Three weight-loading strategies were benchmarked on a cold and warm NVMe page cache:

| Method | Cold 1st | Cold best | Warm avg | Warm MB/s | Mechanism |
|--------|----------|-----------|----------|-----------|-----------|
| `read(2)` | 4704 ms | 2121 ms | 2018 ms | ~1094 | VFS → page cache → copy_to_user → cudaMemcpy |
| `mmap(2)` | 4322 ms | 611 ms | **607 ms** | **~3622** | mmap (fast) → page-fault-driven read → cudaMemcpy |
| `O_DIRECT` | 3669 ms | 3331 ms | 3653 ms | ~612 | generic_file_direct_read → buffer → cudaMemcpy |

**Takeaway:** `mmap(2)` gives the best warm-cache throughput (~3.6 GB/s) and the fastest cold-to-warm transition because it decouples "open" from "read" — the mapping is instant (0 ms) and actual I/O happens lazily during the cudaMemcpy. `read(2)` is simpler and gets a page-cache speedup. `O_DIRECT` bypasses the page cache entirely, yielding consistent but slower cold performance and **no warm-cache benefit**.

---

## 2. Full Per-Run Data

### 2.1 read(2) — VFS read + copy_to_user

```
Run   Cache   Total    Read     Parse  Alloc  H2D     CPU_user  CPU_sys   Throughput
1     cold    4704 ms  3943 ms  1 ms   56 ms  488 ms   464 ms   4069 ms    468 MB/s
2     cold    2439 ms  1695 ms  0 ms   39 ms  470 ms   417 ms   1945 ms    902 MB/s
3     cold    2121 ms  1422 ms  0 ms   37 ms  425 ms   394 ms   1586 ms   1037 MB/s
1     warm    2063 ms  1419 ms  0 ms   33 ms  400 ms   374 ms   1615 ms   1066 MB/s
2     warm    2139 ms  1498 ms  0 ms   34 ms  391 ms   368 ms   1746 ms   1029 MB/s
3     warm    1852 ms  1212 ms  0 ms   34 ms  382 ms   352 ms   1402 ms   1188 MB/s
```

- **Cold → warm gap:** 1st cold (4704 ms) → 3rd cold (2121 ms) shows page cache fillup over 3 runs
- **Warm stable state:** ~2.0 s, ~10.9 GB/s
- **Dominant cost:** `read_ms` (~70% of total) — the blocking `read(2)` syscall into a `Vec<u8>`
- **CPU sys time** is high (1.4–4.0 s) reflecting the kernel-side I/O work (submit_bio, NVMe completion, page cache insertion)

### 2.2 mmap(2) — page-fault-driven load

```
Run   Cache   Total    Read    Parse  Alloc  H2D     CPU_user  CPU_sys   Throughput
1     cold    4322 ms   0 ms   1 ms   43 ms  4165 ms  202 ms   1789 ms    509 MB/s
2     cold     968 ms   0 ms   0 ms   34 ms   635 ms  463 ms    481 ms   2272 MB/s
3     cold     611 ms   0 ms   0 ms   33 ms   489 ms  340 ms    247 ms   3602 MB/s
1     warm     598 ms   0 ms   0 ms   33 ms   476 ms  280 ms    296 ms   3677 MB/s
2     warm     600 ms   0 ms   0 ms   41 ms   469 ms  261 ms    306 ms   3666 MB/s
3     warm     624 ms   0 ms   0 ms   34 ms   494 ms  325 ms    248 ms   3523 MB/s
```

- **`read_ms` is always 0** — `mmap(2)` only sets up page table entries; no data is copied
- **Cold 1st h2d=4165 ms** — the cudaMemcpy touches every page for the first time, triggering major page faults (`filemap_fault → readpage → submit_bio`) inline during the copy
- **Cold 2nd already 968 ms** — only 1 previous pass is enough to populate much of the page cache
- **Warm stable state: ~607 ms**, fastest of all methods — cudaMemcpy reads from hot kernel pages with no copy_to_user overhead
- **CPU sys time drops** from 1789 ms (cold 1st) to ~280 ms (warm) — page faults are resolved from cache, no NVMe I/O needed

### 2.3 O_DIRECT — bypass page cache

```
Run   Cache   Total    Read     Parse  Alloc  H2D     CPU_user  CPU_sys   Throughput
1     cold    3669 ms  2920 ms  0 ms   37 ms  474 ms   436 ms   1665 ms    600 MB/s
2     cold    3331 ms  2642 ms  0 ms   34 ms  417 ms   411 ms   1469 ms    660 MB/s
3     cold    3828 ms  2800 ms  0 ms   43 ms  698 ms   666 ms   1494 ms    575 MB/s
1     warm    4311 ms  3552 ms  0 ms   37 ms  474 ms   447 ms   2409 ms    510 MB/s
2     warm    3487 ms  2760 ms  0 ms   36 ms  428 ms   395 ms   1462 ms    631 MB/s
3     warm    3161 ms  2474 ms  0 ms   33 ms  432 ms   423 ms   1371 ms    696 MB/s
```

- **No warm/cold distinction** — both hover around 3.3–3.8 s because the page cache is bypassed
- **Consistent but slow** — always pays the full NVMe read cost
- **Variance** (3.1–4.3 s) is higher than `read(2)` due to the multiple 2 MiB `pread` chunks (O_DIRECT has stricter alignment requirements)
- **Useful as a baseline** for "what if there were no page cache at all"

---

## 3. Data Path Comparison

```
                    read(2)               mmap(2)                 O_DIRECT
──────────────────────────────────────────────────────────────────────────────
VFS entry           vfs_read              filemap_fault           generic_file_direct_read
Page cache          populated             populated               bypassed
Kernel→user copy    copy_to_user          none (user==page)       DMA→user buffer
GPU copy            cudaMemcpy H→D        cudaMemcpy H→D         cudaMemcpy H→D
CPU copies          2                     1                       1
Double buffering    yes (pcache + Vec)    no                      no
Warm-cache benefit  yes                   yes (strongest)         no
```

---

## 4. How to Run This Test

### 4.1 Quick Run (Automated)

```bash
cd /mnt/d/os/llm-rust-ebpf
bash scripts/step1_test_wsl2.sh
```

This runs cold trace, warm trace, and loader comparison in one shot. Output goes to `results/wsl2/<timestamp>/`.

To run only the loader comparison:

```bash
MODEL_PATH=/home/vitalrubbish/models/tinyllama SUDO_PASS="<password>" \
  sudo -E ./target/release/examples/bench_loaders
```

### 4.2 Manual Build

```bash
cargo build --release --example bench_loaders
```

### 4.3 Run

The benchmark drops the kernel page cache between methods (cold runs), so it needs the sudo password. The `MODEL_PATH` env var sets the model location (defaults to `./models/tinyllama`):

```bash
MODEL_PATH=/home/vitalrubbish/models/tinyllama SUDO_PASS="<password>" \
  sudo -E ./target/release/examples/bench_loaders
```

If sudo requires a terminal, pipe the password via stdin:

```bash
echo "<password>" | sudo -S env MODEL_PATH=/home/vitalrubbish/models/tinyllama \
  SUDO_PASS="<password>" ./target/release/examples/bench_loaders
```

If sudo is not available or you want to skip cache drop, set `SUDO_PASS=""` — the benchmark will still run but cold/warm distinction will blur.

### 4.4 What it does

1. For each loader (`read`, `mmap`, `direct`):
   - Drops page cache (`echo 3 > /proc/sys/vm/drop_caches`)
   - Runs 3 cold loads (cache empty)
   - Runs 3 warm loads (cache hot from the cold runs)
2. Each run creates a fresh `CudaContext`, loads the safetensors file, and reports:
   - `total_ms` — wall clock
   - `read_ms` — file I/O portion
   - `parse_ms` — safetensors JSON header parsing
   - `alloc_ms` — GPU memory allocation (`cuMemAlloc_v2`)
   - `h2d_ms` — host-to-device memcpy
   - `cpu_user_ms` / `cpu_sys_ms` — per-thread CPU time from `getrusage`
   - `throughput` — total bytes / total time in MB/s

### 4.5 Model path

Set the `MODEL_PATH` env var to point to a safetensors file or directory. Default is `./models/tinyllama`. The path accepts both single `.safetensors` files and directories containing sharded `.safetensors` files.

### 4.6 Troubleshooting

- **CUDA_ERROR_NO_DEVICE on WSL2:** The binary needs `rpath` to `/usr/lib/wsl/lib`. The `build.rs` handles this automatically.
- **sudo prompt hangs:** Make sure `SUDO_PASS` is set correctly or the user has passwordless sudo for `sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"`.
- **MODEL_PATH not found:** Verify the path exists and contains `.safetensors` files. The loader accepts both single files and directories.
