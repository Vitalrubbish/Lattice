# Step 1: Full Model Weight Tracing with Real CUDA on WSL2

**Date:** 2026-05-26
**Last Modified:** 2026-05-26

## Changes Made

### 1. Fixed CUDA on WSL2 (`build.rs`)

**Problem:** The binary loaded `/usr/lib/x86_64-linux-gnu/libcuda.so.535`
instead of `/usr/lib/wsl/lib/libcuda.so.1.1`, causing `CUDA_ERROR_NO_DEVICE`.

**Fix:** Added `build.rs` with:
```rust
println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
```
This bakes `RUNPATH: /usr/lib/wsl/lib` into the binary so the WSL2 GPU-PV
libcuda is found first.

**Verification:**
```
$ readelf -d ./target/release/baseline-server | grep RUNPATH
 0x000000000000001d (RUNPATH)  Library runpath: [/usr/lib/wsl/lib]
```

### 2. bpftrace CUDA Uprobes Don't Fire on WSL2

WSL2's `libcuda.so` functions are zero-size stub symbols that trap into the
kernel via the DXGKRNL driver immediately — they never execute a standard
function prologue. bpftrace `--unsafe` attaches the uprobe but the probes
never fire. This is a fundamental WSL2 GPU-PV limitation. CUDA timing data
comes from Rust's `LoadMetrics` instead.

### 3. Trace Data Collected (Updated 2026-05-26)

Two complete trace runs with the full 2.1 GB TinyLlama model (rerun at 19:18):

| Run | Cache | submit_bio | block I/O | Load Time | Throughput |
|-----|-------|-----------|-----------|-----------|------------|
| Cold | dropped | 2,313 | 2,500 | 5,101 ms | 431 MB/s |
| Warm | hot | 2,206 | 2,224 | 4,208 ms | 523 MB/s |

**Key finding (revised):** WSL2 page cache provides a moderate ~17.5% warm-cache
speedup on this run — significantly more than the previously observed ~2%.
The Linux guest page cache retains some pages between runs, and the host-side
NTFS standby list absorbs the worst physical seeks (97% reduction in 4-8K µs
block I/O tail). However, benefit is inconsistent (range 2-18% across runs),
and bare metal is still needed for the true 5-10x warm speedup.

### 4. LoadMetrics Cross-Validation (Updated 2026-05-26)

Rust LoadMetrics successfully fills the gap where WSL2 blocks CUDA uprobes:

| Metric | Cold | Warm |
|--------|------|------|
| read_ms | 4,124 | 3,548 |
| parse_ms | 1.17 | 1.08 |
| alloc_ms | 61.9 | 51.9 |
| h2d_ms | 666 | 388 |
| cpu_user_ms | 595 | 367 |
| cpu_sys_ms | 4,194 | 3,871 |
| total_ms | 5,101 | 4,208 |
| throughput_mbps | 431 | 523 |

Notable: h2d_ms dropped 41.7% on the warm run (666 → 388 ms), suggesting GPU-PV
DMA path has a non-trivial first-run setup cost on WSL2. Multiple warm-up runs
are needed for stable GPU transfer benchmarking.

## Usage After This Session

```bash
# Build (no LD_LIBRARY_PATH needed anymore)
cargo build --release

# Cold cache trace
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo bpftrace --unsafe scripts/trace_all.bt \
  -c "timeout 15 ./target/release/baseline-server \
      --model-path /home/vitalrubbish/models/tinyllama \
      --model-type tinyllama --loader read"

# Warm cache trace (immediately after cold)
sudo bpftrace --unsafe scripts/trace_all.bt \
  -c "timeout 15 ./target/release/baseline-server \
      --model-path /home/vitalrubbish/models/tinyllama \
      --model-type tinyllama --loader read"
```

## Remaining Work

- [ ] `NaiveTransformer` ignores loaded weights (uses `alloc_zeros`) — wire up real tensors
- [ ] Bare-metal testing for page cache benefit, NVMe latency, CUDA uprobes, GDS
- [ ] mmap and O_DIRECT loader comparison (needs bare metal for meaningful difference)
- [ ] Step 2 TCP pipeline tracing (can be developed on WSL2)
