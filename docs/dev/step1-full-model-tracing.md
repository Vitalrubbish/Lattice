# Step 1: Full Model Weight Tracing with Real CUDA on WSL2

**Date:** 2026-05-26

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
function prologue.  bpftrace `--unsafe` attaches the uprobe but the probes
never fire.  This is a fundamental WSL2 GPU-PV limitation.  CUDA timing data
comes from Rust's `LoadMetrics` instead.

### 3. Trace Data Collected

Two complete trace runs with the full 2.1 GB TinyLlama model:

| Run | Cache | submit_bio | block I/O | Load Time | Throughput |
|-----|-------|-----------|-----------|-----------|------------|
| Cold | dropped | 2,321 | 2,306 | 4,266 ms | 516 MB/s |
| Warm | hot | 2,209 | 2,270 | 4,166 ms | 528 MB/s |

Key finding: WSL2 page cache provides negligible benefit (~2%) for large
sequential reads due to the storvsc virtual disk layer.  Bare metal is needed
to measure the true page cache benefit (projected 5-10x warm speedup).

### 4. LoadMetrics Cross-Validation

Rust LoadMetrics successfully fills the gap where WSL2 blocks CUDA uprobes:

| Metric | Cold | Warm |
|--------|------|------|
| read_ms | 3,460 | 3,369 |
| parse_ms | 1.09 | 1.02 |
| alloc_ms | 62.0 | 59.6 |
| h2d_ms | 490 | 488 |
| total_ms | 4,266 | 4,166 |
| throughput_mbps | 516 | 528 |

## Usage After This Session

```bash
# Build (no LD_LIBRARY_PATH needed anymore)
cargo build --release

# Cold cache trace
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo bpftrace --unsafe scripts/trace_all.bt \
  -c "timeout 10 ./target/release/baseline-server \
      --model-path /home/vitalrubbish/models/tinyllama \
      --model-type tinyllama --loader read"

# Warm cache trace (immediately after cold)
sudo bpftrace --unsafe scripts/trace_all.bt \
  -c "timeout 10 ./target/release/baseline-server \
      --model-path /home/vitalrubbish/models/tinyllama \
      --model-type tinyllama --loader read"
```

## Remaining Work

- [ ] `NaiveTransformer` ignores loaded weights (uses `alloc_zeros`) — wire up real tensors
- [ ] Bare-metal testing for page cache benefit, NVMe latency, CUDA uprobes, GDS
- [ ] mmap and O_DIRECT loader comparison (needs bare metal for meaningful difference)
- [ ] Step 2 TCP pipeline tracing (can be developed on WSL2)
