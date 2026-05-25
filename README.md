# Latttice: An Efficient OS Support Layer for Large Language Model (LLM) Inference

A Linux + Rust-based OS support layer tailored for LLM inference workloads, featuring
kernel-level PagedAttention, virtual GPU memory management, coroutine-based
heterogeneous scheduling, and eBPF network offloading.

## Background

LLM inference (GPT, LLaMA, etc.) is split into two stages — **Prefill** (compute-bound)
and **Decode** (memory-bound). Long-context inference generates enormous KV Caches, and
traditional OS memory allocation causes heavy fragmentation that limits GPU utilization
and throughput.

## Goals

- **PagedAttention & virtual GPU memory**: On-demand physical allocation driven by
  kernel page fault handling, with automatic offload/reload of GPU memory.
- **Copy-on-Write for beam search**: Multiple candidate sequences share underlying
  physical KV Cache pages, with forking on write guarded by reference counts.
- **eBPF network offloading**: Parse inference requests at the NIC via XDP/TC hooks,
  bypassing the socket buffer layer for zero-copy data flow.
- **Distributed inference acceleration**: eBPF-based NCCL bypass over AF_XDP sockets
  to avoid the kernel TCP/IP stack for pipeline-parallel activation transfers.

Baselines: vLLM, SGLang.

## Project Steps

| Step | Topic | Weight |
|------|-------|--------|
| 1 | Model weight loading & I/O stack analysis — bpftrace VFS → DMA, compare `read` / `mmap` / `O_DIRECT` / GDS `cuFileRead` | 15% |
| 2 | eBPF distributed inference network acceleration — XDP/TC bypass for NCCL traffic, AF_XDP + GDRCopy | 25% |
| 3 | Continuous batching & KV Cache memory management — CUDA VMM API (`cuMemCreate` / `cuMemMap`), paged GPU memory, block table attention | 35% |
| 4 | Prefix sharing & fine-grained page tables — reference-counted prefix caching, modify NVIDIA open-gpu-kernel-modules for 64 KB allocation granularity | 25% |

## Build

```
cargo build --release
```

Requires Linux + CUDA 12.x. cudarc links `libcuda.so`.

## Run

```
RUST_LOG=info ./target/release/baseline-server \
    --listen 127.0.0.1:8000 \
    --model-path dummy \
    --max-batch 8 \
    --max-seq-len 2048
```

`--model-path dummy` skips weight loading and uses zero buffers.
Pass a directory of `.safetensors` for real weights.

`--loader read` is the only implemented path. The `mmap`, `direct`, and `gds` arms
return errors — those are Step 1.

## Bench

```
CONC=16 PLEN=256 NEW=64 bash scripts/benchmark.sh
```

## bpftrace

```
sudo bpftrace scripts/trace_vfs.bt -c "./target/release/baseline-server ..."
sudo bpftrace scripts/trace_nvme.bt
sudo bpftrace scripts/trace_tcp.bt
```

## Layout

| Crate | Purpose |
|-------|---------|
| `src/cuda` | cudaMalloc / cudaMemcpy + cuBLAS wrappers |
| `src/model/loader` | `read()` + `cudaMemcpy` weight loader |
| `src/model/transformer` | Placeholder forward pass (cuBLAS GEMMs, no real attention) |
| `src/cache/kv_cache` | Contiguous KV Cache, allocated once at `max_batch × max_seq_len` |
| `src/batch/static_batch` | Static batching, padded to max prompt |
| `src/decoder/greedy` | Host-side argmax |
| `src/server/http` | Tokio TCP/JSON, one request per connection |
| `src/server/pipeline` | TCP send/recv between PP stages (not wired into `main`) |