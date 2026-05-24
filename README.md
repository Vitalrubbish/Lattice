# baseline-llm-os

Naive Rust LLM inference server used as the baseline for the course project.

## Build

```
cargo build --release
```

Needs Linux + CUDA 12.x. cudarc links `libcuda.so`.

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

`--loader read` is the only path that's implemented.
The `mmap`, `direct`, `gds` arms return errors — that's step 1.

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

- `src/cuda`          — cudaMalloc/cudaMemcpy + cuBLAS wrappers
- `src/model/loader`  — `read()` + `cudaMemcpy` weight loader
- `src/model/transformer` — placeholder forward (cuBLAS GEMMs, no real attention)
- `src/cache/kv_cache` — contiguous KV cache, allocated once at `max_batch * max_seq_len`
- `src/batch/static_batch` — static batching, padded to max prompt
- `src/decoder/greedy` — host-side argmax
- `src/server/http`   — tokio TCP/JSON, one request per connection
- `src/server/pipeline` — TCP send/recv between PP stages (not wired into `main` yet)
