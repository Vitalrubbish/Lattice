# Step 3: Continuous Batching & Paged KV Cache — Development Document

**Date:** 2026-05-26

## Overview

Step 3 replaces the current static batching + contiguous KV cache with:

1. **Continuous batching scheduler** — dynamically adds/removes requests mid-decode
2. **Paged KV cache via CUDA VMM API** — `cuMemCreate`/`cuMemMap`/`cuMemAddressReserve` for on-demand physical allocation
3. **Block-table attention** — virtual-to-physical address translation in the attention kernel

The implementation is done "manually" — direct CUDA driver API calls via `cudarc` FFI, no dependency on vLLM or CUTLASS.

## Current State (What Exists)

| Component | File | Mechanism |
|-----------|------|-----------|
| Scheduler | `src/batch/static_batch.rs` | `StaticScheduler`: collects N requests, runs prefill for all, then decode for all, releases slots |
| KV Cache | `src/cache/kv_cache.rs` | `KvCache`: contiguous `cudaMalloc` tensors sized `max_batch × kv_heads × max_seq_len × head_dim` |
| Slot mgmt | `src/cache/kv_cache.rs` | `SlotAllocator`: free-list of integer slot IDs |
| Attention | `src/model/transformer.rs` | `NaiveTransformer::forward_step`: fake FFN (GEMM on zero weights), writes KV via `append_step` |
| CUDA runtime | `src/cuda/runtime.rs` | `Blas`: cuBLAS hgemm wrapper |

### Limitations to Address

- **Static batching**: Requests in a batch must finish together. A long-generation request blocks short ones.
- **Contiguous KV cache**: Pre-allocates `max_batch × max_seq_len` per layer — wastes GPU memory when actual usage is lower.
- **No CUDA VMM**: Physical memory is allocated upfront. No on-demand allocation, no memory sharing.
- **Slot-based indexing**: Attention doesn't use a block table; KV cache is indexed by `(slot, position)` directly.

## Design

### 1. Paged KV Cache (`src/cache/paged_kv.rs`)

#### 1.1 Block Definition

```
Block size: BLOCK_SIZE = 16 tokens
Each block stores: [kv_heads × BLOCK_SIZE × head_dim] f16 elements per layer
```

A block is the unit of physical GPU memory allocation. All KV data for a given layer, head, and block of token positions is stored contiguously within one block.

#### 1.2 Block Table

```
BlockTable {
    blocks: Vec<Vec<u32>>,  // [num_requests][max_blocks_per_seq] — maps logical block idx → physical block idx
    num_blocks_per_seq: Vec<usize>,  // current allocated length per sequence
}
```

Each request has a list of physical block indices. The logical position `pos` maps to logical block `pos / BLOCK_SIZE`, which maps to a physical block via the table.

#### 1.3 Physical Block Allocator

```
PhysicalBlockAllocator {
    free_blocks: Mutex<Vec<u32>>,
    total_blocks: u32,
    block_size: usize,        // tokens per block
    block_bytes: usize,       // bytes per block per layer
}
```

Allocation policy:
- On prefill: allocate `ceil(prompt_len / BLOCK_SIZE)` blocks for the request
- On decode: when `pos % BLOCK_SIZE == 0`, allocate one more block
- On request completion: free all blocks belonging to that request

#### 1.4 CUDA VMM Integration

Instead of `cudaMalloc` for each block, use CUDA Virtual Memory Management API:

```
cuMemAddressReserve  → reserve a large contiguous VA region (the "KV pool")
cuMemCreate          → allocate physical GPU memory in block-sized chunks
cuMemMap             → map physical blocks into the VA region on demand
cuMemUnmap           → unmap when a request finishes
cuMemRelease         → free physical memory
cuMemAddressFree     → tear down the VA region on shutdown
```

This gives us:
- **On-demand physical allocation**: only allocate blocks when a request needs them
- **Sparse VA space**: virtual addresses are contiguous, physical blocks are scattered
- **Fine-grained reclaim**: free individual blocks, not the whole cache

Default granularity: CUDA VMM min allocation is 2 MB (hardware limitation). For a block of ~256 KB (16 tokens × 4 heads × 128 dim × 2 bytes), we batch-allocate multiple blocks into one 2 MB physical handle and sub-allocate from it.

#### 1.5 KV Cache Data Structure

```
PagedKvCache {
    cfg: ModelConfig,
    ctx: Arc<CudaContext>,
    max_batch: usize,
    max_seq_len: usize,
    block_size: usize,
    num_layers: usize,

    // CUDA VMM handles
    va_regions: Vec<u64>,        // one reserved VA region per layer
    physical_handles: Vec<Vec<u64>>,  // [layer][handle_idx] — 2MB physical allocations

    // Block allocator
    block_allocator: Arc<PhysicalBlockAllocator>,

    // Per-request metadata
    block_tables: Mutex<Vec<Vec<u32>>>,
    seq_lens: Mutex<Vec<usize>>,
}
```

### 2. Continuous Batching Scheduler (`src/batch/continuous_scheduler.rs`)

#### 2.1 Lifecycle

```
 ┌──────────┐     ┌──────────┐     ┌───────────┐
 │  WAITING │ ──→ │ PREFILL  │ ──→ │ DECODE    │ ──→ DONE
 └──────────┘     └──────────┘     └───────────┘
                                         │
                                   ┌─────┴──────┐
                                   │  PREEMPTED  │ (if OOM for new requests)
                                   └─────────────┘
```

#### 2.2 Scheduling Loop

```
loop {
    // 1. Admit waiting requests into the running set (if budget allows)
    admit_requests();

    // 2. Run one forward step for ALL running requests
    //    - Requests still in prefill: process prompt tokens (may take multiple steps)
    //    - Requests in decode: generate one token
    run_step();

    // 3. Collect completed requests, free their KV blocks
    collect_completed();

    // 4. If no running requests, block on queue
}
```

#### 2.3 Running Request State

```
struct RunningRequest {
    id: u64,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    eos_token_id: u32,
    state: RequestState,
    position: usize,           // next token position to process
    generated: Vec<u32>,
    block_table: Vec<u32>,     // physical block indices
    tx: Sender<InferenceResponse>,
}

enum RequestState {
    Prefill { prompt_pos: usize },  // still consuming prompt
    Decode,
}
```

#### 2.4 Prefill Chunking

Large prompts are split into chunks of `max_prefill_tokens` to avoid blocking decode for too long:

```
fn prefill_step(req: &mut RunningRequest) {
    let chunk_start = req.prefill_pos;
    let chunk_end = min(chunk_start + MAX_PREFILL_TOKENS, req.prompt_tokens.len());
    // run forward pass for tokens[chunk_start..chunk_end]
    req.prefill_pos = chunk_end;
    if chunk_end == req.prompt_tokens.len() {
        req.state = RequestState::Decode;
    }
}
```

### 3. GPU Kernel Changes

#### 3.1 Block-Table Attention

The attention kernel reads the block table to translate logical positions to physical offsets:

```
// For token at logical position `pos`:
logical_block = pos / BLOCK_SIZE
physical_block = block_table[logical_block]
offset_in_block = pos % BLOCK_SIZE
physical_offset = physical_block * BLOCK_SIZE + offset_in_block
```

Since we don't write custom CUDA kernels (we use cuBLAS for GEMM), the block-table translation happens on the CPU side when constructing the attention inputs. The KV cache read/write in `append_step` uses the block table to compute physical addresses.

#### 3.2 Modified `append_step`

```
fn append_step(&self, layer: usize, request_ids: &[usize], positions: &[usize], hidden: &CudaSlice<f16>) {
    for each request:
        logical_block = positions[i] / BLOCK_SIZE
        physical_block = block_tables[request_ids[i]][logical_block]
        offset = physical_block * BLOCK_SIZE + (positions[i] % BLOCK_SIZE)
        // D→D copy to physical_offset in the KV VA region
}
```

### 4. System Integration

#### 4.1 Main Changes

| File | Change |
|------|--------|
| `src/cache/paged_kv.rs` | New: PagedKvCache, PhysicalBlockAllocator, BlockTable |
| `src/cache/cuda_vmm.rs` | New: FFI bindings to CUDA VMM API |
| `src/batch/continuous_scheduler.rs` | New: ContinuousScheduler, RunningRequest |
| `src/main.rs` | Switch from `StaticScheduler` to `ContinuousScheduler` |
| `src/model/transformer.rs` | Update `append_step` to use block-table addressing |
| `Cargo.toml` | No new dependencies needed (cudarc already provides CUDA driver FFI) |

#### 4.2 Configuration

New CLI/config options:
- `--block-size` (default: 16) — tokens per KV block
- `--gpu-memory-utilization` (default: 0.90) — fraction of GPU memory to use for KV cache
- `--max-num-seqs` (default: from `--max-batch`) — max concurrent sequences

### 5. Metrics and Comparison

To compare against vLLM (as required by the task):

| Metric | Measurement |
|--------|-------------|
| Memory fragmentation rate | `1 - (allocated_blocks / total_blocks)` over time |
| Max concurrent requests | Track peak `running_requests.len()` |
| Throughput (tokens/s) | `total_generated_tokens / wall_time` |
| `cuMemMap` latency | Micro-benchmark in `bench_loaders` style |
| Block utilization | `avg_used_blocks_per_seq / max_blocks_per_seq` |

### 6. Implementation Order

1. **CUDA VMM FFI** — `cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`
2. **PhysicalBlockAllocator** — free-list, allocate/release, sub-allocation from 2 MB handles
3. **PagedKvCache** — per-layer VA regions, block tables, modified `append_step`
4. **ContinuousScheduler** — request lifecycle, prefill chunking, dynamic admit/remove
5. **Integration** — wire into `main.rs`, CLI changes
6. **Benchmarking** — fragmentation, throughput, cuMemMap overhead vs contiguous baseline
