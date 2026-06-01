# LLM OS — Efficient OS Support Layer for LLM Inference

A Rust-based LLM inference system that explores OS-level optimizations for KV cache memory management,
model weight loading, and continuous batching, benchmarked against vLLM.

## Language

### Memory Management

**Paged KV Cache**:
GPU KV cache managed in fixed-size blocks with virtual-to-physical address translation,
enabling non-contiguous physical backing for logically contiguous sequences.
_Avoid_: Slot cache, segmented cache

**Block**:
The smallest unit of KV cache allocation. Fixed token count (default 16); byte size varies with model config
(`block_size × kv_heads × head_dim × sizeof(f16)`).
_Avoid_: Page, chunk, slot

**BlockHandle**:
A `(superblock_idx, block_index)` pair identifying a physical block's origin within a superblock.
Returned to the free list on deallocation; reused by subsequent allocations without `cuMemUnmap`.
_Avoid_: Block pointer, block address

**Block Index**:
A numeric identifier (0, 1, 2, …) assigned when a BlockHandle is first installed into `block_info`.
Recycled via `free_block_indices` when a sequence frees blocks — `block_info.len()` is the
peak number of distinct indices ever active, not the current physical block count.
_Avoid_: Block ID, block number

**Superblock**:
A 2 MiB physical GPU memory allocation (via `cuMemCreate`) that is carved into fixed-size blocks.
The allocation granularity of CUDA VMM. One superblock contains `2 MiB / block_bytes` blocks.
All per-layer K and V pools allocate superblocks in lockstep.
_Avoid_: Pool, arena, slab

**Block Table**:
The per-sequence `Vec<u32>` mapping from logical block index (position / block_size) to physical
block index (into `block_info`), enabling PagedAttention address translation.
_Avoid_: Page table (reserved for CPU MMU), block map

**Lockstep Allocation**:
Every per-layer K and V pool allocates and frees blocks with the same `BlockHandle` simultaneously,
so block index N has the same physical offset across all layers. This is why `total_blocks_allocated`
can be queried from any one pool.
_Avoid_: Synchronized allocation, mirrored pools

**Free List**:
The pool of `BlockHandle`s available for allocation without creating new superblocks.
When the free list is empty, `ensure_capacity()` triggers `cuMemCreate` for a new superblock
across all per-layer pools.
_Avoid_: Free pool, available blocks

**Allocator Granularity**:
The minimum physical memory an allocator can request from the hardware.
Baseline (CUDA VMM): 2 MiB superblocks — physical memory grows in 2 MiB increments.
vLLM (PyTorch): the entire pre-allocated pool — physical memory is claimed once at startup.
Determines PME when blocks are partially utilized.
_Avoid_: Allocation unit, chunk size

**Total Blocks Allocated**:
The total count of blocks for which physical GPU memory has been provisioned.
Baseline: `superblock_count × blocks_per_superblock` (grows with load; queried via `total_physical_blocks()`).
vLLM: the entire pre-allocated pool (fixed at startup; queried via `/metrics` or startup log).
⚠️ Distinct from `block_info.len()` (`total_blocks()`), which counts peak distinct block indices ever active.
_Avoid_: Block pool size, max blocks, allocated blocks

### Fragmentation Metrics (UFS — Unified Fragmentation Standard)

**Internal Fragmentation Rate (IFR)**:
`(total_slots - total_tokens) / total_slots`. Waste from the last block of each sequence being partially filled.
Identical across systems given the same block_size and workload. Range [0, 1).
_Avoid_: Internal waste, block padding, slot waste

**Block Utilization (BU)**:
`blocks_in_use / total_blocks_allocated`. Proportion of the physically-backed block pool
actually assigned to sequences. Compare the trend (rising under load = grow-on-demand working),
not the absolute value (vLLM's fixed large pool makes BU perpetually low).
_Avoid_: Pool usage, occupancy

**Physical Memory Efficiency (PME)**:
`ideal_physical_bytes / actual_physical_bytes`. How much of the physically-allocated GPU memory
holds useful KV data. Captures allocator-granularity waste. Range (0, 1].
PME equals BU when there is no granularity waste beyond the block pool itself.
_Avoid_: Memory utilization, VRAM efficiency

**Runtime Fragmentation Index (RFI)**:
`1 − (total_tokens × BPT / actual_active_bytes)`. Combined internal+granularity waste,
sampled over time during active processing. Range [0, 1). 0 = no waste.
_Avoid_: Composite fragmentation, overall waste

### Allocation Strategy

**Pre-allocation**:
Reserving the full GPU memory pool at startup before any requests arrive.
Used by vLLM (`gpu_memory_utilization`). Produces constant `total_blocks_allocated`.
Trades memory efficiency for guaranteed capacity.
_Avoid_: Static allocation, one-shot allocation, upfront allocation

**Grow-on-Demand**:
Creating physical GPU memory incrementally as requests consume blocks.
Used by baseline (CUDA VMM `cuMemCreate`). `total_blocks_allocated` grows with load.
Trades occasional `cuMemCreate` latency (~11 ms per superblock) for memory efficiency.
_Avoid_: Dynamic allocation, lazy allocation, just-in-time allocation

### Benchmarking

**Capacity-at-Workload**:
The maximum number of concurrent sequences a system can admit without OOM,
measured under a specified token-length distribution and max_new_tokens (not worst-case max_seq_len).
Replaces the misleading "max concurrent requests" that used hardcoded blocks-per-seq.
_Avoid_: Max concurrent requests, peak concurrency, max batch size

**EOS-Controlled Benchmark**:
A capacity or fragmentation benchmark where EOS generation is suppressed (via an unreachable
`eos_token_id`) so every sequence generates exactly `max_new_tokens` tokens. Without this,
natural EOS termination frees KV blocks early, giving an unfairly high capacity count.
Baseline tests always use EOS-controlled semantics; vLLM benchmarks must explicitly configure it.
_Avoid_: Unfair capacity comparison, EOS-biased benchmark

**Stress Mode / Concurrency Ramp**:
Running the same benchmark workload at increasing concurrency levels (e.g., 1 → 2 → 4 → 8 → … → 64)
to observe how UFS metrics change with load. Baseline's BU should rise sharply (grow-on-demand),
while vLLM's BU stays flat (pre-allocation). Each level produces one row of the comparison table.
_Avoid_: Multi-level benchmark, scalability test

**Continuous Batching**:
A scheduler that admits new requests and evicts completed ones on every step,
rather than waiting for the whole batch to finish. Enables dynamic request join/leave
and on-the-fly KV cache block allocation during decode.
_Avoid_: Dynamic batching, in-flight batching, iteration-level scheduling

### Measurement Pitfalls

**nvidia-smi Diff Trap**:
Subtracting a post-startup GPU memory baseline from `nvidia-smi` readings to estimate
KV cache memory usage. This hides the pre-allocated block pool because it was already
accounted for in the baseline. Always query the allocator directly for `total_blocks_allocated`
(vLLM: `num_gpu_blocks` from server log or `/metrics`; baseline: `total_physical_blocks()`).
_Avoid_: GPU memory delta, VRAM difference

### Data Loading

**Loader**:
The I/O strategy for moving model weights from storage to GPU VRAM.
Four variants: `read(2)` (buffered, via page cache), `mmap` (demand-paged, zero-copy),
`O_DIRECT` (bypasses page cache), `GDS` (NVMe-to-GPU DMA via cuFileRead).
_Avoid_: Reader, weight loader, deserializer
