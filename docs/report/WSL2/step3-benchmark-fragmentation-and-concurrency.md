# Step 3 Benchmark Report: Fragmentation Rate, Maximum Concurrent Requests, and cuMemMap Overhead

2025-05-28

## Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5070 Laptop (Blackwell) |
| VRAM | 8 GiB total, ~7.9 GiB free |
| CUDA Driver | WSL2, CUDA 12.x |
| Model | TinyLlama (kv_heads=4, head_dim=64, 22 layers, hidden=2048) |
| Block size | 16 tokens |
| Superblock size | 2 MiB |
| VMM map granularity | **2 MiB** (queried via `cuMemGetAllocationGranularity`) |

## Test Code Location

All four tests are in `src/cache/paged_kv.rs` within the `#[cfg(test)] mod tests` block:

| Test Function | Line | What It Measures |
|---|---|---|
| `step3_max_concurrent_requests` | 651 | How many sequences can be allocated before OOM; cuMemMap call count |
| `step3_fragmentation_rate` | 714 | Internal fragmentation under alloc/free/re-alloc cycles |
| `step3_cumemmap_overhead` | 798 | cuMemMap/cuMemUnmap latency vs. mapping size |
| `step3_internal_fragmentation_analysis` | 911 | Per-sequence internal fragmentation with varied lengths |

Run with:
```
cargo test --lib cache::paged_kv::tests::step3_ -- --nocapture
```

## 1. Maximum Concurrent Requests

**Result: 256** concurrent requests (bounded by the test's `max_batch=256`, not by GPU memory).

| Metric | Value |
|--------|-------|
| Model | TinyLlama (kv_heads=4, head_dim=64, 22 layers) |
| Max concurrent requests allocated | 256 |
| Blocks per request (256-token max_seq_len) | 16 |
| Total blocks allocated | 4,096 |
| Superblocks allocated | 16 |
| Physical memory consumed | 32 MiB |
| Blocks per superblock (8192-byte blocks) | 256 |
| cuMemMap calls total | 704 (44 per superblock: 22 layers × K+V) |

**Analysis**: At 32 MiB for 256 concurrent requests at 256 tokens each, the GPU's 8 GiB VRAM could theoretically support **tens of thousands** of concurrent short-context requests. The practical limit is determined by `max_batch` and `max_seq_len` configuration, not by physical memory. Extrapolating: with 8 GiB available and 4096-token max_seq_len (256 blocks per request), theoretical max ~3,500 concurrent requests before exhausting VRAM.

The superblock-level mapping approach means each 2 MiB superblock requires 44 cuMemMap calls (one per layer per K/V region), but each superblock holds 256 blocks. This amortizes mapping overhead to ~35.5 µs per block.

## 2. Fragmentation Rate

### 2.1 Alloc/Free/Re-alloc Cycle Test

| Phase | Blocks In Use | Free Blocks | Internal Frag | Notes |
|---|---|---|---|---|
| 64 sequences, 256 tokens each | 1,024 | 0 | 0.00% | All last blocks perfectly full |
| 50% freed | 512 | 512 | n/a | Free ratio = 0.50 |
| 32 shorter sequences re-added | 768 | 256 | 10.00% | Partial last-block fill |

**Analysis**: External fragmentation is zero by design — the free list reuses block indices regardless of their physical position. Internal fragmentation comes solely from partially-filled last blocks. When block_size=16 and sequence lengths vary, the expected average internal fragmentation is `(block_size/2) / (avg_blocks_per_seq * block_size) ≈ 8 / (avg_seq_len)` — negligible for long sequences.

### 2.2 Per-Sequence Internal Fragmentation (16 tokens/block)

| Seq Length | Blocks | Slots | Wasted | Frag % |
|---|---|---|---|---|
| 1 | 1 | 16 | 15 | 93.8% |
| 7 | 1 | 16 | 9 | 56.2% |
| 15 | 1 | 16 | 1 | 6.2% |
| 16 | 1 | 16 | 0 | 0.0% |
| 17 | 2 | 32 | 15 | 46.9% |
| 100 | 7 | 112 | 12 | 10.7% |
| 127 | 8 | 128 | 1 | 0.8% |
| 128 | 8 | 128 | 0 | 0.0% |

**Overall**: 16 varied-length sequences → **9.95% internal fragmentation**, 5.4 tokens wasted per sequence on average.

## 3. cuMemMap / cuMemUnmap Overhead

### Key Finding: Minimum Mapping Granularity is 2 MiB

Sub-2 MiB `cuMemMap` calls fail with `CUDA_ERROR_NOT_SUPPORTED`. All per-block mapping attempts (8 KiB, 16 KiB, 32 KiB, 64 KiB, 128 KiB, 256 KiB, 512 KiB, 1 MiB) were rejected. Only the full 2 MiB mapping succeeded.

### Latency Measurements

| Operation | Latency |
|---|---|
| Single `cuMemMap` (2 MiB) | ~207 µs |
| Single `cuMemUnmap` (2 MiB) | ~207 µs |
| Total superblock mapping (22 layers × K+V = 44 calls) | ~9.1 ms |
| Amortized per-block (256 blocks per superblock) | **~35.5 µs** |

### Comparison: Before vs After Refactoring

| Approach | cuMemMap Calls per Block | Works? |
|---|---|---|
| Per-block mapping (original) | 44 (one per layer per K/V) | No — fails at 8 KiB granularity |
| Superblock mapping (refactored) | 0 per block; 44 per superblock (256 blocks) | Yes |

The refactored design eliminates per-block `cuMemMap` overhead entirely: each 2 MiB superblock is mapped once into all 22 layers' K and V virtual address regions, costing ~9.1 ms. This cost is amortized over 256 block allocations (for TinyLlama; varies by block_bytes). For larger models with higher kv_heads/head_dim, each superblock holds fewer blocks, so the amortized cost per block increases proportionally.

## 4. Design Impact

The 2 MiB granularity constraint forced a refactoring of the physical-to-virtual mapping strategy:

- **Before**: Each block individually mapped via `cuMemMap` → failed at sub-2 MiB sizes.
- **After**: Entire 2 MiB superblocks mapped once via `cuMemMap`, blocks sub-allocated within the mapped VA range via offset arithmetic. No per-block map/unmap calls.

This is consistent with vLLM's approach of allocating large GPU memory segments and managing fine-grained blocks within them using page table indirection.

## Summary

| Metric | Result |
|---|---|
| VMM map granularity | 2 MiB |
| Max concurrent requests (test config) | 256 (config-limited) |
| External fragmentation | 0% (free-list design) |
| Internal fragmentation (varied lengths) | ~10% (block_size=16) |
| cuMemMap latency (2 MiB) | ~207 µs |
| Superblock mapping cost (all layers) | ~9.1 ms |
| Amortized per-block mapping cost | ~35.5 µs |
| Sub-2 MiB individual block mapping | Not supported |
