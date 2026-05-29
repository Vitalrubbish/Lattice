# Step 3: KV Cache Swapping & Preemption Implementation

**Date:** 2026-05-29

## Summary

Implemented a sequence-level GPU↔host KV cache swapping mechanism to handle VRAM exhaustion in the continuous batching scheduler. When GPU memory is full and new requests arrive, the LRU (Least Recently Used) sequence is evicted to host RAM, freeing GPU blocks for new admission. Evicted sequences are automatically restored when GPU blocks become available.

## Architecture

### Components

1. **`src/cache/swap.rs`** — Low-level swap manager
   - `EvictedSeqData`: Holds per-layer K and V data buffers on host for one sequence
   - `SwapManager`: Manages eviction (GPU→host) and restoration (host→GPU) via synchronous `cuMemcpyDtoH_v2` / `cuMemcpyHtoD_v2`
   - Global epoch counter for LRU tracking (`advance_epoch()`, `current_epoch()`)

2. **`src/cache/paged_kv.rs`** — Modified PagedKvCache
   - Added public accessors: `get_block_table()`, `get_block_va_offset()`, `va_k()`, `va_v()`, `get_seq_len()`, `get_seq_epoch()`
   - Added `seq_last_epoch: Mutex<Vec<u64>>` for per-sequence LRU tracking
   - Added `record_epoch()` to mark sequences as recently accessed
   - Added `alloc_block()` and `append_block_to_sequence()` for on-demand block growth during decode

3. **`src/batch/continuous_scheduler.rs`** — Modified scheduler
   - `RunningRequest` now tracks `num_blocks` for accurate eviction sizing
   - `SwappedRequest` struct holds evicted sequence state (request info, generated tokens, KV data)
   - `admit_waiting()`: When `alloc_sequence()` fails due to VRAM exhaustion, selects an LRU victim, evicts it, retries allocation
   - `try_restore_swapped()`: After GPU blocks are freed (completions), restores evicted sequences
   - `select_victim()`: LRU + block-count-based victim selection (prefers older, larger sequences)
   - `drain_completed_swapped()`: Cleans up swapped sequences that reached completion (e.g., max_new_tokens while evicted)
   - Global epoch advanced each scheduler loop iteration

### Data Flow

```
EVICT:
  GPU VA (K/V blocks) ──cuMemcpyDtoH_v2──→ host Vec<u8> buffers
  → free GPU physical blocks

RESTORE:
  host Vec<u8> buffers ──cuMemcpyHtoD_v2──→ new GPU VA blocks
  → register new block table
```

### Lifecycle

```
WAITING → [admit] → RUNNING → [complete] → DONE
                        │
                        │ (VRAM exhausted, LRU victim)
                        ▼
                    SWAPPED → [blocks freed] → RUNNING (resume)
                        │
                        │ (max_tokens/eos while swapped)
                        ▼
                       DONE
```

## Key Design Decisions

- **Sequence-level eviction**: Entire sequence's KV cache is swapped atomically (simpler than per-block, avoids thrashing)
- **LRU policy**: Tracked via global atomic epoch counter; each sequence records its last-accessed epoch
- **Synchronous copies**: Uses blocking `cuMemcpyDtoH_v2`/`cuMemcpyHtoD_v2` (simpler, preemption is infrequent; future optimization: async + pinned memory)
- **Decode-only victims**: Prefill sequences are not preempted (cheap to let them finish; evicting wastes prompt processing work)
- **Max swapped queue**: Bounded at 256 sequences to prevent host RAM exhaustion

## Files Modified

| File | Changes |
|------|---------|
| `src/cache/swap.rs` | New: 221 lines |
| `src/cache/mod.rs` | Added `swap` module and re-exports |
| `src/cache/paged_kv.rs` | Added accessor methods, epoch tracking, `alloc_block()`, `append_block_to_sequence()` |
| `src/batch/continuous_scheduler.rs` | Added `SwappedRequest`, `select_victim()`, `admit_waiting()` with eviction, `try_restore_swapped()`, `drain_completed_swapped()` |
| `docs/plan/step3-kv-cache-swapping-preemption.md` | New: design document |

## Tests

- `swap_manager_evict_restore_cycle`: GPU allocation → evict → free → restore → verify
- `swap_manager_empty_sequence`: Eviction/restoration of empty block table
- `swap_manager_total_swapped_bytes`: Verify byte tracking is correct
- `swap_manager_epoch_advances`: Verify epoch monotonic advancement
- All existing tests (25 total) continue to pass

## Future Work

- Async copies with pinned host memory + CUDA streams
- Block-level granularity for eviction/restoration (instead of whole-sequence)
- Integration with Step 4 prefix sharing (reference-counted blocks)
- Host RAM eviction policy (swap-to-disk)
