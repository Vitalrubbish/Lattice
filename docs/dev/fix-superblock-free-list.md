# Fix: Superblock Free List Not Including Block 0

**Date**: 2026/05/30

## Problem

When `PhysicalBlockAllocator::add_superblock()` was called (triggered by `PagedKvCache::ensure_capacity()` → CUDA VMM superblock allocation), the function added blocks `1..blocks_per_superblock` to the `free_list` but returned block 0 as a separate handle. In `ensure_capacity()`, this return value was **discarded**, meaning block 0 of every allocated superblock was never tracked in the `free_list`.

This caused the fragmentation rate calculation to be inaccurate because:

1. `free_physical_blocks()` returned `blocks_per_superblock - 1` instead of `blocks_per_superblock` after a superblock was added
2. The orphaned block 0s inflated `blocks_not_free = total_blocks - free_blocks`
3. `RuntimeFragmentationTracker::record()` computed incorrect `active_superblocks` and `memory_allocated_not_free` values

## Fix

Modified `PhysicalBlockAllocator::add_superblock()` in `src/cache/paged_kv.rs`:

- Changed the loop from `1..self.blocks_per_superblock` to `0..self.blocks_per_superblock` so block 0 is also added to the free list
- Removed the return value (`BlockHandle`) since it's no longer needed — all blocks go into the free list directly
- Updated the function signature from `-> BlockHandle` to no return value

## Files Changed

- `src/cache/paged_kv.rs`: Modified `add_superblock()` and updated related tests

## Verification

- All 23 unit tests pass
- Full project compiles successfully
