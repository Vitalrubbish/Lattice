// GPU physical memory sub-allocator — extracted from paged_kv.rs.
//
// Manages 2 MiB superblocks carved into fixed-size blocks.
// Pure CPU-side bookkeeping; all GPU operations happen in pool.rs.

use half::f16;
use parking_lot::Mutex;

/// Granularity of CUDA VMM physical allocations (2 MiB).
pub const SUPERBLOCK_SIZE: usize = 2 * 1024 * 1024; // 2 MiB

// --- Physical block sub-allocator ---

/// Tracks one 2 MiB physical allocation and its VA placement
/// within a specific layer's K or V region.
#[derive(Debug)]
pub struct SuperblockInfo {
    /// CUDA VMM physical memory handle (from cuMemCreate).
    pub phys_handle: u64,
    /// Byte offset within the owning VA region where this superblock starts.
    pub va_base: usize,
}

/// Handle to a logical block carved from a superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHandle {
    /// Index of the superblock in the pool's superblock list.
    pub superblock_idx: u32,
    /// Index of the block within the superblock (0..blocks_per_superblock).
    pub block_index: u32,
}

/// Fixed-size block allocator backed by 2 MiB superblocks.
///
/// Manages a free list of blocks. When the free list is exhausted,
/// the caller must add a new superblock (which populates the free list
/// with `blocks_per_superblock` new blocks).
pub struct PhysicalBlockAllocator {
    /// Size of each block in bytes.
    pub block_bytes: usize,
    /// Number of blocks that fit in one superblock.
    pub blocks_per_superblock: usize,
    /// Free block handles available for allocation.
    free_blocks: Mutex<Vec<BlockHandle>>,
    /// Number of superblocks added so far (used to assign superblock_idx).
    superblock_count: Mutex<usize>,
}

impl PhysicalBlockAllocator {
    /// Create a new allocator given the number of F16 elements per block.
    ///
    /// Computes `block_bytes` and `blocks_per_superblock` from `elem_count`.
    pub fn new(elem_count: usize) -> Self {
        let block_bytes = elem_count * std::mem::size_of::<f16>();
        let blocks_per_superblock = SUPERBLOCK_SIZE / block_bytes;
        assert!(
            blocks_per_superblock > 0,
            "block_bytes ({}) too large; reduce BLOCK_SIZE or model dims",
            block_bytes
        );
        assert_eq!(
            SUPERBLOCK_SIZE % block_bytes,
            0,
            "block_bytes ({}) must divide superblock evenly",
            block_bytes
        );

        Self {
            block_bytes,
            blocks_per_superblock,
            free_blocks: Mutex::new(Vec::new()),
            superblock_count: Mutex::new(0),
        }
    }

    /// Create a new allocator with a pre-computed block size in bytes.
    ///
    /// This is useful for KCMM pools where the block size is configured
    /// directly rather than derived from model dimensions.
    pub fn new_with_block_bytes(block_bytes: usize) -> Self {
        let blocks_per_superblock = SUPERBLOCK_SIZE / block_bytes;
        assert!(
            blocks_per_superblock > 0,
            "block_bytes ({}) too large for superblock", block_bytes
        );
        assert_eq!(
            SUPERBLOCK_SIZE % block_bytes,
            0,
            "block_bytes ({}) must divide superblock evenly", block_bytes
        );

        Self {
            block_bytes,
            blocks_per_superblock,
            free_blocks: Mutex::new(Vec::new()),
            superblock_count: Mutex::new(0),
        }
    }

    /// Try to allocate one block from the free list.
    /// Returns `None` if no free blocks are available (caller must add a superblock).
    pub fn try_allocate(&self) -> Option<BlockHandle> {
        self.free_blocks.lock().pop()
    }

    /// Add a new superblock's blocks to the free list.
    /// Increments the superblock count.
    /// All blocks (including index 0) are added to the free list so that
    /// fragmentation tracking sees the correct free-block count.
    pub fn add_superblock(&self) {
        let mut sb_count = self.superblock_count.lock();
        let sb_idx = *sb_count;
        *sb_count += 1;
        drop(sb_count);

        let mut free = self.free_blocks.lock();
        for i in 0..self.blocks_per_superblock {
            free.push(BlockHandle {
                superblock_idx: sb_idx as u32,
                block_index: i as u32,
            });
        }
    }

    /// Return a block to the free pool.
    pub fn free(&self, handle: BlockHandle) {
        self.free_blocks.lock().push(handle);
    }

    /// Number of blocks currently in the free list.
    pub fn free_count(&self) -> usize {
        self.free_blocks.lock().len()
    }

    /// Total number of blocks across all superblocks (allocated + free).
    pub fn total_blocks_allocated(&self) -> usize {
        *self.superblock_count.lock() * self.blocks_per_superblock
    }

    /// Number of superblocks added to this allocator.
    pub fn superblock_count(&self) -> usize {
        *self.superblock_count.lock()
    }
}

// --- Per-layer KV pool ---

/// Physical memory pool for one layer's K or V cache.
///
/// Each layer has its own allocator and superblock list.
/// In lockstep allocation (PagedKvCache), all K and V pools
/// across all layers have identical free lists.
pub struct LayerKvPool {
    pub allocator: PhysicalBlockAllocator,
    pub superblocks: Mutex<Vec<SuperblockInfo>>,
}

impl LayerKvPool {
    pub fn new(elem_count: usize) -> Self {
        Self {
            allocator: PhysicalBlockAllocator::new(elem_count),
            superblocks: Mutex::new(Vec::new()),
        }
    }

    /// Create a LayerKvPool with a pre-computed block size in bytes.
    pub fn new_with_block_bytes(block_bytes: usize) -> Self {
        Self {
            allocator: PhysicalBlockAllocator::new_with_block_bytes(block_bytes),
            superblocks: Mutex::new(Vec::new()),
        }
    }
}

// --- Utility ---

/// Align `x` up to the next multiple of `align`.
#[inline]
pub fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // --- PhysicalBlockAllocator unit tests (no GPU needed) ---

    #[test]
    fn test_allocator_sizing() {
        let elem_count = 4 * 16 * 128; // 8192
        let alloc = PhysicalBlockAllocator::new(elem_count);
        assert_eq!(alloc.block_bytes, 8192 * 2); // 16384
        assert_eq!(alloc.blocks_per_superblock, 128);
        assert_eq!(alloc.free_count(), 0);
        assert_eq!(alloc.total_blocks_allocated(), 0);
    }

    #[test]
    fn test_allocator_sizing_tinyllama() {
        let elem_count = 4 * 16 * 64;
        let alloc = PhysicalBlockAllocator::new(elem_count);
        assert_eq!(alloc.block_bytes, 8192);
        assert_eq!(alloc.blocks_per_superblock, 256);
    }

    #[test]
    fn test_allocator_free_reuse() {
        let alloc = PhysicalBlockAllocator::new(4 * 16 * 128);
        alloc.free(BlockHandle {
            superblock_idx: 0,
            block_index: 0,
        });
        alloc.free(BlockHandle {
            superblock_idx: 0,
            block_index: 1,
        });
        alloc.free(BlockHandle {
            superblock_idx: 1,
            block_index: 5,
        });
        assert_eq!(alloc.free_count(), 3);
    }

    #[test]
    fn test_allocator_add_superblock() {
        let alloc = PhysicalBlockAllocator::new(4 * 16 * 128);
        assert_eq!(alloc.free_count(), 0);

        alloc.add_superblock();
        assert_eq!(alloc.superblock_count(), 1);
        // All blocks (including block 0) are now in the free list.
        assert_eq!(alloc.free_count(), alloc.blocks_per_superblock);
    }

    #[test]
    fn test_allocator_try_allocate() {
        let alloc = PhysicalBlockAllocator::new(4 * 16 * 128);
        assert!(alloc.try_allocate().is_none());

        alloc.add_superblock();
        let h = alloc.try_allocate().unwrap();
        assert_eq!(h.superblock_idx, 0);
    }

    #[test]
    fn test_new_with_block_bytes() {
        let alloc = PhysicalBlockAllocator::new_with_block_bytes(65536);
        assert_eq!(alloc.block_bytes, 65536);
        assert_eq!(alloc.blocks_per_superblock, 32); // 2MiB / 64KiB
        assert_eq!(alloc.free_count(), 0);
    }

    #[test]
    fn test_new_with_block_bytes_rejects_oversized() {
        // 4 MiB block doesn't fit in a 2 MiB superblock
        let result = std::panic::catch_unwind(|| {
            PhysicalBlockAllocator::new_with_block_bytes(4 * 1024 * 1024);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, SUPERBLOCK_SIZE), 0);
        assert_eq!(align_up(1, SUPERBLOCK_SIZE), SUPERBLOCK_SIZE);
        assert_eq!(align_up(SUPERBLOCK_SIZE, SUPERBLOCK_SIZE), SUPERBLOCK_SIZE);
        assert_eq!(
            align_up(SUPERBLOCK_SIZE + 1, SUPERBLOCK_SIZE),
            4 * 1024 * 1024
        );
    }

    #[test]
    fn test_superblock_block_carving() {
        let elem_count = 4 * 16 * 128;
        let block_bytes = elem_count * std::mem::size_of::<f16>();
        assert_eq!(
            SUPERBLOCK_SIZE % block_bytes,
            0,
            "block_bytes must divide superblock evenly"
        );
    }

    #[test]
    fn test_layer_pool_new_with_block_bytes() {
        let pool = LayerKvPool::new_with_block_bytes(65536);
        assert_eq!(pool.allocator.block_bytes, 65536);
        assert_eq!(pool.allocator.blocks_per_superblock, 32);
    }

    // --- Concurrent alloc/free tests ---

    #[test]
    fn test_allocator_concurrent_alloc_free_no_deadlock() {
        use std::sync::Arc;
        use std::thread;

        let alloc = Arc::new(PhysicalBlockAllocator::new_with_block_bytes(65536));
        // Pre-populate with blocks.
        alloc.add_superblock();
        alloc.add_superblock();
        let total = alloc.blocks_per_superblock * 2;

        let a1 = Arc::clone(&alloc);
        let t1 = thread::spawn(move || {
            for _ in 0..500 {
                if let Some(h) = a1.try_allocate() {
                    a1.free(h);
                }
            }
        });

        let a2 = Arc::clone(&alloc);
        let t2 = thread::spawn(move || {
            for _ in 0..500 {
                if let Some(h) = a2.try_allocate() {
                    a2.free(h);
                }
            }
        });

        t1.join().expect("thread 1 panicked");
        t2.join().expect("thread 2 panicked");

        // After all allocs+frees, the free count should be back to total.
        assert_eq!(alloc.free_count(), total,
            "free count should return to total after all alloc/free cycles");
    }

    #[test]
    fn test_allocator_concurrent_multi_thread_stress() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let alloc = Arc::new(PhysicalBlockAllocator::new_with_block_bytes(65536));
        alloc.add_superblock(); // 32 blocks for 64 KiB blocks
        let blocks_per_sb = alloc.blocks_per_superblock;

        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let a = Arc::clone(&alloc);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait(); // synchronize start
                for _ in 0..200 {
                    if let Some(h) = a.try_allocate() {
                        // Hold briefly to increase contention.
                        thread::sleep(std::time::Duration::from_micros(10));
                        a.free(h);
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(alloc.free_count(), blocks_per_sb,
            "all blocks should be returned to free list");
        assert_eq!(alloc.total_blocks_allocated(), blocks_per_sb);
    }

    // --- Multiple superblock index tests ---

    #[test]
    fn test_multiple_superblocks_correct_indices() {
        let alloc = PhysicalBlockAllocator::new_with_block_bytes(65536);
        // Add 3 superblocks
        alloc.add_superblock();
        alloc.add_superblock();
        alloc.add_superblock();
        assert_eq!(alloc.superblock_count(), 3);

        // Allocate all blocks and track what superblock_idx values we see
        let total_blocks = alloc.blocks_per_superblock * 3;
        let mut seen_indices: Vec<u32> = Vec::new();
        for _ in 0..total_blocks {
            let h = alloc.try_allocate().expect("should have blocks");
            seen_indices.push(h.superblock_idx);
        }

        // We should see all three superblock indices (0, 1, 2)
        seen_indices.sort();
        seen_indices.dedup();
        assert_eq!(seen_indices, vec![0, 1, 2],
            "should have blocks from superblocks 0, 1, and 2");

        // Verify total_blocks_allocated is correct
        assert_eq!(alloc.total_blocks_allocated(), total_blocks);
    }

    #[test]
    fn test_multiple_superblocks_sequential_allocation() {
        let alloc = PhysicalBlockAllocator::new_with_block_bytes(65536);
        let bps = alloc.blocks_per_superblock;

        // Add first superblock and exhaust it
        alloc.add_superblock();
        let first_batch: Vec<_> = (0..bps).map(|_| alloc.try_allocate().unwrap()).collect();
        assert!(alloc.try_allocate().is_none(), "should be exhausted");
        assert_eq!(alloc.superblock_count(), 1);
        assert!(first_batch.iter().all(|h| h.superblock_idx == 0));

        // Add second superblock
        alloc.add_superblock();
        let second_batch: Vec<_> = (0..bps).map(|_| alloc.try_allocate().unwrap()).collect();
        assert!(second_batch.iter().all(|h| h.superblock_idx == 1));
        assert_eq!(alloc.superblock_count(), 2);

        // Free some from first batch and re-allocate — should get them back
        for h in &first_batch[..10] {
            alloc.free(*h);
        }
        let reclaimed: Vec<_> = (0..10).map(|_| alloc.try_allocate().unwrap()).collect();
        assert!(reclaimed.iter().all(|h| h.superblock_idx == 0),
            "reclaimed blocks should be from superblock 0");
    }

    // --- Handle reuse LIFO ordering test ---

    #[test]
    fn test_handle_reuse_lifo_order() {
        let alloc = PhysicalBlockAllocator::new_with_block_bytes(65536);
        alloc.add_superblock();

        let h1 = alloc.try_allocate().unwrap();
        let h2 = alloc.try_allocate().unwrap();

        // Free h1 then h2
        alloc.free(h1);
        alloc.free(h2);

        // Re-allocate: should get h2 back first (LIFO — last pushed = first popped)
        let r1 = alloc.try_allocate().unwrap();
        let r2 = alloc.try_allocate().unwrap();
        assert_eq!(r1, h2, "LIFO: last freed (h2) should be returned first");
        assert_eq!(r2, h1, "LIFO: first freed (h1) should be returned second");
    }

    // --- Misaligned block_bytes test for new() ---

    #[test]
    fn test_new_rejects_misaligned_block_bytes() {
        // 2 MiB / odd_block_bytes → not evenly divisible
        // SUPERBLOCK_SIZE = 2 * 1024 * 1024 = 2,097,152
        // Choose 3 * 2 * 1000 = 6000 bytes → 2,097,152 % 6000 = 1,152 ≠ 0
        let odd_bytes = 6000;
        let result = std::panic::catch_unwind(|| {
            PhysicalBlockAllocator::new_with_block_bytes(odd_bytes);
        });
        assert!(result.is_err(),
            "block_bytes not dividing superblock evenly should panic");
    }
}
