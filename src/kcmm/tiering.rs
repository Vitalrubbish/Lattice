// Tiering engine — GPU↔CPU↔NVMe data migration.
//
// Implements block-granularity eviction and restoration across the
// three-tier storage hierarchy: GPU HBM → CPU DRAM → NVMe SSD.
//
// The EvictionPolicy trait decouples policy (which blocks to evict)
// from mechanism (how to move data between tiers).

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

use anyhow::Result;
use parking_lot::Mutex;
use crate::config::KcmmConfig;
use crate::kcmm::superblock::BlockHandle;

// --- Eviction policy trait ---

/// Pluggable eviction policy — selects victim blocks for eviction.
///
/// Implementations:
///   - LruPolicy: evicts blocks with the oldest `last_access` timestamp.
///   - LfuPolicy: evicts blocks with the lowest access frequency.
///   - FifoPolicy: evicts blocks with the earliest allocation time.
pub trait EvictionPolicy: Send + Sync {
    /// Called when a block is first allocated into the pool.
    ///
    /// For LRU this records the initial access time; for LFU it initialises
    /// the access count to 1; for FIFO it records the allocation timestamp
    /// (which is the sole eviction-ordering key — `on_access` is a no-op).
    fn on_allocate(&self, block: BlockHandle);

    /// Select victim blocks from the candidates, returning up to `count`
    /// blocks ordered by eviction priority (highest priority first).
    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle>;

    /// Called when a block is accessed (for LRU/LFU bookkeeping).
    fn on_access(&self, block: BlockHandle);

    /// Called when a block is evicted (for policy bookkeeping).
    fn on_evict(&self, block: BlockHandle);
}

// --- Default LRU policy ---

/// Least-Recently-Used eviction policy.
///
/// Tracks per-block access timestamps and selects victims with the
/// oldest `last_access` time.
pub struct LruPolicy {
    /// BlockHandle → last access timestamp.
    access_times: Mutex<HashMap<BlockHandle, Instant>>,
}

impl LruPolicy {
    /// Create a new LRU policy with no tracked accesses.
    pub fn new() -> Self {
        Self {
            access_times: Mutex::new(HashMap::new()),
        }
    }
}

impl EvictionPolicy for LruPolicy {
    fn on_allocate(&self, block: BlockHandle) {
        self.access_times.lock().insert(block, Instant::now());
    }

    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle> {
        if candidates.is_empty() || count == 0 {
            return Vec::new();
        }
        let times = self.access_times.lock();
        let mut sorted: Vec<(BlockHandle, Instant)> = candidates
            .iter()
            .filter_map(|h| times.get(h).map(|t| (*h, *t)))
            .collect();
        // Sort by timestamp ascending — oldest (smallest Instant) first.
        sorted.sort_by_key(|(_, t)| *t);
        sorted.truncate(count);
        sorted.into_iter().map(|(h, _)| h).collect()
    }

    fn on_access(&self, block: BlockHandle) {
        self.access_times.lock().insert(block, Instant::now());
    }

    fn on_evict(&self, block: BlockHandle) {
        self.access_times.lock().remove(&block);
    }
}

// --- LFU policy ---

/// Least-Frequently-Used eviction policy.
///
/// Tracks per-block access counts and selects victims with the
/// lowest access frequency.
pub struct LfuPolicy {
    /// BlockHandle → cumulative access count.
    access_counts: Mutex<HashMap<BlockHandle, u64>>,
}

impl LfuPolicy {
    /// Create a new LFU policy with no tracked accesses.
    pub fn new() -> Self {
        Self {
            access_counts: Mutex::new(HashMap::new()),
        }
    }
}

impl EvictionPolicy for LfuPolicy {
    fn on_allocate(&self, block: BlockHandle) {
        self.access_counts.lock().insert(block, 1);
    }

    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle> {
        if candidates.is_empty() || count == 0 {
            return Vec::new();
        }
        let counts = self.access_counts.lock();
        let mut sorted: Vec<(BlockHandle, u64)> = candidates
            .iter()
            .filter_map(|h| counts.get(h).map(|c| (*h, *c)))
            .collect();
        // Sort by access count ascending — least frequently used first.
        sorted.sort_by_key(|(_, c)| *c);
        sorted.truncate(count);
        sorted.into_iter().map(|(h, _)| h).collect()
    }

    fn on_access(&self, block: BlockHandle) {
        let mut counts = self.access_counts.lock();
        *counts.entry(block).or_insert(0) += 1;
    }

    fn on_evict(&self, block: BlockHandle) {
        self.access_counts.lock().remove(&block);
    }
}

// --- FIFO policy ---

/// First-In-First-Out eviction policy.
///
/// Tracks per-block allocation timestamps (set by `on_allocate`) and selects
/// victims with the earliest allocation time.  `on_access` is a no-op for FIFO:
/// once a block enters the pool its eviction priority is fixed by its allocation
/// timestamp and is never refreshed.
pub struct FifoPolicy {
    /// BlockHandle → allocation timestamp (set by `on_allocate`).
    alloc_times: Mutex<HashMap<BlockHandle, Instant>>,
}

impl FifoPolicy {
    /// Create a new FIFO policy with no tracked blocks.
    pub fn new() -> Self {
        Self {
            alloc_times: Mutex::new(HashMap::new()),
        }
    }
}

impl EvictionPolicy for FifoPolicy {
    fn on_allocate(&self, block: BlockHandle) {
        self.alloc_times.lock().insert(block, Instant::now());
    }

    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle> {
        if candidates.is_empty() || count == 0 {
            return Vec::new();
        }
        let times = self.alloc_times.lock();
        let mut sorted: Vec<(BlockHandle, Instant)> = candidates
            .iter()
            .filter_map(|h| times.get(h).map(|t| (*h, *t)))
            .collect();
        // Sort by allocation time ascending — earliest allocated first.
        sorted.sort_by_key(|(_, t)| *t);
        sorted.truncate(count);
        sorted.into_iter().map(|(h, _)| h).collect()
    }

    fn on_access(&self, _block: BlockHandle) {
        // FIFO: access does not affect eviction order.
        // Allocation time (set by `on_allocate`) is the sole ordering key.
    }

    fn on_evict(&self, block: BlockHandle) {
        self.alloc_times.lock().remove(&block);
    }
}

// --- CPU slot allocator ---

/// Manages allocation of byte ranges within the CPU swap buffer.
///
/// Ensures no two concurrent operations (eviction write + restore read)
/// use the same buffer region, preventing data races when the tiering
/// engine performs GPU↔CPU data movement.
///
/// Uses a best-fit free-list approach: free ranges are kept sorted by
/// offset and merged on free to minimise fragmentation.
#[derive(Debug)]
#[allow(dead_code)]
struct CpuSlotAllocator {
    total_size: usize,
    /// Free byte ranges, kept sorted by offset and merged on free.
    free_ranges: Vec<std::ops::Range<usize>>,
}

impl CpuSlotAllocator {
    /// Create a new allocator managing a buffer of `total_size` bytes.
    /// The entire buffer starts as one contiguous free range.
    fn new(total_size: usize) -> Self {
        let free_ranges = if total_size > 0 {
            vec![0..total_size]
        } else {
            Vec::new()
        };
        Self {
            total_size,
            free_ranges,
        }
    }

    /// Allocate a contiguous byte range of at least `size` bytes.
    ///
    /// Uses best-fit (smallest range that satisfies the request) to
    /// minimise fragmentation.  Returns the byte offset into the CPU
    /// buffer, or `None` if no sufficiently large free range exists.
    fn allocate(&mut self, size: usize) -> Option<usize> {
        if size == 0 || self.free_ranges.is_empty() {
            return None;
        }
        // Best-fit: find the smallest range that fits.
        let mut best_idx = None;
        let mut best_len = usize::MAX;
        for (i, range) in self.free_ranges.iter().enumerate() {
            let len = range.end - range.start;
            if len >= size && len < best_len {
                best_len = len;
                best_idx = Some(i);
            }
        }
        let idx = best_idx?;
        let offset = self.free_ranges[idx].start;
        if self.free_ranges[idx].end - self.free_ranges[idx].start == size {
            // Exact fit: remove the range entirely.
            self.free_ranges.remove(idx);
        } else {
            // Split: shrink the range from the left.
            self.free_ranges[idx].start += size;
        }
        Some(offset)
    }

    /// Return a previously allocated byte range to the free pool.
    ///
    /// Merges with adjacent free ranges to minimise fragmentation.
    fn free(&mut self, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        let end = offset + size;

        // Find insertion point to maintain sorted-by-offset order.
        let insert_idx = self
            .free_ranges
            .iter()
            .position(|r| r.start > offset)
            .unwrap_or(self.free_ranges.len());

        // Check if we can merge with the preceding range.
        let merge_prev =
            insert_idx > 0 && self.free_ranges[insert_idx - 1].end == offset;
        // Check if we can merge with the following range.
        let merge_next =
            insert_idx < self.free_ranges.len() && self.free_ranges[insert_idx].start == end;

        match (merge_prev, merge_next) {
            (true, true) => {
                // Merge with both: extend the previous range to cover the next.
                let next_end = self.free_ranges[insert_idx].end;
                self.free_ranges[insert_idx - 1].end = next_end;
                self.free_ranges.remove(insert_idx);
            }
            (true, false) => {
                self.free_ranges[insert_idx - 1].end = end;
            }
            (false, true) => {
                self.free_ranges[insert_idx].start = offset;
            }
            (false, false) => {
                self.free_ranges.insert(insert_idx, offset..end);
            }
        }
    }
}

// --- Tiering engine ---

/// Manages block-granularity data migration across the three-tier hierarchy.
///
/// In step 3 week 13, this is a skeleton. Full implementation
/// (GPU↔CPU evict/restore, NVMe layer, batch optimization) comes in weeks 14-15.
#[allow(dead_code)]
pub struct TieringEngine {
    /// Base address of the CPU swap buffer (file-backed mmap via cpu_cache_path).
    cpu_buffer: *mut u8,
    /// Total size of the CPU swap buffer in bytes.
    cpu_buffer_size: usize,
    /// Path to the mmap'd file (kept for logging / debugging).
    cpu_buffer_path: String,
    /// Whether NVMe tier is enabled.
    nvme_enabled: bool,
    /// Pluggable eviction policy (LRU, LFU, or FIFO).
    pub(crate) eviction_policy: Box<dyn EvictionPolicy>,
    /// Serialises allocation/deallocation of byte ranges within the CPU buffer,
    /// preventing concurrent evict+restore operations from using overlapping regions.
    slot_allocator: Mutex<CpuSlotAllocator>,
}

impl TieringEngine {
    /// Create a new tiering engine.
    ///
    /// Creates (or opens) a file at `config.cpu_cache_path` and mmaps it
    /// as the CPU swap buffer.  Using a file-backed mapping (instead of
    /// `MAP_ANONYMOUS`) enables cross-process sharing of the swap region
    /// and persistence of swapped data across engine restarts.
    pub fn new(config: &KcmmConfig) -> Result<Self> {
        // TODO: add KcmmConfig::cpu_buffer_size; for now estimate.
        let cpu_buffer_size = config.max_blocks * config.block_size * 2;

        let cpu_buffer = if cpu_buffer_size > 0 {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&config.cpu_cache_path)?;
            file.set_len(cpu_buffer_size as u64)?;

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    cpu_buffer_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(anyhow::anyhow!(
                    "mmap CPU swap buffer '{}' ({} bytes) failed: {}",
                    config.cpu_cache_path,
                    cpu_buffer_size,
                    std::io::Error::last_os_error()
                ));
            }
            ptr as *mut u8
        } else {
            std::ptr::null_mut()
        };

        let eviction_policy: Box<dyn EvictionPolicy> = match config.eviction_policy.as_str() {
            "lfu" => Box::new(LfuPolicy::new()),
            "fifo" => Box::new(FifoPolicy::new()),
            _ => Box::new(LruPolicy::new()), // default: LRU
        };

        Ok(Self {
            cpu_buffer,
            cpu_buffer_size,
            cpu_buffer_path: config.cpu_cache_path.clone(),
            nvme_enabled: false,
            eviction_policy,
            slot_allocator: Mutex::new(CpuSlotAllocator::new(cpu_buffer_size)),
        })
    }

    /// Get the CPU buffer base pointer.
    pub fn cpu_buffer_ptr(&self) -> *mut u8 {
        self.cpu_buffer
    }

    /// Get the CPU buffer size.
    pub fn cpu_buffer_size(&self) -> usize {
        self.cpu_buffer_size
    }
}

impl Drop for TieringEngine {
    fn drop(&mut self) {
        if !self.cpu_buffer.is_null() && self.cpu_buffer_size > 0 {
            unsafe {
                libc::munmap(self.cpu_buffer as *mut libc::c_void, self.cpu_buffer_size);
            }
        }
    }
}

// Safety: TieringEngine manages raw mmap'd memory and CUDA resources.
// The Mutex<CpuSlotAllocator> serialises access to the CPU buffer,
// preventing data races from concurrent eviction/restore operations.
unsafe impl Send for TieringEngine {}
unsafe impl Sync for TieringEngine {}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::fs;

    fn test_config() -> KcmmConfig {
        KcmmConfig {
            block_size: 16,
            max_blocks: 1024,
            cpu_cache_path: String::new(), // will be set per test
            tiering: true,
            eviction_policy: "lru".to_string(),
            prefetch_window: 4,
        }
    }

    #[test]
    fn test_tiering_engine_new_with_temp_file() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("kcmm_swap_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 16; // cpu_buffer_size = 16 * 16 * 2 = 512

        let engine = TieringEngine::new(&config).expect("create engine");
        let expected_size = config.max_blocks * config.block_size * 2;

        // Buffer should be non-null and the expected size.
        assert!(!engine.cpu_buffer.is_null());
        assert_eq!(engine.cpu_buffer_size, expected_size);

        // Write a pattern to the mmap'd region via raw pointer.
        unsafe {
            let magic: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
            std::ptr::copy_nonoverlapping(magic.as_ptr(), engine.cpu_buffer, 4);
        }

        // Read back from the file to verify MAP_SHARED persistence.
        let mut file = fs::File::open(&path).expect("reopen file");
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).expect("read back");
        assert_eq!(buf, [0xDE, 0xAD, 0xBE, 0xEF]);

        // engine is dropped here — munmap should succeed.
        drop(engine);

        // Verify the file still exists (munmap doesn't unlink).
        assert!(path.exists());
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_zero_buffer() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("zero_swap");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 0;

        let engine = TieringEngine::new(&config).expect("create engine with zero blocks");
        assert!(engine.cpu_buffer.is_null());
        assert_eq!(engine.cpu_buffer_size, 0);

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_buffer_ptr_accessor() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("accessor_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 4;

        let engine = TieringEngine::new(&config).expect("create engine");
        let expected = config.max_blocks * config.block_size * 2;
        let ptr = engine.cpu_buffer_ptr();
        assert!(!ptr.is_null());
        assert_eq!(engine.cpu_buffer_size(), expected);

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_drop_does_not_crash() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("drop_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 8;

        let engine = TieringEngine::new(&config).expect("create engine");
        // Just dropping should not panic or segfault.
        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_send_sync() {
        // Compile-time check: TieringEngine must implement Send + Sync
        // (already asserted by unsafe impl blocks — this test just
        // exercises the types at runtime).
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("send_sync_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 4;

        let engine = TieringEngine::new(&config).expect("create engine");
        assert_send_sync(&engine);

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_mmap_file_exists_and_has_size() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("size_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 32;

        let engine = TieringEngine::new(&config).expect("create engine");
        let expected_size = config.max_blocks * config.block_size * 2;
        assert_eq!(engine.cpu_buffer_size, expected_size);

        let metadata = fs::metadata(&path).expect("stat file");
        assert_eq!(metadata.len() as usize, expected_size);

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_nonexistent_directory() {
        let mut config = test_config();
        config.cpu_cache_path = "/nonexistent/path/should/fail/kcmm_swap".to_string();

        let result = TieringEngine::new(&config);
        assert!(result.is_err(), "should fail on nonexistent directory");
    }

    // --- Helper to create test BlockHandles ---

    fn bh(sb: u32, blk: u32) -> BlockHandle {
        BlockHandle { superblock_idx: sb, block_index: blk }
    }

    // --- LruPolicy tests ---

    mod lru {
        use super::*;

        #[test]
        fn test_new_is_empty() {
            let policy = LruPolicy::new();
            assert!(policy.access_times.lock().is_empty());
        }

        #[test]
        fn test_on_allocate_inserts_timestamp() {
            let policy = LruPolicy::new();
            let h = bh(0, 0);
            policy.on_allocate(h);
            assert!(policy.access_times.lock().contains_key(&h));
        }

        #[test]
        fn test_on_access_inserts_timestamp() {
            let policy = LruPolicy::new();
            let h = bh(0, 0);
            policy.on_access(h);
            assert!(policy.access_times.lock().contains_key(&h));
        }

        #[test]
        fn test_on_access_updates_timestamp() {
            let policy = LruPolicy::new();
            let h = bh(0, 0);
            policy.on_access(h);
            let t1 = *policy.access_times.lock().get(&h).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
            policy.on_access(h);
            let t2 = *policy.access_times.lock().get(&h).unwrap();
            assert!(t2 > t1, "LRU on_access should update timestamp");
        }

        #[test]
        fn test_on_evict_removes_entry() {
            let policy = LruPolicy::new();
            let h = bh(0, 0);
            policy.on_allocate(h);
            assert!(policy.access_times.lock().contains_key(&h));
            policy.on_evict(h);
            assert!(!policy.access_times.lock().contains_key(&h));
        }

        #[test]
        fn test_select_victims_empty_candidates() {
            let policy = LruPolicy::new();
            let result = policy.select_victims(&[], 5);
            assert!(result.is_empty());
        }

        #[test]
        fn test_select_victims_zero_count() {
            let policy = LruPolicy::new();
            policy.on_allocate(bh(0, 0));
            let result = policy.select_victims(&[bh(0, 0)], 0);
            assert!(result.is_empty());
        }

        #[test]
        fn test_select_victims_oldest_first() {
            let policy = LruPolicy::new();
            let h0 = bh(0, 0);
            let h1 = bh(0, 1);
            let h2 = bh(0, 2);

            policy.on_allocate(h0); // oldest
            std::thread::sleep(std::time::Duration::from_millis(2));
            policy.on_allocate(h1);
            std::thread::sleep(std::time::Duration::from_millis(2));
            policy.on_allocate(h2); // newest

            let victims = policy.select_victims(&[h0, h1, h2], 2);
            assert_eq!(victims.len(), 2);
            assert_eq!(victims[0], h0, "oldest (h0) should be first victim");
            assert_eq!(victims[1], h1, "second oldest (h1) should be second victim");
        }

        #[test]
        fn test_select_victims_respects_count() {
            let policy = LruPolicy::new();
            for i in 0..10 {
                policy.on_allocate(bh(0, i));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let candidates: Vec<BlockHandle> = (0..10).map(|i| bh(0, i)).collect();
            let victims = policy.select_victims(&candidates, 3);
            assert_eq!(victims.len(), 3);
            // bh(0,0) should be oldest
            assert_eq!(victims[0], bh(0, 0));
            assert_eq!(victims[1], bh(0, 1));
            assert_eq!(victims[2], bh(0, 2));
        }

        #[test]
        fn test_select_victims_count_exceeds_candidates() {
            let policy = LruPolicy::new();
            policy.on_allocate(bh(0, 0));
            policy.on_allocate(bh(0, 1));

            let result = policy.select_victims(&[bh(0, 0), bh(0, 1)], 10);
            assert_eq!(result.len(), 2);
        }

        #[test]
        fn test_select_victims_untracked_blocks_skipped() {
            let policy = LruPolicy::new();
            policy.on_allocate(bh(0, 0));
            // bh(0, 1) is NOT tracked
            let result = policy.select_victims(&[bh(0, 0), bh(0, 1)], 2);
            // Only bh(0, 0) has a timestamp; bh(0, 1) is skipped.
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], bh(0, 0));
        }
    }

    // --- LfuPolicy tests ---

    mod lfu {
        use super::*;

        #[test]
        fn test_new_is_empty() {
            let policy = LfuPolicy::new();
            assert!(policy.access_counts.lock().is_empty());
        }

        #[test]
        fn test_on_allocate_sets_count_to_one() {
            let policy = LfuPolicy::new();
            let h = bh(0, 0);
            policy.on_allocate(h);
            assert_eq!(*policy.access_counts.lock().get(&h).unwrap(), 1);
        }

        #[test]
        fn test_on_access_increments_count() {
            let policy = LfuPolicy::new();
            let h = bh(0, 0);
            policy.on_access(h);
            assert_eq!(*policy.access_counts.lock().get(&h).unwrap(), 1);
            policy.on_access(h);
            assert_eq!(*policy.access_counts.lock().get(&h).unwrap(), 2);
            policy.on_access(h);
            assert_eq!(*policy.access_counts.lock().get(&h).unwrap(), 3);
        }

        #[test]
        fn test_on_evict_removes_entry() {
            let policy = LfuPolicy::new();
            let h = bh(0, 0);
            policy.on_allocate(h);
            policy.on_access(h);
            policy.on_access(h);
            assert_eq!(*policy.access_counts.lock().get(&h).unwrap(), 3);
            policy.on_evict(h);
            assert!(!policy.access_counts.lock().contains_key(&h));
        }

        #[test]
        fn test_select_victims_empty_candidates() {
            let policy = LfuPolicy::new();
            assert!(policy.select_victims(&[], 5).is_empty());
        }

        #[test]
        fn test_select_victims_zero_count() {
            let policy = LfuPolicy::new();
            policy.on_allocate(bh(0, 0));
            assert!(policy.select_victims(&[bh(0, 0)], 0).is_empty());
        }

        #[test]
        fn test_select_victims_least_frequent_first() {
            let policy = LfuPolicy::new();
            let h0 = bh(0, 0); // accessed 1 time
            let h1 = bh(0, 1); // accessed 3 times
            let h2 = bh(0, 2); // accessed 2 times

            policy.on_access(h0);
            for _ in 0..3 { policy.on_access(h1); }
            for _ in 0..2 { policy.on_access(h2); }

            let victims = policy.select_victims(&[h0, h1, h2], 3);
            assert_eq!(victims.len(), 3);
            assert_eq!(victims[0], h0, "least frequent (count=1) should be first");
            assert_eq!(victims[1], h2, "second least frequent (count=2) should be second");
            assert_eq!(victims[2], h1, "most frequent (count=3) should be last");
        }

        #[test]
        fn test_select_victims_respects_count() {
            let policy = LfuPolicy::new();
            for i in 0..5 {
                let h = bh(0, i);
                for _ in 0..(i + 1) { policy.on_access(h); }
            }
            let candidates: Vec<BlockHandle> = (0..5).map(|i| bh(0, i)).collect();
            let victims = policy.select_victims(&candidates, 2);
            assert_eq!(victims.len(), 2);
            assert_eq!(victims[0], bh(0, 0)); // count=1
            assert_eq!(victims[1], bh(0, 1)); // count=2
        }

        #[test]
        fn test_select_victims_count_exceeds_candidates() {
            let policy = LfuPolicy::new();
            policy.on_allocate(bh(0, 0));
            policy.on_allocate(bh(0, 1));

            let result = policy.select_victims(&[bh(0, 0), bh(0, 1)], 10);
            assert_eq!(result.len(), 2);
        }

        #[test]
        fn test_select_victims_untracked_blocks_skipped() {
            let policy = LfuPolicy::new();
            policy.on_allocate(bh(0, 0));
            // bh(0, 1) is NOT tracked — should be filtered out.
            let result = policy.select_victims(&[bh(0, 0), bh(0, 1)], 2);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], bh(0, 0));
        }
    }

    // --- FifoPolicy tests ---

    mod fifo {
        use super::*;

        #[test]
        fn test_new_is_empty() {
            let policy = FifoPolicy::new();
            assert!(policy.alloc_times.lock().is_empty());
        }

        #[test]
        fn test_on_allocate_inserts_timestamp() {
            let policy = FifoPolicy::new();
            let h = bh(0, 0);
            policy.on_allocate(h);
            assert!(policy.alloc_times.lock().contains_key(&h));
        }

        #[test]
        fn test_on_access_is_noop() {
            let policy = FifoPolicy::new();
            let h = bh(0, 0);
            // on_access on an untracked block should NOT insert it
            policy.on_access(h);
            assert!(!policy.alloc_times.lock().contains_key(&h),
                "FIFO on_access must not insert untracked blocks");
            // on_allocate inserts; on_access should not update
            policy.on_allocate(h);
            let t1 = *policy.alloc_times.lock().get(&h).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
            policy.on_access(h);
            let t2 = *policy.alloc_times.lock().get(&h).unwrap();
            assert_eq!(t1, t2, "FIFO on_access must NOT refresh timestamp");
        }

        #[test]
        fn test_on_evict_removes_entry() {
            let policy = FifoPolicy::new();
            let h = bh(0, 0);
            policy.on_allocate(h);
            assert!(policy.alloc_times.lock().contains_key(&h));
            policy.on_evict(h);
            assert!(!policy.alloc_times.lock().contains_key(&h));
        }

        #[test]
        fn test_select_victims_empty_candidates() {
            let policy = FifoPolicy::new();
            assert!(policy.select_victims(&[], 5).is_empty());
        }

        #[test]
        fn test_select_victims_zero_count() {
            let policy = FifoPolicy::new();
            policy.on_allocate(bh(0, 0));
            assert!(policy.select_victims(&[bh(0, 0)], 0).is_empty());
        }

        #[test]
        fn test_select_victims_earliest_allocated_first() {
            let policy = FifoPolicy::new();
            let h0 = bh(0, 0);
            let h1 = bh(0, 1);
            let h2 = bh(0, 2);

            policy.on_allocate(h0); // first allocated
            std::thread::sleep(std::time::Duration::from_millis(2));
            policy.on_allocate(h1); // second allocated
            std::thread::sleep(std::time::Duration::from_millis(2));
            policy.on_allocate(h2); // third allocated

            let victims = policy.select_victims(&[h0, h1, h2], 2);
            assert_eq!(victims.len(), 2);
            assert_eq!(victims[0], h0, "earliest allocated should be first");
            assert_eq!(victims[1], h1, "second earliest should be second");
        }

        #[test]
        fn test_select_victims_respects_count() {
            let policy = FifoPolicy::new();
            for i in 0..10 {
                policy.on_allocate(bh(0, i));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let candidates: Vec<BlockHandle> = (0..10).map(|i| bh(0, i)).collect();
            let victims = policy.select_victims(&candidates, 4);
            assert_eq!(victims.len(), 4);
            assert_eq!(victims[0], bh(0, 0));
            assert_eq!(victims[1], bh(0, 1));
            assert_eq!(victims[2], bh(0, 2));
            assert_eq!(victims[3], bh(0, 3));
        }

        #[test]
        fn test_select_victims_count_exceeds_candidates() {
            let policy = FifoPolicy::new();
            policy.on_allocate(bh(0, 0));
            policy.on_allocate(bh(0, 1));

            let result = policy.select_victims(&[bh(0, 0), bh(0, 1)], 10);
            assert_eq!(result.len(), 2);
        }

        #[test]
        fn test_select_victims_untracked_blocks_skipped() {
            let policy = FifoPolicy::new();
            policy.on_allocate(bh(0, 0));
            // bh(0, 1) is NOT tracked — should be filtered out.
            let result = policy.select_victims(&[bh(0, 0), bh(0, 1)], 2);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], bh(0, 0));
        }
    }

    // --- Policy selection in TieringEngine ---

    #[test]
    fn test_tiering_engine_default_policy_is_lru() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("policy_lru_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 4;
        // eviction_policy defaults to "lru"
        let engine = TieringEngine::new(&config).expect("create engine");

        // Verify the policy works like LRU by exercising its behavior.
        let h0 = bh(0, 0);
        let h1 = bh(0, 1);
        engine.eviction_policy.on_access(h0);
        std::thread::sleep(std::time::Duration::from_millis(2));
        engine.eviction_policy.on_access(h1);
        let victims = engine.eviction_policy.select_victims(&[h0, h1], 1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], h0, "LRU should evict oldest access (h0)");

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_selects_lfu_policy() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("policy_lfu_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 4;
        config.eviction_policy = "lfu".to_string();

        let engine = TieringEngine::new(&config).expect("create engine");

        // Verify LFU behavior: least-frequently accessed block evicted first.
        let h0 = bh(0, 0);
        let h1 = bh(0, 1);
        engine.eviction_policy.on_access(h0); // count=1
        engine.eviction_policy.on_access(h1);
        engine.eviction_policy.on_access(h1); // count=2

        let victims = engine.eviction_policy.select_victims(&[h0, h1], 1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], h0, "LFU should evict least frequent (h0, count=1)");

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_selects_fifo_policy() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("policy_fifo_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 4;
        config.eviction_policy = "fifo".to_string();

        let engine = TieringEngine::new(&config).expect("create engine");

        // Verify FIFO behavior: earliest-allocated evicted first.
        let h0 = bh(0, 0);
        let h1 = bh(0, 1);
        engine.eviction_policy.on_allocate(h0); // first allocated
        std::thread::sleep(std::time::Duration::from_millis(2));
        engine.eviction_policy.on_allocate(h1); // second allocated

        let victims = engine.eviction_policy.select_victims(&[h0, h1], 1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], h0, "FIFO should evict earliest allocated (h0)");

        drop(engine);
        drop(dir);
    }

    #[test]
    fn test_tiering_engine_unknown_policy_falls_back_to_lru() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("policy_unknown_test");
        let path_str = path.to_str().expect("valid UTF-8 path");

        let mut config = test_config();
        config.cpu_cache_path = path_str.to_string();
        config.max_blocks = 4;
        config.eviction_policy = "nonexistent".to_string();

        let engine = TieringEngine::new(&config).expect("create engine");

        // Should behave like LRU (fallback), not crash.
        let h = bh(0, 0);
        engine.eviction_policy.on_access(h);
        let victims = engine.eviction_policy.select_victims(&[h], 1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], h);

        drop(engine);
        drop(dir);
    }

    // --- CpuSlotAllocator tests ---

    mod cpu_slot_allocator {
        use super::*;

        #[test]
        fn test_new_empty_buffer() {
            let alloc = CpuSlotAllocator::new(0);
            assert!(alloc.free_ranges.is_empty());
            assert_eq!(alloc.total_size, 0);
        }

        #[test]
        fn test_new_creates_one_free_range() {
            let alloc = CpuSlotAllocator::new(1024);
            assert_eq!(alloc.free_ranges.len(), 1);
            assert_eq!(alloc.free_ranges[0], 0..1024);
        }

        #[test]
        fn test_allocate_exact_fit_removes_range() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let offset = alloc.allocate(1024);
            assert_eq!(offset, Some(0));
            assert!(alloc.free_ranges.is_empty());
        }

        #[test]
        fn test_allocate_split_shrinks_range() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let offset = alloc.allocate(256);
            assert_eq!(offset, Some(0));
            assert_eq!(alloc.free_ranges.len(), 1);
            assert_eq!(alloc.free_ranges[0], 256..1024);
        }

        #[test]
        fn test_allocate_zero_size_returns_none() {
            let mut alloc = CpuSlotAllocator::new(1024);
            assert_eq!(alloc.allocate(0), None);
            // Buffer should be unchanged
            assert_eq!(alloc.free_ranges.len(), 1);
            assert_eq!(alloc.free_ranges[0], 0..1024);
        }

        #[test]
        fn test_allocate_no_space_returns_none() {
            let mut alloc = CpuSlotAllocator::new(64);
            alloc.allocate(64); // consume all
            assert_eq!(alloc.allocate(1), None);
        }

        #[test]
        fn test_allocate_best_fit() {
            // Layout: free ranges [0..64, 128..256, 320..640]
            // Request 48 bytes: best-fit is 0..64 (len=64), not 128..256 (len=128)
            let mut alloc = CpuSlotAllocator {
                total_size: 640,
                free_ranges: vec![0..64, 128..256, 320..640],
            };
            let offset = alloc.allocate(48);
            assert_eq!(offset, Some(0));
            // 0..64 → 48..64 (len 16)
            assert_eq!(alloc.free_ranges[0], 48..64);
        }

        #[test]
        fn test_free_no_merge() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let offset = alloc.allocate(256).unwrap();
            assert_eq!(offset, 0);
            // Free it back — should restore the full range
            alloc.free(offset, 256);
            assert_eq!(alloc.free_ranges.len(), 1);
            assert_eq!(alloc.free_ranges[0], 0..1024);
        }

        #[test]
        fn test_free_merge_with_prev() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let off1 = alloc.allocate(256).unwrap(); // 0..256
            let off2 = alloc.allocate(256).unwrap(); // 256..512
            assert_eq!(off1, 0);
            assert_eq!(off2, 256);
            assert_eq!(alloc.free_ranges, vec![512..1024]);

            // Free off1 first, then off2 — should merge into one range
            alloc.free(off1, 256);
            assert_eq!(alloc.free_ranges, vec![0..256, 512..1024]);
            alloc.free(off2, 256);
            assert_eq!(alloc.free_ranges, vec![0..1024]);
        }

        #[test]
        fn test_free_merge_with_next() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let off2 = alloc.allocate(256).unwrap(); // 0..256
            alloc.allocate(256); // 256..512 — discard
            let off1 = alloc.allocate(256).unwrap(); // 512..768 — this is what we free first
            assert_eq!(off2, 0);
            assert_eq!(off1, 512);

            // Free off1 (512..768) — should merge with next free range 768..1024
            alloc.free(off1, 256);
            assert_eq!(alloc.free_ranges, vec![512..1024]);
        }

        #[test]
        fn test_free_merge_both_sides() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let off1 = alloc.allocate(256).unwrap(); // 0..256
            let off2 = alloc.allocate(256).unwrap(); // 256..512
            let off3 = alloc.allocate(256).unwrap(); // 512..768
            assert_eq!(off1, 0);
            assert_eq!(off2, 256);
            assert_eq!(off3, 512);
            assert_eq!(alloc.free_ranges, vec![768..1024]);

            // Free off1 and off3 first
            alloc.free(off1, 256);
            alloc.free(off3, 256);
            assert_eq!(alloc.free_ranges, vec![0..256, 512..1024]);

            // Now free off2 — should merge both sides: 0..256 + 256..512 + 512..1024
            alloc.free(off2, 256);
            assert_eq!(alloc.free_ranges, vec![0..1024]);
        }

        #[test]
        fn test_free_zero_size_noop() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let before = alloc.free_ranges.clone();
            alloc.free(100, 0);
            assert_eq!(alloc.free_ranges, before);
        }

        #[test]
        fn test_allocate_multiple_sequential() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let off1 = alloc.allocate(128).unwrap();
            let off2 = alloc.allocate(256).unwrap();
            let off3 = alloc.allocate(512).unwrap();
            assert_eq!(off1, 0);
            assert_eq!(off2, 128);
            assert_eq!(off3, 384);
            // 128 used (0..128), 256 used (128..384), 512 used (384..896), 128 free (896..1024)
            assert_eq!(alloc.free_ranges, vec![896..1024]);
        }

        #[test]
        fn test_free_and_reallocate() {
            let mut alloc = CpuSlotAllocator::new(1024);
            let off1 = alloc.allocate(512).unwrap(); // 0..512
            let off2 = alloc.allocate(512).unwrap(); // 512..1024
            assert_eq!(off1, 0);
            assert_eq!(off2, 512);
            assert!(alloc.free_ranges.is_empty());

            // Free off1, then re-allocate a smaller block from that region
            alloc.free(off1, 512);
            let off3 = alloc.allocate(256).unwrap();
            assert_eq!(off3, 0); // best-fit: 0..512 fits 256, returns 0
            assert_eq!(alloc.free_ranges, vec![256..512]);
        }
    }
}
