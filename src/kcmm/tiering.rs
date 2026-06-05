// Tiering engine — GPU↔CPU↔NVMe data migration.
//
// Implements block-granularity eviction and restoration across the
// three-tier storage hierarchy: GPU HBM → CPU DRAM → NVMe SSD.
//
// The EvictionPolicy trait decouples policy (which blocks to evict)
// from mechanism (how to move data between tiers).

use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

use anyhow::Result;
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
    /// Select victim blocks from the candidates, returning up to `count`
    /// blocks ordered by eviction priority (highest priority first).
    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle>;

    /// Called when a block is accessed (for LRU/LFU bookkeeping).
    fn on_access(&mut self, block: BlockHandle);

    /// Called when a block is evicted (for policy bookkeeping).
    fn on_evict(&mut self, block: BlockHandle);
}

// --- Default LRU policy ---

/// Least-Recently-Used eviction policy.
///
/// Selects blocks with the oldest access timestamps.
/// Timestamps are managed externally (via SequenceState::last_access).
pub struct LruPolicy;

impl EvictionPolicy for LruPolicy {
    fn select_victims(&self, _candidates: &[BlockHandle], _count: usize) -> Vec<BlockHandle> {
        // Placeholder — full implementation in Week 14.
        // Will sort candidates by last_access and return the oldest.
        Vec::new()
    }

    fn on_access(&mut self, _block: BlockHandle) {
        // No-op for now.
    }

    fn on_evict(&mut self, _block: BlockHandle) {
        // No-op for now.
    }
}

// --- LFU policy ---

/// Least-Frequently-Used eviction policy.
pub struct LfuPolicy;

impl EvictionPolicy for LfuPolicy {
    fn select_victims(&self, _candidates: &[BlockHandle], _count: usize) -> Vec<BlockHandle> {
        Vec::new()
    }

    fn on_access(&mut self, _block: BlockHandle) {}

    fn on_evict(&mut self, _block: BlockHandle) {}
}

// --- FIFO policy ---

/// First-In-First-Out eviction policy.
pub struct FifoPolicy;

impl EvictionPolicy for FifoPolicy {
    fn select_victims(&self, _candidates: &[BlockHandle], _count: usize) -> Vec<BlockHandle> {
        Vec::new()
    }

    fn on_access(&mut self, _block: BlockHandle) {}

    fn on_evict(&mut self, _block: BlockHandle) {}
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

        Ok(Self {
            cpu_buffer,
            cpu_buffer_size,
            cpu_buffer_path: config.cpu_cache_path.clone(),
            nvme_enabled: false,
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
}
