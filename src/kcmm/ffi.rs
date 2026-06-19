// C FFI API — stable ABI for external inference engine integration.
//
// Exposes KCMM pool operations as C-compatible functions in libkcmm.so.
// Any language with C FFI support (Python, C++, Java, etc.) can call these.
//
// The API is designed to be:
//   - Engine-agnostic: no vLLM or PyTorch dependencies
//   - Thread-safe: each pool_t is an opaque handle to an Arc<KcmmPool>
//   - ABI-stable: uses only C-compatible types (opaque pointers, sized integers)
//
// Error handling:
//   - Functions return 0 on success, -1 on error.
//   - Retrieve the last error message via kcmm_get_last_error().
//   - Functions that return a pointer return NULL on error.
//   - Functions that return a size/count return 0 on error.
//
// Thread safety:
//   - All functions are safe to call from multiple threads concurrently.
//   - Each kcmm_pool_t handle is independent; sharing a handle across threads
//     is safe (backed by Arc<KcmmPool> with internal Mutex protection).

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Arc;

use crate::config::KcmmConfig;
use crate::cuda::CudaContext;
use crate::kcmm::pool::{BlockLocation, KcmmPool, SequencePriority};
use crate::kcmm::metrics::KcmmMetrics;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque handle to a KCMM pool.
/// The C side sees this as an incomplete type; the actual data is
/// `Box<KcmmPoolHandle>` behind the pointer.
#[repr(C)]
pub struct kcmm_pool_t {
    _private: [u8; 0], // opaque
}

/// Internal pool handle stored behind the opaque pointer.
struct KcmmPoolHandle {
    pool: Arc<KcmmPool>,
    last_error: parking_lot::Mutex<String>,
}

impl KcmmPoolHandle {
    fn new(pool: KcmmPool) -> Self {
        Self {
            pool: Arc::new(pool),
            last_error: parking_lot::Mutex::new(String::new()),
        }
    }

    fn set_error(&self, msg: String) {
        *self.last_error.lock() = msg;
    }
}

// ---------------------------------------------------------------------------
// C-compatible configuration struct
// ---------------------------------------------------------------------------

/// Configuration for KCMM pool creation (C-compatible layout).
#[repr(C)]
pub struct kcmm_config_t {
    /// Tokens per block. Default: 16.
    pub block_size: usize,
    /// Maximum number of blocks in the pool. Default: 16384.
    pub max_blocks: usize,
    /// Path to the CPU swap buffer (e.g. "/dev/shm/kcmm_swap").
    /// Use empty string "" for default.
    pub cpu_cache_path: [c_char; 256],
    /// Whether multi-tier storage (GPU→CPU) is enabled. Default: true (1).
    pub tiering: i32,
    /// Eviction policy: 0=LRU, 1=LFU, 2=FIFO. Default: 0.
    pub eviction_policy: i32,
    /// Prefetch look-ahead window in blocks. Default: 4.
    pub prefetch_window: usize,
    /// Maximum batch blocks per eviction/restore operation. Default: 64.
    pub max_batch_blocks: usize,
    /// GPU device ordinal to use. Default: 0.
    pub device_ordinal: usize,
    /// Number of transformer layers (model-specific).
    pub num_layers: usize,
    /// Number of KV attention heads (model-specific).
    pub kv_heads: usize,
    /// Dimension of each attention head (model-specific).
    pub head_dim: usize,
    /// Maximum batch size supported.
    pub max_batch: usize,
    /// Maximum sequence length in tokens.
    pub max_seq_len: usize,
    /// Low watermark threshold for proactive background eviction (0.0–1.0).
    /// Default: 0.2 (20% free blocks triggers background eviction).
    pub low_watermark_threshold: f32,
    /// Background eviction check interval in milliseconds. Default: 100.
    pub background_evict_interval_ms: u64,
    /// Number of attention sink blocks protected by "sink_window" policy.
    /// Default: 1.
    pub attention_sink_blocks: usize,
    /// Number of recent window blocks protected by "sink_window" policy.
    /// Default: 4.
    pub recent_window_blocks: usize,
}

impl Default for kcmm_config_t {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_blocks: 16384,
            cpu_cache_path: [0; 256],
            tiering: 1,
            eviction_policy: 0,
            prefetch_window: 4,
            max_batch_blocks: 64,
            device_ordinal: 0,
            num_layers: 22,
            kv_heads: 4,
            head_dim: 64,
            max_batch: 8,
            max_seq_len: 128,
            low_watermark_threshold: 0.2,
            background_evict_interval_ms: 100,
            attention_sink_blocks: 1,
            recent_window_blocks: 4,
        }
    }
}

/// Runtime statistics for a KCMM pool.
#[repr(C)]
pub struct kcmm_pool_stats_t {
    /// Number of blocks currently in use.
    pub blocks_in_use: u32,
    /// Total number of block indices (including recycled).
    pub total_blocks: u32,
    /// Total physical blocks across all per-layer pools.
    pub total_physical_blocks: u32,
    /// Free physical blocks available.
    pub free_physical_blocks: u32,
    /// Number of active (registered) sequences.
    pub active_sequences: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Blocks per superblock.
    pub blocks_per_superblock: u32,
    /// Superblock count.
    pub superblock_count: u32,
    /// Block size in tokens.
    pub block_size: u32,
    /// Maximum blocks per sequence.
    pub max_blocks_per_seq: u32,
    /// Byte size of each block.
    pub block_bytes: u32,
    /// Whether tiering is enabled.
    pub tiering_enabled: i32,
    /// Whether prefix sharing is enabled.
    pub sharing_enabled: i32,
    /// Physical idle ratio (fraction of allocated superblock capacity idle).
    pub physical_idle_ratio: f32,
}

/// KCMM metrics structure (mirrors `KcmmMetrics` in metrics.rs).
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct kcmm_metrics_t {
    /// Internal fragmentation ratio (0.0 = perfect packing).
    pub ifr: f64,
    /// Physical memory efficiency (1.0 = optimal).
    pub pme: f64,
    /// Block utilization ratio.
    pub bu: f64,
    /// Runtime fragmentation index.
    pub rfi: f64,
    /// Number of blocks currently in GPU.
    pub gpu_blocks: u64,
    /// Number of blocks currently in CPU swap.
    pub cpu_blocks: u64,
    /// Number of blocks currently on NVMe.
    pub nvme_blocks: u64,
    /// Total eviction operations since pool creation.
    pub eviction_count: u64,
    /// Total restoration operations since pool creation.
    pub restoration_count: u64,
}

/// KCMM hint types for the Hint API.
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum kcmm_hint_t {
    KCMM_HINT_MULTI_TURN = 1,
    KCMM_HINT_NEAR_END = 2,
    KCMM_HINT_SYSTEM_PROMPT = 3,
    KCMM_HINT_HIGH_PRIORITY = 4,
    KCMM_HINT_LOW_PRIORITY = 5,
    KCMM_HINT_ATTENTION_SINK = 6,
    KCMM_HINT_HEAVY_HITTER = 7,
    KCMM_HINT_EVICTABLE = 8,
}

/// KCMM protection levels (for block-granularity protection).
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum kcmm_protection_t {
    KCMM_PROTECT_NEVER_EVICT = 0,
    KCMM_PROTECT_PREFERRED = 1,
    KCMM_EVICT_PREFERRED = 2,
}

/// Where a block's data currently resides.
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum kcmm_block_location_t {
    KCMM_LOC_GPU_RESIDENT = 0,
    KCMM_LOC_CPU_RESIDENT = 1,
    KCMM_LOC_NVME_RESIDENT = 2,
    KCMM_LOC_EVICTING = 3,
    KCMM_LOC_RESTORING = 4,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a raw pool pointer to the internal handle reference.
///
/// # Safety
/// The pointer must be a valid, non-null result from `kcmm_pool_create`.
unsafe fn pool_from_ptr<'a>(ptr: *mut kcmm_pool_t) -> &'a KcmmPoolHandle {
    &*(ptr as *const KcmmPoolHandle)
}

/// Parse a C string from a fixed-size buffer, trimming trailing nulls.
unsafe fn c_str_from_fixed(buf: &[c_char]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let slice = std::slice::from_raw_parts(buf.as_ptr() as *const u8, end);
    String::from_utf8_lossy(slice).into_owned()
}

/// Write a Rust string into a fixed-size C buffer. Returns bytes written
/// (excluding null terminator), or 0 if the buffer is too small.
unsafe fn write_c_str_fixed(buf: *mut c_char, max_len: usize, s: &str) -> usize {
    if buf.is_null() || max_len == 0 {
        return 0;
    }
    let bytes = s.as_bytes();
    let copy_len = bytes.len().min(max_len - 1);
    let dst = std::slice::from_raw_parts_mut(buf as *mut u8, max_len);
    dst[..copy_len].copy_from_slice(&bytes[..copy_len]);
    dst[copy_len] = 0; // null terminate
    copy_len
}

/// Parse an eviction policy string from C config enum value.
fn policy_from_i32(val: i32) -> &'static str {
    match val {
        0 => "lru",
        1 => "lfu",
        2 => "fifo",
        _ => "lru",
    }
}

fn default_usize_if_zero(value: usize, default: usize) -> usize {
    if value == 0 { default } else { value }
}

fn default_u64_if_zero(value: u64, default: u64) -> u64 {
    if value == 0 { default } else { value }
}

fn default_f32_if_nonpositive(value: f32, default: f32) -> f32 {
    if value <= 0.0 { default } else { value }
}

/// Convert a BlockLocation to the C enum.
fn block_loc_to_c(loc: &BlockLocation) -> kcmm_block_location_t {
    match loc {
        BlockLocation::GpuResident(..) => kcmm_block_location_t::KCMM_LOC_GPU_RESIDENT,
        BlockLocation::CpuResident(_) => kcmm_block_location_t::KCMM_LOC_CPU_RESIDENT,
        BlockLocation::NvmeResident(_) => kcmm_block_location_t::KCMM_LOC_NVME_RESIDENT,
        BlockLocation::Evicting => kcmm_block_location_t::KCMM_LOC_EVICTING,
        BlockLocation::Restoring => kcmm_block_location_t::KCMM_LOC_RESTORING,
    }
}

// ---------------------------------------------------------------------------
// Pool lifecycle
// ---------------------------------------------------------------------------

/// Create a new KCMM pool.
///
/// Returns an opaque handle on success, or NULL on error.
/// Use `kcmm_get_last_error()` to retrieve the error message.
/// The caller must destroy the pool with `kcmm_pool_destroy()`.
///
/// # Safety
/// `config` must be a valid, non-null pointer to a `kcmm_config_t`.
#[no_mangle]
pub unsafe extern "C" fn kcmm_pool_create(
    config: *const kcmm_config_t,
) -> *mut kcmm_pool_t {
    if config.is_null() {
        return std::ptr::null_mut();
    }

    let cfg = &*config;
    let cpu_path = c_str_from_fixed(&cfg.cpu_cache_path);

    let kcmm_config = KcmmConfig {
        block_size: default_usize_if_zero(cfg.block_size, 16),
        max_blocks: default_usize_if_zero(cfg.max_blocks, 16384),
        cpu_cache_path: if cpu_path.is_empty() {
            "/dev/shm/kcmm_swap".to_string()
        } else {
            cpu_path
        },
        tiering: cfg.tiering != 0,
        eviction_policy: policy_from_i32(cfg.eviction_policy).to_string(),
        prefetch_window: default_usize_if_zero(cfg.prefetch_window, 4),
        max_batch_blocks: default_usize_if_zero(cfg.max_batch_blocks, 64),
        low_watermark_threshold: default_f32_if_nonpositive(cfg.low_watermark_threshold, 0.2),
        background_evict_interval_ms: default_u64_if_zero(cfg.background_evict_interval_ms, 100),
        attention_sink_blocks: default_usize_if_zero(cfg.attention_sink_blocks, 1),
        recent_window_blocks: default_usize_if_zero(cfg.recent_window_blocks, 4),
    };

    // Initialize CUDA context
    let ctx = match CudaContext::new(cfg.device_ordinal) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            // Can't set error on a handle that doesn't exist yet, but the
            // caller will see NULL and can infer an error.
            tracing::error!("kcmm_pool_create: CudaContext::new failed: {:?}", e);
            return std::ptr::null_mut();
        }
    };

    // Create the pool
    let pool = match KcmmPool::new(
        ctx,
        kcmm_config,
        cfg.num_layers,
        cfg.kv_heads,
        cfg.head_dim,
        cfg.max_batch,
        cfg.max_seq_len,
    ) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("kcmm_pool_create: KcmmPool::new failed: {:?}", e);
            return std::ptr::null_mut();
        }
    };

    let handle = KcmmPoolHandle::new(pool);
    let boxed = Box::new(handle);
    Box::into_raw(boxed) as *mut kcmm_pool_t
}

/// Destroy a KCMM pool and free all associated resources.
///
/// Waits for in-flight CUDA operations, unmaps VA regions, and releases
/// physical memory.  The handle is invalid after this call.
///
/// # Safety
/// `pool` must be a valid handle from `kcmm_pool_create`.  Must not be called
/// more than once on the same handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_pool_destroy(pool: *mut kcmm_pool_t) {
    if pool.is_null() {
        return;
    }
    let handle: Box<KcmmPoolHandle> = Box::from_raw(pool as *mut KcmmPoolHandle);
    // Wait for streams before the pool (and its Arc references) drop.
    handle.pool.streams.synchronize_all().ok();
    drop(handle);
}

/// Retrieve the last error message for a pool.
///
/// Writes a null-terminated message into `buf` (up to `max_len` bytes).
/// Returns the number of bytes written (excluding null terminator), or 0
/// if no error has occurred or the buffer is too small.
///
/// # Safety
/// `pool` must be a valid handle. `buf` must be a valid buffer of at least
/// `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_last_error(
    pool: *mut kcmm_pool_t,
    buf: *mut c_char,
    max_len: usize,
) -> usize {
    if pool.is_null() || buf.is_null() || max_len == 0 {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    let err = handle.last_error.lock();
    if err.is_empty() {
        return 0;
    }
    write_c_str_fixed(buf, max_len, &err)
}

/// Clear the last error for a pool.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_clear_error(pool: *mut kcmm_pool_t) {
    if pool.is_null() {
        return;
    }
    let handle = pool_from_ptr(pool);
    handle.last_error.lock().clear();
}

// ---------------------------------------------------------------------------
// Block allocation
// ---------------------------------------------------------------------------

/// Allocate `num_blocks` blocks and write their indices into `out_blocks`.
///
/// `out_blocks` must be pre-allocated with at least `num_blocks` elements.
/// Returns 0 on success, -1 on error (use `kcmm_get_last_error` for details).
///
/// # Safety
/// `pool` must be a valid handle. `out_blocks` must point to a buffer of
/// at least `num_blocks * sizeof(u32)` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_alloc_blocks(
    pool: *mut kcmm_pool_t,
    num_blocks: u32,
    out_blocks: *mut u32,
) -> i32 {
    if pool.is_null() || out_blocks.is_null() || num_blocks == 0 {
        if !pool.is_null() {
            pool_from_ptr(pool).set_error("kcmm_alloc_blocks: null output buffer or zero count"
                .to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let table = match handle.pool.alloc_sequence(num_blocks as usize) {
        Ok(t) => t,
        Err(e) => {
            handle.set_error(format!("kcmm_alloc_blocks: {:#}", e));
            return -1;
        }
    };

    let dst = std::slice::from_raw_parts_mut(out_blocks, num_blocks as usize);
    for (i, &idx) in table.iter().enumerate() {
        dst[i] = idx;
    }
    0
}

/// Free all blocks specified in `blocks`.
///
/// Each block is returned to the per-layer physical allocators.
/// Returns 0 on success, -1 if the pool pointer is null.
///
/// # Safety
/// `pool` must be a valid handle. `blocks` must point to a buffer of
/// at least `num * sizeof(u32)` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_free_blocks(
    pool: *mut kcmm_pool_t,
    blocks: *const u32,
    num: u32,
) -> i32 {
    if pool.is_null() || blocks.is_null() || num == 0 {
        return 0; // freeing nothing is a no-op
    }

    let handle = pool_from_ptr(pool);
    let slice = std::slice::from_raw_parts(blocks, num as usize);
    handle.pool.free_sequence(slice);
    0
}

// ---------------------------------------------------------------------------
// Sequence management
// ---------------------------------------------------------------------------

/// Register a new sequence with its block table.
///
/// Writes the assigned sequence index into `out_seq_idx`.
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `block_table` must point to a buffer of
/// at least `num_blocks * sizeof(u32)` bytes. `out_seq_idx` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn kcmm_register_sequence(
    pool: *mut kcmm_pool_t,
    block_table: *const u32,
    num_blocks: u32,
    out_seq_idx: *mut u32,
) -> i32 {
    if pool.is_null() || block_table.is_null() || out_seq_idx.is_null() {
        if !pool.is_null() {
            pool_from_ptr(pool).set_error(
                "kcmm_register_sequence: null arguments".to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let slice = std::slice::from_raw_parts(block_table, num_blocks as usize);
    let idx = handle.pool.register_sequence(slice.to_vec());
    *out_seq_idx = idx as u32;
    0
}

/// Unregister a sequence and free its blocks.
///
/// Returns 0 on success.  Does nothing if the sequence index is out of bounds.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_unregister_sequence(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
) -> i32 {
    if pool.is_null() {
        return -1;
    }
    let handle = pool_from_ptr(pool);
    handle.pool.unregister_sequence(seq_idx as usize);
    0
}

/// Mark a sequence as recently accessed (hot).
///
/// Updates `last_access` to now and sets `is_active = true`.
/// Call this when a sequence is scheduled for decoding.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_touch(pool: *mut kcmm_pool_t, seq_idx: u32) {
    if pool.is_null() {
        return;
    }
    let handle = pool_from_ptr(pool);
    handle.pool.touch(seq_idx as usize);
}

/// Mark a sequence as cool (eligible for eviction).
///
/// Sets `is_active = false`.  The sequence's blocks become candidates
/// for eviction when memory pressure triggers the tiering engine.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_cool(pool: *mut kcmm_pool_t, seq_idx: u32) {
    if pool.is_null() {
        return;
    }
    let handle = pool_from_ptr(pool);
    handle.pool.cool(seq_idx as usize);
}

/// Update the sequence length for a registered sequence.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_update_seq_len(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
    len: u32,
) {
    if pool.is_null() {
        return;
    }
    let handle = pool_from_ptr(pool);
    handle.pool.update_seq_len(seq_idx as usize, len as usize);
}

/// Get the sequence length for a registered sequence.
///
/// Returns the current sequence length, or 0 if the index is invalid.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_seq_len(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
) -> u32 {
    if pool.is_null() {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    handle.pool.get_seq_len(seq_idx as usize) as u32
}

/// Append a block to an existing sequence's block table.
///
/// Returns 0 on success, -1 if the sequence index is out of bounds.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_append_block_to_sequence(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
    block_idx: u32,
) -> i32 {
    if pool.is_null() {
        return -1;
    }
    let handle = pool_from_ptr(pool);
    // validate that the sequence exists
    {
        let seqs = handle.pool.sequences.lock();
        if (seq_idx as usize) >= seqs.len() {
            handle.set_error(format!(
                "kcmm_append_block_to_sequence: seq_idx {} out of bounds", seq_idx));
            return -1;
        }
    }
    handle.pool.append_block_to_sequence(seq_idx as usize, block_idx);
    0
}

/// Get the block table for a sequence.
///
/// Writes up to `max_blocks` entries into `out_table`. The actual number
/// of blocks is written to `out_count`. Returns 0 on success, -1 on error
/// (e.g., sequence index out of bounds).
///
/// # Safety
/// `pool` must be a valid handle. `out_table` must point to a buffer of at
/// least `max_blocks * sizeof(u32)` bytes. `out_count` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_block_table(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
    out_table: *mut u32,
    max_blocks: u32,
    out_count: *mut u32,
) -> i32 {
    if pool.is_null() || out_table.is_null() || out_count.is_null() {
        if !pool.is_null() {
            pool_from_ptr(pool).set_error(
                "kcmm_get_block_table: null arguments".to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    match handle.pool.get_block_table(seq_idx as usize) {
        Some(table) => {
            let n = table.len().min(max_blocks as usize);
            let dst = std::slice::from_raw_parts_mut(out_table, n);
            dst.copy_from_slice(&table[..n]);
            *out_count = n as u32;
            0
        }
        None => {
            handle.set_error(format!(
                "kcmm_get_block_table: seq_idx {} not found", seq_idx));
            *out_count = 0;
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Block queries
// ---------------------------------------------------------------------------

/// Get the virtual address (byte) offset for a given block index.
///
/// Returns the VA offset in bytes, or 0 if the block index is invalid.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_block_va_offset(
    pool: *mut kcmm_pool_t,
    block_idx: u32,
) -> u64 {
    if pool.is_null() {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    handle.pool.get_block_va_offset(block_idx)
        .map(|o| o as u64)
        .unwrap_or(0)
}

/// Get VA offsets for all blocks in f16-element units.
///
/// Writes up to `max_blocks` entries into `out_offsets`. The actual count
/// of blocks is written to `out_count`.
/// Returns 0 on success, -1 on error.
///
/// Each offset is the byte offset divided by `sizeof(f16)` — this is the
/// format expected by paged-attention CUDA kernels.
///
/// # Safety
/// `pool` must be a valid handle. `out_offsets` must point to a buffer of
/// at least `max_blocks * sizeof(u64)` bytes. `out_count` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_all_block_offsets_f16(
    pool: *mut kcmm_pool_t,
    out_offsets: *mut u64,
    max_blocks: u32,
    out_count: *mut u32,
) -> i32 {
    if pool.is_null() || out_offsets.is_null() || out_count.is_null() {
        if !pool.is_null() {
            pool_from_ptr(pool).set_error(
                "kcmm_get_all_block_offsets_f16: null arguments".to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let offsets = handle.pool.get_all_block_offsets_f16();
    let n = offsets.len().min(max_blocks as usize);
    let dst = std::slice::from_raw_parts_mut(out_offsets, n);
    dst.copy_from_slice(&offsets[..n]);
    *out_count = n as u32;
    0
}

/// Get the location of a block.
///
/// Writes the block location into `out_location`.
/// Returns 0 on success, -1 if the block index is invalid.
///
/// # Safety
/// `pool` must be a valid handle. `out_location` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_block_location(
    pool: *mut kcmm_pool_t,
    block_idx: u32,
    out_location: *mut kcmm_block_location_t,
) -> i32 {
    if pool.is_null() || out_location.is_null() {
        return -1;
    }

    let handle = pool_from_ptr(pool);
    match handle.pool.get_block_location(block_idx) {
        Some(loc) => {
            *out_location = block_loc_to_c(&loc);
            0
        }
        None => {
            handle.set_error(format!(
                "kcmm_get_block_location: block_idx {} not found", block_idx));
            -1
        }
    }
}

/// Get VA offsets for all blocks belonging to a sequence.
///
/// Writes up to `max_blocks` entries into `out_offsets`. The actual count
/// is written to `out_count`. Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `out_offsets` must point to a buffer of
/// at least `max_blocks * sizeof(u64)` bytes. `out_count` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_block_table_va_offsets(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
    out_offsets: *mut u64,
    max_blocks: u32,
    out_count: *mut u32,
) -> i32 {
    if pool.is_null() || out_offsets.is_null() || out_count.is_null() {
        if !pool.is_null() {
            pool_from_ptr(pool).set_error(
                "kcmm_get_block_table_va_offsets: null arguments".to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    match handle.pool.get_block_va_offsets(seq_idx as usize) {
        Some(offsets) => {
            let n = offsets.len().min(max_blocks as usize);
            let dst = std::slice::from_raw_parts_mut(out_offsets, n);
            for (i, &off) in offsets[..n].iter().enumerate() {
                dst[i] = off as u64;
            }
            *out_count = n as u32;
            0
        }
        None => {
            handle.set_error(format!(
                "kcmm_get_block_table_va_offsets: seq_idx {} not found", seq_idx));
            *out_count = 0;
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// VA accessors
// ---------------------------------------------------------------------------

/// Get the K-cache virtual address base for a given layer.
///
/// Returns the VA base in bytes, or 0 if the layer index is out of bounds.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_va_k(
    pool: *mut kcmm_pool_t,
    layer: u32,
) -> u64 {
    if pool.is_null() {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    if (layer as usize) < handle.pool.num_layers {
        handle.pool.va_k(layer as usize)
    } else {
        0
    }
}

/// Get the V-cache virtual address base for a given layer.
///
/// Returns the VA base in bytes, or 0 if the layer index is out of bounds.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_va_v(
    pool: *mut kcmm_pool_t,
    layer: u32,
) -> u64 {
    if pool.is_null() {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    if (layer as usize) < handle.pool.num_layers {
        handle.pool.va_v(layer as usize)
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

/// Write one step of KV data for a batch of sequences.
///
/// `k_src_ptr` and `v_src_ptr` are raw GPU virtual addresses pointing to the
/// source data (post-projection, post-RoPE).  Each source is laid out as
/// [batch, kv_heads * head_dim] in F16.
///
/// `seq_indices` and `positions` must each have `batch` elements.
/// The K and V data for each sequence in the batch is copied to the
/// corresponding logical block in the pool's per-layer VA regions using
/// `cuMemcpyDtoDAsync`.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `seq_indices` and `positions` must each
/// point to `batch * sizeof(u32)` bytes. `k_src_ptr` and `v_src_ptr` must
/// be valid GPU virtual addresses.
#[no_mangle]
pub unsafe extern "C" fn kcmm_append_kv_step(
    pool: *mut kcmm_pool_t,
    layer_idx: u32,
    seq_indices: *const u32,
    positions: *const u32,
    batch: u32,
    k_src_ptr: u64,
    v_src_ptr: u64,
) -> i32 {
    if pool.is_null() || seq_indices.is_null() || positions.is_null() {
        if !pool.is_null() {
            pool_from_ptr(pool).set_error(
                "kcmm_append_kv_step: null arguments".to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let pool = &handle.pool;

    let seqs = std::slice::from_raw_parts(seq_indices, batch as usize);
    let poss = std::slice::from_raw_parts(positions, batch as usize);
    let batch_usize = batch as usize;

    use cudarc::driver::sys::{self, CUdeviceptr};

    let va_k = match pool.va_k.get(layer_idx as usize) {
        Some(&v) => v,
        None => {
            handle.set_error(format!(
                "kcmm_append_kv_step: layer_idx {} out of bounds", layer_idx));
            return -1;
        }
    };
    let va_v = match pool.va_v.get(layer_idx as usize) {
        Some(&v) => v,
        None => {
            handle.set_error(format!(
                "kcmm_append_kv_step: layer_idx {} out of bounds", layer_idx));
            return -1;
        }
    };

    let k_src: CUdeviceptr = k_src_ptr;
    let v_src: CUdeviceptr = v_src_ptr;
    let eb = std::mem::size_of::<half::f16>();
    let step = pool.elem_per_block / pool.block_size; // kv_heads * head_dim
    let nbytes = step * eb;

    let seqs_lock = pool.sequences.lock();
    let info_lock = pool.block_info.lock();

    for b in 0..batch_usize {
        let seq_idx = seqs[b] as usize;
        let pos = poss[b] as usize;

        let seq = match seqs_lock.get(seq_idx) {
            Some(s) => s,
            None => {
                handle.set_error(format!(
                    "kcmm_append_kv_step: seq_idx {} out of bounds", seqs[b]));
                return -1;
            }
        };

        let logical_block = pos / pool.block_size;
        let offset_in_block = pos % pool.block_size;

        if logical_block >= seq.block_table.len() {
            handle.set_error(format!(
                "kcmm_append_kv_step: logical_block {} >= allocated {} for seq {}",
                logical_block, seq.block_table.len(), seq_idx));
            return -1;
        }

        let block_idx = seq.block_table[logical_block] as usize;
        let bi = match info_lock.get(block_idx) {
            Some(bi) if bi.in_use => bi,
            _ => {
                handle.set_error(format!(
                    "kcmm_append_kv_step: block_idx {} not in use", block_idx));
                return -1;
            }
        };

        let dst_off = bi.va_offset / eb + offset_in_block * step;
        let src_off = b * step;

        let dk = va_k + (dst_off * eb) as u64;
        let dv = va_v + (dst_off * eb) as u64;
        let sk = k_src + (src_off * eb) as u64;
        let sv = v_src + (src_off * eb) as u64;

        let r = sys::lib().cuMemcpyDtoDAsync_v2(
            dk, sk, nbytes, std::ptr::null_mut(),
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            handle.set_error(format!(
                "kcmm_append_kv_step: cuMemcpyDtoDAsync K failed: {:?}", r));
            return -1;
        }
        let r = sys::lib().cuMemcpyDtoDAsync_v2(
            dv, sv, nbytes, std::ptr::null_mut(),
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            handle.set_error(format!(
                "kcmm_append_kv_step: cuMemcpyDtoDAsync V failed: {:?}", r));
            return -1;
        }
    }

    0
}

/// Write one step of KV data using vLLM-style physical slot ids.
///
/// `slot_mapping` is a CPU-side i64 array with `batch` elements. Non-negative
/// slots are interpreted as `slot = block_idx * block_size + offset_in_block`;
/// negative slots are padding and are skipped. `k_src_ptr` and `v_src_ptr` are
/// raw GPU virtual addresses pointing to source rows laid out as
/// [batch, kv_heads * head_dim] in F16.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `slot_mapping` must point to
/// `batch * sizeof(i64)` bytes. `k_src_ptr` and `v_src_ptr` must be valid GPU
/// virtual addresses.
#[no_mangle]
pub unsafe extern "C" fn kcmm_append_kv_slots(
    pool: *mut kcmm_pool_t,
    layer_idx: u32,
    slot_mapping: *const i64,
    batch: u32,
    k_src_ptr: u64,
    v_src_ptr: u64,
) -> i32 {
    if pool.is_null() || slot_mapping.is_null() {
        if !pool.is_null() {
            pool_from_ptr(pool)
                .set_error("kcmm_append_kv_slots: null arguments".to_string());
        }
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let pool = &handle.pool;
    let slots = std::slice::from_raw_parts(slot_mapping, batch as usize);
    let batch_usize = batch as usize;

    use cudarc::driver::sys::{self, CUdeviceptr};

    let va_k = match pool.va_k.get(layer_idx as usize) {
        Some(&v) => v,
        None => {
            handle.set_error(format!(
                "kcmm_append_kv_slots: layer_idx {} out of bounds",
                layer_idx
            ));
            return -1;
        }
    };
    let va_v = match pool.va_v.get(layer_idx as usize) {
        Some(&v) => v,
        None => {
            handle.set_error(format!(
                "kcmm_append_kv_slots: layer_idx {} out of bounds",
                layer_idx
            ));
            return -1;
        }
    };

    let k_src: CUdeviceptr = k_src_ptr;
    let v_src: CUdeviceptr = v_src_ptr;
    let eb = std::mem::size_of::<half::f16>();
    let step = pool.elem_per_block / pool.block_size; // kv_heads * head_dim
    let nbytes = step * eb;

    let info_lock = pool.block_info.lock();

    for b in 0..batch_usize {
        let slot = slots[b];
        if slot < 0 {
            continue;
        }

        let slot = slot as usize;
        let block_idx = slot / pool.block_size;
        let offset_in_block = slot % pool.block_size;

        let bi = match info_lock.get(block_idx) {
            Some(bi) if bi.in_use => bi,
            _ => {
                handle.set_error(format!(
                    "kcmm_append_kv_slots: block_idx {} from slot {} not in use",
                    block_idx, slots[b]
                ));
                return -1;
            }
        };

        let dst_off = bi.va_offset / eb + offset_in_block * step;
        let src_off = b * step;

        let dk = va_k + (dst_off * eb) as u64;
        let dv = va_v + (dst_off * eb) as u64;
        let sk = k_src + (src_off * eb) as u64;
        let sv = v_src + (src_off * eb) as u64;

        let r = sys::lib().cuMemcpyDtoDAsync_v2(dk, sk, nbytes, std::ptr::null_mut());
        if r != sys::CUresult::CUDA_SUCCESS {
            handle.set_error(format!(
                "kcmm_append_kv_slots: cuMemcpyDtoDAsync K failed: {:?}",
                r
            ));
            return -1;
        }
        let r = sys::lib().cuMemcpyDtoDAsync_v2(dv, sv, nbytes, std::ptr::null_mut());
        if r != sys::CUresult::CUDA_SUCCESS {
            handle.set_error(format!(
                "kcmm_append_kv_slots: cuMemcpyDtoDAsync V failed: {:?}",
                r
            ));
            return -1;
        }
    }

    0
}

// ---------------------------------------------------------------------------
// Tiering operations
// ---------------------------------------------------------------------------

/// Trigger eviction of up to `count` blocks.
///
/// The tiering engine selects victim blocks according to the current
/// eviction policy and copies their data from GPU to CPU memory.
/// Returns the actual number of blocks evicted on success, or 0 if
/// tiering is disabled or no blocks could be evicted.
///
/// Returns 0 with no error if tiering is disabled — callers should check
/// `kcmm_is_tiering_enabled()` first if they depend on eviction.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_evict_blocks(
    pool: *mut kcmm_pool_t,
    count: u32,
) -> u32 {
    if pool.is_null() || count == 0 {
        return 0;
    }

    let handle = pool_from_ptr(pool);
    let tiering = match handle.pool.tiering.as_ref() {
        Some(t) => t,
        None => return 0,
    };

    // Collect all GPU-resident block handles as candidates.
    let candidates: Vec<super::superblock::BlockHandle> = {
        let info = handle.pool.block_info.lock();
        info.iter()
            .filter(|bi| bi.in_use)
            .filter_map(|bi| {
                if let BlockLocation::GpuResident(h, _) = &bi.location {
                    Some(*h)
                } else {
                    None
                }
            })
            .collect()
    };

    if candidates.is_empty() {
        return 0;
    }

    match tiering.evict_blocks(&handle.pool, &candidates, count as usize) {
        Ok(evicted) => evicted.len() as u32,
        Err(e) => {
            handle.set_error(format!("kcmm_evict_blocks: {:#}", e));
            0
        }
    }
}

/// Restore a single evicted block from CPU memory back to GPU.
///
/// If the block is already `GpuResident`, this is a no-op and the current
/// VA offset is returned. If the block is `CpuResident`, a new GPU physical
/// block is allocated and data is copied from CPU to GPU.
///
/// Returns the GPU VA offset in bytes on success, or 0 on error.
/// Use `kcmm_get_last_error()` for details.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_restore_evicted_block(
    pool: *mut kcmm_pool_t,
    block_idx: u32,
) -> u64 {
    if pool.is_null() {
        return 0;
    }

    let handle = pool_from_ptr(pool);
    match handle.pool.restore_evicted_block(block_idx) {
        Ok(va) => va,
        Err(e) => {
            handle.set_error(format!("kcmm_restore_evicted_block: {:#}", e));
            0
        }
    }
}

/// Restore multiple evicted blocks from CPU memory back to GPU.
///
/// When the batch size is ≥4, the scatter-kernel batch path is used;
/// otherwise each block is restored individually.
/// Blocks already in GPU are silently skipped.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `block_indices` must point to a buffer
/// of at least `count * sizeof(u32)` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_restore_evicted_blocks(
    pool: *mut kcmm_pool_t,
    block_indices: *const u32,
    count: u32,
) -> i32 {
    if pool.is_null() || block_indices.is_null() || count == 0 {
        return 0;
    }

    let handle = pool_from_ptr(pool);
    let slice = std::slice::from_raw_parts(block_indices, count as usize);
    match handle.pool.restore_evicted_blocks(slice) {
        Ok(()) => 0,
        Err(e) => {
            handle.set_error(format!("kcmm_restore_evicted_blocks: {:#}", e));
            -1
        }
    }
}

/// Check whether a block is currently resident in GPU memory.
///
/// Returns 1 if the block is `GpuResident`, 0 otherwise.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_is_gpu_resident(
    pool: *mut kcmm_pool_t,
    block_idx: u32,
) -> i32 {
    if pool.is_null() {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    match handle.pool.get_block_location(block_idx) {
        Some(BlockLocation::GpuResident(..)) => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Metrics and statistics
// ---------------------------------------------------------------------------

/// Collect a UFS-compatible metrics snapshot.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `out` must be a non-null pointer to a
/// `kcmm_metrics_t` struct.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_metrics(
    pool: *mut kcmm_pool_t,
    out: *mut kcmm_metrics_t,
) -> i32 {
    if pool.is_null() || out.is_null() {
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let ufs = handle.pool.collect_metrics();
    let metrics = KcmmMetrics::from_ufs(&ufs);

    // Check if we can get tiering-level counters.
    if let Some(ref tiering) = handle.pool.tiering {
        let te = tiering;
        // We can optionally enrich with tiering stats
        // For now, use the basic metrics
        _ = te;
    }

    *out = kcmm_metrics_t {
        ifr: metrics.ifr,
        pme: metrics.pme,
        bu: metrics.bu,
        rfi: metrics.rfi,
        gpu_blocks: metrics.gpu_blocks,
        cpu_blocks: metrics.cpu_blocks,
        nvme_blocks: metrics.nvme_blocks,
        eviction_count: metrics.eviction_count,
        restoration_count: metrics.restoration_count,
    };
    0
}

/// Get pool runtime statistics.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `out` must be a non-null pointer to a
/// `kcmm_pool_stats_t` struct.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_pool_stats(
    pool: *mut kcmm_pool_t,
    out: *mut kcmm_pool_stats_t,
) -> i32 {
    if pool.is_null() || out.is_null() {
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let p = &handle.pool;

    *out = kcmm_pool_stats_t {
        blocks_in_use: p.blocks_in_use() as u32,
        total_blocks: p.total_blocks() as u32,
        total_physical_blocks: p.total_physical_blocks() as u32,
        free_physical_blocks: p.free_physical_blocks() as u32,
        active_sequences: p.active_sequences() as u32,
        num_layers: p.num_layers() as u32,
        blocks_per_superblock: p.blocks_per_superblock() as u32,
        superblock_count: p.superblock_count() as u32,
        block_size: p.block_size() as u32,
        max_blocks_per_seq: p.max_blocks_per_seq() as u32,
        block_bytes: p.block_bytes() as u32,
        tiering_enabled: if p.tiering.is_some() { 1 } else { 0 },
        sharing_enabled: if p.sharing.is_some() { 1 } else { 0 },
        physical_idle_ratio: p.physical_idle_ratio(),
    };
    0
}

/// Get the number of blocks currently in use.
///
/// Returns the count, or 0 if `pool` is null.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_blocks_in_use(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() {
        return 0;
    }
    pool_from_ptr(pool).pool.blocks_in_use() as u32
}

/// Get the total number of block indices (including recycled slots).
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_total_blocks(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() {
        return 0;
    }
    pool_from_ptr(pool).pool.total_blocks() as u32
}

/// Get the number of free physical blocks.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_free_physical_blocks(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() {
        return 0;
    }
    pool_from_ptr(pool).pool.free_physical_blocks() as u32
}

/// Check whether free blocks are available.
///
/// Returns 1 if free blocks exist, 0 otherwise.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_has_free_blocks(pool: *mut kcmm_pool_t) -> i32 {
    if pool.is_null() {
        return 0;
    }
    if pool_from_ptr(pool).pool.has_free_blocks() { 1 } else { 0 }
}

/// Get the number of active (registered) sequences.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_active_sequences(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() {
        return 0;
    }
    pool_from_ptr(pool).pool.active_sequences() as u32
}

// ---------------------------------------------------------------------------
// Policy configuration
// ---------------------------------------------------------------------------

/// Set the eviction policy at runtime.
///
/// Policy names: "lru", "lfu", or "fifo".
/// Returns 0 on success, -1 if tiering is disabled or the policy is invalid.
///
/// # Safety
/// `pool` must be a valid handle. `policy` must be a null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn kcmm_set_eviction_policy(
    pool: *mut kcmm_pool_t,
    policy: *const c_char,
) -> i32 {
    if pool.is_null() || policy.is_null() {
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let policy_str = match CStr::from_ptr(policy).to_str() {
        Ok(s) => s.to_ascii_lowercase(),
        Err(_) => {
            handle.set_error("kcmm_set_eviction_policy: invalid UTF-8".to_string());
            return -1;
        }
    };

    match handle.pool.tiering.as_ref() {
        Some(t) => {
            if let Err(e) = t.set_policy(&policy_str) {
                handle.set_error(format!("kcmm_set_eviction_policy: {:#}", e));
                return -1;
            }
            0
        }
        None => {
            handle.set_error(
                "kcmm_set_eviction_policy: tiering is disabled".to_string());
            -1
        }
    }
}

/// Get the current eviction policy name.
///
/// Writes the null-terminated policy name into `out_policy` (up to `max_len`
/// bytes). Returns the number of bytes written (excluding null terminator),
/// or 0 if tiering is disabled.
///
/// # Safety
/// `pool` must be a valid handle. `out_policy` must point to a buffer of
/// at least `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_eviction_policy(
    pool: *mut kcmm_pool_t,
    out_policy: *mut c_char,
    max_len: u32,
) -> u32 {
    if pool.is_null() || out_policy.is_null() || max_len == 0 {
        return 0;
    }

    let handle = pool_from_ptr(pool);
    let policy = match handle.pool.tiering.as_ref() {
        Some(t) => t.current_policy_name(),
        None => return 0,
    };

    write_c_str_fixed(out_policy, max_len as usize, &policy) as u32
}

/// Check whether tiering is enabled.
///
/// Returns 1 if tiering is enabled, 0 otherwise.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_is_tiering_enabled(pool: *mut kcmm_pool_t) -> i32 {
    if pool.is_null() {
        return 0;
    }
    if pool_from_ptr(pool).pool.tiering.is_some() { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Hint API
// ---------------------------------------------------------------------------

/// Apply a hint to a sequence.
///
/// Hints influence eviction decisions for the sequence's blocks:
///   - `KCMM_HINT_MULTI_TURN`: protect blocks from eviction (fake "recent access")
///   - `KCMM_HINT_NEAR_END`: mark blocks as cold (preferred victims)
///   - `KCMM_HINT_SYSTEM_PROMPT`: high cache value, protect from eviction
///   - `KCMM_HINT_HIGH_PRIORITY`: SLO-critical, strong protection
///   - `KCMM_HINT_LOW_PRIORITY`: background batch, prefer eviction
///   - `KCMM_HINT_ATTENTION_SINK`: initial tokens, high sharing value
///   - `KCMM_HINT_HEAVY_HITTER`: high-attention tokens, protect
///   - `KCMM_HINT_EVICTABLE`: can be discarded without restore
///
/// Returns 0 on success, -1 if the sequence index is invalid.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_hint(
    pool: *mut kcmm_pool_t,
    seq_idx: u32,
    hint: kcmm_hint_t,
) -> i32 {
    if pool.is_null() {
        return -1;
    }

    let handle = pool_from_ptr(pool);

    // Verify the sequence exists.
    {
        let seqs = handle.pool.sequences.lock();
        if (seq_idx as usize) >= seqs.len() {
            handle.set_error(format!(
                "kcmm_hint: seq_idx {} out of bounds", seq_idx));
            return -1;
        }
        // We have a valid sequence — apply the hint.
        // Currently we need the seq's block table to apply per-block hints.
        let seq = &seqs[seq_idx as usize];
        let block_table = seq.block_table.clone();

        // Apply the appropriate hint behavior:
        match hint {
            kcmm_hint_t::KCMM_HINT_MULTI_TURN
            | kcmm_hint_t::KCMM_HINT_SYSTEM_PROMPT
            | kcmm_hint_t::KCMM_HINT_ATTENTION_SINK
            | kcmm_hint_t::KCMM_HINT_HEAVY_HITTER => {
                // Protect: touch the sequence (update last_access to now)
                drop(seqs);
                handle.pool.touch(seq_idx as usize);

                // Also signal the eviction policy per-block
                if let Some(ref tiering) = handle.pool.tiering {
                    for &blk_idx in &block_table {
                        if let Some(h) = handle.pool.get_block_handle(blk_idx) {
                            tiering.eviction_policy.lock().on_access(h);
                        }
                    }
                }
            }
            kcmm_hint_t::KCMM_HINT_HIGH_PRIORITY => {
                // Strong protection: touch + set High priority class
                drop(seqs);
                handle.pool.touch(seq_idx as usize);
                handle.pool.set_sequence_priority(
                    seq_idx as usize,
                    SequencePriority::High,
                );
            }
            kcmm_hint_t::KCMM_HINT_NEAR_END => {
                // Mark cold but keep normal priority
                drop(seqs);
                handle.pool.cool(seq_idx as usize);
            }
            kcmm_hint_t::KCMM_HINT_LOW_PRIORITY => {
                // Mark cold + set Low priority class
                drop(seqs);
                handle.pool.cool(seq_idx as usize);
                handle.pool.set_sequence_priority(
                    seq_idx as usize,
                    SequencePriority::Low,
                );
            }
            kcmm_hint_t::KCMM_HINT_EVICTABLE => {
                // Mark cold + set Evictable (can discard without restore)
                drop(seqs);
                handle.pool.cool(seq_idx as usize);
                handle.pool.set_sequence_priority(
                    seq_idx as usize,
                    SequencePriority::Evictable,
                );
            }
        }
    }

    0
}

/// Set the protection level for specific blocks within a sequence.
///
/// This is a more precise version of `kcmm_hint` that operates at the
/// individual block level rather than the whole sequence.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle. `block_ids` must point to a buffer of
/// at least `num_blocks * sizeof(u32)` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_protect(
    pool: *mut kcmm_pool_t,
    _seq_idx: u32,
    block_ids: *const u32,
    num_blocks: u32,
    level: kcmm_protection_t,
) -> i32 {
    if pool.is_null() || block_ids.is_null() || num_blocks == 0 {
        return -1;
    }

    let handle = pool_from_ptr(pool);
    let blocks = std::slice::from_raw_parts(block_ids, num_blocks as usize);

    // Apply per-block protection hints.
    // Currently, we only update the eviction policy's access tracking.
    if let Some(ref tiering) = handle.pool.tiering {
        for &blk_idx in blocks {
            let h = match handle.pool.get_block_handle(blk_idx) {
                Some(h) => h,
                None => {
                    handle.set_error(format!(
                        "kcmm_protect: block_idx {} not found", blk_idx));
                    return -1;
                }
            };

            match level {
                kcmm_protection_t::KCMM_PROTECT_NEVER_EVICT
                | kcmm_protection_t::KCMM_PROTECT_PREFERRED => {
                    // Refresh access timestamp (makes it appear "recently used")
                    tiering.eviction_policy.lock().on_access(h);
                }
                kcmm_protection_t::KCMM_EVICT_PREFERRED => {
                    // Mark as evictable: no-op for now — the policy will
                    // prefer these since we don't touch their timestamps.
                    // Future: could explicitly push to front of LRU queue.
                }
            }
        }
    }

    // Ignore _seq_idx for now (verified conceptually; per-block ops
    // are sufficient for the current EvictionPolicy trait).

    0
}

// ---------------------------------------------------------------------------
// Prefix sharing
// ---------------------------------------------------------------------------

/// Share prefix blocks from a source sequence to a destination sequence.
///
/// The destination sequence reuses the source's prefix blocks instead of
/// allocating its own. The `num_blocks` prefix blocks are shared.
///
/// Returns 0 on success, -1 if sharing is unavailable or arguments are invalid.
///
/// # Safety
/// `pool` must be a valid handle. `out_blocks` must point to a buffer of at
/// least `num_blocks * sizeof(u32)` bytes.
#[no_mangle]
pub unsafe extern "C" fn kcmm_share_prefix(
    pool: *mut kcmm_pool_t,
    _src_seq: u32,
    _dst_seq: u32,
    num_blocks: u32,
    out_blocks: *mut u32,
) -> i32 {
    if pool.is_null() || out_blocks.is_null() || num_blocks == 0 {
        return -1;
    }

    let handle = pool_from_ptr(pool);
    if handle.pool.sharing.is_none() {
        handle.set_error(
            "kcmm_share_prefix: sharing is not enabled (step 4 feature)".to_string());
        return -1;
    }

    // Prefix sharing is a step 4 feature — return placeholder.
    // The out_blocks buffer is zeroed to indicate no shared blocks.
    let dst = std::slice::from_raw_parts_mut(out_blocks, num_blocks as usize);
    dst.fill(0);

    0
}

// ---------------------------------------------------------------------------
// Utility: low watermark check
// ---------------------------------------------------------------------------

/// Check whether free block ratio is below the low watermark.
///
/// Returns 1 if below the threshold, 0 otherwise.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_below_low_watermark(
    pool: *mut kcmm_pool_t,
    threshold: f32,
) -> i32 {
    if pool.is_null() {
        return 0;
    }
    let handle = pool_from_ptr(pool);
    if handle.pool.below_low_watermark(threshold) { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Block size / config accessors
// ---------------------------------------------------------------------------

/// Get the block size in tokens.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_block_size(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() { 0 } else { pool_from_ptr(pool).pool.block_size() as u32 }
}

/// Get the maximum blocks per sequence.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_max_blocks_per_seq(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() { 0 } else { pool_from_ptr(pool).pool.max_blocks_per_seq() as u32 }
}

/// Get the byte size of each block.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_block_bytes(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() { 0 } else { pool_from_ptr(pool).pool.block_bytes() as u32 }
}

/// Get the number of transformer layers.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_num_layers(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() { 0 } else { pool_from_ptr(pool).pool.num_layers() as u32 }
}

/// Get the maximum batch size.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_max_batch(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() { 0 } else { pool_from_ptr(pool).pool.max_batch as u32 }
}

/// Get the maximum sequence length in tokens.
#[no_mangle]
pub unsafe extern "C" fn kcmm_get_max_seq_len(pool: *mut kcmm_pool_t) -> u32 {
    if pool.is_null() { 0 } else { pool_from_ptr(pool).pool.max_seq_len as u32 }
}

/// Synchronize all CUDA streams (evict, restore, prefetch).
///
/// Blocks the calling CPU thread until all GPU operations complete.
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `pool` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn kcmm_synchronize(pool: *mut kcmm_pool_t) -> i32 {
    if pool.is_null() {
        return -1;
    }
    let handle = pool_from_ptr(pool);
    match handle.pool.streams.synchronize_all() {
        Ok(()) => 0,
        Err(e) => {
            handle.set_error(format!("kcmm_synchronize: {:#}", e));
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{size_of, MaybeUninit};
    use std::ptr::addr_of;

    unsafe fn config_field_offset<T>(
        f: impl FnOnce(*const kcmm_config_t) -> *const T,
    ) -> usize {
        let uninit = MaybeUninit::<kcmm_config_t>::uninit();
        let base = uninit.as_ptr();
        f(base) as usize - base as usize
    }

    #[test]
    fn kcmm_config_layout_matches_c_header_on_lp64() {
        assert_eq!(size_of::<usize>(), 8);
        assert_eq!(size_of::<kcmm_config_t>(), 376);

        unsafe {
            assert_eq!(
                config_field_offset(|base| addr_of!((*base).max_seq_len)),
                336
            );
            assert_eq!(
                config_field_offset(|base| addr_of!((*base).low_watermark_threshold)),
                344
            );
            assert_eq!(
                config_field_offset(|base| addr_of!((*base).background_evict_interval_ms)),
                352
            );
            assert_eq!(
                config_field_offset(|base| addr_of!((*base).attention_sink_blocks)),
                360
            );
            assert_eq!(
                config_field_offset(|base| addr_of!((*base).recent_window_blocks)),
                368
            );
        }
    }
}
