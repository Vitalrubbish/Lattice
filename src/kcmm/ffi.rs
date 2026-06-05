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
// Full implementation in step 3 week 16.

/// Opaque handle to a KCMM pool.
/// Actually wraps `Arc<KcmmPool>`.
#[repr(C)]
pub struct kcmm_pool_t {
    _private: [u8; 0], // opaque
}

/// KCMM metrics structure (mirrors `KcmmMetrics` in metrics.rs).
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct kcmm_metrics_t {
    pub ifr: f64,
    pub pme: f64,
    pub bu: f64,
    pub rfi: f64,
    pub gpu_blocks: u64,
    pub cpu_blocks: u64,
    pub nvme_blocks: u64,
    pub eviction_count: u64,
    pub restoration_count: u64,
}

/// KCMM hint types for the Hint API (section 1.6.6).
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

// The C function signatures are declared here for documentation.
// Actual #[no_mangle] extern "C" functions will be added in week 16.

// extern "C" {
//     pub fn kcmm_pool_create(block_size: usize, max_blocks: usize,
//                             cpu_cache_path: *const c_char) -> *mut kcmm_pool_t;
//     pub fn kcmm_pool_destroy(pool: *mut kcmm_pool_t);
//     pub fn kcmm_alloc_blocks(pool: *mut kcmm_pool_t, seq_id: u64,
//                              num_blocks: usize, out_blocks: *mut u32) -> c_int;
//     pub fn kcmm_free_blocks(pool: *mut kcmm_pool_t, seq_id: u64,
//                             blocks: *const u32, num: usize);
//     pub fn kcmm_touch(pool: *mut kcmm_pool_t, seq_id: u64);
//     pub fn kcmm_cool(pool: *mut kcmm_pool_t, seq_id: u64);
//     pub fn kcmm_get_metrics(pool: *mut kcmm_pool_t, out: *mut kcmm_metrics_t);
//     pub fn kcmm_share_prefix(pool: *mut kcmm_pool_t, src_seq: u64,
//                              dst_seq: u64, num_blocks: usize,
//                              out: *mut u32) -> c_int;
// }
