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
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::driver::sys;
use parking_lot::Mutex;
use crate::config::KcmmConfig;
use crate::cuda::CudaContext;
use crate::kcmm::pool::{BlockLocation, KcmmPool};
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

    // --- Memcpy batching infrastructure ---

    /// GPU staging buffer for gather/scatter kernels (f16 elements).
    /// Holds one layer's worth of batched data: `max_batch_blocks × half_count`.
    gpu_staging: Option<CudaSlice<half::f16>>,
    /// CPU staging buffer for batched transfers.
    /// Holds `max_batch_blocks × num_layers × 2 × block_bytes` bytes for
    /// all-layer gather before scatter to individual CPU slots.
    cpu_staging: Vec<u8>,
    /// Gather kernel: N scattered GPU sources → contiguous GPU destination.
    gather_kernel: Option<CudaFunction>,
    /// Scatter kernel: contiguous GPU source → N scattered GPU destinations.
    scatter_kernel: Option<CudaFunction>,
    /// CUDA device handle (for temporary GPU allocations during batching).
    device: Option<Arc<CudaDevice>>,
    /// Maximum blocks per batch (from config).
    max_batch_blocks: usize,
}

/// Per-block state carried from async-submit to finalize during batched eviction.
struct EvictContext {
    block_idx: u32,
    handle: BlockHandle,
    cpu_offset: usize,
    total_bytes: usize,
}

/// Per-block state carried from async-submit to finalize during restoration.
struct RestoreContext {
    block_idx: u32,
    cpu_offset: usize,
    total_bytes: usize,
    new_handle: BlockHandle,
    va_offset: usize,
}

impl TieringEngine {
    /// Create a new tiering engine.
    ///
    /// Creates (or opens) a file at `config.cpu_cache_path` and mmaps it
    /// as the CPU swap buffer.  Using a file-backed mapping (instead of
    /// `MAP_ANONYMOUS`) enables cross-process sharing of the swap region
    /// and persistence of swapped data across engine restarts.
    ///
    /// `num_layers` and `block_bytes` determine the total swap buffer size:
    /// every evicted block needs `num_layers * 2 * block_bytes` bytes of
    /// CPU storage (K+V for every layer).
    ///
    /// When `device` and `config.max_batch_blocks` are provided, GPU/CPU
    /// staging buffers are allocated and the gather/scatter kernels are
    /// compiled for batched memcpy operations.
    pub fn new(
        config: &KcmmConfig,
        num_layers: usize,
        block_bytes: usize,
        device: Option<Arc<CudaDevice>>,
    ) -> Result<Self> {
        // Total buffer size: enough to hold all blocks if every one is evicted.
        let per_block_bytes = num_layers * 2 * block_bytes;
        let cpu_buffer_size = config.max_blocks * per_block_bytes;

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

        let max_batch_blocks = config.max_batch_blocks;

        // Compile gather/scatter kernels and allocate staging buffers if a
        // device is available and batching is enabled.
        let (gpu_staging, gather_kernel, scatter_kernel, device) =
            if let Some(dev) = device {
                if max_batch_blocks > 0 {
                    let (gk, sk) = Self::compile_kv_gather_kernels(&dev)?;
                    let half_count = block_bytes / std::mem::size_of::<half::f16>();
                    let gpu_staging_elems = max_batch_blocks * half_count;
                    let gs = dev.alloc_zeros::<half::f16>(gpu_staging_elems)
                        .with_context(|| "alloc GPU staging buffer (f16) for memcpy batching")?;
                    (Some(gs), Some(gk), Some(sk), Some(dev))
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };

        // CPU staging buffer: holds all layers for a full batch.
        // Layout: [batch*K0][batch*V0][batch*K1][batch*V1]...
        let cpu_staging_size = if max_batch_blocks > 0 && num_layers > 0 {
            max_batch_blocks * per_block_bytes
        } else {
            0
        };
        let cpu_staging = vec![0u8; cpu_staging_size];

        Ok(Self {
            cpu_buffer,
            cpu_buffer_size,
            cpu_buffer_path: config.cpu_cache_path.clone(),
            nvme_enabled: false,
            eviction_policy,
            slot_allocator: Mutex::new(CpuSlotAllocator::new(cpu_buffer_size)),
            gpu_staging,
            cpu_staging,
            gather_kernel,
            scatter_kernel,
            device,
            max_batch_blocks,
        })
    }

    /// Compile the gather/scatter KV kernels via NVRTC.
    fn compile_kv_gather_kernels(
        device: &Arc<CudaDevice>,
    ) -> Result<(CudaFunction, CudaFunction)> {
        let mut include_paths = vec!["/usr/include".into()];
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".into());
        include_paths.push(format!("{cuda_home}/include"));

        let opts = cudarc::nvrtc::CompileOptions {
            ftz: Some(true),
            use_fast_math: Some(true),
            include_paths,
            ..Default::default()
        };

        let src = include_str!("../cuda/kernels/kv_gather.cu");
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts)
            .context("compile kv_gather kernel via NVRTC")?;
        device
            .load_ptx(ptx, "kv_gather", &["gather_kv_layer", "scatter_kv_layer"])
            .context("load kv_gather PTX module")?;

        let gather = device
            .get_func("kv_gather", "gather_kv_layer")
            .ok_or_else(|| anyhow::anyhow!("kernel not found: kv_gather::gather_kv_layer"))?;
        let scatter = device
            .get_func("kv_gather", "scatter_kv_layer")
            .ok_or_else(|| anyhow::anyhow!("kernel not found: kv_gather::scatter_kv_layer"))?;

        Ok((gather, scatter))
    }

    /// Get the CPU buffer base pointer.
    pub fn cpu_buffer_ptr(&self) -> *mut u8 {
        self.cpu_buffer
    }

    /// Get the CPU buffer size.
    pub fn cpu_buffer_size(&self) -> usize {
        self.cpu_buffer_size
    }

    /// Allocate a CPU swap buffer slot of at least `size` bytes.
    ///
    /// Returns the byte offset into the CPU buffer, or an error if
    /// the buffer is exhausted.
    pub fn alloc_cpu_slot(&self, size: usize) -> Result<usize> {
        let mut allocator = self.slot_allocator.lock();
        allocator
            .allocate(size)
            .ok_or_else(|| anyhow::anyhow!("CPU swap buffer exhausted (need {} bytes)", size))
    }

    /// Return a previously allocated CPU swap buffer slot to the free pool.
    pub fn free_cpu_slot(&self, offset: usize, size: usize) {
        let mut allocator = self.slot_allocator.lock();
        allocator.free(offset, size);
    }

    // --- Eviction ---

    /// Phase 1 of single-block eviction: allocate CPU slot, mark Evicting, submit
    /// async D2H copies.  Does **not** synchronise the stream — the caller must
    /// synchronise before calling `evict_finalize`.
    ///
    /// On failure the block is rolled back to `GpuResident` and the CPU slot freed.
    fn evict_submit_async(
        &self,
        pool: &KcmmPool,
        block_idx: u32,
        handle: BlockHandle,
    ) -> Result<EvictContext> {
        let block_bytes = pool.block_bytes;
        let total_bytes = pool.num_layers * 2 * block_bytes;

        // 1. Allocate CPU slot
        let cpu_offset = self.alloc_cpu_slot(total_bytes)?;

        // 2. Mark as Evicting — concurrent access will see this and back off
        pool.set_block_location(block_idx, BlockLocation::Evicting)?;

        // 3. Submit all async D2H copies (no synchronise here — caller batches it)
        if let Err(e) = self.evict_single_block_all_layers(pool, handle, cpu_offset) {
            // Rollback: return CPU slot, restore location
            self.free_cpu_slot(cpu_offset, total_bytes);
            // Best-effort restore to GpuResident — the original VA is still valid
            // since we haven't released physical resources yet.
            match pool.block_va_offset(handle) {
                Ok(va_off) => {
                    if let Err(rollback_err) = pool.set_block_location(
                        block_idx,
                        BlockLocation::GpuResident(handle, va_off as u64),
                    ) {
                        tracing::error!(
                            block_idx,
                            ?handle,
                            error = %rollback_err,
                            "KCMM: CRITICAL — failed to rollback location after memcpy submit error; block stuck as Evicting"
                        );
                    }
                }
                Err(va_err) => {
                    tracing::error!(
                        block_idx,
                        ?handle,
                        error = %va_err,
                        "KCMM: CRITICAL — failed to compute VA offset during rollback; block stuck as Evicting"
                    );
                }
            }
            return Err(e);
        }

        Ok(EvictContext {
            block_idx,
            handle,
            cpu_offset,
            total_bytes,
        })
    }

    /// Phase 3 of single-block eviction: release GPU physical resources and mark
    /// `CpuResident`.  Must only be called after `pool.streams.evict.synchronize()`
    /// has confirmed that all D2H copies completed successfully.
    fn evict_finalize(&self, pool: &KcmmPool, ctx: EvictContext) -> Result<()> {
        // Release GPU physical resources (return to per-layer free lists)
        pool.release_block_physical(ctx.block_idx)?;

        // Mark as CpuResident
        pool.set_block_location(
            ctx.block_idx,
            BlockLocation::CpuResident(ctx.cpu_offset),
        )?;

        tracing::debug!(
            ctx.block_idx,
            ?ctx.handle,
            ctx.cpu_offset,
            ctx.total_bytes,
            "KCMM: evicted block to CPU"
        );

        Ok(())
    }

    /// Evict a single block from GPU to CPU (convenience wrapper).
    ///
    /// For batched eviction use `evict_blocks` which amortises the CUDA
    /// stream synchronise across all victims.
    #[allow(dead_code)]
    fn evict_single_block(
        &self,
        pool: &KcmmPool,
        block_idx: u32,
        handle: BlockHandle,
    ) -> Result<()> {
        let ctx = self.evict_submit_async(pool, block_idx, handle)?;
        pool.streams.evict.synchronize()?;
        self.evict_finalize(pool, ctx)
    }

    /// Evict up to `count` blocks from GPU to CPU.
    ///
    /// Uses the configured eviction policy to select victims from
    /// `candidates`, then copies each victim's data for all layers
    /// (K and V) to the CPU swap buffer and releases the GPU physical
    /// resources.
    ///
    /// The eviction is performed in three phases to amortise CUDA
    /// stream synchronisation across all blocks:
    ///
    ///   1. Submit all async D2H copies (no synchronise)
    ///   2. One `cuStreamSynchronize` for the entire batch
    ///   3. Release physical resources and update block locations
    ///
    /// Returns the list of successfully evicted block handles.
    pub fn evict_blocks(
        &self,
        pool: &KcmmPool,
        candidates: &[BlockHandle],
        count: usize,
    ) -> Result<Vec<BlockHandle>> {
        if candidates.is_empty() || count == 0 {
            return Ok(Vec::new());
        }

        // 1. Select victims
        let victims = self.eviction_policy.select_victims(candidates, count);
        if victims.is_empty() {
            return Ok(Vec::new());
        }

        // If memcpy batching is available and the victim count justifies the
        // gather-kernel launch overhead, use the batched path.
        const MIN_BATCH_FOR_GATHER: usize = 4;
        if victims.len() >= MIN_BATCH_FOR_GATHER
            && self.gather_kernel.is_some()
            && self.gpu_staging.is_some()
        {
            return self.evict_blocks_batched(pool, &victims);
        }

        // 2. Phase 1 — Submit all async D2H copies
        let mut pending: Vec<EvictContext> = Vec::with_capacity(victims.len());
        for &victim in &victims {
            let block_idx = pool.find_block_idx(victim).ok_or_else(|| {
                anyhow::anyhow!("victim block {:?} not found in pool", victim)
            })?;

            match self.evict_submit_async(pool, block_idx, victim) {
                Ok(ctx) => pending.push(ctx),
                Err(e) => {
                    tracing::warn!(
                        block_idx,
                        ?victim,
                        error = %e,
                        "KCMM: evict_submit_async failed, skipping"
                    );
                    // Continue — partial success is better than total failure.
                }
            }
        }

        if pending.is_empty() {
            return Ok(Vec::new());
        }

        // 3. Phase 2 — ONE synchronise for the entire batch.
        //    If this fails, *all* pending blocks may be in an unknown state
        //    (some D2H copies may have completed, others not).  We
        //    conservatively roll back every pending block.
        if let Err(e) = pool.streams.evict.synchronize() {
            tracing::error!(
                pending_count = pending.len(),
                error = %e,
                "KCMM: batch evict synchronize failed; rolling back all pending blocks"
            );
            for ctx in pending {
                self.free_cpu_slot(ctx.cpu_offset, ctx.total_bytes);
                match pool.block_va_offset(ctx.handle) {
                    Ok(va_off) => {
                        let _ = pool.set_block_location(
                            ctx.block_idx,
                            BlockLocation::GpuResident(ctx.handle, va_off as u64),
                        );
                    }
                    Err(va_err) => {
                        tracing::error!(
                            ctx.block_idx,
                            ?ctx.handle,
                            error = %va_err,
                            "KCMM: CRITICAL — failed to rollback after batch sync failure"
                        );
                    }
                }
            }
            return Err(e);
        }

        // 4. Phase 3 — Finalize all blocks (no GPU ops, fast)
        let mut evicted = Vec::with_capacity(pending.len());
        for ctx in pending {
            let handle = ctx.handle;
            match self.evict_finalize(pool, ctx) {
                Ok(()) => {
                    self.eviction_policy.on_evict(handle);
                    evicted.push(handle);
                }
                Err(e) => {
                    tracing::error!(
                        ?handle,
                        error = %e,
                        "KCMM: evict_finalize failed, skipping"
                    );
                }
            }
        }

        Ok(evicted)
    }

    /// Batched eviction using gather kernel + single D2H per layer.
    ///
    /// Instead of issuing `4 × N` individual `memcpy_d2h_async` calls
    /// (K0, V0, K1, V1 per block), this method issues only 4 batched
    /// transfers (one per layer×KV pair) by first gathering same-layer
    /// KV data from scattered GPU VAs into the contiguous GPU staging
    /// buffer via the `gather_kv_layer` kernel.
    ///
    /// Phases:
    ///   1. Alloc CPU slot + mark Evicting (per block, CPU-side, fast)
    ///   2. For each layer×KV: gather → batched D2H (all queued on evict stream)
    ///   3. One `cuStreamSynchronize`
    ///   4. CPU scatter: staging → per-block CPU slots
    ///   5. Finalize: release physical + mark CpuResident
    fn evict_blocks_batched(
        &self,
        pool: &KcmmPool,
        victims: &[BlockHandle],
    ) -> Result<Vec<BlockHandle>> {
        let n = victims.len();
        let block_bytes = pool.block_bytes;
        let num_layers = pool.num_layers;
        let per_block_bytes = num_layers * 2 * block_bytes;
        let half_count = block_bytes / std::mem::size_of::<half::f16>();

        let gather_kernel = self.gather_kernel.as_ref().unwrap();
        let gpu_staging = self.gpu_staging.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();

        // --- Phase 1: Alloc CPU slots + mark Evicting ---
        let mut pending: Vec<EvictContext> = Vec::with_capacity(n);
        for &victim in victims {
            let block_idx = pool.find_block_idx(victim).ok_or_else(|| {
                anyhow::anyhow!("victim block {:?} not found in pool", victim)
            })?;

            let cpu_offset = match self.alloc_cpu_slot(per_block_bytes) {
                Ok(off) => off,
                Err(e) => {
                    tracing::warn!(block_idx, ?victim, error = %e,
                        "KCMM: CPU slot alloc failed in batched evict, skipping");
                    continue;
                }
            };

            if let Err(e) = pool.set_block_location(block_idx, BlockLocation::Evicting) {
                self.free_cpu_slot(cpu_offset, per_block_bytes);
                tracing::warn!(block_idx, ?victim, error = %e,
                    "KCMM: mark Evicting failed in batched evict, skipping");
                continue;
            }

            pending.push(EvictContext {
                block_idx,
                handle: victim,
                cpu_offset,
                total_bytes: per_block_bytes,
            });
        }

        if pending.is_empty() {
            return Ok(Vec::new());
        }

        // --- Phase 2: GPU gather + batched D2H per layer ---
        // Note: `htod_sync_copy_into` uses the default stream and therefore
        // synchronizes all streams.  This means each layer's upload implicitly
        // waits for the previous layer's GPU work to finish.  This adds ~120 µs
        // (4 layers × 30 µs) but is correct and the overhead is negligible
        // compared to the ~6.8 ms saved by eliminating per-block driver calls.
        let mut layer_idx: usize = 0;
        let _layer_count = num_layers * 2; // K+V per layer
        for l in 0..num_layers {
            for is_v in [false, true] {
                let actual_n = pending.len(); // may be < n if slots failed

                // Build host-side pointer array for this layer×KV
                let mut ptrs_host: Vec<u64> = Vec::with_capacity(actual_n);
                for ctx in &pending {
                    let gpu_va = if is_v {
                        pool.gpu_va_for_block(ctx.handle, l, true)?
                    } else {
                        pool.gpu_va_for_block(ctx.handle, l, false)?
                    };
                    ptrs_host.push(gpu_va);
                }

                // Upload to device (sync — see note above).
                // Allocate a fresh device array each iteration (small: N × 8 bytes).
                let mut ptrs_dev_layer = device
                    .alloc_zeros::<u64>(actual_n)
                    .context("alloc ptrs layer device array")?;
                device.htod_sync_copy_into(&ptrs_host, &mut ptrs_dev_layer)?;

                // Launch gather kernel: scattered blocks → contiguous GPU staging
                // SAFETY: all source VAs are valid GPU addresses allocated by the pool.
                unsafe {
                    crate::cuda::kernels::launch_kv_gather(
                        gather_kernel,
                        &ptrs_dev_layer,
                        gpu_staging,
                        half_count,
                        actual_n,
                    )
                    .context("launch gather_kv_layer for batched eviction")?;
                }

                // Batched D2H: GPU staging → CPU staging[this layer's region]
                let layer_byte_offset = layer_idx * actual_n * block_bytes;
                let nbytes = actual_n * block_bytes;
                let gpu_staging_ptr =
                    CudaContext::device_ptr(&gpu_staging) as sys::CUdeviceptr;
                unsafe {
                    pool.streams.evict.memcpy_d2h_async(
                        (self.cpu_staging.as_ptr() as *mut u8)
                            .add(layer_byte_offset),
                        gpu_staging_ptr,
                        nbytes,
                    )?;
                }

                layer_idx += 1;
            }
        }

        // --- Phase 3: Synchronise ---
        if let Err(e) = pool.streams.evict.synchronize() {
            tracing::error!(
                pending_count = pending.len(),
                error = %e,
                "KCMM: batched evict synchronize failed; rolling back"
            );
            for ctx in &pending {
                self.free_cpu_slot(ctx.cpu_offset, ctx.total_bytes);
                let _ = pool.set_block_location(
                    ctx.block_idx,
                    BlockLocation::GpuResident(
                        ctx.handle,
                        pool.block_va_offset(ctx.handle).unwrap_or(0) as u64,
                    ),
                );
            }
            return Err(e);
        }

        // --- Phase 4: CPU scatter — staging → per-block CPU slots ---
        // CPU staging layout: [batch_K0][batch_V0][batch_K1][batch_V1]
        // Per-block slot layout: [K0][V0][K1][V1]
        let actual_n = pending.len();
        for (i, ctx) in pending.iter().enumerate() {
            let mut slot_off = ctx.cpu_offset;
            let mut layer_idx2: usize = 0;
            for _l in 0..num_layers {
                for _is_v in 0..2 {
                    let staging_off = layer_idx2 * actual_n * block_bytes + i * block_bytes;
                    // SAFETY: both staging and cpu_buffer are valid, non-overlapping.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.cpu_staging.as_ptr().add(staging_off),
                            self.cpu_buffer.add(slot_off),
                            block_bytes,
                        );
                    }
                    slot_off += block_bytes;
                    layer_idx2 += 1;
                }
            }
        }

        // --- Phase 5: Finalize ---
        let mut evicted = Vec::with_capacity(actual_n);
        for ctx in pending {
            let handle = ctx.handle;
            match self.evict_finalize(pool, ctx) {
                Ok(()) => {
                    self.eviction_policy.on_evict(handle);
                    evicted.push(handle);
                }
                Err(e) => {
                    tracing::error!(?handle, error = %e,
                        "KCMM: evict_finalize failed in batched path, skipping");
                }
            }
        }

        Ok(evicted)
    }

    /// Copy all K and V layers for a block from GPU to CPU.
    ///
    /// Data layout in CPU buffer (conceptual):
    ///
    /// `[K layer 0][V layer 0][K layer 1][V layer 1]...[K layer N][V layer N]`
    fn evict_single_block_all_layers(
        &self,
        pool: &KcmmPool,
        handle: BlockHandle,
        cpu_offset: usize,
    ) -> Result<()> {
        let block_bytes = pool.block_bytes;
        let num_layers = pool.num_layers;
        let mut byte_offset = cpu_offset;

        for l in 0..num_layers {
            // Copy K layer
            let gpu_va_k = pool.gpu_va_for_block(handle, l, false)?;
            unsafe {
                pool.streams.evict.memcpy_d2h_async(
                    self.cpu_buffer.add(byte_offset),
                    gpu_va_k,
                    block_bytes,
                )?;
            }
            byte_offset += block_bytes;

            // Copy V layer
            let gpu_va_v = pool.gpu_va_for_block(handle, l, true)?;
            unsafe {
                pool.streams.evict.memcpy_d2h_async(
                    self.cpu_buffer.add(byte_offset),
                    gpu_va_v,
                    block_bytes,
                )?;
            }
            byte_offset += block_bytes;
        }

        Ok(())
    }

    // --- Restoration ---

    /// Batched restoration using CPU gather + single H2D per layer + scatter
    /// kernel.
    ///
    /// Mirror of `evict_blocks_batched`: instead of `4 × N` individual
    /// `memcpy_h2d_async` calls, issues only 4 batched H2D transfers.
    ///
    /// Each entry in `blocks` is `(block_idx, cpu_offset)`.  The caller
    /// must have already verified that each block is in `CpuResident` state.
    ///
    /// Phases:
    ///   1. Mark Restoring + alloc GPU block (per block, fast)
    ///   2. For each layer×KV: CPU gather → batched H2D → scatter kernel
    ///      (all queued on restore stream)
    ///   3. One `cuStreamSynchronize`
    ///   4. Finalize: mark GpuResident + free CPU slot
    pub(crate) fn restore_blocks_batched(
        &self,
        pool: &KcmmPool,
        blocks: &[(u32, usize)], // (block_idx, cpu_offset)
    ) -> Result<()> {
        let n = blocks.len();
        let block_bytes = pool.block_bytes;
        let num_layers = pool.num_layers;
        let per_block_bytes = num_layers * 2 * block_bytes;
        let half_count = block_bytes / std::mem::size_of::<half::f16>();

        let scatter_kernel = self.scatter_kernel.as_ref().unwrap();
        let gpu_staging = self.gpu_staging.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();

        // --- Phase 1: Mark Restoring + alloc GPU blocks ---
        let mut pending: Vec<RestoreContext> = Vec::with_capacity(n);
        for &(block_idx, cpu_offset) in blocks {
            // Mark as Restoring
            if let Err(e) = pool.set_block_location(block_idx, BlockLocation::Restoring) {
                tracing::warn!(block_idx, cpu_offset, error = %e,
                    "KCMM: mark Restoring failed in batched restore, skipping");
                continue;
            }

            // Allocate new GPU physical block
            let (va_offset, sb_idx, blk_in_sb) = match pool.alloc_one_block_internal() {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(block_idx, cpu_offset, error = %e,
                        "KCMM: GPU alloc failed in batched restore, rolling back");
                    let _ = pool.set_block_location(
                        block_idx,
                        BlockLocation::CpuResident(cpu_offset),
                    );
                    continue;
                }
            };

            let new_handle = BlockHandle {
                superblock_idx: sb_idx,
                block_index: blk_in_sb,
            };

            // Update BlockInfo with new physical allocation
            if let Err(e) = pool.update_block_physical(block_idx, va_offset, sb_idx, blk_in_sb) {
                let _ = pool.release_block_physical(block_idx);
                let _ = pool.set_block_location(
                    block_idx,
                    BlockLocation::CpuResident(cpu_offset),
                );
                tracing::warn!(block_idx, cpu_offset, error = %e,
                    "KCMM: update_block_physical failed in batched restore, skipping");
                continue;
            }

            pending.push(RestoreContext {
                block_idx,
                cpu_offset,
                total_bytes: per_block_bytes,
                new_handle,
                va_offset,
            });
        }

        if pending.is_empty() {
            return Ok(());
        }

        // --- Phase 2: CPU gather → batched H2D → scatter kernel per layer ---
        let actual_n = pending.len();
        let mut layer_idx: usize = 0;
        for l in 0..num_layers {
            for is_v in [false, true] {
                // Step 2a: CPU gather — copy from each block's CPU slot to staging
                for (i, ctx) in pending.iter().enumerate() {
                    let slot_src = ctx.cpu_offset + layer_idx * block_bytes;
                    let staging_dst = layer_idx * actual_n * block_bytes + i * block_bytes;
                    // SAFETY: source (CPU buffer slot) and dest (CPU staging) are
                    // valid, non-overlapping memory regions.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.cpu_buffer.add(slot_src),
                            (self.cpu_staging.as_ptr() as *mut u8).add(staging_dst),
                            block_bytes,
                        );
                    }
                }

                // Step 2b: Batched H2D — CPU staging[this layer] → GPU staging
                let layer_byte_offset = layer_idx * actual_n * block_bytes;
                let nbytes = actual_n * block_bytes;
                let gpu_staging_ptr =
                    CudaContext::device_ptr(&gpu_staging) as sys::CUdeviceptr;
                unsafe {
                    pool.streams.restore.memcpy_h2d_async(
                        gpu_staging_ptr,
                        self.cpu_staging.as_ptr().add(layer_byte_offset),
                        nbytes,
                    )?;
                }

                // Step 2c: Build destination pointer array for scatter kernel
                let mut ptrs_host: Vec<u64> = Vec::with_capacity(actual_n);
                for ctx in &pending {
                    let gpu_va = if is_v {
                        pool.va_v(l) + ctx.va_offset as u64
                    } else {
                        pool.va_k(l) + ctx.va_offset as u64
                    };
                    ptrs_host.push(gpu_va);
                }

                // Upload to device.
                let mut ptrs_dev_layer = device
                    .alloc_zeros::<u64>(actual_n)
                    .context("alloc ptrs layer device array for restore")?;
                device.htod_sync_copy_into(&ptrs_host, &mut ptrs_dev_layer)?;

                // Launch scatter kernel: GPU staging → scattered block VAs
                unsafe {
                    crate::cuda::kernels::launch_kv_scatter(
                        scatter_kernel,
                        gpu_staging,
                        &ptrs_dev_layer,
                        half_count,
                        actual_n,
                    )
                    .context("launch scatter_kv_layer for batched restore")?;
                }

                layer_idx += 1;
            }
        }

        // --- Phase 3: Synchronise ---
        if let Err(e) = pool.streams.restore.synchronize() {
            tracing::error!(
                pending_count = actual_n,
                error = %e,
                "KCMM: batched restore synchronize failed; rolling back"
            );
            for ctx in &pending {
                let _ = pool.release_block_physical(ctx.block_idx);
                let _ = pool.set_block_location(
                    ctx.block_idx,
                    BlockLocation::CpuResident(ctx.cpu_offset),
                );
            }
            return Err(e);
        }

        // --- Phase 4: Finalize ---
        for ctx in pending {
            let block_idx = ctx.block_idx;
            let new_handle = ctx.new_handle;
            if let Err(e) = self.restore_finalize(pool, ctx) {
                tracing::error!(
                    block_idx,
                    ?new_handle,
                    error = %e,
                    "KCMM: restore_finalize failed in batched path"
                );
            }
        }

        Ok(())
    }

    /// Copy all K and V layers for a block from CPU to GPU.
    ///
    /// Reads from the CPU swap buffer at `cpu_offset` and writes to each
    /// layer's K+V VA region at `va_offset`.  Uses the dedicated `restore`
    /// CUDA stream for asynchronous H2D transfers.
    ///
    /// Data layout in CPU buffer (same as eviction):
    /// `[K layer 0][V layer 0][K layer 1][V layer 1]...[K layer N][V layer N]`
    fn restore_block_all_layers(
        &self,
        pool: &KcmmPool,
        cpu_offset: usize,
        va_offset: usize,
    ) -> Result<()> {
        let block_bytes = pool.block_bytes;
        let num_layers = pool.num_layers;
        let mut byte_offset = cpu_offset;
        let va_off = va_offset as u64;

        for l in 0..num_layers {
            // Restore K layer
            let gpu_va_k = pool.va_k(l) + va_off;
            unsafe {
                pool.streams.restore.memcpy_h2d_async(
                    gpu_va_k,
                    self.cpu_buffer.add(byte_offset) as *const u8,
                    block_bytes,
                )?;
            }
            byte_offset += block_bytes;

            // Restore V layer
            let gpu_va_v = pool.va_v(l) + va_off;
            unsafe {
                pool.streams.restore.memcpy_h2d_async(
                    gpu_va_v,
                    self.cpu_buffer.add(byte_offset) as *const u8,
                    block_bytes,
                )?;
            }
            byte_offset += block_bytes;
        }

        Ok(())
    }

    // --- Restoration ---

    /// Phase 1 of single-block restoration: allocate GPU physical block,
    /// update BlockInfo, submit async H2D copies.  Does **not** synchronise
    /// the stream — the caller must synchronise before `restore_finalize`.
    ///
    /// On failure the new physical allocation is released and the block
    /// reverts to `CpuResident`.
    fn restore_submit_async(
        &self,
        pool: &KcmmPool,
        block_idx: u32,
        cpu_offset: usize,
    ) -> Result<RestoreContext> {
        let block_bytes = pool.block_bytes;
        let total_bytes = pool.num_layers * 2 * block_bytes;

        // 1. Mark as Restoring — concurrent access will see this and back off
        pool.set_block_location(block_idx, BlockLocation::Restoring)?;

        // 2. Allocate new GPU physical block
        let (va_offset, sb_idx, blk_in_sb) = pool.alloc_one_block_internal()?;
        let new_handle = BlockHandle {
            superblock_idx: sb_idx,
            block_index: blk_in_sb,
        };

        // 3. Update BlockInfo with the new physical allocation
        pool.update_block_physical(block_idx, va_offset, sb_idx, blk_in_sb)?;

        // 4. H2D memcpy for all layers (async, no synchronise — caller batches it)
        if let Err(e) = self.restore_block_all_layers(pool, cpu_offset, va_offset) {
            // Rollback: release the new physical allocation and revert to CpuResident.
            if let Err(phys_err) = pool.release_block_physical(block_idx) {
                tracing::error!(
                    block_idx,
                    ?new_handle,
                    error = %phys_err,
                    "KCMM: CRITICAL — failed to release physical block during restore rollback"
                );
            }
            if let Err(loc_err) = pool.set_block_location(
                block_idx,
                BlockLocation::CpuResident(cpu_offset),
            ) {
                tracing::error!(
                    block_idx,
                    cpu_offset,
                    error = %loc_err,
                    "KCMM: CRITICAL — failed to revert location during restore rollback; block stuck as Restoring"
                );
            }
            return Err(e);
        }

        Ok(RestoreContext {
            block_idx,
            cpu_offset,
            total_bytes,
            new_handle,
            va_offset,
        })
    }

    /// Phase 3 of single-block restoration: mark `GpuResident`, free the CPU
    /// slot, and notify the policy.  Must only be called after
    /// `pool.streams.restore.synchronize()` confirmed that all H2D copies
    /// completed successfully.
    fn restore_finalize(&self, pool: &KcmmPool, ctx: RestoreContext) -> Result<()> {
        // Mark as GpuResident
        pool.set_block_location(
            ctx.block_idx,
            BlockLocation::GpuResident(ctx.new_handle, ctx.va_offset as u64),
        )?;

        // Free CPU slot — the data is now on GPU, the CPU copy is stale
        self.free_cpu_slot(ctx.cpu_offset, ctx.total_bytes);

        // Notify policy — the block was just "accessed" via restore
        self.eviction_policy.on_access(ctx.new_handle);

        tracing::debug!(
            ctx.block_idx,
            ?ctx.new_handle,
            ctx.cpu_offset,
            ctx.va_offset,
            ctx.total_bytes,
            "KCMM: restored block from CPU to GPU"
        );

        Ok(())
    }

    /// Restore a single block from CPU back to GPU.
    ///
    /// Called by `KcmmPool::restore_evicted_block` when a `CpuResident`
    /// block needs to be brought back into GPU HBM (e.g. when its owning
    /// sequence becomes active again).
    ///
    /// # Flow
    ///
    /// 1. Mark `Restoring` (blocks concurrent access).
    /// 2. Allocate a new GPU physical block via `pool.alloc_one_block_internal`.
    /// 3. Update `BlockInfo` with the new physical allocation.
    /// 4. H2D memcpy all layers from CPU buffer to GPU (async, on restore stream).
    /// 5. Synchronise the restore stream.
    /// 6. Mark `GpuResident` with the new handle and VA offset.
    /// 7. Free the CPU swap buffer slot.
    /// 8. Notify the eviction policy (`on_access`).
    ///
    /// On copy failure the new physical allocation is released and the
    /// block reverts to `CpuResident`.
    pub(crate) fn restore_block(
        &self,
        pool: &KcmmPool,
        block_idx: u32,
        cpu_offset: usize,
    ) -> Result<()> {
        let ctx = self.restore_submit_async(pool, block_idx, cpu_offset)?;
        pool.streams.restore.synchronize()?;
        self.restore_finalize(pool, ctx)
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
            max_batch_blocks: 64,
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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");
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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine with zero blocks");
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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");
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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");
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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");
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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");
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

        let result = TieringEngine::new(&config, 1, 16, None);
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
        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");

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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");

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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");

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

        let engine = TieringEngine::new(&config, 1, 16, None).expect("create engine");

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

    // --- TieringEngine eviction tests (GPU required) ---

    mod eviction_gpu {
        use super::*;
        use crate::cuda::CudaContext;
        use std::sync::Arc;

        fn make_pool_with_tiering() -> (Arc<CudaContext>, KcmmPool, tempfile::TempDir) {
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let dir = tempfile::tempdir().expect("create temp dir");
            let path = dir.path().join("kcmm_evict_test");
            let path_str = path.to_str().expect("valid UTF-8 path").to_string();

            let config = KcmmConfig {
                block_size: 16,
                max_blocks: 256,   // enough for CPU buffer = 256 * 16 * 2 = 8192 bytes
                cpu_cache_path: path_str,
                tiering: true,      // enable tiering
                eviction_policy: "lru".to_string(),
                prefetch_window: 4,
                max_batch_blocks: 64,
            };

            let pool = KcmmPool::new(
                ctx.clone(),
                config,
                2,  // num_layers — small to keep tests fast
                4,  // kv_heads
                64, // head_dim
                4,  // max_batch
                64, // max_seq_len
            )
            .expect("create KcmmPool with tiering");
            (ctx, pool, dir)
        }

        #[test]
        fn test_evict_single_block_location_transition() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate a block and get its handle
            let block_idx = pool.alloc_block().expect("alloc block");
            let handle = pool.get_block_handle(block_idx).expect("get handle");
            let initial_loc = pool.get_block_location(block_idx).expect("get location");
            assert!(
                matches!(initial_loc, BlockLocation::GpuResident(_, _)),
                "new block should be GpuResident"
            );

            // Evict the block
            let evicted = tiering
                .evict_blocks(&pool, &[handle], 1)
                .expect("evict blocks");
            assert_eq!(evicted.len(), 1);
            assert_eq!(evicted[0], handle);

            // Verify location is CpuResident
            let loc = pool.get_block_location(block_idx).expect("get location after evict");
            assert!(
                matches!(loc, BlockLocation::CpuResident(_)),
                "evicted block should be CpuResident, got {:?}",
                loc
            );
            if let BlockLocation::CpuResident(offset) = loc {
                // Offset should be within the CPU buffer
                let total = pool.num_layers * 2 * pool.block_bytes;
                assert!(offset < tiering.cpu_buffer_size());
                // First eviction should be at offset 0
                assert_eq!(offset, 0);
                let _ = total; // suppress unused warning
            }
        }

        #[test]
        fn test_evict_multiple_blocks_cpu_offsets_sequential() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate 3 blocks
            let table = pool.alloc_sequence(3).expect("alloc 3 blocks");
            let handles: Vec<BlockHandle> = table
                .iter()
                .map(|&idx| pool.get_block_handle(idx).expect("get handle"))
                .collect();
            assert_eq!(handles.len(), 3);

            // Record initial locations
            for &h in &handles {
                let idx = pool.find_block_idx(h).expect("find block");
                assert!(matches!(
                    pool.get_block_location(idx).expect("get location"),
                    BlockLocation::GpuResident(_, _)
                ));
            }

            // Evict all 3
            let evicted = tiering
                .evict_blocks(&pool, &handles, 3)
                .expect("evict 3 blocks");
            assert_eq!(evicted.len(), 3);

            // Verify each block is CpuResident with sequential offsets
            let total_per_block = pool.num_layers * 2 * pool.block_bytes;
            let mut offsets: Vec<usize> = Vec::new();
            for &h in &evicted {
                let idx = pool.find_block_idx(h).expect("find block");
                let loc = pool.get_block_location(idx).expect("get location");
                match loc {
                    BlockLocation::CpuResident(off) => {
                        assert!(off < tiering.cpu_buffer_size());
                        offsets.push(off);
                    }
                    other => panic!("expected CpuResident, got {:?}", other),
                }
            }
            // Offsets should be sequential: 0, total_per_block, 2*total_per_block
            offsets.sort();
            for (i, &off) in offsets.iter().enumerate() {
                assert_eq!(off, i * total_per_block,
                    "block {} should be at offset {}", i, i * total_per_block);
            }
        }

        #[test]
        fn test_evict_empty_candidates_returns_empty() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            let result = tiering.evict_blocks(&pool, &[], 5).expect("evict empty");
            assert!(result.is_empty());
        }

        #[test]
        fn test_evict_zero_count_returns_empty() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            let table = pool.alloc_sequence(2).expect("alloc");
            let handles: Vec<BlockHandle> = table
                .iter()
                .map(|&idx| pool.get_block_handle(idx).expect("get handle"))
                .collect();

            let result = tiering.evict_blocks(&pool, &handles, 0).expect("evict 0");
            assert!(result.is_empty());

            // Blocks should still be GpuResident
            for &h in &handles {
                let idx = pool.find_block_idx(h).expect("find block");
                assert!(matches!(
                    pool.get_block_location(idx).expect("get location"),
                    BlockLocation::GpuResident(_, _)
                ));
            }
        }

        #[test]
        fn test_evict_count_exceeds_candidates() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            let table = pool.alloc_sequence(2).expect("alloc");
            let handles: Vec<BlockHandle> = table
                .iter()
                .map(|&idx| pool.get_block_handle(idx).expect("get handle"))
                .collect();

            // Ask for 10 evictions, only 2 available
            let evicted = tiering
                .evict_blocks(&pool, &handles, 10)
                .expect("evict");
            assert_eq!(evicted.len(), 2);
        }

        #[test]
        fn test_evict_then_new_allocation_reuses_physical() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate and evict 2 blocks
            let table = pool.alloc_sequence(2).expect("alloc 2");
            let handles: Vec<BlockHandle> = table
                .iter()
                .map(|&idx| pool.get_block_handle(idx).expect("get handle"))
                .collect();

            let physical_total_before = pool.total_physical_blocks();
            let _ = tiering
                .evict_blocks(&pool, &handles, 2)
                .expect("evict");

            // Evicted blocks should NOT reduce total_physical_blocks (we only
            // returned them to the free list — they're still in the superblock).
            let physical_total_after = pool.total_physical_blocks();
            assert_eq!(physical_total_before, physical_total_after,
                "physical total should be unchanged after eviction");
        }

        #[test]
        fn test_alloc_cpu_slot_and_free_roundtrip() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            let size = 1024;
            let off1 = tiering.alloc_cpu_slot(size).expect("alloc slot 1");
            assert_eq!(off1, 0);

            let off2 = tiering.alloc_cpu_slot(size).expect("alloc slot 2");
            assert_eq!(off2, 1024);

            // Free off1
            tiering.free_cpu_slot(off1, size);

            // Re-allocate — should get off1 back (best-fit: only range that fits)
            let off3 = tiering.alloc_cpu_slot(size).expect("alloc slot 3");
            assert_eq!(off3, 0, "reclaimed slot should be at offset 0");

            // Free all
            tiering.free_cpu_slot(off2, size);
            tiering.free_cpu_slot(off3, size);
        }

        #[test]
        fn test_evict_preserves_lru_policy_state() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate 3 blocks
            let table = pool.alloc_sequence(3).expect("alloc");
            let h0 = pool.get_block_handle(table[0]).expect("get handle");
            let h1 = pool.get_block_handle(table[1]).expect("get handle");
            let h2 = pool.get_block_handle(table[2]).expect("get handle");

            // Register with LRU policy (on_allocate)
            tiering.eviction_policy.on_allocate(h0);
            tiering.eviction_policy.on_allocate(h1);
            tiering.eviction_policy.on_allocate(h2);

            // Evict h0 (oldest) — should be removed from policy tracking
            let evicted = tiering
                .evict_blocks(&pool, &[h0, h1, h2], 1)
                .expect("evict");
            assert_eq!(evicted.len(), 1);
            // LRU should select h0 (oldest on_allocate)
            assert_eq!(evicted[0], h0);

            // h0 should no longer be tracked by policy (on_evict called)
            let remaining = tiering.eviction_policy.select_victims(&[h0, h1, h2], 2);
            // h0 should be skipped (no tracking) — only h1, h2 selected
            assert_eq!(remaining.len(), 2);
            assert!(!remaining.contains(&h0));
        }

        #[test]
        fn test_evict_single_block_data_integrity() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            let block_idx = pool.alloc_block().expect("alloc block");
            let handle = pool.get_block_handle(block_idx).expect("get handle");
            let block_bytes = pool.block_bytes;

            // Write a known pattern to the GPU block's K and V layers
            // We use layer 0, K cache for simplicity
            let num_elements = block_bytes / 2; // f16 = 2 bytes each
            let pattern: Vec<u16> = (0..num_elements).map(|i| (i % 256) as u16).collect();
            let gpu_va_k0 = pool.gpu_va_for_block(handle, 0, false).expect("va k0");

            // Copy pattern to GPU (H2D on evict stream)
            unsafe {
                pool.streams.evict
                    .memcpy_h2d_async(
                        gpu_va_k0,
                        pattern.as_ptr() as *const u8,
                        block_bytes,
                    )
                    .expect("h2d memcpy");
            }
            pool.streams.evict.synchronize().expect("sync");

            // Evict the block
            let evicted = tiering
                .evict_blocks(&pool, &[handle], 1)
                .expect("evict");
            assert_eq!(evicted.len(), 1);

            // Read back the CPU buffer at the evicted offset
            let loc = pool.get_block_location(block_idx).expect("get location");
            let cpu_offset = match loc {
                BlockLocation::CpuResident(off) => off,
                _ => panic!("expected CpuResident"),
            };

            // The K layer data is at cpu_offset
            let cpu_base = tiering.cpu_buffer_ptr();
            let readback: Vec<u16> = unsafe {
                let src = cpu_base.add(cpu_offset) as *const u16;
                std::slice::from_raw_parts(src, num_elements).to_vec()
            };

            assert_eq!(readback, pattern,
                "CPU buffer should contain the exact pattern written to GPU K layer");
        }
    }

    // --- TieringEngine restore tests (GPU required) ---

    mod restore_gpu {
        use super::*;
        use crate::cuda::CudaContext;
        use std::sync::Arc;

        fn make_pool_with_tiering() -> (Arc<CudaContext>, KcmmPool, tempfile::TempDir) {
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let dir = tempfile::tempdir().expect("create temp dir");
            let path = dir.path().join("kcmm_restore_test");
            let path_str = path.to_str().expect("valid UTF-8 path").to_string();

            let config = KcmmConfig {
                block_size: 16,
                max_blocks: 256,
                cpu_cache_path: path_str,
                tiering: true,
                eviction_policy: "lru".to_string(),
                prefetch_window: 4,
                max_batch_blocks: 64,
            };

            let pool = KcmmPool::new(
                ctx.clone(),
                config,
                2,  // num_layers — small for fast tests
                4,  // kv_heads
                64, // head_dim
                4,  // max_batch
                64, // max_seq_len
            )
            .expect("create KcmmPool with tiering");
            (ctx, pool, dir)
        }

        #[test]
        fn test_restore_single_block_location_transition() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate a block and evict it
            let block_idx = pool.alloc_block().expect("alloc block");
            let handle = pool.get_block_handle(block_idx).expect("get handle");

            let evicted = tiering
                .evict_blocks(&pool, &[handle], 1)
                .expect("evict");
            assert_eq!(evicted.len(), 1);

            // Verify CpuResident
            let loc = pool.get_block_location(block_idx).expect("get location");
            assert!(
                matches!(loc, BlockLocation::CpuResident(_)),
                "should be CpuResident after eviction"
            );

            // Restore the block
            let va_offset = pool
                .restore_evicted_block(block_idx)
                .expect("restore evicted block");
            assert!(va_offset > 0, "restored VA offset should be positive");

            // Verify GpuResident
            let loc = pool.get_block_location(block_idx).expect("get location after restore");
            assert!(
                matches!(loc, BlockLocation::GpuResident(_, _)),
                "should be GpuResident after restore, got {:?}",
                loc
            );
        }

        #[test]
        fn test_restore_already_gpu_resident_is_noop() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();

            // Allocate a block — it starts as GpuResident
            let block_idx = pool.alloc_block().expect("alloc block");
            let loc_before = pool.get_block_location(block_idx).expect("get location");
            let va_before = match loc_before {
                BlockLocation::GpuResident(_, va) => va,
                _ => panic!("expected GpuResident"),
            };

            // Restore on an already-GpuResident block should return the same VA offset
            let va_after = pool
                .restore_evicted_block(block_idx)
                .expect("restore on GpuResident should be noop");
            assert_eq!(va_before, va_after,
                "restore on GpuResident block should return same VA offset");

            // Location should still be GpuResident
            let loc = pool.get_block_location(block_idx).expect("get location");
            assert!(matches!(loc, BlockLocation::GpuResident(_, _)));
        }

        #[test]
        fn test_evict_then_restore_multiple_blocks() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate 3 blocks
            let table = pool.alloc_sequence(3).expect("alloc 3");
            let handles: Vec<BlockHandle> = table
                .iter()
                .map(|&idx| pool.get_block_handle(idx).expect("get handle"))
                .collect();

            // Evict all 3
            tiering
                .evict_blocks(&pool, &handles, 3)
                .expect("evict all");

            for &h in &handles {
                let idx = pool.find_block_idx(h).expect("find block");
                assert!(matches!(
                    pool.get_block_location(idx).expect("get location"),
                    BlockLocation::CpuResident(_)
                ));
            }

            // Restore each block
            for &idx in &table {
                let va = pool
                    .restore_evicted_block(idx)
                    .expect("restore block");
                assert!(va > 0);
                assert!(matches!(
                    pool.get_block_location(idx).expect("get location"),
                    BlockLocation::GpuResident(_, _)
                ));
            }
        }

        #[test]
        fn test_restore_cpu_slot_freed() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");
            let total_per_block = pool.num_layers * 2 * pool.block_bytes;

            // Allocate and evict a block
            let block_idx = pool.alloc_block().expect("alloc block");
            let handle = pool.get_block_handle(block_idx).expect("get handle");

            tiering
                .evict_blocks(&pool, &[handle], 1)
                .expect("evict");

            let cpu_offset = match pool.get_block_location(block_idx).expect("get location") {
                BlockLocation::CpuResident(off) => off,
                _ => panic!("expected CpuResident"),
            };
            assert_eq!(cpu_offset, 0, "first eviction should be at offset 0");

            // Restore — should free the CPU slot
            pool.restore_evicted_block(block_idx).expect("restore");

            // Allocate a new CPU slot — should get the same offset back
            let new_offset = tiering.alloc_cpu_slot(total_per_block).expect("alloc cpu slot");
            assert_eq!(new_offset, 0,
                "CPU slot should be freed after restore and reallocatable at offset 0");
        }

        #[test]
        fn test_restore_then_evict_again() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate
            let block_idx = pool.alloc_block().expect("alloc block");
            let orig_handle = pool.get_block_handle(block_idx).expect("get handle");

            // Evict
            tiering
                .evict_blocks(&pool, &[orig_handle], 1)
                .expect("evict 1");
            assert!(matches!(
                pool.get_block_location(block_idx).expect("get location"),
                BlockLocation::CpuResident(_)
            ));

            // Restore
            pool.restore_evicted_block(block_idx).expect("restore");
            assert!(matches!(
                pool.get_block_location(block_idx).expect("get location"),
                BlockLocation::GpuResident(_, _)
            ));

            // Get the NEW handle (physical allocation changes on restore)
            let new_handle = pool.get_block_handle(block_idx).expect("get new handle");

            // Evict again — should work with the new handle
            let evicted2 = tiering
                .evict_blocks(&pool, &[new_handle], 1)
                .expect("evict 2");
            assert_eq!(evicted2.len(), 1);

            assert!(matches!(
                pool.get_block_location(block_idx).expect("get location after 2nd evict"),
                BlockLocation::CpuResident(_)
            ));
        }

        #[test]
        fn test_restore_preserves_policy_state() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            // Allocate and evict
            let block_idx = pool.alloc_block().expect("alloc block");
            let handle = pool.get_block_handle(block_idx).expect("get handle");

            tiering
                .evict_blocks(&pool, &[handle], 1)
                .expect("evict");

            // After eviction, the old handle should NOT be tracked by LRU
            let victims = tiering.eviction_policy.select_victims(&[handle], 1);
            assert!(victims.is_empty(),
                "evicted block's old handle should not be tracked by policy");

            // Restore — policy should be notified (on_access with new handle)
            pool.restore_evicted_block(block_idx).expect("restore");

            // The new handle should be tracked by policy (on_access)
            let new_handle = pool.get_block_handle(block_idx).expect("get new handle");
            let victims = tiering.eviction_policy.select_victims(&[new_handle], 1);
            assert_eq!(victims.len(), 1,
                "restored block's new handle should be tracked by policy");
            assert_eq!(victims[0], new_handle);
        }

        #[test]
        fn test_restore_data_integrity_roundtrip() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");

            let block_idx = pool.alloc_block().expect("alloc block");
            let handle = pool.get_block_handle(block_idx).expect("get handle");
            let block_bytes = pool.block_bytes;
            let num_elements = block_bytes / 2; // f16 = 2 bytes each

            // Write a known pattern to the GPU block's layer-0 K cache
            let pattern: Vec<u16> = (0..num_elements).map(|i| (i % 256) as u16).collect();
            let gpu_va_k0 = pool.gpu_va_for_block(handle, 0, false).expect("va k0");
            unsafe {
                pool.streams.evict
                    .memcpy_h2d_async(
                        gpu_va_k0,
                        pattern.as_ptr() as *const u8,
                        block_bytes,
                    )
                    .expect("h2d memcpy");
            }
            pool.streams.evict.synchronize().expect("sync");

            // Evict the block (GPU → CPU)
            tiering
                .evict_blocks(&pool, &[handle], 1)
                .expect("evict");

            // Restore the block (CPU → GPU)
            pool.restore_evicted_block(block_idx).expect("restore");

            // Read back from the NEW GPU location and verify data integrity
            let new_handle = pool.get_block_handle(block_idx).expect("get new handle");
            let new_gpu_va = pool.gpu_va_for_block(new_handle, 0, false).expect("va k0 new");

            let mut readback: Vec<u16> = vec![0u16; num_elements];
            unsafe {
                pool.streams.restore.memcpy_d2h_async(
                    readback.as_mut_ptr() as *mut u8,
                    new_gpu_va,
                    block_bytes,
                )
                .expect("d2h memcpy");
            }
            pool.streams.restore.synchronize().expect("sync");

            assert_eq!(readback, pattern,
                "restored GPU data should match the original pattern (full roundtrip)");
        }

        #[test]
        fn test_restore_invalid_block_idx_errors() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();

            let result = pool.restore_evicted_block(999);
            assert!(result.is_err(), "restore on invalid block index should error");
        }

        #[test]
        fn test_restore_evicting_block_errors() {
            // This tests that the Evicting guard state is respected.
            // We can't easily create an Evicting block without a race,
            // but we can test that the error path exists for the variant.
            let (_ctx, pool, _dir) = make_pool_with_tiering();

            let block_idx = pool.alloc_block().expect("alloc block");
            // Manually set to Evicting to simulate in-flight transfer
            pool.set_block_location(block_idx, BlockLocation::Evicting)
                .expect("set Evicting");

            let result = pool.restore_evicted_block(block_idx);
            assert!(result.is_err(), "restore on Evicting block should error");
        }

        #[test]
        fn test_restore_restoring_block_errors() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();

            let block_idx = pool.alloc_block().expect("alloc block");
            // Manually set to Restoring to simulate concurrent restore
            pool.set_block_location(block_idx, BlockLocation::Restoring)
                .expect("set Restoring");

            let result = pool.restore_evicted_block(block_idx);
            assert!(result.is_err(), "restore on Restoring block should error");
        }
    }

    mod gather_scatter_kernels {
        use super::*;
        use crate::cuda::CudaContext;
        use crate::cuda::kernels::{
            launch_kv_gather, launch_kv_scatter,
        };
        use cudarc::driver::CudaSlice;
        use half::f16;
        use std::sync::Arc;

        /// Create a CUDA context and compile the gather/scatter kernels.
        fn setup() -> (Arc<CudaContext>, CudaFunction, CudaFunction) {
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
            let (gather, scatter) =
                TieringEngine::compile_kv_gather_kernels(&ctx.device)
                    .expect("compile kernels");
            (ctx, gather, scatter)
        }

        /// Fill a GPU buffer with a known pattern and return the pattern.
        fn fill_pattern(ctx: &CudaContext, slice: &mut CudaSlice<f16>, n: usize, seed: u16) -> Vec<f16> {
            let pattern: Vec<f16> = (0..n)
                .map(|i| f16::from_f32((i.wrapping_mul(seed as usize).wrapping_add(13) % 1000) as f32))
                .collect();
            ctx.h2d_sync(&pattern, slice).expect("h2d sync");
            pattern
        }

        /// Read back a GPU buffer.
        fn readback(ctx: &CudaContext, slice: &CudaSlice<f16>, n: usize) -> Vec<f16> {
            let mut host = vec![f16::ZERO; n];
            ctx.d2h_sync(slice, &mut host).expect("d2h sync");
            host
        }

        #[test]
        fn test_gather_single_block() {
            let (ctx, gather, _scatter) = setup();
            let half_count = 64;
            let num_blocks = 1;

            // Create 1 source buffer with a known pattern
            let mut src = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc src");
            let pattern = fill_pattern(&ctx, &mut src, half_count, 7);

            // Build src_ptrs device array
            let src_ptr = CudaContext::device_ptr(&src);
            let src_ptrs_host = vec![src_ptr];
            let mut src_ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");
            ctx.h2d_sync(&src_ptrs_host, &mut src_ptrs_dev).expect("h2d ptrs");

            // Allocate staging buffer
            let mut staging = ctx.device.alloc_zeros::<f16>(half_count * num_blocks)
                .expect("alloc staging");

            // Launch gather
            unsafe {
                launch_kv_gather(&gather, &src_ptrs_dev, &mut staging, half_count, num_blocks)
                    .expect("gather kernel");
            }
            ctx.synchronize().expect("sync");

            // Read back staging and verify
            let result = readback(&ctx, &staging, half_count * num_blocks);
            assert_eq!(&result[..half_count], &pattern[..],
                "gather single block: data mismatch");
        }

        #[test]
        fn test_gather_multiple_blocks() {
            let (ctx, gather, _scatter) = setup();
            let half_count = 32;
            let num_blocks = 4;

            // Create N source buffers with distinct patterns
            let mut srcs: Vec<CudaSlice<f16>> = Vec::new();
            let mut patterns: Vec<Vec<f16>> = Vec::new();
            let mut ptrs_host: Vec<u64> = Vec::new();

            for i in 0..num_blocks {
                let mut src = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc src");
                let pat = fill_pattern(&ctx, &mut src, half_count, (i as u16 + 1) * 11);
                ptrs_host.push(CudaContext::device_ptr(&src));
                patterns.push(pat);
                srcs.push(src);
            }

            let mut ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");
            ctx.h2d_sync(&ptrs_host, &mut ptrs_dev).expect("h2d ptrs");

            let mut staging = ctx.device.alloc_zeros::<f16>(half_count * num_blocks)
                .expect("alloc staging");

            unsafe {
                launch_kv_gather(&gather, &ptrs_dev, &mut staging, half_count, num_blocks)
                    .expect("gather kernel");
            }
            ctx.synchronize().expect("sync");

            let result = readback(&ctx, &staging, half_count * num_blocks);
            for i in 0..num_blocks {
                let start = i * half_count;
                let end = start + half_count;
                assert_eq!(&result[start..end], &patterns[i][..],
                    "gather block {}: data mismatch", i);
            }
        }

        #[test]
        fn test_scatter_single_block() {
            let (ctx, _gather, scatter) = setup();
            let half_count = 64;
            let num_blocks = 1;

            // Create contiguous source with a known pattern
            let mut src = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc src");
            let pattern = fill_pattern(&ctx, &mut src, half_count, 13);

            // Create destination buffer (zeroed)
            let mut dst = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc dst");
            let dst_ptr = CudaContext::device_ptr(&dst);
            let dst_ptrs_host = vec![dst_ptr];
            let mut dst_ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");
            ctx.h2d_sync(&dst_ptrs_host, &mut dst_ptrs_dev).expect("h2d ptrs");

            // Launch scatter
            unsafe {
                launch_kv_scatter(&scatter, &src, &dst_ptrs_dev, half_count, num_blocks)
                    .expect("scatter kernel");
            }
            ctx.synchronize().expect("sync");

            // Read back destination and verify
            let result = readback(&ctx, &dst, half_count);
            assert_eq!(&result[..], &pattern[..],
                "scatter single block: data mismatch");
        }

        #[test]
        fn test_scatter_multiple_blocks() {
            let (ctx, _gather, scatter) = setup();
            let half_count = 32;
            let num_blocks = 4;

            // Create contiguous source: interleaved patterns for blocks 0,1,2,3
            let total = half_count * num_blocks;
            let mut src = ctx.device.alloc_zeros::<f16>(total).expect("alloc src");
            let mut expected: Vec<Vec<f16>> = Vec::new();
            let mut src_host: Vec<f16> = Vec::with_capacity(total);
            for i in 0..num_blocks {
                let pat: Vec<f16> = (0..half_count)
                    .map(|j| f16::from_f32(((i * 100 + j) % 500) as f32))
                    .collect();
                src_host.extend_from_slice(&pat);
                expected.push(pat);
            }
            ctx.h2d_sync(&src_host, &mut src).expect("h2d src");

            // Create destination buffers (zeroed)
            let mut dsts: Vec<CudaSlice<f16>> = Vec::new();
            let mut dst_ptrs_host: Vec<u64> = Vec::new();
            for _ in 0..num_blocks {
                let dst = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc dst");
                dst_ptrs_host.push(CudaContext::device_ptr(&dst));
                dsts.push(dst);
            }
            let mut dst_ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");
            ctx.h2d_sync(&dst_ptrs_host, &mut dst_ptrs_dev).expect("h2d ptrs");

            unsafe {
                launch_kv_scatter(&scatter, &src, &dst_ptrs_dev, half_count, num_blocks)
                    .expect("scatter kernel");
            }
            ctx.synchronize().expect("sync");

            for i in 0..num_blocks {
                let result = readback(&ctx, &dsts[i], half_count);
                assert_eq!(&result[..], &expected[i][..],
                    "scatter block {}: data mismatch", i);
            }
        }

        #[test]
        fn test_gather_scatter_roundtrip() {
            let (ctx, gather, scatter) = setup();
            let half_count = 64;
            let num_blocks = 3;

            // Create source buffers with distinct patterns (the "original" data)
            let mut srcs: Vec<CudaSlice<f16>> = Vec::new();
            let mut original: Vec<Vec<f16>> = Vec::new();
            let mut src_ptrs_host: Vec<u64> = Vec::new();
            for i in 0..num_blocks {
                let mut src = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc src");
                let pat = fill_pattern(&ctx, &mut src, half_count, (i as u16 + 3) * 17);
                src_ptrs_host.push(CudaContext::device_ptr(&src));
                original.push(pat);
                srcs.push(src);
            }

            let mut ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");

            // ---- Gather: scattered → contiguous staging ----
            ctx.h2d_sync(&src_ptrs_host, &mut ptrs_dev).expect("h2d gather ptrs");
            let mut staging = ctx.device.alloc_zeros::<f16>(half_count * num_blocks)
                .expect("alloc staging");
            unsafe {
                launch_kv_gather(&gather, &ptrs_dev, &mut staging, half_count, num_blocks)
                    .expect("gather kernel");
            }
            ctx.synchronize().expect("sync after gather");

            // ---- Scatter: contiguous staging → new scattered destinations ----
            // Create fresh destination buffers (zeroed)
            let mut dsts: Vec<CudaSlice<f16>> = Vec::new();
            let mut dst_ptrs_host: Vec<u64> = Vec::new();
            for _ in 0..num_blocks {
                let dst = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc dst");
                dst_ptrs_host.push(CudaContext::device_ptr(&dst));
                dsts.push(dst);
            }
            let mut dst_ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");
            ctx.h2d_sync(&dst_ptrs_host, &mut dst_ptrs_dev).expect("h2d scatter ptrs");

            unsafe {
                launch_kv_scatter(&scatter, &staging, &dst_ptrs_dev, half_count, num_blocks)
                    .expect("scatter kernel");
            }
            ctx.synchronize().expect("sync after scatter");

            // Verify roundtrip identity
            for i in 0..num_blocks {
                let result = readback(&ctx, &dsts[i], half_count);
                assert_eq!(&result[..], &original[i][..],
                    "gather-scatter roundtrip block {}: data mismatch", i);
            }
        }

        #[test]
        fn test_gather_edge_max_batch() {
            let (ctx, gather, _scatter) = setup();
            let half_count = 16; // small blocks for memory efficiency
            let num_blocks = 64; // KcmmConfig default max batch

            let mut patterns: Vec<Vec<f16>> = Vec::new();
            let mut ptrs_host: Vec<u64> = Vec::new();
            let mut _srcs: Vec<CudaSlice<f16>> = Vec::new();

            for i in 0..num_blocks {
                let mut src = ctx.device.alloc_zeros::<f16>(half_count).expect("alloc src");
                let pat = fill_pattern(&ctx, &mut src, half_count, (i as u16).wrapping_mul(3));
                ptrs_host.push(CudaContext::device_ptr(&src));
                patterns.push(pat);
                _srcs.push(src);
            }

            let mut ptrs_dev = ctx.device.alloc_zeros::<u64>(num_blocks).expect("alloc ptrs");
            ctx.h2d_sync(&ptrs_host, &mut ptrs_dev).expect("h2d ptrs");

            let mut staging = ctx.device.alloc_zeros::<f16>(half_count * num_blocks)
                .expect("alloc staging");

            unsafe {
                launch_kv_gather(&gather, &ptrs_dev, &mut staging, half_count, num_blocks)
                    .expect("gather kernel max batch");
            }
            ctx.synchronize().expect("sync");

            let result = readback(&ctx, &staging, half_count * num_blocks);
            for i in 0..num_blocks {
                let start = i * half_count;
                assert_eq!(&result[start..start + half_count], &patterns[i][..],
                    "gather max batch block {}: mismatch", i);
            }
        }
    }
}
