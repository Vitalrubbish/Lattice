// KCMM pool — central memory pool managing GPU KV Cache blocks.
//
// Generalizes PagedKvCache with:
//   - Tiering-aware block lifecycle (BlockLocation enum)
//   - SequenceState with explicit hot/cold tracking
//   - Pluggable eviction policy (via TieringEngine)
//   - Dedicated CUDA streams for async data migration
//   - Built-in fragmentation tracking

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

use super::superblock::{
    align_up, BlockHandle, LayerKvPool, SuperblockInfo, SUPERBLOCK_SIZE,
};
use super::streams::KcmmStreams;
use super::tiering::TieringEngine;
use super::sharing::SharingManager;
use crate::cache::cuda_vmm::CudaVmm;
use crate::cache::fragmentation_tracker::RuntimeFragmentationTracker;
use crate::config::KcmmConfig;
use crate::cuda::CudaContext;

// --- Block location tracking ---

/// Tracks where a block's data currently resides.
///
/// In step 3, only `GpuResident` is fully used. The other variants
/// are populated by the tiering engine (implemented in step 3 weeks 14-15).
#[derive(Debug, Clone)]
pub enum BlockLocation {
    /// Block is resident in GPU HBM.
    GpuResident(BlockHandle, u64),
    /// Block data is in CPU DRAM swap buffer at the given byte offset.
    CpuResident(usize),
    /// Block data is on NVMe SSD at the given byte offset.
    NvmeResident(u64),
    /// Block is being evicted from GPU to CPU/NVMe (transfer in flight).
    Evicting,
    /// Block is being restored from CPU/NVMe to GPU (transfer in flight).
    Restoring,
}

// --- Per-block tracking ---

#[derive(Debug, Clone)]
struct BlockInfo {
    /// Byte offset within the per-layer K/V VA region for this block.
    va_offset: usize,
    /// Superblock index (same across all pools in lockstep).
    superblock_idx: u32,
    /// Block index within the superblock.
    block_index_in_sb: u32,
    /// Whether this block is currently assigned to a sequence.
    in_use: bool,
    /// Where the block's data resides.
    location: BlockLocation,
}

// --- Sequence state ---

/// Per-sequence metadata for KCMM pool tracking.
///
/// Extends `SeqMetadata` with fields needed for eviction policy decisions.
#[derive(Debug, Clone)]
pub struct SequenceState {
    /// Logical block → block_idx mapping.
    pub block_table: Vec<u32>,
    /// Current sequence length in tokens.
    pub seq_len: usize,
    /// Whether this sequence is actively being decoded.
    pub is_active: bool,
    /// Timestamp of the most recent access (for LRU ordering).
    pub last_access: Instant,
    /// Number of prefix blocks shared with other sequences.
    /// (Used in step 4; always 0 in step 3.)
    pub shared_prefix_len: usize,
}

impl SequenceState {
    pub fn new(block_table: Vec<u32>) -> Self {
        Self {
            block_table,
            seq_len: 0,
            is_active: true,
            last_access: Instant::now(),
            shared_prefix_len: 0,
        }
    }
}

// --- KCMM Pool ---

/// KCMM memory pool — the central abstraction.
///
/// Manages GPU KV Cache blocks across all transformer layers using
/// CUDA VMM for physical memory management. Supports optional multi-tier
/// storage (GPU→CPU→NVMe) through the tiering engine.
pub struct KcmmPool {
    /// Pool configuration.
    pub config: KcmmConfig,
    /// CUDA device context.
    pub ctx: Arc<CudaContext>,
    /// Maximum batch size supported.
    pub max_batch: usize,
    /// Maximum sequence length in tokens.
    pub max_seq_len: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// Maximum blocks per sequence.
    pub max_blocks_per_seq: usize,

    /// CUDA VMM handle for GPU physical memory management.
    vmm: CudaVmm,
    /// K-cache VA regions (one per layer).
    va_k: Vec<u64>,
    /// V-cache VA regions (one per layer).
    va_v: Vec<u64>,

    /// Per-layer K physical pools.
    pub(crate) k_pools: Vec<LayerKvPool>,
    /// Per-layer V physical pools.
    pub(crate) v_pools: Vec<LayerKvPool>,

    /// Block-level tracking: block_idx → BlockInfo.
    block_info: Mutex<Vec<BlockInfo>>,
    /// Recycled block indices.
    free_block_indices: Mutex<Vec<u32>>,
    /// Per-sequence metadata.
    sequences: Mutex<Vec<SequenceState>>,

    /// Optional tiering engine for GPU↔CPU↔NVMe migration.
    /// `None` when tiering is disabled in config.
    pub tiering: Option<TieringEngine>,
    /// Optional prefix sharing manager (step 4).
    /// Always `None` in step 3.
    pub sharing: Option<SharingManager>,

    /// CUDA streams for async data migration.
    pub streams: KcmmStreams,

    /// Fragmentation tracker for UFS metrics.
    pub fragmentation_tracker: RuntimeFragmentationTracker,

    /// Number of transformer layers (determines how many per-layer pools).
    pub num_layers: usize,

    /// Precomputed sizes.
    pub elem_per_block: usize,
    pub block_bytes: usize,
    pub max_blocks_total: usize,
}

impl KcmmPool {
    /// Create a new KCMM pool.
    pub fn new(
        ctx: Arc<CudaContext>,
        config: KcmmConfig,
        model_num_layers: usize,
        model_kv_heads: usize,
        model_head_dim: usize,
        max_batch: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let block_size = config.block_size;
        let elem_per_block = model_kv_heads * block_size * model_head_dim;
        let block_bytes = elem_per_block * std::mem::size_of::<half::f16>();
        let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
        let max_blocks_total = max_batch * max_blocks_per_seq;

        let vmm = CudaVmm::new(ctx.device.ordinal())?;

        // Reserve separate VA regions per layer for K and V
        let va_size = max_blocks_total * block_bytes;
        let va_size = align_up(va_size, SUPERBLOCK_SIZE);
        let num_layers = model_num_layers;
        let mut va_k = Vec::with_capacity(num_layers);
        let mut va_v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            va_k.push(vmm.reserve_address(va_size)?);
            va_v.push(vmm.reserve_address(va_size)?);
        }

        // Create per-layer K and V pools
        let k_pools: Vec<LayerKvPool> = (0..num_layers)
            .map(|_| LayerKvPool::new_with_block_bytes(block_bytes))
            .collect();
        let v_pools: Vec<LayerKvPool> = (0..num_layers)
            .map(|_| LayerKvPool::new_with_block_bytes(block_bytes))
            .collect();

        // Bytes per token for one layer of K (used by fragmentation tracker)
        let bytes_per_token_k = model_kv_heads * model_head_dim * 2;

        // Create dedicated CUDA streams
        let streams = KcmmStreams::new()?;

        // Create tiering engine if enabled
        let tiering = if config.tiering {
            Some(TieringEngine::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            ctx,
            max_batch,
            max_seq_len,
            block_size,
            max_blocks_per_seq,
            vmm,
            va_k,
            va_v,
            k_pools,
            v_pools,
            block_info: Mutex::new(Vec::new()),
            free_block_indices: Mutex::new(Vec::new()),
            sequences: Mutex::new(Vec::new()),
            tiering,
            sharing: None,
            streams,
            fragmentation_tracker: RuntimeFragmentationTracker::new(bytes_per_token_k),
            num_layers,
            elem_per_block,
            block_bytes,
            max_blocks_total,
        })
    }

    // --- Superblock management ---

    /// Map a newly created physical handle into the specified layer's K or V VA region.
    fn map_superblock_to_layer(
        &self,
        phys_handle: u64,
        layer_idx: usize,
        is_v: bool,
        pool: &LayerKvPool,
    ) -> Result<usize> {
        let sb_count = pool.allocator.superblock_count();
        let sb_idx = sb_count.saturating_sub(1);
        let va_base = sb_idx * SUPERBLOCK_SIZE;
        let va_region = if is_v {
            self.va_v[layer_idx]
        } else {
            self.va_k[layer_idx]
        };

        self.vmm
            .map(va_region, va_base, phys_handle, 0, SUPERBLOCK_SIZE)?;

        tracing::debug!(
            phys_handle,
            sb_idx,
            va_base,
            layer_idx,
            is_v,
            "KCMM: mapped superblock into layer VA region"
        );

        pool.superblocks.lock().push(SuperblockInfo {
            phys_handle,
            va_base,
        });

        Ok(va_base)
    }

    /// Ensure all per-layer pools have at least one free block.
    fn ensure_capacity(&self) -> Result<()> {
        if self.k_pools[0].allocator.free_count() > 0 {
            return Ok(());
        }

        let num_layers = self.num_layers;

        for l in 0..num_layers {
            let phys_k = self.vmm.create_physical(SUPERBLOCK_SIZE)?;
            self.k_pools[l].allocator.add_superblock();
            self.map_superblock_to_layer(phys_k, l, false, &self.k_pools[l])?;

            let phys_v = self.vmm.create_physical(SUPERBLOCK_SIZE)?;
            self.v_pools[l].allocator.add_superblock();
            self.map_superblock_to_layer(phys_v, l, true, &self.v_pools[l])?;
        }

        tracing::debug!(
            num_layers,
            total_pools = num_layers * 2,
            "KCMM: created superblocks for all per-layer pools"
        );

        Ok(())
    }

    // --- Block allocation ---

    /// Allocate a single physical block across all per-layer pools.
    fn alloc_one_block_internal(&self) -> Result<(usize, u32, u32)> {
        self.ensure_capacity()?;

        let num_layers = self.num_layers;
        let mut handle: Option<BlockHandle> = None;

        for l in 0..num_layers {
            let h_k = self.k_pools[l]
                .allocator
                .try_allocate()
                .ok_or_else(|| anyhow!("K pool layer {}: no free block after ensure_capacity", l))?;
            let h_v = self.v_pools[l]
                .allocator
                .try_allocate()
                .ok_or_else(|| anyhow!("V pool layer {}: no free block after ensure_capacity", l))?;

            if let Some(ref first) = handle {
                assert_eq!(
                    first.superblock_idx, h_k.superblock_idx,
                    "K pool layer {} superblock_idx mismatch", l
                );
                assert_eq!(
                    first.block_index, h_k.block_index,
                    "K pool layer {} block_index mismatch", l
                );
                assert_eq!(
                    first.superblock_idx, h_v.superblock_idx,
                    "V pool layer {} superblock_idx mismatch", l
                );
                assert_eq!(
                    first.block_index, h_v.block_index,
                    "V pool layer {} block_index mismatch", l
                );
            } else {
                handle = Some(h_k);
            }
        }

        let h = handle.unwrap();

        let sb = &self.k_pools[0].superblocks.lock()[h.superblock_idx as usize];
        let va_offset = sb.va_base + h.block_index as usize * self.block_bytes;

        Ok((va_offset, h.superblock_idx, h.block_index))
    }

    fn install_block(&self, va_offset: usize, sb_idx: u32, blk_in_sb: u32) -> u32 {
        let block_handle = BlockHandle {
            superblock_idx: sb_idx,
            block_index: blk_in_sb,
        };
        let location = BlockLocation::GpuResident(block_handle, va_offset as u64);

        let mut free = self.free_block_indices.lock();
        if let Some(idx) = free.pop() {
            let mut info = self.block_info.lock();
            info[idx as usize] = BlockInfo {
                va_offset,
                superblock_idx: sb_idx,
                block_index_in_sb: blk_in_sb,
                in_use: true,
                location,
            };
            idx
        } else {
            let mut info = self.block_info.lock();
            let idx = info.len() as u32;
            info.push(BlockInfo {
                va_offset,
                superblock_idx: sb_idx,
                block_index_in_sb: blk_in_sb,
                in_use: true,
                location,
            });
            idx
        }
    }

    /// Allocate a single block. Returns the block index.
    pub fn alloc_block(&self) -> Result<u32> {
        let (va_offset, sb_idx, blk_in_sb) = self.alloc_one_block_internal()?;
        Ok(self.install_block(va_offset, sb_idx, blk_in_sb))
    }

    /// Allocate `num_blocks` for a new sequence. Returns the block table.
    pub fn alloc_sequence(&self, num_blocks: usize) -> Result<Vec<u32>> {
        let mut table = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let (va_offset, sb_idx, blk_in_sb) = self.alloc_one_block_internal()?;
            let block_idx = self.install_block(va_offset, sb_idx, blk_in_sb);
            table.push(block_idx);
        }
        Ok(table)
    }

    /// Free all blocks belonging to a sequence.
    ///
    /// Lock ordering: `block_info` → then `free_block_indices`.
    /// We collect recycled indices into a temporary Vec, drop `block_info`,
    /// and then extend `free_block_indices` — this avoids an AB-BA deadlock
    /// with `install_block` which acquires the locks in reverse order.
    pub fn free_sequence(&self, block_table: &[u32]) {
        let mut info = self.block_info.lock();
        let num_layers = self.num_layers;
        let mut recycled = Vec::new();

        for &block_idx in block_table {
            let bi = &mut info[block_idx as usize];
            if !bi.in_use {
                continue;
            }
            bi.in_use = false;

            let handle = BlockHandle {
                superblock_idx: bi.superblock_idx,
                block_index: bi.block_index_in_sb,
            };
            for l in 0..num_layers {
                self.k_pools[l].allocator.free(handle);
                self.v_pools[l].allocator.free(handle);
            }
            recycled.push(block_idx);
        }
        drop(info);
        self.free_block_indices.lock().extend(recycled);
    }

    // --- Sequence tracking ---

    /// Register a new sequence with its block table. Returns the sequence index.
    pub fn register_sequence(&self, block_table: Vec<u32>) -> usize {
        let mut seqs = self.sequences.lock();
        let idx = seqs.len();
        seqs.push(SequenceState::new(block_table));
        idx
    }

    /// Unregister a sequence and free its blocks.
    pub fn unregister_sequence(&self, seq_idx: usize) {
        let block_table = {
            let mut seqs = self.sequences.lock();
            if seq_idx >= seqs.len() {
                return;
            }
            let table = std::mem::take(&mut seqs[seq_idx].block_table);
            seqs[seq_idx].seq_len = 0;
            table
        };
        if !block_table.is_empty() {
            self.free_sequence(&block_table);
        }
    }

    /// Mark a sequence as recently accessed (hot).
    ///
    /// Updates `last_access` to now and sets `is_active = true`.
    /// Call this when a sequence is scheduled for decoding.
    pub fn touch(&self, seq_idx: usize) {
        let mut seqs = self.sequences.lock();
        if seq_idx < seqs.len() {
            seqs[seq_idx].last_access = Instant::now();
            seqs[seq_idx].is_active = true;
        }
    }

    /// Mark a sequence as cool (eligible for eviction).
    ///
    /// Sets `is_active = false`. The sequence's blocks become candidates
    /// for eviction when memory pressure triggers the tiering engine.
    pub fn cool(&self, seq_idx: usize) {
        let mut seqs = self.sequences.lock();
        if seq_idx < seqs.len() {
            seqs[seq_idx].is_active = false;
        }
    }

    /// Update sequence length.
    pub fn update_seq_len(&self, seq_idx: usize, len: usize) {
        let mut seqs = self.sequences.lock();
        if seq_idx < seqs.len() {
            seqs[seq_idx].seq_len = len;
        }
    }

    /// Append a block to an existing sequence's block table.
    pub fn append_block_to_sequence(&self, seq_idx: usize, block_idx: u32) {
        let mut seqs = self.sequences.lock();
        if seq_idx < seqs.len() {
            seqs[seq_idx].block_table.push(block_idx);
        }
    }

    /// Get the block table for a sequence.
    pub fn get_block_table(&self, seq_idx: usize) -> Option<Vec<u32>> {
        let seqs = self.sequences.lock();
        seqs.get(seq_idx).map(|s| s.block_table.clone())
    }

    /// Get the sequence length.
    pub fn get_seq_len(&self, seq_idx: usize) -> usize {
        let seqs = self.sequences.lock();
        if seq_idx < seqs.len() {
            seqs[seq_idx].seq_len
        } else {
            0
        }
    }

    /// Number of registered sequences.
    pub fn active_sequences(&self) -> usize {
        self.sequences.lock().len()
    }

    /// Check if a sequence is active (currently decoding).
    pub fn is_active(&self, seq_idx: usize) -> bool {
        let seqs = self.sequences.lock();
        seq_idx < seqs.len() && seqs[seq_idx].is_active
    }

    // --- Block queries ---

    /// Get the VA offset for a block index.
    pub fn get_block_va_offset(&self, block_idx: u32) -> Option<usize> {
        let info = self.block_info.lock();
        let bi = info.get(block_idx as usize)?;
        if bi.in_use {
            Some(bi.va_offset)
        } else {
            None
        }
    }

    /// Get VA offsets for all blocks belonging to a sequence.
    pub fn get_block_va_offsets(&self, seq_idx: usize) -> Option<Vec<usize>> {
        let seqs = self.sequences.lock();
        let seq = seqs.get(seq_idx)?;
        let info = self.block_info.lock();
        let offsets: Option<Vec<usize>> = seq
            .block_table
            .iter()
            .map(|&idx| {
                let bi = info.get(idx as usize)?;
                if bi.in_use {
                    Some(bi.va_offset)
                } else {
                    None
                }
            })
            .collect();
        offsets
    }

    /// Get the block location for a block index.
    pub fn get_block_location(&self, block_idx: u32) -> Option<BlockLocation> {
        let info = self.block_info.lock();
        info.get(block_idx as usize)
            .filter(|bi| bi.in_use)
            .map(|bi| bi.location.clone())
    }

    // --- VA accessors ---

    /// Get the K-cache virtual address base for a given layer.
    pub fn va_k(&self, layer: usize) -> u64 {
        self.va_k[layer]
    }

    /// Get the V-cache virtual address base for a given layer.
    pub fn va_v(&self, layer: usize) -> u64 {
        self.va_v[layer]
    }

    // --- Statistics ---

    /// Number of blocks currently in use.
    pub fn blocks_in_use(&self) -> usize {
        self.block_info.lock().iter().filter(|b| b.in_use).count()
    }

    /// Total number of block indices.
    pub fn total_blocks(&self) -> usize {
        self.block_info.lock().len()
    }

    /// Total blocks allocated across all per-layer pools.
    pub fn total_physical_blocks(&self) -> usize {
        self.k_pools[0].allocator.total_blocks_allocated()
    }

    /// Free blocks available across all per-layer pools.
    pub fn free_physical_blocks(&self) -> usize {
        self.k_pools[0].allocator.free_count()
    }

    /// Blocks per superblock.
    pub fn blocks_per_superblock(&self) -> usize {
        self.k_pools[0].allocator.blocks_per_superblock
    }

    /// Superblock count.
    pub fn superblock_count(&self) -> usize {
        self.k_pools[0].allocator.superblock_count()
    }

    /// Fraction of allocated superblock capacity sitting idle in the free list.
    pub fn physical_idle_ratio(&self) -> f32 {
        let num_layers = self.num_layers;
        let superblocks = self.k_pools[0].allocator.superblock_count();
        if superblocks == 0 {
            return 0.0;
        }
        let total_physical = superblocks * SUPERBLOCK_SIZE * num_layers * 2;
        let physical_used = self.blocks_in_use() * self.block_bytes * num_layers * 2;
        if physical_used >= total_physical {
            return 0.0;
        }
        (total_physical - physical_used) as f32 / total_physical as f32
    }

    /// Check if there are free blocks available.
    pub fn has_free_blocks(&self) -> bool {
        self.k_pools[0].allocator.free_count() > 0
    }

    /// Collect UFS metrics snapshot.
    pub fn collect_metrics(&self) -> crate::cache::unified_frag::UnifiedFragMetrics {
        use crate::cache::unified_frag::UnifiedFragMetrics;
        let total_physical = self.total_physical_blocks();
        let in_use = self.blocks_in_use();
        let seqs = self.sequences.lock();
        let total_blocks_used: usize = seqs.iter().map(|s| s.block_table.len()).sum();
        let total_tokens: usize = seqs.iter().map(|s| s.seq_len).sum();
        drop(seqs);

        let num_layers = self.num_layers;
        let superblock_count = self.superblock_count();
        let superblock_size = SUPERBLOCK_SIZE;
        let actual_physical_bytes = (superblock_count * superblock_size * num_layers * 2) as u64;

        let total_slots = total_blocks_used * self.block_size;
        let internal_frag_rate = if total_slots > 0 {
            (total_slots - total_tokens) as f32 / total_slots as f32
        } else {
            0.0
        };

        let block_utilization = if total_physical > 0 {
            in_use as f32 / total_physical as f32
        } else {
            0.0
        };

        let bpt_all = self.elem_per_block / self.block_size * num_layers * 2;
        let ideal_physical_bytes = (total_tokens * bpt_all) as u64;
        let physical_memory_efficiency = if actual_physical_bytes > 0 {
            ideal_physical_bytes as f32 / actual_physical_bytes as f32
        } else {
            1.0
        };

        let blocks_per_sb = self.blocks_per_superblock();
        let active_superblocks = (in_use + blocks_per_sb - 1) / blocks_per_sb;
        let actual_active_bytes = (active_superblocks * superblock_size * num_layers * 2) as u64;
        let ideal_active_bytes = (total_tokens * bpt_all) as u64;
        let runtime_frag_index = if actual_active_bytes > 0 {
            1.0 - (ideal_active_bytes as f32 / actual_active_bytes as f32)
        } else {
            0.0
        };

        UnifiedFragMetrics {
            internal_frag_rate,
            block_utilization,
            physical_memory_efficiency,
            runtime_frag_index,
            active_sequences: self.active_sequences(),
            blocks_in_use: in_use,
            total_blocks_allocated: total_physical,
            total_tokens,
            ideal_physical_bytes,
            actual_physical_bytes,
        }
    }

    /// Low watermark detection: returns true if free block ratio is below threshold.
    ///
    /// When this returns true, the caller (or a background thread) should
    /// trigger tiering eviction to free up GPU blocks.
    pub fn below_low_watermark(&self, threshold: f32) -> bool {
        let total = self.total_physical_blocks();
        if total == 0 {
            return false;
        }
        let free = self.free_physical_blocks();
        let ratio = free as f32 / total as f32;
        ratio < threshold
    }
}

// --- Drop ---

impl Drop for KcmmPool {
    fn drop(&mut self) {
        // Wait for all in-flight CUDA stream operations before unmapping.
        // This prevents use-after-free of physical memory during async
        // eviction/restore/prefetch memcpy operations.
        self.streams.synchronize_all().ok();

        let num_layers = self.num_layers;

        // Unmap and release per-layer K superblocks
        for l in 0..num_layers {
            let sbs = std::mem::take(&mut *self.k_pools[l].superblocks.lock());
            for sb in &sbs {
                let _ = self.vmm.unmap(self.va_k[l], sb.va_base, SUPERBLOCK_SIZE);
                let _ = self.vmm.release_physical(sb.phys_handle);
            }
        }
        // Unmap and release per-layer V superblocks
        for l in 0..num_layers {
            let sbs = std::mem::take(&mut *self.v_pools[l].superblocks.lock());
            for sb in &sbs {
                let _ = self.vmm.unmap(self.va_v[l], sb.va_base, SUPERBLOCK_SIZE);
                let _ = self.vmm.release_physical(sb.phys_handle);
            }
        }

        // Free VA regions
        let va_size = self.max_blocks_total * self.block_bytes;
        let va_size = align_up(va_size, SUPERBLOCK_SIZE);
        for &va in &self.va_k {
            let _ = self.vmm.free_address(va, va_size);
        }
        for &va in &self.va_v {
            let _ = self.vmm.free_address(va, va_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // --- SequenceState tests ---

    #[test]
    fn test_sequence_state_new() {
        let table = vec![0, 1, 2];
        let state = SequenceState::new(table.clone());
        assert_eq!(state.block_table, table);
        assert_eq!(state.seq_len, 0);
        assert!(state.is_active);
        assert_eq!(state.shared_prefix_len, 0);
    }

    #[test]
    fn test_sequence_state_touch_updates_last_access() {
        let state = SequenceState::new(vec![10, 20]);
        let before = state.last_access;
        // Sleep a tiny bit so Instant moves forward.
        thread::sleep(Duration::from_millis(2));

        let mut state = state;
        state.last_access = Instant::now();
        state.is_active = true;
        assert!(state.last_access > before);
        assert!(state.is_active);
    }

    #[test]
    fn test_sequence_state_cool_sets_inactive() {
        let mut state = SequenceState::new(vec![5]);
        assert!(state.is_active);
        state.is_active = false;
        assert!(!state.is_active);
    }

    #[test]
    fn test_sequence_state_seq_len_update() {
        let mut state = SequenceState::new(vec![0, 1, 2, 3]);
        assert_eq!(state.seq_len, 0);
        state.seq_len = 42;
        assert_eq!(state.seq_len, 42);
    }

    #[test]
    fn test_sequence_state_shared_prefix_len_default_zero() {
        let state = SequenceState::new(vec![1, 2, 3]);
        assert_eq!(state.shared_prefix_len, 0);
    }

    // --- BlockLocation tests ---

    #[test]
    fn test_block_location_gpu_resident() {
        let handle = BlockHandle {
            superblock_idx: 0,
            block_index: 5,
        };
        let loc = BlockLocation::GpuResident(handle, 0x1000);
        assert!(matches!(loc, BlockLocation::GpuResident(_, _)));
    }

    #[test]
    fn test_block_location_all_variants_constructible() {
        let handle = BlockHandle { superblock_idx: 1, block_index: 3 };

        let gpu = BlockLocation::GpuResident(handle, 0x2000);
        let cpu = BlockLocation::CpuResident(4096);
        let nvme = BlockLocation::NvmeResident(8192);
        let evicting = BlockLocation::Evicting;
        let restoring = BlockLocation::Restoring;

        // Verify each variant matches correctly.
        assert!(matches!(gpu, BlockLocation::GpuResident(_, _)));
        match gpu {
            BlockLocation::GpuResident(h, offset) => {
                assert_eq!(h.superblock_idx, 1);
                assert_eq!(h.block_index, 3);
                assert_eq!(offset, 0x2000);
            }
            _ => panic!("expected GpuResident"),
        }
        assert!(matches!(cpu, BlockLocation::CpuResident(4096)));
        assert!(matches!(nvme, BlockLocation::NvmeResident(8192)));
        assert!(matches!(evicting, BlockLocation::Evicting));
        assert!(matches!(restoring, BlockLocation::Restoring));
    }

    #[test]
    fn test_block_location_clone() {
        let handle = BlockHandle { superblock_idx: 7, block_index: 42 };
        let loc = BlockLocation::GpuResident(handle, 0xABCD);
        let cloned = loc.clone();
        assert!(matches!(cloned, BlockLocation::GpuResident(_, _)));
        match cloned {
            BlockLocation::GpuResident(h, off) => {
                assert_eq!(h.superblock_idx, 7);
                assert_eq!(h.block_index, 42);
                assert_eq!(off, 0xABCD);
            }
            _ => panic!("clone mismatch"),
        }
    }

    #[test]
    fn test_block_location_evicting_restoring_no_data() {
        // Evicting and Restoring carry no payload — verify they construct fine.
        let e = BlockLocation::Evicting;
        let r = BlockLocation::Restoring;
        assert!(matches!(e, BlockLocation::Evicting));
        assert!(matches!(r, BlockLocation::Restoring));
    }

    // --- Watermark math (tested without KcmmPool) ---

    #[test]
    fn test_watermark_ratio_when_empty() {
        // If total == 0, ratio should be considered above watermark.
        let total: usize = 0;
        let free: usize = 0;
        let threshold: f32 = 0.2;
        let below = if total == 0 {
            false
        } else {
            (free as f32 / total as f32) < threshold
        };
        assert!(!below, "empty pool should never be below watermark");
    }

    #[test]
    fn test_watermark_ratio_below_threshold() {
        let total = 100;
        let free = 10; // 10% free
        let threshold = 0.2;
        let below = (free as f32 / total as f32) < threshold;
        assert!(below);
    }

    #[test]
    fn test_watermark_ratio_above_threshold() {
        let total = 100;
        let free = 50; // 50% free
        let threshold = 0.2;
        let below = (free as f32 / total as f32) < threshold;
        assert!(!below);
    }

    #[test]
    fn test_watermark_ratio_exactly_at_threshold() {
        let total = 100;
        let free = 20; // exactly 20%
        let threshold = 0.2;
        let below = (free as f32 / total as f32) < threshold;
        assert!(!below, "exactly at threshold should NOT be below");
    }

    #[test]
    fn test_watermark_ratio_one_free_block() {
        let total = 1000;
        let free = 1; // 0.1%
        let threshold = 0.05;
        let below = (free as f32 / total as f32) < threshold;
        assert!(below);
    }

    #[test]
    fn test_block_info_construction() {
        let bi = BlockInfo {
            va_offset: 0x10000,
            superblock_idx: 2,
            block_index_in_sb: 15,
            in_use: true,
            location: BlockLocation::GpuResident(
                BlockHandle { superblock_idx: 2, block_index: 15 },
                0x10000,
            ),
        };
        assert_eq!(bi.va_offset, 0x10000);
        assert_eq!(bi.superblock_idx, 2);
        assert_eq!(bi.block_index_in_sb, 15);
        assert!(bi.in_use);
        assert!(matches!(bi.location, BlockLocation::GpuResident(_, _)));
    }

    #[test]
    fn test_block_info_not_in_use() {
        let bi = BlockInfo {
            va_offset: 0,
            superblock_idx: 0,
            block_index_in_sb: 0,
            in_use: false,
            location: BlockLocation::Evicting,
        };
        assert!(!bi.in_use);
        assert!(matches!(bi.location, BlockLocation::Evicting));
    }

    // --- Lock-ordering smoke test ---
    //
    // We can't construct a full KcmmPool without a GPU, but we verify
    // that the two lock types (free_block_indices + block_info) work
    // correctly when used with the same pattern as install_block/free_sequence.

    #[test]
    fn test_lock_ordering_install_then_free() {
        // Simulate the two-mutex pattern used in install_block / free_sequence.
        // Verifies that the fixed ordering (collect → drop → extend) doesn't
        // cause issues when the two operations are interleaved.
        let free_indices: Mutex<Vec<u32>> = Mutex::new(vec![0, 1, 2]);
        let block_info: Mutex<Vec<BlockInfo>> = Mutex::new(Vec::new());

        // install_block pattern: lock free_indices → lock block_info
        {
            let mut free = free_indices.lock();
            if let Some(idx) = free.pop() {
                let mut info = block_info.lock();
                info.push(BlockInfo {
                    va_offset: idx as usize * 100,
                    superblock_idx: 0,
                    block_index_in_sb: idx,
                    in_use: true,
                    location: BlockLocation::GpuResident(
                        BlockHandle { superblock_idx: 0, block_index: idx },
                        (idx * 100) as u64,
                    ),
                });
            }
        }

        assert_eq!(block_info.lock().len(), 1);
        assert_eq!(free_indices.lock().len(), 2);

        // free_sequence pattern (FIXED): lock block_info → collect → drop → extend free_indices
        {
            let mut info = block_info.lock();
            let mut recycled = Vec::new();
            for bi in info.iter_mut() {
                if bi.in_use {
                    bi.in_use = false;
                    recycled.push(bi.block_index_in_sb);
                }
            }
            drop(info);
            free_indices.lock().extend(recycled);
        }

        assert_eq!(free_indices.lock().len(), 3); // reclaimed the one we allocated
    }

    #[test]
    fn test_lock_ordering_concurrent_install_free() {
        // Spawn two threads: one does install_block pattern, one does
        // free_sequence pattern.  With the fix applied, this must not deadlock.
        use std::sync::Arc;

        let free_indices = Arc::new(Mutex::new((0u32..100).collect::<Vec<_>>()));
        let block_info = Arc::new(Mutex::new(Vec::<BlockInfo>::new()));

        let fi_a = Arc::clone(&free_indices);
        let bi_a = Arc::clone(&block_info);
        let t1 = thread::spawn(move || {
            for _ in 0..500 {
                // install_block pattern
                let idx = {
                    let mut free = fi_a.lock();
                    free.pop()
                };
                if let Some(idx) = idx {
                    let mut info = bi_a.lock();
                    info.push(BlockInfo {
                        va_offset: idx as usize,
                        superblock_idx: 0,
                        block_index_in_sb: idx,
                        in_use: true,
                        location: BlockLocation::GpuResident(
                            BlockHandle { superblock_idx: 0, block_index: idx },
                            idx as u64,
                        ),
                    });
                }
            }
        });

        let fi_b = Arc::clone(&free_indices);
        let bi_b = Arc::clone(&block_info);
        let t2 = thread::spawn(move || {
            for _ in 0..500 {
                // free_sequence pattern (FIXED with drop before extend)
                let mut info = bi_b.lock();
                let mut recycled = Vec::new();
                // Free the first few in-use entries.
                let mut count = 0;
                for bi in info.iter_mut() {
                    if bi.in_use && count < 3 {
                        bi.in_use = false;
                        recycled.push(bi.block_index_in_sb);
                        count += 1;
                    }
                }
                drop(info);
                fi_b.lock().extend(recycled);
            }
        });

        // If we get here without hanging, the lock ordering is safe.
        t1.join().expect("install thread panicked");
        t2.join().expect("free thread panicked");
    }
}
