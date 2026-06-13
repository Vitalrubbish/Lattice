// KCMM pool — central memory pool managing GPU KV Cache blocks.
//
// Generalizes PagedKvCache with:
//   - Tiering-aware block lifecycle (BlockLocation enum)
//   - SequenceState with explicit hot/cold tracking
//   - Pluggable eviction policy (via TieringEngine)
//   - Dedicated CUDA streams for async data migration
//   - Built-in fragmentation tracking

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaSlice, DevicePtr};
use cudarc::driver::sys::{self, CUdeviceptr};
use half::f16;
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
pub(crate) struct BlockInfo {
    /// Byte offset within the per-layer K/V VA region for this block.
    pub(crate) va_offset: usize,
    /// Superblock index (same across all pools in lockstep).
    pub(crate) superblock_idx: u32,
    /// Block index within the superblock.
    pub(crate) block_index_in_sb: u32,
    /// Whether this block is currently assigned to a sequence.
    pub(crate) in_use: bool,
    /// Where the block's data resides.
    pub(crate) location: BlockLocation,
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
    pub(crate) va_k: Vec<u64>,
    /// V-cache VA regions (one per layer).
    pub(crate) va_v: Vec<u64>,

    /// Per-layer K physical pools.
    pub(crate) k_pools: Vec<LayerKvPool>,
    /// Per-layer V physical pools.
    pub(crate) v_pools: Vec<LayerKvPool>,

    /// Block-level tracking: block_idx → BlockInfo.
    pub(crate) block_info: Mutex<Vec<BlockInfo>>,
    /// Recycled block indices.
    pub(crate) free_block_indices: Mutex<Vec<u32>>,
    /// Per-sequence metadata.
    pub(crate) sequences: Mutex<Vec<SequenceState>>,

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
            Some(TieringEngine::new(
                &config,
                num_layers,
                block_bytes,
                Some(ctx.device.clone()),
            )?)
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
    pub(crate) fn alloc_one_block_internal(&self) -> Result<(usize, u32, u32)> {
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

        let idx = {
            let mut free = self.free_block_indices.lock();
            if let Some(recycled) = free.pop() {
                let mut info = self.block_info.lock();
                info[recycled as usize] = BlockInfo {
                    va_offset,
                    superblock_idx: sb_idx,
                    block_index_in_sb: blk_in_sb,
                    in_use: true,
                    location,
                };
                recycled
            } else {
                let mut info = self.block_info.lock();
                let new_idx = info.len() as u32;
                info.push(BlockInfo {
                    va_offset,
                    superblock_idx: sb_idx,
                    block_index_in_sb: blk_in_sb,
                    in_use: true,
                    location,
                });
                new_idx
            }
        };

        // Register with the eviction policy so the tiering engine
        // knows about this block for victim selection.
        if let Some(ref tiering) = self.tiering {
            tiering.eviction_policy.lock().on_allocate(block_handle);
        }

        idx
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
        let mut gpu_handles = Vec::new();
        let mut cpu_offsets = Vec::new();

        for &block_idx in block_table {
            let bi = &mut info[block_idx as usize];
            if !bi.in_use {
                continue;
            }

            match &bi.location {
                BlockLocation::GpuResident(handle, _) => {
                    bi.in_use = false;
                    gpu_handles.push(*handle);
                    recycled.push(block_idx);
                }
                BlockLocation::CpuResident(cpu_offset) => {
                    bi.in_use = false;
                    cpu_offsets.push(*cpu_offset);
                    recycled.push(block_idx);
                }
                BlockLocation::NvmeResident(_) => {
                    bi.in_use = false;
                    tracing::warn!(
                        block_idx,
                        "KCMM: freeing NvmeResident block without NVMe cleanup support"
                    );
                    recycled.push(block_idx);
                }
                BlockLocation::Evicting | BlockLocation::Restoring => {
                    tracing::warn!(
                        block_idx,
                        ?bi.location,
                        "KCMM: freeing block while migration is in flight; physical cleanup skipped"
                    );
                }
            }
        }
        drop(info);

        for handle in gpu_handles {
            for l in 0..num_layers {
                self.k_pools[l].allocator.free(handle);
                self.v_pools[l].allocator.free(handle);
            }
        }

        if !cpu_offsets.is_empty() {
            if let Some(ref tiering) = self.tiering {
                let cpu_slot_bytes = self.num_layers * 2 * self.block_bytes;
                for cpu_offset in cpu_offsets {
                    tiering.free_cpu_slot(cpu_offset, cpu_slot_bytes);
                }
            } else {
                tracing::warn!(
                    count = cpu_offsets.len(),
                    "KCMM: freeing CpuResident blocks with tiering disabled; CPU slots not released"
                );
            }
        }

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

    /// Get VA offsets for all blocks in f16-element units.
    ///
    /// Returns a flat Vec where index = block_idx and value = va_offset
    /// divided by `sizeof(f16)`.  Inactive blocks yield 0.
    /// This matches the `PagedKvCache` API consumed by the paged-attention
    /// CUDA kernel.
    pub fn get_all_block_offsets_f16(&self) -> Vec<u64> {
        let info = self.block_info.lock();
        info.iter().map(|bi| {
            if bi.in_use {
                (bi.va_offset / std::mem::size_of::<f16>()) as u64
            } else {
                0u64
            }
        }).collect()
    }

    /// Get the block location for a block index.
    pub fn get_block_location(&self, block_idx: u32) -> Option<BlockLocation> {
        let info = self.block_info.lock();
        info.get(block_idx as usize)
            .filter(|bi| bi.in_use)
            .map(|bi| bi.location.clone())
    }

    /// Get the `BlockHandle` for a logical block index.
    pub fn get_block_handle(&self, block_idx: u32) -> Option<BlockHandle> {
        let info = self.block_info.lock();
        info.get(block_idx as usize)
            .filter(|bi| bi.in_use)
            .map(|bi| BlockHandle {
                superblock_idx: bi.superblock_idx,
                block_index: bi.block_index_in_sb,
            })
    }

    /// Find the logical block index for a given `BlockHandle`.
    pub fn find_block_idx(&self, handle: BlockHandle) -> Option<u32> {
        let info = self.block_info.lock();
        info.iter()
            .position(|bi| {
                bi.in_use
                    && bi.superblock_idx == handle.superblock_idx
                    && bi.block_index_in_sb == handle.block_index
            })
            .map(|i| i as u32)
    }

    /// Update the `BlockLocation` for a block.
    ///
    /// Returns an error if the block index is out of bounds or not in use.
    pub fn set_block_location(&self, block_idx: u32, location: BlockLocation) -> Result<()> {
        let mut info = self.block_info.lock();
        let bi = info
            .get_mut(block_idx as usize)
            .ok_or_else(|| anyhow!("block_idx {} out of bounds", block_idx))?;
        if !bi.in_use {
            return Err(anyhow!("block_idx {} is not in use", block_idx));
        }
        bi.location = location;
        Ok(())
    }

    /// Update the physical allocation fields for a block index.
    ///
    /// Used during `restore_block` when a new physical block is allocated
    /// to replace the previously-evicted one.  The `BlockHandle` changes
    /// because the new physical slot may be in a different superblock.
    ///
    /// Returns an error if the block index is out of bounds or not in use.
    pub(crate) fn update_block_physical(
        &self,
        block_idx: u32,
        va_offset: usize,
        sb_idx: u32,
        blk_in_sb: u32,
    ) -> Result<()> {
        let mut info = self.block_info.lock();
        let bi = info
            .get_mut(block_idx as usize)
            .ok_or_else(|| anyhow!("block_idx {} out of bounds", block_idx))?;
        if !bi.in_use {
            return Err(anyhow!("block_idx {} is not in use", block_idx));
        }
        bi.va_offset = va_offset;
        bi.superblock_idx = sb_idx;
        bi.block_index_in_sb = blk_in_sb;
        Ok(())
    }

    /// Restore an evicted block from CPU memory back to GPU.
    ///
    /// If the block is already `GpuResident`, this is a no-op and the
    /// current VA offset is returned.  If the block is `CpuResident`,
    /// the tiering engine allocates a new GPU physical block, copies
    /// all K+V layer data from the CPU swap buffer to GPU, and marks
    /// the block as `GpuResident`.
    ///
    /// Returns the GPU VA offset of the restored (or already-resident) block.
    pub fn restore_evicted_block(&self, block_idx: u32) -> Result<u64> {
        let tiering = self
            .tiering
            .as_ref()
            .ok_or_else(|| anyhow!("tiering is disabled; cannot restore evicted block"))?;

        // Extract the CPU offset (if CpuResident) without holding the lock
        // across the call to tiering.restore_block — parking_lot::Mutex is
        // not re-entrant and restore_block will lock block_info itself.
        let cpu_offset = {
            let info = self.block_info.lock();
            let bi = info
                .get(block_idx as usize)
                .ok_or_else(|| anyhow!("block_idx {} out of bounds", block_idx))?;
            if !bi.in_use {
                return Err(anyhow!("block_idx {} is not in use", block_idx));
            }
            match &bi.location {
                BlockLocation::CpuResident(offset) => Some(*offset),
                BlockLocation::GpuResident(_, va_offset) => return Ok(*va_offset),
                BlockLocation::Evicting | BlockLocation::Restoring => {
                    return Err(anyhow!(
                        "block {} is in transit ({:?}); cannot restore",
                        block_idx,
                        bi.location
                    ));
                }
                BlockLocation::NvmeResident(_) => {
                    return Err(anyhow!(
                        "block {} is NVMe-resident; NVMe restore not yet implemented",
                        block_idx
                    ));
                }
            }
        }; // lock dropped here

        let cpu_offset =
            cpu_offset.ok_or_else(|| anyhow!("block {} is not CpuResident", block_idx))?;
        tiering.restore_block(self, block_idx, cpu_offset)?;

        // Read back the new VA offset
        let info = self.block_info.lock();
        let bi = info.get(block_idx as usize).unwrap();
        match &bi.location {
            BlockLocation::GpuResident(_, va_offset) => Ok(*va_offset),
            other => Err(anyhow!(
                "restore did not result in GpuResident (got {:?})",
                other
            )),
        }
    }

    /// Restore multiple evicted blocks from CPU memory back to GPU.
    ///
    /// This is the batch counterpart to `restore_evicted_block`.  It extracts
    /// the CPU offset for each block, validates that all are `CpuResident`,
    /// and delegates to the tiering engine's batched restore path.  When
    /// batching infrastructure is available and the batch size is ≥4 blocks,
    /// the scatter-kernel path is used; otherwise each block is restored
    /// individually via the single-block path.
    ///
    /// Blocks that are already `GpuResident` are silently skipped (no-op).
    pub fn restore_evicted_blocks(&self, block_indices: &[u32]) -> Result<()> {
        let tiering = self
            .tiering
            .as_ref()
            .ok_or_else(|| anyhow!("tiering is disabled; cannot restore evicted blocks"))?;

        // Collect (block_idx, cpu_offset) pairs, skipping already-resident blocks.
        let mut blocks: Vec<(u32, usize)> = Vec::with_capacity(block_indices.len());
        {
            let info = self.block_info.lock();
            for &block_idx in block_indices {
                let bi = info
                    .get(block_idx as usize)
                    .ok_or_else(|| anyhow!("block_idx {} out of bounds", block_idx))?;
                if !bi.in_use {
                    return Err(anyhow!("block_idx {} is not in use", block_idx));
                }
                match &bi.location {
                    BlockLocation::CpuResident(offset) => {
                        blocks.push((block_idx, *offset));
                    }
                    BlockLocation::GpuResident(..) => {
                        // Already resident — skip.
                    }
                    BlockLocation::Evicting | BlockLocation::Restoring => {
                        return Err(anyhow!(
                            "block {} is in transit ({:?}); cannot restore",
                            block_idx,
                            bi.location
                        ));
                    }
                    BlockLocation::NvmeResident(_) => {
                        return Err(anyhow!(
                            "block {} is NVMe-resident; NVMe restore not yet implemented",
                            block_idx
                        ));
                    }
                }
            }
        } // lock dropped

        if blocks.is_empty() {
            return Ok(());
        }

        tiering.restore_blocks(self, &blocks)
    }

    /// Compute the byte offset of a block within each layer's VA region.
    ///
    /// All K/V pools across all layers are in lockstep — the same
    /// `BlockHandle` maps to the same VA offset in every layer's VA region.
    pub fn block_va_offset(&self, handle: BlockHandle) -> Result<usize> {
        let sbs = self.k_pools[0].superblocks.lock();
        let sb = sbs
            .get(handle.superblock_idx as usize)
            .ok_or_else(|| {
                anyhow!(
                    "superblock_idx {} out of bounds",
                    handle.superblock_idx
                )
            })?;
        Ok(sb.va_base + handle.block_index as usize * self.block_bytes)
    }

    /// Get the raw GPU virtual address for a block at a given layer and K/V split.
    ///
    /// `is_v` selects the V cache (`true`) or K cache (`false`).
    pub fn gpu_va_for_block(
        &self,
        handle: BlockHandle,
        layer: usize,
        is_v: bool,
    ) -> Result<CUdeviceptr> {
        let va_offset = self.block_va_offset(handle)? as u64;
        let va_base = if is_v {
            self.va_v
                .get(layer)
                .ok_or_else(|| anyhow!("V layer {} out of bounds", layer))?
        } else {
            self.va_k
                .get(layer)
                .ok_or_else(|| anyhow!("K layer {} out of bounds", layer))?
        };
        Ok(*va_base + va_offset)
    }

    /// Release a block's physical GPU resources back to all per-layer allocators.
    ///
    /// After this call the block's physical slot may be re-used by a future
    /// `alloc_one_block_internal`.  The logical block index remains in-use
    /// (its `BlockLocation` should be updated to `CpuResident` or `NvmeResident`
    /// before or after this call).
    pub fn release_block_physical(&self, block_idx: u32) -> Result<()> {
        let info = self.block_info.lock();
        let bi = info
            .get(block_idx as usize)
            .ok_or_else(|| anyhow!("block_idx {} out of bounds", block_idx))?;
        if !bi.in_use {
            return Err(anyhow!("block_idx {} is not in use", block_idx));
        }
        let handle = BlockHandle {
            superblock_idx: bi.superblock_idx,
            block_index: bi.block_index_in_sb,
        };

        let num_layers = self.num_layers;
        for l in 0..num_layers {
            self.k_pools[l].allocator.free(handle);
            self.v_pools[l].allocator.free(handle);
        }
        Ok(())
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

    // --- KV cache write (paged attention) ---

    /// Write one step of KV data for a batch of sequences.
    /// Convenience wrapper — copies the same source to both K and V caches.
    pub fn append_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],
        positions: &[usize],
        hidden: &CudaSlice<f16>,
    ) -> Result<()> {
        self.append_kv_step(layer_idx, seq_indices, positions, hidden, hidden)
    }

    /// Write one step of KV data for a batch of sequences, using separate
    /// K and V sources (post-projection, post-RoPE).  Each source is expected
    /// to be laid out as [batch, kv_heads * head_dim] in F16.
    ///
    /// This is a port of `PagedKvCache::append_kv_step` that works with
    /// `KcmmPool`'s internal `sequences` / `block_info` layout.
    pub fn append_kv_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],
        positions: &[usize],
        k_src: &CudaSlice<f16>,
        v_src: &CudaSlice<f16>,
    ) -> Result<()> {
        let batch = seq_indices.len();
        let step = self.elem_per_block / self.block_size; // kv_heads * head_dim
        let eb = std::mem::size_of::<f16>();
        let nbytes = step * eb;

        let va_k = self.va_k[layer_idx];
        let va_v = self.va_v[layer_idx];
        let k_base: CUdeviceptr = *k_src.device_ptr();
        let v_base: CUdeviceptr = *v_src.device_ptr();
        let seqs = self.sequences.lock();
        let info = self.block_info.lock();

        for b in 0..batch {
            let seq_idx = seq_indices[b];
            let pos = positions[b];
            let seq = &seqs[seq_idx];

            let logical_block = pos / self.block_size;
            let offset_in_block = pos % self.block_size;

            if logical_block >= seq.block_table.len() {
                return Err(anyhow!(
                    "logical block {} >= allocated {} for seq {}",
                    logical_block,
                    seq.block_table.len(),
                    seq_idx
                ));
            }

            let block_idx = seq.block_table[logical_block] as usize;
            let bi = &info[block_idx];
            let dst_off = bi.va_offset / eb + offset_in_block * step;
            let src_off = b * step;

            let dk = va_k + (dst_off * eb) as u64;
            let dv = va_v + (dst_off * eb) as u64;
            let sk = k_base + (src_off * eb) as u64;
            let sv = v_base + (src_off * eb) as u64;

            unsafe {
                let r = sys::lib().cuMemcpyDtoDAsync_v2(
                    dk, sk, nbytes, std::ptr::null_mut(),
                );
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync K: {:?}", r));
                }
                let r = sys::lib().cuMemcpyDtoDAsync_v2(
                    dv, sv, nbytes, std::ptr::null_mut(),
                );
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync V: {:?}", r));
                }
            }
        }
        Ok(())
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

    // --- Accessors for KvCacheBackend trait ---

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Maximum blocks per sequence.
    pub fn max_blocks_per_seq(&self) -> usize {
        self.max_blocks_per_seq
    }

    /// Bytes per block.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// --- KvCacheBackend impl ---

use crate::cache::backend::KvCacheBackend;

impl KvCacheBackend for KcmmPool {
    fn alloc_block(&self) -> Result<u32> {
        self.alloc_block()
    }
    fn alloc_sequence(&self, num_blocks: usize) -> Result<Vec<u32>> {
        self.alloc_sequence(num_blocks)
    }
    fn free_sequence(&self, block_table: &[u32]) {
        self.free_sequence(block_table)
    }
    fn append_block_to_sequence(&self, seq_idx: usize, block_idx: u32) {
        self.append_block_to_sequence(seq_idx, block_idx)
    }
    fn register_sequence(&self, block_table: Vec<u32>) -> usize {
        self.register_sequence(block_table)
    }
    fn unregister_sequence(&self, seq_idx: usize) {
        self.unregister_sequence(seq_idx)
    }
    fn update_seq_len(&self, seq_idx: usize, len: usize) {
        self.update_seq_len(seq_idx, len)
    }
    fn get_seq_len(&self, seq_idx: usize) -> usize {
        self.get_seq_len(seq_idx)
    }
    fn get_block_table(&self, seq_idx: usize) -> Option<Vec<u32>> {
        self.get_block_table(seq_idx)
    }
    fn get_block_va_offsets(&self, seq_idx: usize) -> Option<Vec<usize>> {
        self.get_block_va_offsets(seq_idx)
    }
    fn get_block_va_offset(&self, block_idx: u32) -> Option<usize> {
        self.get_block_va_offset(block_idx)
    }
    fn va_k(&self, layer: usize) -> u64 {
        self.va_k(layer)
    }
    fn va_v(&self, layer: usize) -> u64 {
        self.va_v(layer)
    }
    fn get_all_block_offsets_f16(&self) -> Vec<u64> {
        self.get_all_block_offsets_f16()
    }
    fn append_kv_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],
        positions: &[usize],
        k_src: &CudaSlice<f16>,
        v_src: &CudaSlice<f16>,
    ) -> Result<()> {
        self.append_kv_step(layer_idx, seq_indices, positions, k_src, v_src)
    }
    fn block_size(&self) -> usize {
        self.block_size
    }
    fn max_blocks_per_seq(&self) -> usize {
        self.max_blocks_per_seq
    }
    fn block_bytes(&self) -> usize {
        self.block_bytes
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn blocks_in_use(&self) -> usize {
        self.blocks_in_use()
    }
    fn has_free_blocks(&self) -> bool {
        self.has_free_blocks()
    }
    fn active_sequences(&self) -> usize {
        self.active_sequences()
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

    // --- GPU-dependent KcmmPool lifecycle tests ---

    mod gpu {
        use super::*;
        use crate::cuda::CudaContext;

        fn make_pool() -> (Arc<CudaContext>, KcmmPool) {
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let config = KcmmConfig {
                block_size: 16,
                max_blocks: 1024,
                cpu_cache_path: String::new(),
                tiering: false,
                eviction_policy: "lru".to_string(),
                prefetch_window: 4,
                max_batch_blocks: 64,
            };
            let pool = KcmmPool::new(
                ctx.clone(),
                config,
                22,  // num_layers (matching tiny_llama)
                4,   // kv_heads
                64,  // head_dim
                8,   // max_batch
                128, // max_seq_len
            )
            .expect("create KcmmPool");
            (ctx, pool)
        }

        fn make_pool_with_tiering() -> (Arc<CudaContext>, KcmmPool, tempfile::TempDir) {
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let dir = tempfile::tempdir().expect("create temp dir");
            let path = dir.path().join("kcmm_pool_swap_test");
            let config = KcmmConfig {
                block_size: 16,
                max_blocks: 256,
                cpu_cache_path: path.to_str().expect("valid UTF-8 path").to_string(),
                tiering: true,
                eviction_policy: "lru".to_string(),
                prefetch_window: 4,
                max_batch_blocks: 64,
            };
            let pool = KcmmPool::new(
                ctx.clone(),
                config,
                2,  // num_layers, kept small for test speed
                4,  // kv_heads
                64, // head_dim
                4,  // max_batch
                64, // max_seq_len
            )
            .expect("create KcmmPool with tiering");
            (ctx, pool, dir)
        }

        #[test]
        fn test_pool_construction() {
            let (_, pool) = make_pool();
            assert_eq!(pool.block_size, 16);
            assert_eq!(pool.max_batch, 8);
            assert_eq!(pool.max_seq_len, 128);
            assert_eq!(pool.num_layers, 22);
            assert_eq!(pool.max_blocks_per_seq, 8); // 128/16
            assert_eq!(pool.max_blocks_total, 64); // 8*8
            assert_eq!(pool.active_sequences(), 0);
            assert_eq!(pool.blocks_in_use(), 0);
            assert_eq!(pool.total_blocks(), 0);
            assert!(!pool.has_free_blocks());
            assert!(pool.tiering.is_none()); // tiering disabled
            assert!(pool.sharing.is_none()); // step 3: no sharing
        }

        #[test]
        fn test_alloc_single_block() {
            let (_, pool) = make_pool();
            let block_idx = pool.alloc_block().expect("alloc block");
            assert_eq!(pool.blocks_in_use(), 1);
            assert_eq!(pool.total_blocks(), 1);

            let va = pool.get_block_va_offset(block_idx);
            assert!(va.is_some());
            assert!(va.unwrap() > 0);
        }

        #[test]
        fn test_alloc_sequence() {
            let (_, pool) = make_pool();
            let table = pool.alloc_sequence(5).expect("alloc sequence");
            assert_eq!(table.len(), 5);
            assert_eq!(pool.blocks_in_use(), 5);

            for &idx in &table {
                assert!(pool.get_block_va_offset(idx).is_some());
            }
        }

        #[test]
        fn test_register_and_unregister() {
            let (_, pool) = make_pool();
            let table = pool.alloc_sequence(3).expect("alloc");
            let seq_idx = pool.register_sequence(table.clone());
            assert_eq!(pool.active_sequences(), 1);
            assert_eq!(pool.get_seq_len(seq_idx), 0);

            pool.update_seq_len(seq_idx, 45);
            assert_eq!(pool.get_seq_len(seq_idx), 45);

            pool.unregister_sequence(seq_idx);
            assert_eq!(pool.blocks_in_use(), 0);
        }

        #[test]
        fn test_append_block_to_sequence() {
            let (_, pool) = make_pool();
            let table = pool.alloc_sequence(2).expect("alloc");
            let seq_idx = pool.register_sequence(table);
            assert_eq!(pool.get_block_table(seq_idx).unwrap().len(), 2);

            let new_block = pool.alloc_block().expect("alloc extra");
            pool.append_block_to_sequence(seq_idx, new_block);
            assert_eq!(pool.get_block_table(seq_idx).unwrap().len(), 3);
        }

        #[test]
        fn test_touch_and_cool() {
            let (_, pool) = make_pool();
            let table = pool.alloc_sequence(2).expect("alloc");
            let seq_idx = pool.register_sequence(table);

            assert!(pool.is_active(seq_idx));

            pool.cool(seq_idx);
            assert!(!pool.is_active(seq_idx));

            pool.touch(seq_idx);
            assert!(pool.is_active(seq_idx));
        }

        #[test]
        fn test_touch_and_cool_out_of_bounds() {
            let (_, pool) = make_pool();
            // Should not panic on invalid indices
            pool.touch(999);
            pool.cool(999);
        }

        #[test]
        fn test_update_seq_len_out_of_bounds() {
            let (_, pool) = make_pool();
            // Should not panic on invalid index
            pool.update_seq_len(999, 42);
            assert_eq!(pool.get_seq_len(999), 0);
        }

        #[test]
        fn test_unregister_out_of_bounds() {
            let (_, pool) = make_pool();
            // Should not panic on invalid index
            pool.unregister_sequence(999);
        }

        #[test]
        fn test_get_block_va_offsets() {
            let (_, pool) = make_pool();
            let table = pool.alloc_sequence(4).expect("alloc");
            let seq_idx = pool.register_sequence(table);

            let offsets = pool.get_block_va_offsets(seq_idx);
            assert!(offsets.is_some());
            let offsets = offsets.unwrap();
            assert_eq!(offsets.len(), 4);
            // All offsets should be distinct and non-zero
            for o in &offsets {
                assert!(*o > 0, "VA offset should be positive");
            }
            let mut sorted = offsets.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), 4);
        }

        #[test]
        fn test_get_block_va_offsets_invalid_seq() {
            let (_, pool) = make_pool();
            assert_eq!(pool.get_block_va_offsets(999), None);
        }

        #[test]
        fn test_get_block_location_gpu_resident() {
            let (_, pool) = make_pool();
            let block_idx = pool.alloc_block().expect("alloc");

            let loc = pool.get_block_location(block_idx);
            assert!(loc.is_some());
            assert!(matches!(loc.unwrap(), BlockLocation::GpuResident(_, _)));
        }

        #[test]
        fn test_get_block_location_invalid_index() {
            let (_, pool) = make_pool();
            assert!(pool.get_block_location(999).is_none());
        }

        #[test]
        fn test_va_k_and_va_v() {
            let (_, pool) = make_pool();
            for l in 0..22 {
                assert!(pool.va_k(l) > 0, "va_k layer {} zero", l);
                assert!(pool.va_v(l) > 0, "va_v layer {} zero", l);
            }
        }

        #[test]
        fn test_blocks_in_use_and_total() {
            let (_, pool) = make_pool();
            assert_eq!(pool.blocks_in_use(), 0);
            assert_eq!(pool.total_blocks(), 0);

            let t1 = pool.alloc_sequence(3).expect("alloc");
            assert_eq!(pool.blocks_in_use(), 3);
            assert_eq!(pool.total_blocks(), 3);

            let t2 = pool.alloc_sequence(5).expect("alloc");
            assert_eq!(pool.blocks_in_use(), 8);
            assert_eq!(pool.total_blocks(), 8);

            pool.free_sequence(&t1);
            assert_eq!(pool.blocks_in_use(), 5);
            // total_blocks stays 8 (indices are recycled, not removed)
            assert_eq!(pool.total_blocks(), 8);

            pool.free_sequence(&t2);
            assert_eq!(pool.blocks_in_use(), 0);
        }

        #[test]
        fn test_free_sequence_after_eviction_releases_cpu_slot_without_double_free() {
            let (_ctx, pool, _dir) = make_pool_with_tiering();
            let tiering = pool.tiering.as_ref().expect("tiering enabled");
            let total_per_block = pool.num_layers * 2 * pool.block_bytes;

            let table = pool.alloc_sequence(2).expect("alloc sequence");
            let total_physical = pool.total_physical_blocks();
            let handles: Vec<BlockHandle> = table
                .iter()
                .map(|&idx| pool.get_block_handle(idx).expect("get handle"))
                .collect();

            tiering
                .evict_blocks(&pool, &[handles[0]], 1)
                .expect("evict one block");
            assert!(matches!(
                pool.get_block_location(table[0])
                    .expect("get evicted location"),
                BlockLocation::CpuResident(0)
            ));
            assert_eq!(
                pool.free_physical_blocks(),
                total_physical - 1,
                "one GPU-resident sequence block should still occupy physical memory"
            );

            pool.free_sequence(&table);

            assert_eq!(pool.blocks_in_use(), 0);
            assert_eq!(
                pool.free_physical_blocks(),
                total_physical,
                "freeing a CpuResident block must not return its old GPU handle twice"
            );

            let recycled_cpu_offset = tiering
                .alloc_cpu_slot(total_per_block)
                .expect("alloc CPU slot after sequence free");
            assert_eq!(
                recycled_cpu_offset, 0,
                "freeing a CpuResident block should release its CPU swap slot"
            );
        }

        #[test]
        fn test_collect_metrics() {
            let (_, pool) = make_pool();
            let table = pool.alloc_sequence(4).expect("alloc");
            let seq_idx = pool.register_sequence(table.clone());
            pool.update_seq_len(seq_idx, 50);

            let metrics = pool.collect_metrics();
            assert_eq!(metrics.active_sequences, 1);
            assert_eq!(metrics.blocks_in_use, 4);
            assert_eq!(metrics.total_tokens, 50);
            assert!(metrics.ideal_physical_bytes > 0);
            assert!(metrics.actual_physical_bytes > 0);
            assert!(metrics.internal_frag_rate >= 0.0);
            assert!(metrics.block_utilization > 0.0);
            assert!(metrics.physical_memory_efficiency > 0.0);

            pool.free_sequence(&table);
        }

        #[test]
        fn test_below_low_watermark() {
            let (_, pool) = make_pool();
            // With no blocks, total_physical_blocks = 0 → not below watermark
            assert!(!pool.below_low_watermark(0.2));

            // Allocate some blocks
            let table = pool.alloc_sequence(10).expect("alloc");
            let total = pool.total_physical_blocks();
            let free = pool.free_physical_blocks();
            let ratio = free as f32 / total as f32;

            let threshold = 0.01; // very low threshold
            assert_eq!(pool.below_low_watermark(threshold), ratio < threshold);

            pool.free_sequence(&table);
        }

        #[test]
        fn test_alloc_many_blocks_across_superblocks() {
            // Use a larger config to get a VA region big enough for multiple
            // superblock positions.
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let config = KcmmConfig {
                block_size: 16,
                max_blocks: 4096,
                cpu_cache_path: String::new(),
                tiering: false,
                eviction_policy: "lru".to_string(),
                prefetch_window: 4,
                max_batch_blocks: 64,
            };
            // max_seq_len=16384 → max_blocks_per_seq=1024. max_batch=4 → 4096 blocks.
            let pool = KcmmPool::new(
                ctx, config, 22, 4, 64, 4, 16384,
            ).expect("create pool with large VA");

            let sb_blocks = pool.k_pools[0].allocator.blocks_per_superblock;
            let count = sb_blocks + 10;
            let table = pool.alloc_sequence(count).expect("alloc many");
            assert_eq!(table.len(), count);
            assert!(pool.superblock_count() >= 2);
            assert_eq!(pool.blocks_in_use(), count);

            pool.free_sequence(&table);
            assert_eq!(pool.blocks_in_use(), 0);
        }

        #[test]
        fn test_lockstep_invariant() {
            let (_, pool) = make_pool();
            let num_layers = pool.num_layers;

            let table = pool.alloc_sequence(10).expect("alloc");
            for l in 1..num_layers {
                assert_eq!(
                    pool.k_pools[0].allocator.free_count(),
                    pool.k_pools[l].allocator.free_count(),
                    "K pool layer {} diverged", l
                );
                assert_eq!(
                    pool.v_pools[0].allocator.free_count(),
                    pool.v_pools[l].allocator.free_count(),
                    "V pool layer {} diverged", l
                );
            }

            pool.free_sequence(&table);
            for l in 1..num_layers {
                assert_eq!(
                    pool.k_pools[0].allocator.free_count(),
                    pool.k_pools[l].allocator.free_count()
                );
            }
        }
    }
}
