use anyhow::{anyhow, Result};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::f16;
use parking_lot::Mutex;
use std::sync::Arc;

use super::cuda_vmm::CudaVmm;
use crate::config::ModelConfig;
use crate::cuda::CudaContext;
use crate::kcmm::superblock::{
    align_up, BlockHandle, LayerKvPool, SuperblockInfo, SUPERBLOCK_SIZE,
};

/// Tokens per block — matches typical vLLM default.
pub const BLOCK_SIZE: usize = 16;
pub const BLOCK_BYTES: usize = 8192; // BLOCK_SIZE * kv_heads * head_dim * sizeof(f16)

// --- Per-block info (tracked by PagedKvCache) ---

#[derive(Debug, Clone)]
struct BlockInfo {
    /// Byte offset within each per-layer K/V VA region for this block.
    /// Same value across all layers because all pools allocate in lockstep.
    va_offset: usize,
    /// Superblock index (same across all pools in lockstep).
    superblock_idx: u32,
    /// Block index within the superblock.
    block_index_in_sb: u32,
    in_use: bool,
}

/// Per-request metadata for KV cache lookups.
pub struct SeqMetadata {
    pub block_table: Vec<u32>, // logical_block → block_idx
    pub seq_len: usize,
}

// --- Main cache ---

pub struct PagedKvCache {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,

    vmm: CudaVmm,
    /// K cache: one VA region per layer.
    va_k: Vec<u64>,
    /// V cache: one VA region per layer.
    va_v: Vec<u64>,

    /// Per-layer K physical pools.
    pub(crate) k_pools: Vec<LayerKvPool>,
    /// Per-layer V physical pools.
    pub(crate) v_pools: Vec<LayerKvPool>,

    /// Block-level tracking: block_idx → BlockInfo.
    block_info: Mutex<Vec<BlockInfo>>,

    /// Recycled block indices (blocks whose physical backing is still valid
    /// but are not currently assigned to any sequence).
    free_block_indices: Mutex<Vec<u32>>,

    /// Per-sequence metadata.
    pub seq_metadata: Mutex<Vec<SeqMetadata>>,

    /// Precomputed sizes.
    pub elem_per_block: usize,
    pub block_bytes: usize,
    pub max_blocks_total: usize,
}

impl PagedKvCache {
    pub fn new(
        ctx: Arc<CudaContext>,
        cfg: ModelConfig,
        max_batch: usize,
        max_seq_len: usize,
        block_size: usize,
    ) -> Result<Self> {
        let elem_per_block = cfg.kv_heads() * block_size * cfg.head_dim();
        let block_bytes = elem_per_block * std::mem::size_of::<f16>();
        let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
        let max_blocks_total = max_batch * max_blocks_per_seq;

        let vmm = CudaVmm::new(ctx.device.ordinal())?;

        // Reserve separate VA regions per layer for K and V
        let va_size = max_blocks_total * block_bytes;
        let va_size = align_up(va_size, SUPERBLOCK_SIZE);
        let mut va_k = Vec::with_capacity(cfg.num_hidden_layers);
        let mut va_v = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            va_k.push(vmm.reserve_address(va_size)?);
            va_v.push(vmm.reserve_address(va_size)?);
        }

        let num_layers = cfg.num_hidden_layers;
        let k_pools: Vec<LayerKvPool> = (0..num_layers)
            .map(|_| LayerKvPool::new(elem_per_block))
            .collect();
        let v_pools: Vec<LayerKvPool> = (0..num_layers)
            .map(|_| LayerKvPool::new(elem_per_block))
            .collect();

        Ok(Self {
            cfg,
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
            seq_metadata: Mutex::new(Vec::new()),
            elem_per_block,
            block_bytes,
            max_blocks_total,
        })
    }

    // --- Superblock management ---

    /// Map a newly created physical handle into the specified layer's K or V VA region.
    /// Returns the VA base offset within that region.
    fn map_superblock_to_layer(
        &self,
        phys_handle: u64,
        layer_idx: usize,
        is_v: bool,
        pool: &LayerKvPool,
    ) -> Result<usize> {
        let sb_count = pool.allocator.superblock_count();
        // sb_count was already incremented by add_superblock, so the new
        // superblock has index (sb_count - 1).
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
            "mapped superblock into layer VA region"
        );

        pool.superblocks.lock().push(SuperblockInfo {
            phys_handle,
            va_base,
        });

        Ok(va_base)
    }

    /// Ensure all per-layer pools have at least one free block.
    /// Creates new superblocks for ALL pools simultaneously if needed.
    fn ensure_capacity(&self) -> Result<()> {
        // All pools have the same free count (lockstep allocation).
        if self.k_pools[0].allocator.free_count() > 0 {
            return Ok(());
        }

        let num_layers = self.cfg.num_hidden_layers;

        // Create superblocks for all K and V pools simultaneously.
        // We add the superblock (which fills free lists), then create
        // the physical handle and map it.
        for l in 0..num_layers {
            // K pool
            let phys_k = self.vmm.create_physical(SUPERBLOCK_SIZE)?;
            self.k_pools[l].allocator.add_superblock();
            self.map_superblock_to_layer(phys_k, l, false, &self.k_pools[l])?;

            // V pool
            let phys_v = self.vmm.create_physical(SUPERBLOCK_SIZE)?;
            self.v_pools[l].allocator.add_superblock();
            self.map_superblock_to_layer(phys_v, l, true, &self.v_pools[l])?;
        }

        tracing::debug!(
            num_layers,
            total_pools = num_layers * 2,
            "created superblocks for all per-layer pools"
        );

        Ok(())
    }

    // --- Block allocation ---

    /// Allocate a single physical block across all per-layer pools.
    /// Returns (va_offset, superblock_idx, block_index_in_sb) computed
    /// from any pool (all pools produce the same values in lockstep).
    fn alloc_one_block_internal(&self) -> Result<(usize, u32, u32)> {
        self.ensure_capacity()?;

        // Allocate from all K and V pools simultaneously.
        // All pools have identical free lists, so they all return
        // handles with the same (superblock_idx, block_index).
        let num_layers = self.cfg.num_hidden_layers;
        let mut handle: Option<BlockHandle> = None;

        for l in 0..num_layers {
            let h_k = self.k_pools[l].allocator.try_allocate().ok_or_else(|| {
                anyhow!("K pool layer {}: no free block after ensure_capacity", l)
            })?;
            let h_v = self.v_pools[l].allocator.try_allocate().ok_or_else(|| {
                anyhow!("V pool layer {}: no free block after ensure_capacity", l)
            })?;

            if let Some(ref first) = handle {
                assert_eq!(
                    first.superblock_idx, h_k.superblock_idx,
                    "K pool layer {} superblock_idx mismatch",
                    l
                );
                assert_eq!(
                    first.block_index, h_k.block_index,
                    "K pool layer {} block_index mismatch",
                    l
                );
                assert_eq!(
                    first.superblock_idx, h_v.superblock_idx,
                    "V pool layer {} superblock_idx mismatch",
                    l
                );
                assert_eq!(
                    first.block_index, h_v.block_index,
                    "V pool layer {} block_index mismatch",
                    l
                );
            } else {
                handle = Some(h_k);
            }
        }

        let h = handle.unwrap();

        // Compute VA offset (same for all layers — use layer 0 K pool for reference).
        let sb = &self.k_pools[0].superblocks.lock()[h.superblock_idx as usize];
        let va_offset = sb.va_base + h.block_index as usize * self.block_bytes;

        Ok((va_offset, h.superblock_idx, h.block_index))
    }

    /// Assign or reuse a block index and populate its BlockInfo.
    fn install_block(&self, va_offset: usize, sb_idx: u32, blk_in_sb: u32) -> u32 {
        let mut free = self.free_block_indices.lock();
        if let Some(idx) = free.pop() {
            let mut info = self.block_info.lock();
            info[idx as usize] = BlockInfo {
                va_offset,
                superblock_idx: sb_idx,
                block_index_in_sb: blk_in_sb,
                in_use: true,
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
            });
            idx
        }
    }

    /// Allocate a single block. Returns the block index.
    /// Used during decode when a sequence needs to grow its block table.
    pub fn alloc_block(&self) -> Result<u32> {
        let (va_offset, sb_idx, blk_in_sb) = self.alloc_one_block_internal()?;
        Ok(self.install_block(va_offset, sb_idx, blk_in_sb))
    }

    /// Append a block to an existing sequence's block table.
    pub fn append_block_to_sequence(&self, seq_idx: usize, block_idx: u32) {
        let mut meta = self.seq_metadata.lock();
        if seq_idx < meta.len() {
            meta[seq_idx].block_table.push(block_idx);
        }
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
        let num_layers = self.cfg.num_hidden_layers;
        let mut recycled = Vec::new();

        for &block_idx in block_table {
            let bi = &mut info[block_idx as usize];
            if !bi.in_use {
                continue;
            }
            bi.in_use = false;

            // Return blocks to all per-layer free lists
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

    /// Register a new sequence with its block table. Returns the sequence index.
    pub fn register_sequence(&self, block_table: Vec<u32>) -> usize {
        let mut meta = self.seq_metadata.lock();
        let idx = meta.len();
        meta.push(SeqMetadata {
            block_table,
            seq_len: 0,
        });
        idx
    }

    /// Unregister a sequence and free its blocks.
    pub fn unregister_sequence(&self, seq_idx: usize) {
        let block_table = {
            let mut meta = self.seq_metadata.lock();
            if seq_idx >= meta.len() {
                return;
            }
            let table = std::mem::take(&mut meta[seq_idx].block_table);
            meta[seq_idx].seq_len = 0;
            table
        };
        if !block_table.is_empty() {
            self.free_sequence(&block_table);
        }
    }

    /// Update sequence length.
    pub fn update_seq_len(&self, seq_idx: usize, len: usize) {
        let mut meta = self.seq_metadata.lock();
        if seq_idx < meta.len() {
            meta[seq_idx].seq_len = len;
        }
    }

    /// Get the block table for a given sequence index.
    /// Returns the block count (0 if sequence not found).
    pub fn seq_block_count(&self, seq_idx: usize) -> usize {
        let meta = self.seq_metadata.lock();
        if seq_idx < meta.len() {
            meta[seq_idx].block_table.len()
        } else {
            0
        }
    }

    /// Get the block table VA offsets for a sequence.
    /// Returns None if the sequence is not found or any block is invalid.
    pub fn get_block_va_offsets(&self, seq_idx: usize) -> Option<Vec<usize>> {
        let meta = self.seq_metadata.lock();
        let seq = meta.get(seq_idx)?;
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

    /// Get the VA offset for a given block index.
    pub fn get_block_va_offset(&self, block_idx: u32) -> Option<usize> {
        let info = self.block_info.lock();
        let bi = info.get(block_idx as usize)?;
        if bi.in_use {
            Some(bi.va_offset)
        } else {
            None
        }
    }

    /// Get VA offsets for all blocks (in element offset, not byte offset).
    pub fn get_all_block_offsets_f16(&self) -> Vec<u64> {
        let info = self.block_info.lock();
        info.iter()
            .map(|bi| {
                if bi.in_use {
                    (bi.va_offset / std::mem::size_of::<f16>()) as u64
                } else {
                    0u64
                }
            })
            .collect()
    }

    /// Get the block table for a given sequence index.
    pub fn get_block_table(&self, seq_idx: usize) -> Option<Vec<u32>> {
        let meta = self.seq_metadata.lock();
        meta.get(seq_idx).map(|s| s.block_table.clone())
    }

    /// Get the K-cache virtual address base for a given layer.
    pub fn va_k(&self, layer: usize) -> u64 {
        self.va_k[layer]
    }

    /// Get the V-cache virtual address base for a given layer.
    pub fn va_v(&self, layer: usize) -> u64 {
        self.va_v[layer]
    }

    /// Get the sequence length.
    pub fn get_seq_len(&self, seq_idx: usize) -> usize {
        let meta = self.seq_metadata.lock();
        if seq_idx < meta.len() {
            meta[seq_idx].seq_len
        } else {
            0
        }
    }

    /// Number of active (registered) sequences.
    pub fn active_sequences(&self) -> usize {
        self.seq_metadata.lock().len()
    }

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
    pub fn append_kv_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],
        positions: &[usize],
        k_src: &CudaSlice<f16>,
        v_src: &CudaSlice<f16>,
    ) -> Result<()> {
        let batch = seq_indices.len();
        let kv = self.cfg.kv_heads();
        let hd = self.cfg.head_dim();
        let step = kv * hd;
        let eb = std::mem::size_of::<f16>();
        let nbytes = step * eb;

        let va_k = self.va_k[layer_idx];
        let va_v = self.va_v[layer_idx];
        let k_base: CUdeviceptr = *k_src.device_ptr();
        let v_base: CUdeviceptr = *v_src.device_ptr();
        let meta = self.seq_metadata.lock();
        let info = self.block_info.lock();

        for b in 0..batch {
            let seq_idx = seq_indices[b];
            let pos = positions[b];
            let seq = &meta[seq_idx];

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
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                    dk,
                    sk,
                    nbytes,
                    std::ptr::null_mut(),
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync K: {:?}", r));
                }
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                    dv,
                    sv,
                    nbytes,
                    std::ptr::null_mut(),
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync V: {:?}", r));
                }
            }
        }
        Ok(())
    }

    // --- Statistics ---

    /// Fraction of allocated superblock capacity sitting idle in the free list.
    /// This is NOT fragmentation — in a fixed-size block allocator, every free
    /// block can satisfy any allocation request. It measures physical memory
    /// idle rate due to CUDA VMM's 2 MiB allocation granularity.
    ///
    /// Returns 0.0 when no superblocks have been allocated.
    pub fn physical_idle_ratio(&self) -> f32 {
        let num_layers = self.cfg.num_hidden_layers;
        let superblocks = self.k_pools[0].allocator.superblock_count();
        if superblocks == 0 {
            return 0.0;
        }
        // Each logical superblock position consumes num_layers * 2 physical superblocks.
        let total_physical = superblocks * SUPERBLOCK_SIZE * num_layers * 2;
        let physical_used = self.blocks_in_use() * self.block_bytes * num_layers * 2;
        if physical_used >= total_physical {
            return 0.0;
        }
        (total_physical - physical_used) as f32 / total_physical as f32
    }

    pub fn blocks_in_use(&self) -> usize {
        self.block_info.lock().iter().filter(|b| b.in_use).count()
    }

    pub fn total_blocks(&self) -> usize {
        self.block_info.lock().len()
    }

    pub fn internal_fragmentation(&self) -> f32 {
        let meta = self.seq_metadata.lock();
        if meta.is_empty() {
            return 0.0;
        }
        let total_blocks_used: usize = meta.iter().map(|s| s.block_table.len()).sum();
        let total_slots: usize = total_blocks_used * self.block_size;
        let total_tokens: usize = meta.iter().map(|s| s.seq_len).sum();
        if total_slots == 0 {
            return 0.0;
        }
        (total_slots - total_tokens) as f32 / total_slots as f32
    }

    /// Total blocks allocated across all per-layer pools.
    ///
    /// Per-layer K and V pools allocate and free in lockstep (see
    /// CONTEXT.md "Lockstep Allocation"), so every pool has the same
    /// number of physical blocks. The value from `k_pools[0]` therefore
    /// represents all pools.
    pub fn total_physical_blocks(&self) -> usize {
        self.k_pools[0].allocator.total_blocks_allocated()
    }

    /// Free blocks available across all per-layer pools.
    ///
    /// Lockstep allocation keeps the per-layer K/V pools synchronized,
    /// so the free-block count is identical in every pool. Querying
    /// `k_pools[0]` is sufficient; see CONTEXT.md "Lockstep Allocation".
    pub fn free_physical_blocks(&self) -> usize {
        self.k_pools[0].allocator.free_count()
    }

    /// Check if there are free blocks available.
    pub fn has_free_blocks(&self) -> bool {
        self.k_pools[0].allocator.free_count() > 0
    }

    /// Blocks per superblock (same for all per-layer pools).
    pub fn blocks_per_superblock(&self) -> usize {
        self.k_pools[0].allocator.blocks_per_superblock
    }

    /// Superblock count (number of logical superblock positions, each
    /// backed by num_layers * 2 physical superblocks).
    pub fn superblock_count(&self) -> usize {
        self.k_pools[0].allocator.superblock_count()
    }

    pub fn stats(&self) -> CacheStats {
        let meta = self.seq_metadata.lock();
        let active_seqs = meta.len();
        let total_blocks_used: usize = meta.iter().map(|s| s.block_table.len()).sum();
        let total_tokens: usize = meta.iter().map(|s| s.seq_len).sum();
        let allocated = self.total_physical_blocks();
        let in_use = self.blocks_in_use();
        let free_pool = self.k_pools[0].allocator.free_count();
        let total_slots = total_blocks_used * self.block_size;
        let internal_frag = if total_slots > 0 {
            (total_slots - total_tokens) as f32 / total_slots as f32
        } else {
            0.0
        };
        let sb_count = self.k_pools[0].allocator.superblock_count();
        let num_layers = self.cfg.num_hidden_layers;

        CacheStats {
            active_sequences: active_seqs,
            total_blocks_allocated: allocated,
            blocks_in_use: in_use,
            free_blocks_in_pool: free_pool,
            blocks_per_superblock: self.k_pools[0].allocator.blocks_per_superblock,
            superblocks_allocated: sb_count,
            block_bytes: self.block_bytes,
            total_tokens_stored: total_tokens,
            internal_fragmentation: internal_frag,
            physical_memory_mib: (sb_count * SUPERBLOCK_SIZE * num_layers * 2) as f32
                / (1024.0 * 1024.0),
        }
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
        self.cfg.num_hidden_layers
    }
}

// --- KvCacheBackend impl ---

use super::backend::KvCacheBackend;

impl KvCacheBackend for PagedKvCache {
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
        self.cfg.num_hidden_layers
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

impl Drop for PagedKvCache {
    fn drop(&mut self) {
        let num_layers = self.cfg.num_hidden_layers;

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

        // Free all VA regions
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

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub active_sequences: usize,
    pub total_blocks_allocated: usize,
    pub blocks_in_use: usize,
    pub free_blocks_in_pool: usize,
    pub blocks_per_superblock: usize,
    pub superblocks_allocated: usize,
    pub block_bytes: usize,
    pub total_tokens_stored: usize,
    pub internal_fragmentation: f32,
    pub physical_memory_mib: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Block address computation tests ---

    #[test]
    fn test_block_address_formulas() {
        let block_size = 16;
        let kv_heads = 4;
        let head_dim = 128;
        let step = kv_heads * head_dim;
        let block_bytes = step * block_size * std::mem::size_of::<f16>(); // 16384

        assert_eq!(0usize * block_bytes, 0);
        assert_eq!(5usize * block_bytes, 5 * 16384);

        let pos = 47;
        let logical_block = pos / block_size;
        let offset_in_block = pos % block_size;
        assert_eq!(logical_block, 2);
        assert_eq!(offset_in_block, 15);

        let phys_block = 7usize;
        let dst_off = phys_block * block_size * step + offset_in_block * step;
        assert_eq!(dst_off, 7 * 16 * 512 + 15 * 512);
    }

    #[test]
    fn test_logical_to_physical_translation() {
        let block_table = vec![3u32, 7, 1];
        let block_size = 16;
        let step = 512;

        let pos = 25;
        let logical_block = pos / block_size;
        let offset_in_block = pos % block_size;
        assert_eq!(logical_block, 1);
        assert_eq!(offset_in_block, 9);

        let phys_block = block_table[logical_block] as usize;
        assert_eq!(phys_block, 7);

        let dst_off = phys_block * block_size * step + offset_in_block * step;
        assert_eq!(dst_off, 7 * 16 * 512 + 9 * 512);
    }

    #[test]
    fn test_seq_metadata_block_table() {
        let seq = SeqMetadata {
            block_table: vec![0, 1, 2, 3],
            seq_len: 64,
        };
        assert_eq!(seq.block_table.len(), 4);
        assert_eq!(63 / 16, 3);
        assert!(3 < seq.block_table.len());
    }

    #[test]
    fn test_block_size_constant() {
        assert_eq!(BLOCK_SIZE, 16);
    }

    #[test]
    fn test_max_blocks_calculation() {
        let max_seq_len = 2048;
        let block_size = 16;
        let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
        assert_eq!(max_blocks_per_seq, 128);

        let max_batch = 8;
        let max_blocks_total = max_batch * max_blocks_per_seq;
        assert_eq!(max_blocks_total, 1024);
    }

    // --- GPU-dependent lifecycle tests ---

    mod gpu {
        use super::*;
        use crate::cuda::CudaContext;

        fn make_cache() -> (Arc<CudaContext>, PagedKvCache) {
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let cfg = ModelConfig::tiny_llama();
            let cache =
                PagedKvCache::new(ctx.clone(), cfg, 8, 128, 16).expect("create PagedKvCache");
            (ctx, cache)
        }

        #[test]
        fn test_cache_construction() {
            let (_, cache) = make_cache();
            assert_eq!(cache.block_size, 16);
            assert_eq!(cache.max_batch, 8);
            assert_eq!(cache.max_seq_len, 128);
            assert_eq!(cache.max_blocks_per_seq, 8); // 128/16
            assert_eq!(cache.max_blocks_total, 64); // 8 * 8
            assert_eq!(cache.active_sequences(), 0);
            assert_eq!(cache.blocks_in_use(), 0);
            assert_eq!(cache.total_blocks(), 0);
            assert_eq!(cache.total_physical_blocks(), 0);
            assert!(!cache.has_free_blocks());
            assert_eq!(cache.free_physical_blocks(), 0);
        }

        #[test]
        fn test_alloc_single_block() {
            let (_, cache) = make_cache();
            let block_idx = cache.alloc_block().expect("alloc single block");
            assert_eq!(cache.blocks_in_use(), 1);
            assert_eq!(cache.total_blocks(), 1);

            let va = cache.get_block_va_offset(block_idx);
            assert!(va.is_some());
            assert!(va.unwrap() > 0);
        }

        #[test]
        fn test_alloc_sequence_multiple_blocks() {
            let (_, cache) = make_cache();
            let num_blocks = 5;
            let table = cache.alloc_sequence(num_blocks).expect("alloc sequence");
            assert_eq!(table.len(), num_blocks);
            assert_eq!(cache.blocks_in_use(), num_blocks);
            assert_eq!(cache.total_blocks(), num_blocks);

            // Each block should have a valid VA offset
            for &idx in &table {
                assert!(
                    cache.get_block_va_offset(idx).is_some(),
                    "block {} should have VA offset",
                    idx
                );
            }

            // All indices should be distinct
            let mut sorted = table.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                num_blocks,
                "all block indices should be distinct"
            );
        }

        #[test]
        fn test_register_and_unregister_sequence() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(3).expect("alloc");
            let seq_idx = cache.register_sequence(table.clone());
            assert_eq!(cache.active_sequences(), 1);
            assert_eq!(cache.seq_block_count(seq_idx), 3);

            cache.unregister_sequence(seq_idx);
            assert_eq!(cache.active_sequences(), 1); // seq_metadata still holds the slot
                                                     // Blocks should be freed
            assert_eq!(cache.blocks_in_use(), 0);
        }

        #[test]
        fn test_update_seq_len_and_get_seq_len() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(2).expect("alloc");
            let seq_idx = cache.register_sequence(table);
            assert_eq!(cache.get_seq_len(seq_idx), 0);

            cache.update_seq_len(seq_idx, 32);
            assert_eq!(cache.get_seq_len(seq_idx), 32);

            cache.update_seq_len(seq_idx, 16);
            assert_eq!(cache.get_seq_len(seq_idx), 16);
        }

        #[test]
        fn test_get_seq_len_invalid_index() {
            let (_, cache) = make_cache();
            assert_eq!(cache.get_seq_len(999), 0);
            assert_eq!(cache.seq_block_count(999), 0);
            assert!(cache.get_block_table(999).is_none());
        }

        #[test]
        fn test_get_block_va_offset_invalid_index() {
            let (_, cache) = make_cache();
            assert_eq!(cache.get_block_va_offset(999), None);
        }

        #[test]
        fn test_free_sequence_returns_blocks_to_pool() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(4).expect("alloc");
            assert_eq!(cache.blocks_in_use(), 4);

            cache.free_sequence(&table);
            assert_eq!(cache.blocks_in_use(), 0);

            // After freeing, re-allocating should reuse indices
            let new_table = cache.alloc_sequence(2).expect("re-alloc");
            assert_eq!(cache.blocks_in_use(), 2);
            assert!(
                new_table.iter().all(|&idx| idx < 4),
                "re-alloc should reuse freed indices: got {:?}",
                new_table
            );
        }

        #[test]
        fn test_append_block_to_sequence() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(2).expect("alloc");
            let seq_idx = cache.register_sequence(table);
            assert_eq!(cache.seq_block_count(seq_idx), 2);

            let new_block = cache.alloc_block().expect("alloc extra block");
            cache.append_block_to_sequence(seq_idx, new_block);
            assert_eq!(cache.seq_block_count(seq_idx), 3);
        }

        #[test]
        fn test_get_block_table() {
            let (_, cache) = make_cache();
            let table = vec![5u32, 10, 15];
            let seq_idx = cache.register_sequence(table.clone());
            let retrieved = cache.get_block_table(seq_idx).expect("should have table");
            assert_eq!(retrieved, table);
        }

        #[test]
        fn test_alloc_many_blocks_across_superblocks() {
            // Use a larger config to get a VA region big enough for multiple
            // superblock positions (each requires 2 MiB of VA space).
            let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
            let cfg = ModelConfig::tiny_llama();
            // max_seq_len=16384 → max_blocks_per_seq=1024. max_batch=2 → 2048 blocks.
            // block_bytes=8192, so va_size = 2048*8192 = 16 MiB → fits 8 superblocks.
            let cache =
                PagedKvCache::new(ctx, cfg, 2, 16384, 16).expect("create cache with large VA");

            let sb_blocks = cache.blocks_per_superblock();
            assert!(sb_blocks > 0);

            // Fill first superblock, then allocate more to trigger ensure_capacity
            let count = sb_blocks + 10; // force a second superblock
            let table = cache.alloc_sequence(count).expect("alloc many blocks");
            assert_eq!(table.len(), count);
            assert!(
                cache.superblock_count() >= 2,
                "should have at least 2 superblocks, got {}",
                cache.superblock_count()
            );

            for &idx in &table {
                assert!(cache.get_block_va_offset(idx).is_some());
            }

            cache.free_sequence(&table);
            assert_eq!(cache.blocks_in_use(), 0);
        }

        #[test]
        fn test_lockstep_invariant_across_layers() {
            // Verify that all per-layer K and V pools remain in lockstep:
            // they should always have the same free_count and superblock_count.
            let (_, cache) = make_cache();
            let num_layers = cache.cfg.num_hidden_layers;

            // Allocate some blocks
            let table = cache.alloc_sequence(10).expect("alloc");
            for l in 1..num_layers {
                assert_eq!(
                    cache.k_pools[0].allocator.free_count(),
                    cache.k_pools[l].allocator.free_count(),
                    "K pool layer {} free_count diverged from layer 0",
                    l
                );
                assert_eq!(
                    cache.v_pools[0].allocator.free_count(),
                    cache.v_pools[l].allocator.free_count(),
                    "V pool layer {} free_count diverged from layer 0",
                    l
                );
                assert_eq!(
                    cache.k_pools[0].allocator.superblock_count(),
                    cache.k_pools[l].allocator.superblock_count(),
                    "K pool layer {} superblock_count diverged",
                    l
                );
            }

            cache.free_sequence(&table);
            for l in 1..num_layers {
                assert_eq!(
                    cache.k_pools[0].allocator.free_count(),
                    cache.k_pools[l].allocator.free_count()
                );
            }
        }

        #[test]
        fn test_stats_accurate() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(4).expect("alloc");
            let seq_idx = cache.register_sequence(table.clone());
            cache.update_seq_len(seq_idx, 50);

            let stats = cache.stats();
            assert_eq!(stats.active_sequences, 1);
            assert_eq!(stats.blocks_in_use, 4);
            assert!(stats.total_blocks_allocated >= 4);
            assert_eq!(stats.total_tokens_stored, 50);
            assert!(stats.block_bytes > 0);
            assert!(stats.physical_memory_mib > 0.0);

            cache.free_sequence(&table);
            let after = cache.stats();
            assert_eq!(after.blocks_in_use, 0);
        }

        #[test]
        fn test_internal_fragmentation_zero_when_full() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(8).expect("alloc");
            let seq_idx = cache.register_sequence(table.clone());
            cache.update_seq_len(seq_idx, 128); // exactly 8 * 16

            let frag = cache.internal_fragmentation();
            assert!(
                (frag - 0.0).abs() < 0.001,
                "full blocks should have 0 fragmentation, got {}",
                frag
            );

            cache.free_sequence(&table);
        }

        #[test]
        fn test_internal_fragmentation_nonzero_with_partial() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(3).expect("alloc");
            let seq_idx = cache.register_sequence(table.clone());
            cache.update_seq_len(seq_idx, 20); // 3*16=48 slots, 20 used → (48-20)/48 = 0.583

            let frag = cache.internal_fragmentation();
            assert!(frag > 0.0, "partial block should have fragmentation > 0");
            assert!(frag < 1.0, "fragmentation should be < 1.0");

            cache.free_sequence(&table);
        }

        #[test]
        fn test_physical_idle_ratio() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(10).expect("alloc");

            let ratio = cache.physical_idle_ratio();
            // After allocating some blocks from a superblock, some blocks are idle
            assert!(
                ratio >= 0.0 && ratio <= 1.0,
                "physical_idle_ratio should be in [0,1], got {}",
                ratio
            );

            cache.free_sequence(&table);
        }

        #[test]
        fn test_has_free_blocks_after_alloc_and_free() {
            let (_, cache) = make_cache();
            assert!(!cache.has_free_blocks());

            let table = cache.alloc_sequence(8).expect("alloc");
            assert_eq!(
                cache.free_physical_blocks(),
                cache.k_pools[0].allocator.free_count()
            );

            cache.free_sequence(&table);
            // After freeing, there may be free blocks if alloc didn't consume all
            let free = cache.free_physical_blocks();
            let total = cache.total_physical_blocks();
            assert_eq!(free, total, "after freeing all, free should equal total");
        }

        #[test]
        fn test_va_k_and_va_v_nonzero() {
            let (_, cache) = make_cache();
            let num_layers = cache.cfg.num_hidden_layers;
            for l in 0..num_layers {
                assert!(cache.va_k(l) > 0, "va_k layer {} should be non-zero", l);
                assert!(cache.va_v(l) > 0, "va_v layer {} should be non-zero", l);
            }
        }

        #[test]
        fn test_get_all_block_offsets_f16() {
            let (_, cache) = make_cache();
            let table = cache.alloc_sequence(3).expect("alloc");
            let offsets = cache.get_all_block_offsets_f16();
            assert_eq!(offsets.len(), cache.total_blocks());
            // Active blocks should have non-zero offsets
            for &idx in &table {
                assert!(
                    offsets[idx as usize] > 0,
                    "active block {} should have non-zero f16 offset",
                    idx
                );
            }
            cache.free_sequence(&table);
        }
    }
}
