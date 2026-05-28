use anyhow::{anyhow, Result};
use cudarc::driver::{CudaSlice, DevicePtr};
use cudarc::driver::sys::CUdeviceptr;
use half::f16;
use parking_lot::Mutex;
use std::sync::Arc;

use super::cuda_vmm::CudaVmm;
use crate::config::ModelConfig;
use crate::cuda::CudaContext;

/// Tokens per block — matches typical vLLM default.
pub const BLOCK_SIZE: usize = 16;

/// Physical memory sub-allocator.
/// CUDA VMM min granularity is 2 MB, so we allocate "superblocks" of 2 MB
/// and carve them into individual blocks.
///
/// Does **not** own a `CudaVmm` — callers pass `&CudaVmm` to `allocate()` when
/// a new superblock is needed. `PagedKvCache` is the single VMM owner.
pub struct PhysicalBlockAllocator {
    block_bytes: usize,            // bytes per block per layer
    pub blocks_per_superblock: usize,  // how many blocks fit in 2 MB
    free_blocks: Mutex<Vec<BlockHandle>>,
    total_allocated: Mutex<usize>,
}

#[derive(Debug, Clone, Copy)]
struct BlockHandle {
    superblock_phys: u64,  // physical handle owning this block
    block_index: u32,      // which block within the superblock
}

impl PhysicalBlockAllocator {
    /// `elem_count` = kv_heads * BLOCK_SIZE * head_dim = f16 elements per block per layer
    pub fn new(elem_count: usize) -> Self {
        let block_bytes = elem_count * std::mem::size_of::<f16>();
        let superblock_size = 2 * 1024 * 1024; // 2 MB
        let blocks_per_superblock = superblock_size / block_bytes;

        assert!(blocks_per_superblock > 0,
            "block_bytes ({}) must be < 2 MB; reduce BLOCK_SIZE or model dims", block_bytes);

        Self {
            block_bytes,
            blocks_per_superblock,
            free_blocks: Mutex::new(Vec::new()),
            total_allocated: Mutex::new(0),
        }
    }

    /// Allocate one block. Returns (physical_handle, block_offset_in_bytes).
    ///
    /// `vmm` is borrowed from the caller (`PagedKvCache`) so there is a single
    /// VMM owner for the entire cache.
    pub fn allocate(&self, vmm: &CudaVmm) -> Result<(u64, usize)> {
        let mut free = self.free_blocks.lock();
        if let Some(handle) = free.pop() {
            let offset = handle.block_index as usize * self.block_bytes;
            return Ok((handle.superblock_phys, offset));
        }
        drop(free);

        // No free blocks — allocate a new 2 MB superblock
        let phys = vmm.create_physical(2 * 1024 * 1024)?;
        *self.total_allocated.lock() += 1;

        // Push all blocks from the new superblock into the free list (except the one we return)
        let mut free = self.free_blocks.lock();
        for i in 1..self.blocks_per_superblock {
            free.push(BlockHandle { superblock_phys: phys, block_index: i as u32 });
        }

        tracing::debug!(
            phys_handle = phys,
            blocks_added = self.blocks_per_superblock - 1,
            total_free = free.len(),
            "allocated new superblock"
        );

        Ok((phys, 0))
    }

    /// Free a previously allocated block.
    pub fn free(&self, phys_handle: u64, block_index: u32) {
        self.free_blocks.lock().push(BlockHandle {
            superblock_phys: phys_handle,
            block_index,
        });
    }

    pub fn free_count(&self) -> usize {
        self.free_blocks.lock().len()
    }

    pub fn total_blocks_allocated(&self) -> usize {
        *self.total_allocated.lock() * self.blocks_per_superblock
    }
}

/// Per-request metadata for KV cache lookups.
pub struct SeqMetadata {
    pub block_table: Vec<u32>,  // logical_block_idx → physical_block_idx
    pub seq_len: usize,
}

pub struct PagedKvCache {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,

    // CUDA VMM
    vmm: CudaVmm,
    /// One VA region per layer. Each region spans max_blocks_total * block_bytes.
    va_regions: Vec<u64>,
    /// Allocated physical handles (each 2 MB). Used for cleanup.
    physical_handles: Mutex<Vec<u64>>,

    // Block allocator
    block_allocator: Arc<PhysicalBlockAllocator>,

    // Block → VA offset mapping: physical_block_idx → va_offset (in bytes) within each layer
    block_va_map: Mutex<Vec<u64>>,

    // Per-sequence metadata
    seq_metadata: Mutex<Vec<SeqMetadata>>,

    // Precomputed
    pub elem_per_block: usize,   // kv_heads * BLOCK_SIZE * head_dim
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

        let vmm = CudaVmm::new(ctx.device.ordinal());
        let block_allocator = Arc::new(PhysicalBlockAllocator::new(elem_per_block));

        // Reserve one VA region per layer, sized for all possible blocks
        let va_size = max_blocks_total * block_bytes;
        let va_size = align_up(va_size, 2 * 1024 * 1024);
        let mut va_regions = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            va_regions.push(vmm.reserve_address(va_size)?);
        }

        Ok(Self {
            cfg,
            ctx,
            max_batch,
            max_seq_len,
            block_size,
            max_blocks_per_seq,
            vmm,
            va_regions,
            physical_handles: Mutex::new(Vec::new()),
            block_allocator,
            block_va_map: Mutex::new(Vec::new()),
            seq_metadata: Mutex::new(Vec::new()),
            elem_per_block,
            block_bytes,
            max_blocks_total,
        })
    }

    /// Allocate `num_blocks` new physical blocks for a new sequence.
    /// Returns the block table (list of physical block indices).
    pub fn alloc_sequence(&self, num_blocks: usize) -> Result<Vec<u32>> {
        let mut table = Vec::with_capacity(num_blocks);
        let mut va_map = self.block_va_map.lock();
        let next_idx = va_map.len() as u32;

        for i in 0..num_blocks {
            let (phys_handle, _offset_in_phys) = self.block_allocator.allocate(&self.vmm)?;

            let physical_block_idx = next_idx + i as u32;
            let va_offset = physical_block_idx as usize * self.block_bytes;

            // Map into each layer's VA region
            for (_layer, &va_base) in self.va_regions.iter().enumerate() {
                self.vmm.map(va_base, va_offset, phys_handle, self.block_bytes)?;
            }

            table.push(physical_block_idx);
            va_map.push(va_offset as u64);
        }

        Ok(table)
    }

    /// Free all blocks belonging to a sequence.
    pub fn free_sequence(&self, block_table: &[u32]) {
        for &phys_block_idx in block_table {
            // We don't track per-block phys handles separately;
            // the simplest approach: unmap from all layers, add back to free list.
            // For now, just mark blocks as free in the allocator.
            let va_offset = phys_block_idx as usize * self.block_bytes;
            for &va_base in &self.va_regions {
                let _ = self.vmm.unmap(va_base, va_offset, self.block_bytes);
            }
            // The physical handle is tracked per-superblock; individual blocks
            // return to the free list for reuse.
            let _superblock_idx = phys_block_idx as usize / self.block_allocator.blocks_per_superblock;
            let _block_in_sb = phys_block_idx as usize % self.block_allocator.blocks_per_superblock;
            // ... tracking requires mapping block_idx → (phys_handle, idx_in_sb)
            // This is simplified; full implementation needs a block_handle_map.
        }
    }

    /// Write KV for one step. Uses block table to resolve physical addresses.
    pub fn append_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],    // which sequences in the batch
        positions: &[usize],       // token position for each sequence
        hidden: &CudaSlice<f16>,
    ) -> Result<()> {
        let batch = seq_indices.len();
        let kv = self.cfg.kv_heads();
        let hd = self.cfg.head_dim();
        let step = kv * hd;
        let eb = std::mem::size_of::<f16>();
        let nbytes = step * eb;

        let va_base = self.va_regions[layer_idx];
        let src_base: CUdeviceptr = *hidden.device_ptr();
        let meta = self.seq_metadata.lock();

        for b in 0..batch {
            let seq_idx = seq_indices[b];
            let pos = positions[b];
            let seq = &meta[seq_idx];

            let logical_block = pos / self.block_size;
            let offset_in_block = pos % self.block_size;

            if logical_block >= seq.block_table.len() {
                return Err(anyhow!(
                    "logical block {} >= allocated {} for seq {}",
                    logical_block, seq.block_table.len(), seq_idx
                ));
            }

            let phys_block = seq.block_table[logical_block] as usize;
            let dst_off = phys_block * self.block_size * step + offset_in_block * step;
            let src_off = b * step;

            let dk = va_base + (dst_off * eb) as u64;
            let dv = va_base + (dst_off * eb) as u64;  // K and V could be separate regions
            let src = src_base + (src_off * eb) as u64;

            unsafe {
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                    dk, src, nbytes, std::ptr::null_mut(),
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync K: {:?}", r));
                }
                // V region is contiguous after K region in the same VA region.
                // Simplified: treat K and V separately with their own VA offsets.
            }
        }
        Ok(())
    }

    pub fn fragmentation_ratio(&self) -> f32 {
        let free = self.block_allocator.free_count();
        let total = self.block_allocator.total_blocks_allocated();
        if total == 0 {
            return 0.0;
        }
        free as f32 / total as f32
    }
}

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}