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
const SUPERBLOCK_SIZE: usize = 2 * 1024 * 1024; // 2 MB

// --- Physical block sub-allocator ---

/// Tracks one 2 MB physical allocation and its VA placement.
struct SuperblockInfo {
    phys_handle: u64,
    /// Byte offset within each VA region where this superblock starts.
    va_base: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BlockHandle {
    superblock_idx: u32,
    block_index: u32,
}

pub struct PhysicalBlockAllocator {
    block_bytes: usize,
    pub blocks_per_superblock: usize,
    free_blocks: Mutex<Vec<BlockHandle>>,
    /// Number of superblocks allocated.
    superblock_count: Mutex<usize>,
}

impl PhysicalBlockAllocator {
    pub fn new(elem_count: usize) -> Self {
        let block_bytes = elem_count * std::mem::size_of::<f16>();
        let blocks_per_superblock = SUPERBLOCK_SIZE / block_bytes;
        assert!(blocks_per_superblock > 0,
            "block_bytes ({}) too large; reduce BLOCK_SIZE or model dims", block_bytes);
        assert_eq!(SUPERBLOCK_SIZE % block_bytes, 0,
            "block_bytes ({}) must divide superblock evenly", block_bytes);

        Self {
            block_bytes,
            blocks_per_superblock,
            free_blocks: Mutex::new(Vec::new()),
            superblock_count: Mutex::new(0),
        }
    }

    /// Allocate one block. Returns the handle.
    /// If a new superblock was created, `new_superblock_phys` is `Some(phys_handle)`.
    pub fn allocate(&self, vmm: &CudaVmm) -> Result<(BlockHandle, Option<u64>)> {
        // Try free list first
        {
            let mut free = self.free_blocks.lock();
            if let Some(handle) = free.pop() {
                return Ok((handle, None));
            }
        }

        // Need a new superblock
        let phys = vmm.create_physical(SUPERBLOCK_SIZE)?;
        let mut sb_count = self.superblock_count.lock();
        let sb_idx = *sb_count;
        *sb_count += 1;
        drop(sb_count);

        // Push remaining blocks into free list
        {
            let mut free = self.free_blocks.lock();
            for i in 1..self.blocks_per_superblock {
                free.push(BlockHandle {
                    superblock_idx: sb_idx as u32,
                    block_index: i as u32,
                });
            }
        }

        tracing::debug!(
            phys_handle = phys,
            sb_idx,
            blocks_added = self.blocks_per_superblock - 1,
            "allocated new superblock"
        );

        Ok((BlockHandle { superblock_idx: sb_idx as u32, block_index: 0 }, Some(phys)))
    }

    /// Return a block to the free pool.
    pub fn free(&self, handle: BlockHandle) {
        self.free_blocks.lock().push(handle);
    }

    pub fn free_count(&self) -> usize {
        self.free_blocks.lock().len()
    }

    pub fn total_blocks_allocated(&self) -> usize {
        *self.superblock_count.lock() * self.blocks_per_superblock
    }

    pub fn superblock_count(&self) -> usize {
        *self.superblock_count.lock()
    }

    /// Drain all physical handles — the caller must know the mapping to release them.
    /// Returns (phys_handles, superblock_count).
    pub fn superblock_metrics(&self) -> usize {
        *self.superblock_count.lock()
    }
}

// --- Per-block info (tracked by PagedKvCache) ---

#[derive(Debug, Clone, Copy)]
struct BlockInfo {
    /// VA offset within each K/V region for this block.
    va_offset: usize,
    /// Superblock that owns this block (for cleanup).
    superblock_idx: u32,
    block_index_in_sb: u32,
    in_use: bool,
}

/// Per-request metadata for KV cache lookups.
pub struct SeqMetadata {
    pub block_table: Vec<u32>,  // logical_block_idx → block_idx
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

    block_allocator: Arc<PhysicalBlockAllocator>,

    /// Superblock info: phys_handle + va_base for each superblock.
    superblocks: Mutex<Vec<SuperblockInfo>>,

    /// Block-level tracking: block_idx → BlockInfo.
    block_info: Mutex<Vec<BlockInfo>>,

    /// Recycled block indices.
    free_block_indices: Mutex<Vec<u32>>,

    /// Per-sequence metadata.
    seq_metadata: Mutex<Vec<SeqMetadata>>,

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
        let block_allocator = Arc::new(PhysicalBlockAllocator::new(elem_per_block));

        // Reserve separate VA regions per layer for K and V
        let va_size = max_blocks_total * block_bytes;
        let va_size = align_up(va_size, SUPERBLOCK_SIZE);
        let mut va_k = Vec::with_capacity(cfg.num_hidden_layers);
        let mut va_v = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            va_k.push(vmm.reserve_address(va_size)?);
            va_v.push(vmm.reserve_address(va_size)?);
        }

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
            block_allocator,
            superblocks: Mutex::new(Vec::new()),
            block_info: Mutex::new(Vec::new()),
            free_block_indices: Mutex::new(Vec::new()),
            seq_metadata: Mutex::new(Vec::new()),
            elem_per_block,
            block_bytes,
            max_blocks_total,
        })
    }

    /// Map a newly created superblock into all K/V VA regions.
    fn map_superblock(&self, phys_handle: u64, sb_idx: usize) -> Result<usize> {
        let va_base = sb_idx * SUPERBLOCK_SIZE;
        let num_layers = self.cfg.num_hidden_layers;

        for l in 0..num_layers {
            self.vmm.map(self.va_k[l], va_base, phys_handle, 0, SUPERBLOCK_SIZE)?;
            self.vmm.map(self.va_v[l], va_base, phys_handle, 0, SUPERBLOCK_SIZE)?;
        }

        tracing::debug!(
            phys_handle, sb_idx, va_base,
            num_layers, "mapped superblock into all KV regions"
        );

        self.superblocks.lock().push(SuperblockInfo {
            phys_handle,
            va_base,
        });

        Ok(va_base)
    }

    /// Allocate `num_blocks` for a new sequence. Returns the block table.
    pub fn alloc_sequence(&self, num_blocks: usize) -> Result<Vec<u32>> {
        let mut table = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            let (handle, new_phys) = self.block_allocator.allocate(&self.vmm)?;

            // Map the superblock into VA if this is the first block from it
            if let Some(phys_handle) = new_phys {
                self.map_superblock(phys_handle, handle.superblock_idx as usize)?;
            }

            // Compute VA offset for this block
            let sb = &self.superblocks.lock()[handle.superblock_idx as usize];
            let va_offset = sb.va_base + handle.block_index as usize * self.block_bytes;

            // Reuse or allocate a block index
            let block_idx = {
                let mut free = self.free_block_indices.lock();
                if let Some(idx) = free.pop() {
                    let mut info = self.block_info.lock();
                    info[idx as usize] = BlockInfo {
                        va_offset,
                        superblock_idx: handle.superblock_idx,
                        block_index_in_sb: handle.block_index,
                        in_use: true,
                    };
                    idx
                } else {
                    let mut info = self.block_info.lock();
                    let idx = info.len() as u32;
                    info.push(BlockInfo {
                        va_offset,
                        superblock_idx: handle.superblock_idx,
                        block_index_in_sb: handle.block_index,
                        in_use: true,
                    });
                    idx
                }
            };

            table.push(block_idx);
        }

        Ok(table)
    }

    /// Free all blocks belonging to a sequence.
    pub fn free_sequence(&self, block_table: &[u32]) {
        let mut info = self.block_info.lock();
        for &block_idx in block_table {
            let bi = &mut info[block_idx as usize];
            if !bi.in_use {
                continue;
            }
            bi.in_use = false;

            // Return block to free list
            self.block_allocator.free(BlockHandle {
                superblock_idx: bi.superblock_idx,
                block_index: bi.block_index_in_sb,
            });

            // Recycle the block index
            self.free_block_indices.lock().push(block_idx);
        }
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
            let meta = self.seq_metadata.lock();
            if seq_idx >= meta.len() {
                return;
            }
            meta[seq_idx].block_table.clone()
        };
        self.free_sequence(&block_table);
    }

    /// Update sequence length.
    pub fn update_seq_len(&self, seq_idx: usize, len: usize) {
        let mut meta = self.seq_metadata.lock();
        if seq_idx < meta.len() {
            meta[seq_idx].seq_len = len;
        }
    }

    pub fn block_allocator(&self) -> &Arc<PhysicalBlockAllocator> {
        &self.block_allocator
    }

    /// Write one step of KV data for a batch of sequences.
    pub fn append_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],
        positions: &[usize],
        hidden: &CudaSlice<f16>,
    ) -> Result<()> {
        let batch = seq_indices.len();
        let kv = self.cfg.kv_heads();
        let hd = self.cfg.head_dim();
        let step = kv * hd;
        let eb = std::mem::size_of::<f16>();
        let nbytes = step * eb;

        let va_k = self.va_k[layer_idx];
        let va_v = self.va_v[layer_idx];
        let src_base: CUdeviceptr = *hidden.device_ptr();
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
                    logical_block, seq.block_table.len(), seq_idx
                ));
            }

            let block_idx = seq.block_table[logical_block] as usize;
            let bi = &info[block_idx];
            let dst_off = bi.va_offset / eb + offset_in_block * step;
            let src_off = b * step;

            let dk = va_k + (dst_off * eb) as u64;
            let dv = va_v + (dst_off * eb) as u64;
            let src = src_base + (src_off * eb) as u64;

            unsafe {
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                    dk, src, nbytes, std::ptr::null_mut(),
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync K: {:?}", r));
                }
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                    dv, src, nbytes, std::ptr::null_mut(),
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync V: {:?}", r));
                }
            }
        }
        Ok(())
    }

    // --- Statistics ---

    pub fn fragmentation_ratio(&self) -> f32 {
        let free = self.block_allocator.free_count();
        let total = self.block_allocator.total_blocks_allocated();
        if total == 0 {
            return 0.0;
        }
        free as f32 / total as f32
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

    pub fn stats(&self) -> CacheStats {
        let meta = self.seq_metadata.lock();
        let active_seqs = meta.len();
        let total_blocks_used: usize = meta.iter().map(|s| s.block_table.len()).sum();
        let total_tokens: usize = meta.iter().map(|s| s.seq_len).sum();
        let allocated = self.total_blocks();
        let in_use = self.blocks_in_use();
        let free_pool = self.block_allocator.free_count();
        let total_slots = total_blocks_used * self.block_size;
        let internal_frag = if total_slots > 0 {
            (total_slots - total_tokens) as f32 / total_slots as f32
        } else {
            0.0
        };
        let sb_count = self.block_allocator.superblock_count();

        CacheStats {
            active_sequences: active_seqs,
            total_blocks_allocated: allocated,
            blocks_in_use: in_use,
            free_blocks_in_pool: free_pool,
            blocks_per_superblock: self.block_allocator.blocks_per_superblock,
            superblocks_allocated: sb_count,
            block_bytes: self.block_bytes,
            total_tokens_stored: total_tokens,
            internal_fragmentation: internal_frag,
            physical_memory_mib: (sb_count * SUPERBLOCK_SIZE) as f32 / (1024.0 * 1024.0),
        }
    }
}

// --- Drop ---

impl Drop for PagedKvCache {
    fn drop(&mut self) {
        // Unmap all superblocks from all VA regions
        let sbs = std::mem::take(&mut *self.superblocks.lock());
        let num_layers = self.cfg.num_hidden_layers;
        for sb in &sbs {
            for l in 0..num_layers {
                let _ = self.vmm.unmap(self.va_k[l], sb.va_base, SUPERBLOCK_SIZE);
                let _ = self.vmm.unmap(self.va_v[l], sb.va_base, SUPERBLOCK_SIZE);
            }
            let _ = self.vmm.release_physical(sb.phys_handle);
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

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

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
        alloc.free(BlockHandle { superblock_idx: 0, block_index: 0 });
        alloc.free(BlockHandle { superblock_idx: 0, block_index: 1 });
        alloc.free(BlockHandle { superblock_idx: 1, block_index: 5 });
        assert_eq!(alloc.free_count(), 3);
    }

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

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, SUPERBLOCK_SIZE), 0);
        assert_eq!(align_up(1, SUPERBLOCK_SIZE), SUPERBLOCK_SIZE);
        assert_eq!(align_up(SUPERBLOCK_SIZE, SUPERBLOCK_SIZE), SUPERBLOCK_SIZE);
        assert_eq!(align_up(SUPERBLOCK_SIZE + 1, SUPERBLOCK_SIZE), 4 * 1024 * 1024);
    }

    #[test]
    fn test_superblock_block_carving() {
        let elem_count = 4 * 16 * 128;
        let block_bytes = elem_count * std::mem::size_of::<f16>();
        assert_eq!(SUPERBLOCK_SIZE % block_bytes, 0,
            "block_bytes must divide superblock evenly");
    }

    // --- Step 3 GPU benchmarks (require CUDA device) ---

    #[test]
    fn step3_max_concurrent_requests() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = ModelConfig::tiny_llama();

        let max_batch = 256;
        let max_seq_len = 256;
        let block_size = 16;
        let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;

        let cache = PagedKvCache::new(
            ctx, cfg.clone(), max_batch, max_seq_len, block_size,
        ).expect("create PagedKvCache");

        println!("\n=== Step 3: Maximum Concurrent Requests ===");
        println!("model: tiny_llama (kv_heads={}, head_dim={}, layers={})",
            cfg.kv_heads(), cfg.head_dim(), cfg.num_hidden_layers);
        println!("GPU map granularity: {} bytes", cache.vmm.map_granularity);
        println!("block_size={}, max_seq_len={}, blocks_per_seq={}",
            block_size, max_seq_len, max_blocks_per_seq);
        println!("block_bytes={}, blocks_per_superblock={}",
            cache.block_bytes, cache.block_allocator.blocks_per_superblock);

        // Allocate until OOM
        let mut allocated = 0usize;
        for _ in 0..max_batch {
            match cache.alloc_sequence(max_blocks_per_seq) {
                Ok(table) => {
                    cache.register_sequence(table);
                    allocated += 1;
                }
                Err(e) => {
                    println!("alloc_sequence failed at seq {}: {:?}", allocated, e);
                    break;
                }
            }
        }

        let stats = cache.stats();
        println!("\nResults:");
        println!("  max concurrent requests:  {}", allocated);
        println!("  total blocks allocated:   {}", stats.total_blocks_allocated);
        println!("  blocks in use:            {}", stats.blocks_in_use);
        println!("  free blocks in pool:      {}", stats.free_blocks_in_pool);
        println!("  superblocks allocated:    {}", stats.superblocks_allocated);
        println!("  physical memory:          {:.2} MiB", stats.physical_memory_mib);
        println!("  physical blocks / request: {}", stats.blocks_in_use as f32 / allocated.max(1) as f32);

        // Check how maps are batched
        let maps_per_superblock = cfg.num_hidden_layers * 2;
        let total_map_calls = stats.superblocks_allocated * maps_per_superblock;
        println!("  total cuMemMap calls:     {} ({} per superblock)",
            total_map_calls, maps_per_superblock);

        // Free all
        for i in 0..allocated {
            cache.unregister_sequence(i);
        }
        let after = cache.stats();
        println!("\nAfter freeing all:");
        println!("  blocks in use:            {}", after.blocks_in_use);
        println!("  free blocks in pool:      {}", after.free_blocks_in_pool);
        println!("  free ratio:               {:.4}", cache.fragmentation_ratio());

        assert!(allocated > 0, "should allocate at least some sequences");
        println!("=== End Max Concurrent Requests ===\n");
    }

    #[test]
    fn step3_fragmentation_rate() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = ModelConfig::tiny_llama();

        let max_batch = 64;
        let max_seq_len = 256;
        let block_size = 16;
        let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;

        let cache = PagedKvCache::new(
            ctx, cfg.clone(), max_batch, max_seq_len, block_size,
        ).expect("create PagedKvCache");

        println!("\n=== Step 3: Fragmentation Rate ===");

        // Phase 1: allocate all sequences
        let mut tables: Vec<Vec<u32>> = Vec::new();
        let mut seq_indices: Vec<usize> = Vec::new();
        for _i in 0..max_batch {
            if let Ok(table) = cache.alloc_sequence(max_blocks_per_seq) {
                let si = cache.register_sequence(table.clone());
                cache.update_seq_len(si, max_seq_len);
                tables.push(table);
                seq_indices.push(si);
            }
        }

        let stats = cache.stats();
        println!("After allocating {} sequences:", tables.len());
        println!("  internal_fragmentation: {:.4}", stats.internal_fragmentation);
        println!("  blocks in use: {}", stats.blocks_in_use);
        println!("  physical memory: {:.2} MiB", stats.physical_memory_mib);
        println!("  superblocks: {}", stats.superblocks_allocated);

        // Phase 2: free 50% to create holes
        for (idx, table) in tables.iter().enumerate() {
            if idx % 2 != 0 {
                cache.free_sequence(table);
            }
        }

        let mid_stats = cache.stats();
        println!("\nAfter freeing 50% of sequences:");
        println!("  blocks in use:         {}", mid_stats.blocks_in_use);
        println!("  free blocks in pool:   {}", mid_stats.free_blocks_in_pool);
        println!("  free block ratio:      {:.4}", cache.fragmentation_ratio());

        // Phase 3: re-allocate with shorter sequences
        let half_blocks = max_blocks_per_seq / 2;
        let half_seq_len = half_blocks * block_size / 2;
        let mut new_count = 0usize;
        for _ in 0..(max_batch / 2) {
            if let Ok(table) = cache.alloc_sequence(half_blocks) {
                let si = cache.register_sequence(table);
                cache.update_seq_len(si, half_seq_len);
                new_count += 1;
            } else {
                break;
            }
        }

        let final_stats = cache.stats();
        println!("\nAfter re-allocating {} shorter sequences:", new_count);
        println!("  active sequences:     {}", final_stats.active_sequences);
        println!("  blocks in use:        {}", final_stats.blocks_in_use);
        println!("  free blocks in pool:  {}", final_stats.free_blocks_in_pool);
        println!("  total blocks:         {}", final_stats.total_blocks_allocated);
        println!("  superblocks:          {}", final_stats.superblocks_allocated);

        let wasted_tokens_in_last_block = new_count * (block_size - (half_seq_len % block_size));
        println!("\nInternal fragmentation:");
        println!("  total tokens stored:     {}", final_stats.total_tokens_stored);
        println!("  total slots allocated:   {}", final_stats.blocks_in_use * block_size);
        println!("  internal_fragmentation:  {:.4} ({:.2}%)",
            final_stats.internal_fragmentation,
            final_stats.internal_fragmentation * 100.0);
        println!("  wasted slots in last blocks: ~{}", wasted_tokens_in_last_block);
        println!("=== End Fragmentation Rate ===\n");

        assert!(!tables.is_empty());
    }

    #[test]
    fn step3_cumemmap_overhead() {
        let vmm = CudaVmm::new(0).expect("cuda device 0");
        let cfg = ModelConfig::tiny_llama();
        let num_layers = cfg.num_hidden_layers;

        println!("\n=== Step 3: cuMemMap/cuMemUnmap Overhead ===");
        println!("GPU map granularity: {} bytes", vmm.map_granularity);
        println!("num_layers={}, maps per superblock = {} (K+V per layer)",
            num_layers, num_layers * 2);

        // Setup: one 2MB VA region per layer, one 2MB physical handle
        let va_k: Vec<u64> = (0..num_layers)
            .map(|_| vmm.reserve_address(SUPERBLOCK_SIZE).expect("reserve K"))
            .collect();
        let va_v: Vec<u64> = (0..num_layers)
            .map(|_| vmm.reserve_address(SUPERBLOCK_SIZE).expect("reserve V"))
            .collect();

        let _warmup = 4;
        let iters = 16;

        // --- Per-layer mapping benchmark (mimics per-block approach) ---
        let per_layer_sizes = [8192, 16384, 32768, 65536, 131072, 262144, 524288, SUPERBLOCK_SIZE];

        println!("\nPer-call latency vs. mapping size:");
        println!("  {:>8}  {:>12}  {:>12}", "size", "map (µs)", "unmap (µs)");

        for &size in &per_layer_sizes {
            if size > SUPERBLOCK_SIZE || size < vmm.map_granularity {
                continue;
            }
            let phys = vmm.create_physical(size).expect("create phys");

            // Warmup
            for _ in 0..2 {
                for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                    vmm.map(vk, 0, phys, 0, size).unwrap();
                    vmm.unmap(vk, 0, size).unwrap();
                    vmm.map(vv, 0, phys, 0, size).unwrap();
                    vmm.unmap(vv, 0, size).unwrap();
                }
            }

            let start = std::time::Instant::now();
            for _ in 0..iters {
                for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                    vmm.map(vk, 0, phys, 0, size).unwrap();
                    vmm.map(vv, 0, phys, 0, size).unwrap();
                    vmm.unmap(vk, 0, size).unwrap();
                    vmm.unmap(vv, 0, size).unwrap();
                }
            }
            let elapsed = start.elapsed();
            let total_ops = iters * num_layers * 2 * 2; // map+unmap × K+V
            let avg_us = elapsed.as_micros() as f64 / total_ops as f64;

            println!("  {:>8}  {:>12.2}  {:>12.2}", size, avg_us, avg_us);

            vmm.release_physical(phys).expect("release");
        }

        // --- Full superblock mapping (our new approach) ---
        println!("\nFull-superblock (2MB) mapping per layer:");
        let phys = vmm.create_physical(SUPERBLOCK_SIZE).expect("create phys");

        // Warmup
        for (_i, (&vk, &vv)) in va_k.iter().zip(va_v.iter()).enumerate() {
            vmm.map(vk, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.map(vv, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.unmap(vk, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.unmap(vv, 0, SUPERBLOCK_SIZE).unwrap();
        }

        let start = std::time::Instant::now();
        for _ in 0..iters {
            for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                vmm.map(vk, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
                vmm.map(vv, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
                vmm.unmap(vk, 0, SUPERBLOCK_SIZE).unwrap();
                vmm.unmap(vv, 0, SUPERBLOCK_SIZE).unwrap();
            }
        }
        let elapsed = start.elapsed();
        let total_ops = iters * num_layers * 2 * 2;
        let avg_us = elapsed.as_micros() as f64 / total_ops as f64;
        println!("  avg per 2MB map/unmap:  {:.2} µs", avg_us);
        println!("  total for {} layers:    {:.2} µs", num_layers, avg_us * num_layers as f64 * 2.0);

        // Cleanup
        for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
            vmm.unmap(vk, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.unmap(vv, 0, SUPERBLOCK_SIZE).unwrap();
        }
        vmm.release_physical(phys).expect("release phys");
        for v in va_k.iter().chain(va_v.iter()) {
            vmm.free_address(*v, SUPERBLOCK_SIZE).expect("free va");
        }

        println!("=== End cuMemMap/cuMemUnmap Overhead ===\n");
    }

    #[test]
    fn step3_internal_fragmentation_analysis() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = ModelConfig::tiny_llama();

        let max_batch = 16;
        let max_seq_len = 128;
        let block_size = 16;
        let cache = PagedKvCache::new(
            ctx, cfg, max_batch, max_seq_len, block_size,
        ).expect("create PagedKvCache");

        println!("\n=== Step 3: Internal Fragmentation Analysis ===");
        println!("block_size={} tokens", block_size);

        let seq_lengths = [1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 100, 127, 128, 7];
        let mut total_wasted = 0usize;
        let mut total_slots = 0usize;

        for &sl in &seq_lengths {
            let blocks_needed = (sl + block_size - 1) / block_size;
            let table = cache.alloc_sequence(blocks_needed).expect("alloc");
            let seq_idx = cache.register_sequence(table);
            cache.update_seq_len(seq_idx, sl);

            let slots = blocks_needed * block_size;
            let waste = slots - sl;
            let frag = waste as f32 / slots as f32;
            total_wasted += waste;
            total_slots += slots;
            println!(
                "  seq_len={:3}  blocks={:2}  slots={:3}  waste={:2}  frag={:.3}",
                sl, blocks_needed, slots, waste, frag
            );
        }

        let overall_frag = total_wasted as f32 / total_slots as f32;
        println!("\nSummary:");
        println!("  total sequences:         {}", seq_lengths.len());
        println!("  total slots allocated:   {}", total_slots);
        println!("  total tokens stored:     {}", total_slots - total_wasted);
        println!("  total wasted slots:      {}", total_wasted);
        println!("  overall internal frag:   {:.4} ({:.2}%)",
            overall_frag, overall_frag * 100.0);
        println!("  average waste per seq:   {:.1} tokens",
            total_wasted as f32 / seq_lengths.len() as f32);

        let stats = cache.stats();
        println!("\n  cache.internal_fragmentation() = {:.4}", stats.internal_fragmentation);
        println!("=== End Internal Fragmentation Analysis ===\n");

        assert_eq!(total_slots - total_wasted, seq_lengths.iter().sum::<usize>());
    }
}
