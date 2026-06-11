// Shared cache-backend trait — abstracts over PagedKvCache and KcmmPool.
//
// Both backends manage GPU KV-cache blocks backed by CUDA VMM.  This trait
// exposes the common operations needed by the transformer forward pass and
// the continuous scheduler, so either backend can be used without changing
// the hot path.
//
// KCMM-specific operations (touch, cool, tiering eviction, restore) are
// NOT part of this trait — the scheduler calls them directly on the
// concrete `KcmmPool` handle when KCMM mode is enabled.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::f16;

pub trait KvCacheBackend: Send + Sync {
    // --- Block allocation ---

    /// Allocate a single block. Returns the block index.
    fn alloc_block(&self) -> Result<u32>;

    /// Allocate `num_blocks` for a new sequence. Returns the block table.
    fn alloc_sequence(&self, num_blocks: usize) -> Result<Vec<u32>>;

    /// Free all blocks belonging to a sequence.
    fn free_sequence(&self, block_table: &[u32]);

    /// Append a block to an existing sequence's block table.
    fn append_block_to_sequence(&self, seq_idx: usize, block_idx: u32);

    // --- Sequence management ---

    /// Register a new sequence with its block table. Returns the sequence index.
    fn register_sequence(&self, block_table: Vec<u32>) -> usize;

    /// Unregister a sequence and free its blocks.
    fn unregister_sequence(&self, seq_idx: usize);

    /// Update sequence length.
    fn update_seq_len(&self, seq_idx: usize, len: usize);

    /// Get sequence length.
    fn get_seq_len(&self, seq_idx: usize) -> usize;

    /// Get the block table for a given sequence index.
    fn get_block_table(&self, seq_idx: usize) -> Option<Vec<u32>>;

    /// Get VA offsets for all blocks belonging to a sequence.
    /// Returns None if the sequence is not found or any block is invalid.
    fn get_block_va_offsets(&self, seq_idx: usize) -> Option<Vec<usize>>;

    /// Get the VA offset for a given block index.
    fn get_block_va_offset(&self, block_idx: u32) -> Option<usize>;

    // --- VA layout (consumed by paged-attention CUDA kernel) ---

    /// Get the K-cache virtual address base for a given layer.
    fn va_k(&self, layer: usize) -> u64;

    /// Get the V-cache virtual address base for a given layer.
    fn va_v(&self, layer: usize) -> u64;

    /// Get VA offsets for all blocks in f16-element units.
    ///
    /// Returns a flat Vec where index = block_idx and value = va_offset
    /// divided by `sizeof(f16)`.  Inactive blocks yield 0.
    fn get_all_block_offsets_f16(&self) -> Vec<u64>;

    // --- KV write (used by forward_step_paged) ---

    /// Write one step of KV data for a batch of sequences, using separate
    /// K and V sources (post-projection, post-RoPE).
    fn append_kv_step(
        &self,
        layer_idx: usize,
        seq_indices: &[usize],
        positions: &[usize],
        k_src: &CudaSlice<f16>,
        v_src: &CudaSlice<f16>,
    ) -> Result<()>;

    // --- Config accessors ---

    /// Tokens per block.
    fn block_size(&self) -> usize;

    /// Maximum blocks per sequence.
    fn max_blocks_per_seq(&self) -> usize;

    /// Bytes per block (computed from elem_per_block × sizeof(f16)).
    fn block_bytes(&self) -> usize;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    // --- Pool stats ---

    /// Number of blocks currently in use.
    fn blocks_in_use(&self) -> usize;

    /// Check if there are free blocks available.
    fn has_free_blocks(&self) -> bool;

    /// Number of active (registered) sequences.
    fn active_sequences(&self) -> usize;
}
