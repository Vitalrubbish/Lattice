// src/cache/swap.rs
//
// GPU↔host KV cache swapping mechanism.
// When VRAM is exhausted, running sequences can be preempted
// by evicting their KV blocks to host memory.

use anyhow::{anyhow, Result};
use cudarc::driver::sys;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::backend::KvCacheBackend;

/// KV cache data for one evicted sequence, stored on the host.
#[derive(Debug, Clone)]
pub struct EvictedSeqData {
    /// K data per layer: each element is a Vec<u8> of `num_blocks * block_bytes`.
    pub k_layers: Vec<Vec<u8>>,
    /// V data per layer.
    pub v_layers: Vec<Vec<u8>>,
    /// Number of KV blocks in this sequence.
    pub num_blocks: usize,
    /// Sequence length at the time of eviction.
    pub seq_len: usize,
}

impl EvictedSeqData {
    /// Create an empty placeholder for KCMM mode, where the tiering engine
    /// manages CPU-side buffers internally and `EvictedSeqData` is unused.
    pub fn dummy() -> Self {
        Self {
            k_layers: Vec::new(),
            v_layers: Vec::new(),
            num_blocks: 0,
            seq_len: 0,
        }
    }
}

/// Manages GPU↔host swapping of KV cache blocks.
pub struct SwapManager {
    total_swapped_bytes: Mutex<usize>,
}

impl SwapManager {
    pub fn new() -> Self {
        Self {
            total_swapped_bytes: Mutex::new(0),
        }
    }

    /// Total bytes currently held in host swap buffers.
    pub fn total_swapped_bytes(&self) -> usize {
        *self.total_swapped_bytes.lock()
    }

    /// Evict a sequence's KV blocks from GPU to host memory.
    ///
    /// Copies K and V data for all layers from GPU VA to host buffers,
    /// then returns the host-side data. The caller must call
    /// `cache.unregister_sequence()` to release the GPU blocks afterward.
    pub fn evict_sequence(
        &self,
        cache: &dyn KvCacheBackend,
        seq_idx: usize,
    ) -> Result<EvictedSeqData> {
        let va_offsets = cache
            .get_block_va_offsets(seq_idx)
            .ok_or_else(|| anyhow!("seq {} not found or has invalid blocks", seq_idx))?;

        let num_blocks = va_offsets.len();
        if num_blocks == 0 {
            return Ok(EvictedSeqData {
                k_layers: Vec::new(),
                v_layers: Vec::new(),
                num_blocks: 0,
                seq_len: cache.get_seq_len(seq_idx),
            });
        }

        let block_bytes = cache.block_bytes();
        let num_layers = cache.num_layers();
        let total_bytes = num_blocks * block_bytes;

        let mut k_layers: Vec<Vec<u8>> = Vec::with_capacity(num_layers);
        let mut v_layers: Vec<Vec<u8>> = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let va_k = cache.va_k(l);
            let va_v = cache.va_v(l);

            let mut k_buf: Vec<u8> = vec![0u8; total_bytes];
            let mut v_buf: Vec<u8> = vec![0u8; total_bytes];

            for (i, &va_offset) in va_offsets.iter().enumerate() {
                let src_k = va_k + va_offset as u64;
                let src_v = va_v + va_offset as u64;
                let dst_off = i * block_bytes;

                // D2H copy: K
                unsafe {
                    let r = sys::lib().cuMemcpyDtoH_v2(
                        k_buf[dst_off..].as_mut_ptr() as *mut std::ffi::c_void,
                        src_k,
                        block_bytes,
                    );
                    if r != sys::CUresult::CUDA_SUCCESS {
                        return Err(anyhow!("cuMemcpyDtoH K failed at layer {}: {:?}", l, r));
                    }
                }

                // D2H copy: V
                unsafe {
                    let r = sys::lib().cuMemcpyDtoH_v2(
                        v_buf[dst_off..].as_mut_ptr() as *mut std::ffi::c_void,
                        src_v,
                        block_bytes,
                    );
                    if r != sys::CUresult::CUDA_SUCCESS {
                        return Err(anyhow!("cuMemcpyDtoH V failed at layer {}: {:?}", l, r));
                    }
                }
            }

            k_layers.push(k_buf);
            v_layers.push(v_buf);
        }

        {
            let mut swapped = self.total_swapped_bytes.lock();
            *swapped += total_bytes * num_layers * 2; // K+V
        }

        tracing::debug!(
            seq_idx,
            num_blocks,
            total_bytes_per_layer = total_bytes,
            num_layers,
            "evicted sequence to host"
        );

        Ok(EvictedSeqData {
            k_layers,
            v_layers,
            num_blocks,
            seq_len: cache.get_seq_len(seq_idx),
        })
    }

    /// Restore a previously evicted sequence's KV blocks from host back to GPU.
    ///
    /// Allocates new GPU blocks by calling `cache.alloc_sequence()`,
    /// copies the host-side data back to the new GPU blocks,
    /// and returns the new block table.
    ///
    /// The caller should then call `cache.register_sequence()` with the returned
    /// block table.
    pub fn restore_sequence(
        &self,
        cache: &dyn KvCacheBackend,
        data: &EvictedSeqData,
    ) -> Result<Vec<u32>> {
        if data.num_blocks == 0 {
            return Ok(Vec::new());
        }

        // Allocate new GPU blocks
        let block_table = cache
            .alloc_sequence(data.num_blocks)
            .map_err(|e| anyhow!("restore alloc_sequence failed: {}", e))?;

        let block_bytes = cache.block_bytes();
        let num_layers = cache.num_layers();
        let total_bytes = data.num_blocks * block_bytes;

        // Collect VA offsets for the newly allocated blocks
        let va_offsets: Vec<usize> = block_table
            .iter()
            .map(|&idx| {
                cache
                    .get_block_va_offset(idx)
                    .ok_or_else(|| anyhow!("newly allocated block {} not found", idx))
            })
            .collect::<Result<Vec<_>>>()?;

        for l in 0..num_layers {
            let va_k = cache.va_k(l);
            let va_v = cache.va_v(l);

            let k_data = &data.k_layers[l];
            let v_data = &data.v_layers[l];

            for (i, &va_offset) in va_offsets.iter().enumerate() {
                let dst_k = va_k + va_offset as u64;
                let dst_v = va_v + va_offset as u64;
                let src_off = i * block_bytes;

                // H2D copy: K
                unsafe {
                    let r = sys::lib().cuMemcpyHtoD_v2(
                        dst_k,
                        k_data[src_off..].as_ptr() as *const std::ffi::c_void,
                        block_bytes,
                    );
                    if r != sys::CUresult::CUDA_SUCCESS {
                        return Err(anyhow!("cuMemcpyHtoD K failed at layer {}: {:?}", l, r));
                    }
                }

                // H2D copy: V
                unsafe {
                    let r = sys::lib().cuMemcpyHtoD_v2(
                        dst_v,
                        v_data[src_off..].as_ptr() as *const std::ffi::c_void,
                        block_bytes,
                    );
                    if r != sys::CUresult::CUDA_SUCCESS {
                        return Err(anyhow!("cuMemcpyHtoD V failed at layer {}: {:?}", l, r));
                    }
                }
            }
        }

        {
            let mut swapped = self.total_swapped_bytes.lock();
            let dec = total_bytes * num_layers * 2;
            *swapped = swapped.saturating_sub(dec);
        }

        tracing::debug!(
            num_blocks = data.num_blocks,
            total_bytes_per_layer = total_bytes,
            num_layers,
            "restored sequence from host"
        );

        Ok(block_table)
    }

    /// Free host buffers associated with evicted data.
    /// Called when a swapped sequence is dropped (e.g., when it finally completes).
    pub fn drop_swapped(&self, data: &EvictedSeqData) -> usize {
        let freed = data
            .k_layers
            .iter()
            .chain(data.v_layers.iter())
            .map(|v| v.len())
            .sum::<usize>();
        {
            let mut swapped = self.total_swapped_bytes.lock();
            *swapped = swapped.saturating_sub(freed);
        }
        tracing::debug!(freed_bytes = freed, "dropped swapped sequence data");
        freed
    }
}

/// Global epoch counter for LRU tracking.
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Advance the epoch and return the new value.
pub fn advance_epoch() -> u64 {
    GLOBAL_EPOCH.fetch_add(1, Ordering::Relaxed)
}

/// Read the current epoch (without advancing).
pub fn current_epoch() -> u64 {
    GLOBAL_EPOCH.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::paged_kv::PagedKvCache;
    use crate::config::ModelConfig;
    use crate::cuda::CudaContext;

    #[test]
    fn swap_manager_evict_restore_cycle() {
        use std::sync::Arc;
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = ModelConfig::tiny_llama();
        let max_batch = 8;
        let max_seq_len = 64;
        let block_size = 16;

        let cache =
            PagedKvCache::new(ctx.clone(), cfg, max_batch, max_seq_len, block_size)
                .expect("create PagedKvCache");

        let swap = SwapManager::new();

        // Allocate a sequence
        let blocks_needed = 3;
        let block_table = cache.alloc_sequence(blocks_needed).expect("alloc");
        let seq_idx = cache.register_sequence(block_table.clone());
        cache.update_seq_len(seq_idx, 48);

        // Evict
        let evicted = swap.evict_sequence(&cache, seq_idx).expect("evict");

        assert_eq!(evicted.num_blocks, blocks_needed);
        assert_eq!(evicted.k_layers.len(), cache.cfg.num_hidden_layers);
        assert_eq!(evicted.v_layers.len(), cache.cfg.num_hidden_layers);

        // Free GPU blocks
        cache.free_sequence(&block_table);

        // Restore
        let new_table = swap
            .restore_sequence(&cache, &evicted)
            .expect("restore");

        assert_eq!(new_table.len(), blocks_needed);

        // Cleanup
        swap.drop_swapped(&evicted);
        cache.free_sequence(&new_table);

        println!("=== swap_manager_evict_restore_cycle: PASS ===");
    }

    #[test]
    fn swap_manager_empty_sequence() {
        use std::sync::Arc;
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = ModelConfig::tiny_llama();

        let cache =
            PagedKvCache::new(ctx.clone(), cfg, 4, 64, 16).expect("create PagedKvCache");

        let swap = SwapManager::new();

        // Empty block table
        let seq_idx = cache.register_sequence(Vec::new());
        let evicted = swap.evict_sequence(&cache, seq_idx).expect("evict empty");
        assert_eq!(evicted.num_blocks, 0);

        let new_table = swap
            .restore_sequence(&cache, &evicted)
            .expect("restore empty");
        assert!(new_table.is_empty());

        swap.drop_swapped(&evicted);
        println!("=== swap_manager_empty_sequence: PASS ===");
    }

    #[test]
    fn swap_manager_epoch_advances() {
        let e1 = advance_epoch();
        let e2 = advance_epoch();
        let e3 = advance_epoch();
        assert!(e2 > e1);
        assert!(e3 > e2);
        // fetch_add returns previous value, so current_epoch > e3
        assert!(current_epoch() > e3);
        println!("=== swap_manager_epoch_advances: PASS ===");
    }

    #[test]
    fn swap_manager_total_swapped_bytes() {
        use std::sync::Arc;
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
        let cfg = ModelConfig::tiny_llama();
        let cache =
            PagedKvCache::new(ctx.clone(), cfg, 4, 64, 16).expect("create PagedKvCache");

        let swap = SwapManager::new();
        assert_eq!(swap.total_swapped_bytes(), 0);

        let blocks_needed = 2;
        let block_table = cache.alloc_sequence(blocks_needed).expect("alloc");
        let seq_idx = cache.register_sequence(block_table.clone());
        cache.update_seq_len(seq_idx, 32);

        let evicted = swap.evict_sequence(&cache, seq_idx).expect("evict");
        let expected_bytes =
            blocks_needed * cache.block_bytes * cache.cfg.num_hidden_layers * 2;
        assert_eq!(swap.total_swapped_bytes(), expected_bytes);

        // Restore should free swap
        let new_table = swap
            .restore_sequence(&cache, &evicted)
            .expect("restore");
        assert_eq!(swap.total_swapped_bytes(), 0);

        cache.free_sequence(&new_table);
        swap.drop_swapped(&evicted);
        println!("=== swap_manager_total_swapped_bytes: PASS ===");
    }
}
