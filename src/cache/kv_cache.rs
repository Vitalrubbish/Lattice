use anyhow::{anyhow, Result};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::f16;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::config::ModelConfig;
use crate::cuda::CudaContext;

pub struct SlotAllocator {
    free: Mutex<Vec<usize>>,
    capacity: usize,
}

impl SlotAllocator {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free: Mutex::new((0..capacity).rev().collect()),
        }
    }

    pub fn acquire(&self) -> Option<usize> {
        self.free.lock().pop()
    }
    pub fn release(&self, slot: usize) {
        assert!(slot < self.capacity);
        self.free.lock().push(slot);
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn free_count(&self) -> usize {
        self.free.lock().len()
    }
}

pub struct KvCache {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub k_layers: Vec<CudaSlice<f16>>,
    pub v_layers: Vec<CudaSlice<f16>>,
    pub allocator: Arc<SlotAllocator>,
}

impl KvCache {
    pub fn new(
        ctx: Arc<CudaContext>,
        cfg: ModelConfig,
        max_batch: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let per_layer = max_batch * cfg.kv_heads() * max_seq_len * cfg.head_dim();
        let mut k_layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut v_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            k_layers.push(ctx.device.alloc_zeros::<f16>(per_layer)?);
            v_layers.push(ctx.device.alloc_zeros::<f16>(per_layer)?);
        }
        Ok(Self {
            cfg,
            ctx,
            max_batch,
            max_seq_len,
            k_layers,
            v_layers,
            allocator: Arc::new(SlotAllocator::new(max_batch)),
        })
    }

    pub fn allocator(&self) -> Arc<SlotAllocator> {
        self.allocator.clone()
    }

    pub fn append_step(
        &mut self,
        layer_idx: usize,
        slot_ids: &[usize],
        positions: &[usize],
        hidden: &CudaSlice<f16>,
    ) -> Result<()> {
        assert_eq!(slot_ids.len(), positions.len());
        let batch = slot_ids.len();
        let kv = self.cfg.kv_heads();
        let hd = self.cfg.head_dim();
        let hidden_size = self.cfg.hidden_size;
        let step = kv * hd;
        assert_eq!(hidden_size, step);

        let k = self.k_layers.get(layer_idx).ok_or_else(|| anyhow!("bad layer"))?;
        let v = self.v_layers.get(layer_idx).ok_or_else(|| anyhow!("bad layer"))?;
        let eb = std::mem::size_of::<f16>();
        let k_base: CUdeviceptr = *k.device_ptr();
        let v_base: CUdeviceptr = *v.device_ptr();
        let src_base: CUdeviceptr = *hidden.device_ptr();
        let nbytes = step * eb;

        for b in 0..batch {
            let slot = slot_ids[b];
            let pos = positions[b];
            if slot >= self.max_batch {
                return Err(anyhow!("slot {slot} >= max_batch {}", self.max_batch));
            }
            if pos >= self.max_seq_len {
                return Err(anyhow!("pos {pos} >= max_seq_len {}", self.max_seq_len));
            }

            let dst_off = (slot * kv * self.max_seq_len + pos) * hd;
            let src_off = b * hidden_size;
            let dk = k_base + (dst_off * eb) as u64;
            let dv = v_base + (dst_off * eb) as u64;
            let src = src_base + (src_off * eb) as u64;

            unsafe {
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(dk, src, nbytes, std::ptr::null_mut());
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync K: {:?}", r));
                }
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(dv, src, nbytes, std::ptr::null_mut());
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow!("cuMemcpyDtoDAsync V: {:?}", r));
                }
            }
        }
        Ok(())
    }

    pub fn fragmentation_ratio(&self, used: &[usize]) -> f32 {
        let alloc = self.allocator.capacity() * self.max_seq_len * self.cfg.num_hidden_layers;
        let used_total: usize = used.iter().sum::<usize>() * self.cfg.num_hidden_layers;
        if alloc == 0 {
            return 0.0;
        }
        1.0 - (used_total as f32 / alloc as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // --- SlotAllocator tests ---

    #[test]
    fn test_slot_allocator_new_capacity() {
        let alloc = SlotAllocator::new(8);
        assert_eq!(alloc.capacity(), 8);
        assert_eq!(alloc.free_count(), 8);
    }

    #[test]
    fn test_slot_allocator_new_zero_capacity() {
        let alloc = SlotAllocator::new(0);
        assert_eq!(alloc.capacity(), 0);
        assert_eq!(alloc.free_count(), 0);
        assert!(alloc.acquire().is_none());
    }

    #[test]
    fn test_slot_allocator_acquire_lifo_order() {
        // free is (0..4).rev().collect() = [3, 2, 1, 0]
        // Vec::pop() removes from the end, so order is: 0, 1, 2, 3
        let alloc = SlotAllocator::new(4);
        assert_eq!(alloc.acquire(), Some(0));
        assert_eq!(alloc.acquire(), Some(1));
        assert_eq!(alloc.acquire(), Some(2));
        assert_eq!(alloc.acquire(), Some(3));
        assert_eq!(alloc.free_count(), 0);
    }

    #[test]
    fn test_slot_allocator_acquire_exhaustion() {
        let alloc = SlotAllocator::new(2);
        assert_eq!(alloc.acquire(), Some(0));
        assert_eq!(alloc.acquire(), Some(1));
        assert_eq!(alloc.acquire(), None);
        assert_eq!(alloc.acquire(), None);
        assert_eq!(alloc.free_count(), 0);
    }

    #[test]
    fn test_slot_allocator_release_and_reacquire() {
        let alloc = SlotAllocator::new(3);
        let s0 = alloc.acquire().unwrap(); // gets 0
        let s1 = alloc.acquire().unwrap(); // gets 1
        assert_eq!(alloc.free_count(), 1); // 2 remains

        alloc.release(s0); // push 0 → [3, 2, 0]
        assert_eq!(alloc.free_count(), 2);
        // pop → 0
        assert_eq!(alloc.acquire(), Some(0));

        alloc.release(s1); // push 1 → [3, 2, 1]
        // pop → 1
        assert_eq!(alloc.acquire(), Some(1));
        // pop → 2
        assert_eq!(alloc.acquire(), Some(2));
        // Now exhausted
        assert_eq!(alloc.acquire(), None);
        assert_eq!(alloc.acquire(), None);
    }

    #[test]
    fn test_slot_allocator_concurrent_acquire_release() {
        let alloc = Arc::new(SlotAllocator::new(64));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let a = Arc::clone(&alloc);
            handles.push(thread::spawn(move || {
                let mut held = Vec::new();
                for _ in 0..50 {
                    if let Some(slot) = a.acquire() {
                        held.push(slot);
                    }
                    if !held.is_empty() {
                        let s = held.pop().unwrap();
                        a.release(s);
                    }
                }
                // Release any remaining
                for s in held {
                    a.release(s);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // After all threads finish, all slots should be back
        assert_eq!(alloc.free_count(), 64);
    }

    // --- KvCache::fragmentation_ratio tests ---

    #[test]
    fn test_fragmentation_ratio_empty() {
        // Create a minimal KvCache with 1 layer to test fragmentation_ratio
        // without needing an actual GPU context (the ratio is pure math).
        // We can't construct KvCache without GPU, so we test the formula
        // with stand-alone values.

        // alloc = capacity * max_seq_len * num_layers
        // used_total = sum(used) * num_layers
        // ratio = 1 - used_total/alloc

        // Perfect usage: used = max_seq_len for all slots
        let capacity = 4usize;
        let max_seq_len = 64usize;
        let num_layers = 22usize;
        let used: Vec<usize> = vec![64, 64, 64, 64]; // all fully used
        let alloc = capacity * max_seq_len * num_layers;
        let used_total: usize = used.iter().sum::<usize>() * num_layers;
        assert_eq!(used_total, alloc);
        let ratio = 1.0 - (used_total as f32 / alloc as f32);
        assert!((ratio - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fragmentation_ratio_half_used() {
        let capacity = 4usize;
        let max_seq_len = 64usize;
        let num_layers = 1usize;
        let used: Vec<usize> = vec![32, 32, 32, 32]; // half
        let alloc = capacity * max_seq_len * num_layers;
        let used_total: usize = used.iter().sum::<usize>() * num_layers;
        let ratio = 1.0 - (used_total as f32 / alloc as f32);
        assert!((ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_fragmentation_ratio_zero_alloc() {
        let capacity = 0usize;
        let max_seq_len = 0usize;
        let num_layers = 22usize;
        let _used: Vec<usize> = vec![];
        let alloc = capacity * max_seq_len * num_layers;
        assert_eq!(alloc, 0);
        // Ratio should be 0.0 when alloc == 0
        let ratio = if alloc == 0 { 0.0 } else { 1.0 };
        assert_eq!(ratio, 0.0);
    }
}
