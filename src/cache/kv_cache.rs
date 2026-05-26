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
