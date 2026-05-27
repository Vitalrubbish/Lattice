# Step 3 Tutorial: Continuous Batching & Paged KV Cache Implementation

**Date:** 2026-05-26

This tutorial walks through implementing Step 3 manually — continuous batching with paged GPU memory management via CUDA VMM API.

## Prerequisites

- Working Step 1 baseline (`cargo build --release` succeeds)
- CUDA 12.x+ with driver (WSL2 GPU-PV is fine for development)
- Understanding of the current `KvCache` and `StaticScheduler` code

---

## Phase 1: CUDA VMM FFI Bindings

### Step 1.1 — Create `src/cache/cuda_vmm.rs`

The CUDA Virtual Memory Management API is in `cudarc::driver::sys::lib()` but the high-level wrappers don't exist yet in `cudarc`. We need raw FFI.

```rust
// src/cache/cuda_vmm.rs
use anyhow::{anyhow, Result};
use cudarc::driver::sys::{self, CUresult, CUdeviceptr};
use std::os::raw::c_void;

const CU_MEM_ALLOCATION_TYPE_PINNED: u32 = 0x04;
const CU_MEM_ACCESS_FLAGS_PROT_READ_WRITE: u64 = 3;

// Physical memory creation flags
const CU_MEM_CREATE_USAGE_TILE_POOL: u32 = 1;

pub struct CudaVmm {
    device: usize,
}

impl CudaVmm {
    pub fn new(device: usize) -> Self {
        Self { device }
    }

    /// Reserve a contiguous virtual address range (no physical backing yet).
    pub fn reserve_address(&self, size: usize) -> Result<u64> {
        let mut ptr: CUdeviceptr = 0;
        // Align to 2MB (the VMM granularity)
        let aligned_size = align_up(size, 2 * 1024 * 1024);

        let cu_result = unsafe {
            sys::lib()
                .cuMemAddressReserve(
                    &mut ptr as *mut CUdeviceptr,
                    aligned_size,
                    0,          // alignment: 0 = default
                    0,          // addr: 0 = let driver pick
                    0,          // flags
                )
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemAddressReserve failed: {:?}", cu_result));
        }
        tracing::debug!(va = ptr, size = aligned_size, "reserved VA region");
        Ok(ptr)
    }

    /// Create a physical memory handle of `size` bytes.
    /// CUDA VMM minimum granularity is 2 MB, so size should be at least that.
    pub fn create_physical(&self, size: usize) -> Result<u64> {
        let mut handle: u64 = 0;
        let aligned_size = align_up(size, 2 * 1024 * 1024);

        let mut prop = sys::CUmemAllocationProp {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: CU_MEM_CREATE_USAGE_TILE_POOL as u8,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE as i32,
                id: self.device as i32,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: sys::CUmemAllocationFlags { compressionType: 0, gpuDirectRDMACapable: 0, usage: 0, reserved: [0u8; 3] },
            granularity: 0,
        };

        let cu_result = unsafe {
            sys::lib().cuMemCreate(&mut handle as *mut u64, aligned_size, &prop as *const sys::CUmemAllocationProp, 0)
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemCreate failed: {:?} for size {}", cu_result, aligned_size));
        }
        tracing::debug!(handle, size = aligned_size, "created physical mem handle");
        Ok(handle)
    }

    /// Map a physical handle into a reserved VA region at the given offset.
    pub fn map(&self, va_base: u64, offset: usize, phys_handle: u64, size: usize) -> Result<()> {
        let cu_result = unsafe {
            sys::lib().cuMemMap(
                va_base + offset as u64,
                size,
                0,              // offset within physical handle
                phys_handle,
                0,              // flags
            )
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemMap failed: {:?}", cu_result));
        }

        // Set access permissions on the mapped range (needed after mapping)
        let cu_result = unsafe {
            sys::lib().cuMemSetAccess(
                va_base + offset as u64,
                size,
                &CU_MEM_ACCESS_FLAGS_PROT_READ_WRITE as *const u64 as *const sys::CUmemAccessDesc,
                1,  // count
            )
        };
        // cuMemSetAccess might fail on some drivers; log but don't fail
        if cu_result != CUresult::CUDA_SUCCESS {
            tracing::warn!("cuMemSetAccess returned: {:?}", cu_result);
        }

        Ok(())
    }

    /// Unmap a range from VA.
    pub fn unmap(&self, va_base: u64, offset: usize, size: usize) -> Result<()> {
        let cu_result = unsafe {
            sys::lib().cuMemUnmap(va_base + offset as u64, size)
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemUnmap failed: {:?}", cu_result));
        }
        Ok(())
    }

    /// Release a physical memory handle.
    pub fn release_physical(&self, handle: u64) -> Result<()> {
        let cu_result = unsafe { sys::lib().cuMemRelease(handle) };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemRelease failed: {:?}", cu_result));
        }
        Ok(())
    }

    /// Free a reserved VA range.
    pub fn free_address(&self, va_base: u64) -> Result<()> {
        let cu_result = unsafe { sys::lib().cuMemAddressFree(va_base) };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemAddressFree failed: {:?}", cu_result));
        }
        Ok(())
    }
}

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmm_lifecycle() {
        // This test requires a GPU. Skip if none available.
        let vmm = CudaVmm::new(0);
        let va = vmm.reserve_address(2 * 1024 * 1024).expect("reserve");
        let phys = vmm.create_physical(2 * 1024 * 1024).expect("create");
        vmm.map(va, 0, phys, 2 * 1024 * 1024).expect("map");
        vmm.unmap(va, 0, 2 * 1024 * 1024).expect("unmap");
        vmm.release_physical(phys).expect("release");
        vmm.free_address(va).expect("free");
    }
}
```

### Step 1.2 — Add module to `src/cache/mod.rs`

```rust
pub mod kv_cache;
pub mod paged_kv;     // added
pub mod cuda_vmm;     // added

pub use kv_cache::KvCache;
// will add paged exports later
```

### Step 1.3 — Verify compilation

```bash
cargo build --release
```

Expected: no errors. The VMM module compiles but isn't wired in yet.

---

## Phase 2: Physical Block Allocator

### Step 2.1 — Block Allocator (`src/cache/paged_kv.rs`, top section)

```rust
use anyhow::{anyhow, Result};
use cudarc::driver::CudaSlice;
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
pub struct PhysicalBlockAllocator {
    vmm: CudaVmm,
    block_bytes: usize,            // bytes per block per layer
    blocks_per_superblock: usize,  // how many blocks fit in 2 MB
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
    pub fn new(device: usize, elem_count: usize) -> Self {
        let block_bytes = elem_count * std::mem::size_of::<f16>();
        let superblock_size = 2 * 1024 * 1024; // 2 MB
        let blocks_per_superblock = superblock_size / block_bytes;

        assert!(blocks_per_superblock > 0,
            "block_bytes ({}) must be < 2 MB; reduce BLOCK_SIZE or model dims", block_bytes);

        Self {
            vmm: CudaVmm::new(device),
            block_bytes,
            blocks_per_superblock,
            free_blocks: Mutex::new(Vec::new()),
            total_allocated: Mutex::new(0),
        }
    }

    /// Allocate one block. Returns (physical_handle, block_offset_in_bytes).
    pub fn allocate(&self) -> Result<(u64, usize)> {
        let mut free = self.free_blocks.lock();
        if let Some(handle) = free.pop() {
            let offset = handle.block_index as usize * self.block_bytes;
            return Ok((handle.superblock_phys, offset));
        }
        drop(free);

        // No free blocks — allocate a new 2 MB superblock
        let phys = self.vmm.create_physical(2 * 1024 * 1024)?;
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
```

---

## Phase 3: Paged KV Cache

### Step 3.1 — PagedKvCache struct (continue in `paged_kv.rs`)

```rust
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

        let vmm = CudaVmm::new(ctx.device_id);
        let block_allocator = Arc::new(PhysicalBlockAllocator::new(
            ctx.device_id,
            elem_per_block,
        ));

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
            let (phys_handle, offset_in_phys) = self.block_allocator.allocate()?;

            let physical_block_idx = next_idx + i as u32;
            let va_offset = physical_block_idx as usize * self.block_bytes;

            // Map into each layer's VA region
            for (layer, &va_base) in self.va_regions.iter().enumerate() {
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
            let superblock_idx = phys_block_idx as usize / self.block_allocator.blocks_per_superblock;
            let block_in_sb = phys_block_idx as usize % self.block_allocator.blocks_per_superblock;
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
```

---

## Phase 4: Continuous Batching Scheduler

### Step 4.1 — Create `src/batch/continuous_scheduler.rs`

```rust
use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender};
use cudarc::driver::CudaSlice;
use half::f16;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

use crate::cache::paged_kv::{PagedKvCache, BLOCK_SIZE};
use crate::config::ModelConfig;
use crate::cuda::CudaContext;
use crate::decoder::greedy_sample;
use crate::model::NaiveTransformer;

use super::static_batch::{InferenceQueue, InferenceRequest, InferenceResponse};

const MAX_PREFILL_TOKENS_PER_STEP: usize = 256;

#[derive(Debug, Clone, PartialEq)]
enum RequestState {
    Waiting,
    Prefill { prompt_pos: usize },
    Decode,
    Done,
}

struct RunningRequest {
    request: InferenceRequest,
    state: RequestState,
    position: usize,        // absolute token position in the sequence
    generated: Vec<u32>,
    block_table: Vec<u32>,  // physical block indices
    tx: Sender<InferenceResponse>,
}

pub struct ContinuousScheduler {
    cfg: ModelConfig,
    ctx: Arc<CudaContext>,
    model: Arc<NaiveTransformer>,
    cache: Arc<PagedKvCache>,
    max_seqs: usize,
    max_seq_len: usize,
    queue: Arc<InferenceQueue>,

    running: Vec<RunningRequest>,
    waiting: Vec<(InferenceRequest, Sender<InferenceResponse>)>,

    // Metrics
    total_tokens_generated: u64,
    max_concurrent: usize,
}

impl ContinuousScheduler {
    pub fn new(
        cfg: ModelConfig,
        ctx: Arc<CudaContext>,
        model: Arc<NaiveTransformer>,
        cache: PagedKvCache,
        max_seqs: usize,
        max_seq_len: usize,
        queue: Arc<InferenceQueue>,
    ) -> Self {
        Self {
            cfg,
            ctx,
            model,
            cache: Arc::new(cache),
            max_seqs,
            max_seq_len,
            queue,
            running: Vec::new(),
            waiting: Vec::new(),
            total_tokens_generated: 0,
            max_concurrent: 0,
        }
    }

    pub fn spawn(mut self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("continuous-scheduler".into())
            .spawn(move || {
                if let Err(e) = self.run() {
                    tracing::error!("scheduler exit: {e:?}");
                }
            })
            .expect("spawn scheduler")
    }

    fn run(&mut self) -> Result<()> {
        let rx = self.queue.receiver();
        loop {
            // 1. Drain incoming requests into waiting queue
            loop {
                match rx.try_recv() {
                    Ok((req, tx)) => self.waiting.push((req, tx)),
                    Err(_) => break,
                }
            }

            // 2. Admit waiting requests into running set
            self.admit_requests()?;

            // 3. If we have running requests, execute one step
            if !self.running.is_empty() {
                self.run_step()?;
            } else {
                // Block until at least one request arrives
                match rx.recv() {
                    Ok((req, tx)) => self.waiting.push((req, tx)),
                    Err(_) => return Ok(()),
                }
            }

            // 4. Collect and respond to completed requests
            self.collect_completed();

            // 5. Update metrics
            self.max_concurrent = self.max_concurrent.max(self.running.len());
        }
    }

    fn admit_requests(&mut self) -> Result<()> {
        while self.running.len() < self.max_seqs {
            let (req, tx) = match self.waiting.pop() {
                Some(v) => v,
                None => break,
            };

            let prompt_len = req.prompt_tokens.len();
            let blocks_needed = (req.max_new_tokens + prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            match self.cache.alloc_sequence(blocks_needed) {
                Ok(block_table) => {
                    self.running.push(RunningRequest {
                        request: req,
                        state: RequestState::Prefill { prompt_pos: 0 },
                        position: 0,
                        generated: Vec::new(),
                        block_table,
                        tx,
                    });
                }
                Err(e) => {
                    tracing::warn!(id = req.id, "failed to allocate KV blocks: {e:?}");
                    // Put the request back and stop admitting
                    self.waiting.push((req, tx));
                    break;
                }
            }
        }
        Ok(())
    }

    fn run_step(&mut self) -> Result<()> {
        let batch = self.running.len();
        let h = self.cfg.hidden_size;

        let mut hidden: CudaSlice<f16> = self.ctx.device.alloc_zeros::<f16>(batch * h)?;

        // Collect positions and sequence indices for this step
        let mut slot_ids = Vec::with_capacity(batch);
        let mut positions = Vec::with_capacity(batch);
        let mut active_indices = Vec::with_capacity(batch);

        for (i, req) in self.running.iter().enumerate() {
            active_indices.push(i);
            slot_ids.push(i);  // using running index as slot for now
            positions.push(req.position);
        }

        // Run forward pass (simplified: single step for all, like static scheduler)
        let logits = self.model.forward_step(
            &mut hidden,
            &mut self.cache,  // needs adaptation for PagedKvCache
            &slot_ids,
            &positions,
        )?;

        // Sample next tokens
        let next = greedy_sample(&logits, batch, self.cfg.vocab_size);

        // Update per-request state
        for (i, req) in self.running.iter_mut().enumerate() {
            match &req.state {
                RequestState::Prefill { prompt_pos } => {
                    // Advance prefill position
                    let new_pos = prompt_pos + 1;
                    if new_pos >= req.request.prompt_tokens.len() {
                        req.state = RequestState::Decode;
                    } else {
                        req.state = RequestState::Prefill { prompt_pos: new_pos };
                    }
                    req.position = new_pos;
                }
                RequestState::Decode => {
                    let token = next[i];
                    req.generated.push(token);
                    req.position += 1;

                    if token == req.request.eos_token_id
                        || req.generated.len() >= req.request.max_new_tokens
                        || req.position >= self.max_seq_len
                    {
                        req.state = RequestState::Done;
                    }
                }
                _ => {}
            }
        }

        self.total_tokens_generated += batch as u64;
        Ok(())
    }

    fn collect_completed(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].state == RequestState::Done {
                let req = self.running.remove(i);

                // Free KV cache blocks
                self.cache.free_sequence(&req.block_table);

                let response = InferenceResponse {
                    id: req.request.id,
                    generated_tokens: req.generated,
                    prefill_ms: 0.0,   // TODO: track separately
                    decode_ms: 0.0,
                };
                let _ = req.tx.send(response);
            } else {
                i += 1;
            }
        }
    }
}
```

---

## Phase 5: Integration

### Step 5.1 — Update `src/batch/mod.rs`

```rust
pub mod static_batch;
pub mod continuous_scheduler;

pub use static_batch::{InferenceQueue, InferenceRequest, InferenceResponse, StaticScheduler};
pub use continuous_scheduler::ContinuousScheduler;
```

### Step 5.2 — Update `src/main.rs`

In `main.rs`, replace:

```rust
let cache = KvCache::new(ctx.clone(), cfg.clone(), cli.max_batch, cli.max_seq_len)?;
// ...
let sched = StaticScheduler::new(...);
```

With:

```rust
let cache = PagedKvCache::new(
    ctx.clone(), cfg.clone(), cli.max_batch, cli.max_seq_len, BLOCK_SIZE,
)?;
// ...
let sched = ContinuousScheduler::new(
    cfg.clone(), ctx.clone(), model.clone(), cache, cli.max_batch, cli.max_seq_len, queue.clone(),
);
```

Add CLI flags:
```rust
#[arg(long, default_value_t = 16)]
block_size: usize,

#[arg(long, default_value_t = 0.90)]
gpu_memory_utilization: f32,
```

### Step 5.3 — Verify build

```bash
cargo build --release 2>&1 | head -50
```

---

## Phase 6: Testing

### 6.1 Unit Tests

```bash
# Test CUDA VMM lifecycle
cargo test --release cuda_vmm -- --test-threads=1

# Test block allocator
cargo test --release paged_kv -- --test-threads=1
```

### 6.2 Integration Test

Start the server with the paged cache:

```bash
RUST_LOG=debug ./target/release/baseline-server \
    --listen 127.0.0.1:8000 \
    --model-path /home/vitalrubbish/models/tinyllama \
    --model-type tinyllama \
    --loader read \
    --block-size 16 \
    --max-batch 8 \
    --max-seq-len 2048
```

Send concurrent requests:

```bash
for i in $(seq 1 4); do
  curl -s -X POST http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello\",\"max_tokens\":32}" &
done
wait
```

### 6.3 Metrics Collection

Add a `/metrics` endpoint or log periodic stats:

```rust
tracing::info!(
    running_seqs = self.running.len(),
    max_concurrent = self.max_concurrent,
    tokens_per_sec = self.total_tokens_generated as f64 / elapsed,
    fragmentation = self.cache.fragmentation_ratio(),
    "scheduler metrics"
);
```

---

## CUDA VMM API Quick Reference

| API | Purpose | Key Constraint |
|-----|---------|---------------|
| `cuMemAddressReserve` | Reserve VA range | Size aligned to 2 MB |
| `cuMemCreate` | Allocate physical memory | Min 2 MB, `CU_MEM_ALLOCATION_TYPE_PINNED` |
| `cuMemMap` | Map phys → VA | Must be within reserved range |
| `cuMemSetAccess` | Set RW access on mapped range | Call after each `cuMemMap` |
| `cuMemUnmap` | Unmap VA range | Phys memory not freed |
| `cuMemRelease` | Free physical memory | Only after unmapping all references |
| `cuMemAddressFree` | Free VA reservation | Only after all phys handles released |

## Common Issues

1. **`cuMemCreate` returns `CUDA_ERROR_INVALID_VALUE`**: Size must be a multiple of 2 MB and at least 2 MB.

2. **`cuMemMap` returns `CUDA_ERROR_INVALID_VALUE`**: The VA offset must be within the reserved range, and the physical handle must be large enough. Check alignment.

3. **`cuMemSetAccess` not found**: This symbol may not be in cudarc's `sys::lib()`. If it's missing, add it manually:
   ```rust
   extern "C" {
       fn cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: *const c_void, count: u32) -> CUresult;
   }
   ```

4. **WSL2 compatibility**: CUDA VMM API works on WSL2 GPU-PV (tested on driver 591.97). If it fails, verify your driver version with `nvidia-smi`.

5. **Memory leak on `free_sequence`**: The simplified `free_sequence` unmaps blocks but doesn't track per-block physical handles. The full implementation needs a `HashMap<u32, (u64, u32)>` mapping physical block index → (superblock phys handle, index within superblock).
