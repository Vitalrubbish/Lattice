// src/cache/cuda_vmm.rs

use anyhow::{anyhow, Context, Result};
use cudarc::driver::sys::{self, CUresult, CUdeviceptr};

pub struct CudaVmm {
    device: usize,
    pub map_granularity: usize,
}

impl CudaVmm {
    /// Create a new VMM handle for the given device ordinal.
    /// Initializes the CUDA driver if not already done.
    pub fn new(device: usize) -> Result<Self> {
        cudarc::driver::result::init()
            .map_err(|e| anyhow!("cuInit failed: {e:?}"))
            .with_context(|| format!("cuda init dev {device}"))?;

        let map_granularity = Self::query_granularity(device)?;

        Ok(Self { device, map_granularity })
    }

    fn query_granularity(device: usize) -> Result<usize> {
        let prop = sys::CUmemAllocationProp {
            type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device as i32,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
                compressionType: 0,
                gpuDirectRDMACapable: 0,
                usage: 0,
                reserved: [0u8; 4],
            },
        };
        let mut granularity: usize = 0;
        let cu_result = unsafe {
            sys::lib().cuMemGetAllocationGranularity(
                &mut granularity as *mut usize,
                &prop as *const sys::CUmemAllocationProp,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemGetAllocationGranularity failed: {:?}", cu_result));
        }
        tracing::info!(granularity, device, "queried VMM map granularity");
        Ok(granularity)
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

        let prop = sys::CUmemAllocationProp {
            type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device as i32,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
                compressionType: 0,
                gpuDirectRDMACapable: 0,
                usage: 0,
                reserved: [0u8; 4],
            },
        };

        let cu_result = unsafe {
            sys::lib().cuMemCreate(
                &mut handle as *mut u64,
                aligned_size,
                &prop as *const sys::CUmemAllocationProp,
                0,
            )
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!(
                "cuMemCreate failed: {:?} for size {}",
                cu_result,
                aligned_size
            ));
        }
        tracing::debug!(handle, size = aligned_size, "created physical mem handle");
        Ok(handle)
    }

    /// Map a range of a physical handle into a reserved VA region.
    ///
    /// `va_offset` = byte offset within the VA region.
    /// `phys_offset` = byte offset within the physical handle (for sub-allocation).
    pub fn map(
        &self,
        va_base: u64,
        va_offset: usize,
        phys_handle: u64,
        phys_offset: usize,
        size: usize,
    ) -> Result<()> {
        let cu_result = unsafe {
            sys::lib().cuMemMap(
                va_base + va_offset as u64,
                size,
                phys_offset,
                phys_handle,
                0,              // flags
            )
        };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemMap failed: {:?}", cu_result));
        }

        // Set access permissions on the mapped range (needed after mapping)
        let desc = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device as i32,
            },
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let cu_result = unsafe {
            sys::lib().cuMemSetAccess(
                va_base + va_offset as u64,
                size,
                &desc as *const sys::CUmemAccessDesc,
                1, // count
            )
        };
        // cuMemSetAccess might fail on some drivers; log but don't fail
        if cu_result != CUresult::CUDA_SUCCESS {
            tracing::warn!("cuMemSetAccess returned: {:?}", cu_result);
        }

        Ok(())
    }

    /// Unmap a range from VA.
    pub fn unmap(&self, va_base: u64, va_offset: usize, size: usize) -> Result<()> {
        let cu_result = unsafe {
            sys::lib().cuMemUnmap(va_base + va_offset as u64, size)
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
    pub fn free_address(&self, va_base: u64, size: usize) -> Result<()> {
        let cu_result = unsafe { sys::lib().cuMemAddressFree(va_base, size) };
        if cu_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemAddressFree failed: {:?}", cu_result));
        }
        Ok(())
    }

    /// Batch-map multiple blocks (same physical handle, different offsets).
    /// Maps each block into a separate layer's K and V VA region.
    pub fn batch_map_blocks(
        &self,
        va_k: &[u64],
        va_v: &[u64],
        va_offset: usize,
        phys_handle: u64,
        block_offsets: &[usize],
        block_bytes: usize,
    ) -> Result<()> {
        for &offset in block_offsets {
            for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                self.map(vk, va_offset, phys_handle, offset, block_bytes)?;
                self.map(vv, va_offset, phys_handle, offset, block_bytes)?;
            }
        }
        Ok(())
    }

    /// Batch-unmap blocks from all layers.
    pub fn batch_unmap_blocks(
        &self,
        va_k: &[u64],
        va_v: &[u64],
        va_offsets: &[usize],
        block_bytes: usize,
    ) -> Result<()> {
        for &offset in va_offsets {
            for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                self.unmap(vk, offset, block_bytes)?;
                self.unmap(vv, offset, block_bytes)?;
            }
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
        let vmm = CudaVmm::new(0).expect("cuda init");
        let va = vmm.reserve_address(2 * 1024 * 1024).expect("reserve");
        let phys = vmm.create_physical(2 * 1024 * 1024).expect("create");
        vmm.map(va, 0, phys, 0, 2 * 1024 * 1024).expect("map");
        vmm.unmap(va, 0, 2 * 1024 * 1024).expect("unmap");
        vmm.release_physical(phys).expect("release");
        vmm.free_address(va, 2 * 1024 * 1024).expect("free");
    }
}