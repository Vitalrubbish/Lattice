// src/cache/cuda_vmm.rs
// cudarc 0.11.9 bindgen incorrectly maps CU_MEM_ALLOCATION_TYPE_PINNED as 1;
// the actual CUDA driver value is 0x04. We transmute the raw value to pass
// the correct discriminant at runtime.
const CU_MEM_ALLOCATION_TYPE_PINNED_RAW: u32 = 0x04;

use anyhow::{anyhow, Result};
use cudarc::driver::sys::{self, CUresult, CUdeviceptr};

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

        let prop = sys::CUmemAllocationProp {
            type_: unsafe {
                // transmute needed: cudarc bindgen says PINNED=1 but driver expects 0x04
                std::mem::transmute::<u32, sys::CUmemAllocationType>(
                    CU_MEM_ALLOCATION_TYPE_PINNED_RAW,
                )
            },
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

    /// Map a physical handle into a reserved VA region at the given offset.
    pub fn map(
        &self,
        va_base: u64,
        offset: usize,
        phys_handle: u64,
        size: usize,
    ) -> Result<()> {
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
        let desc = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device as i32,
            },
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let cu_result = unsafe {
            sys::lib().cuMemSetAccess(
                va_base + offset as u64,
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
    pub fn free_address(&self, va_base: u64, size: usize) -> Result<()> {
        let cu_result = unsafe { sys::lib().cuMemAddressFree(va_base, size) };
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
        vmm.free_address(va, 2 * 1024 * 1024).expect("free");
    }
}