// CUDA stream management for KCMM async data migration.
//
// Provides dedicated CUDA streams for eviction (GPU→CPU), restoration
// (CPU→GPU), and prefetch (speculative CPU→GPU) operations. Using
// separate streams with CU_STREAM_NON_BLOCKING allows these operations
// to overlap with the inference compute stream.

use anyhow::{anyhow, Result};
use cudarc::driver::sys::{self, CUdeviceptr};

/// Wrapper around a CUDA stream for async operations.
pub struct CudaStream {
    pub(crate) inner: sys::CUstream,
}

// CUstream is Send + Sync (it's a pointer-like handle).
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream with the given flags.
    pub fn new(flags: sys::CUstream_flags) -> Result<Self> {
        let mut stream: sys::CUstream = std::ptr::null_mut();
        let cu_result = unsafe {
            sys::lib().cuStreamCreate(&mut stream as *mut sys::CUstream, flags as u32)
        };
        if cu_result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuStreamCreate failed: {:?}", cu_result));
        }
        Ok(Self { inner: stream })
    }

    /// Synchronize the stream — block until all operations complete.
    pub fn synchronize(&self) -> Result<()> {
        let cu_result = unsafe { sys::lib().cuStreamSynchronize(self.inner) };
        if cu_result != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuStreamSynchronize failed: {:?}", cu_result));
        }
        Ok(())
    }

    /// Query whether all operations on the stream have completed.
    pub fn is_done(&self) -> bool {
        let cu_result = unsafe { sys::lib().cuStreamQuery(self.inner) };
        cu_result == sys::CUresult::CUDA_SUCCESS
    }

    /// Asynchronous Device-to-Host memcpy (GPU → CPU) on this stream.
    ///
    /// # Safety
    /// `dst` must point to a valid host buffer of at least `nbytes` bytes.
    /// `src` must be a valid GPU virtual address.
    pub unsafe fn memcpy_d2h_async(
        &self,
        dst: *mut u8,
        src: CUdeviceptr,
        nbytes: usize,
    ) -> Result<()> {
        let r = sys::lib().cuMemcpyDtoHAsync_v2(
            dst as *mut std::ffi::c_void,
            src,
            nbytes,
            self.inner,
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyDtoHAsync failed: {:?}", r));
        }
        Ok(())
    }

    /// Asynchronous Host-to-Device memcpy (CPU → GPU) on this stream.
    ///
    /// # Safety
    /// `src` must point to a valid host buffer of at least `nbytes` bytes.
    /// `dst` must be a valid GPU virtual address.
    pub unsafe fn memcpy_h2d_async(
        &self,
        dst: CUdeviceptr,
        src: *const u8,
        nbytes: usize,
    ) -> Result<()> {
        let r = sys::lib().cuMemcpyHtoDAsync_v2(
            dst,
            src as *const std::ffi::c_void,
            nbytes,
            self.inner,
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyHtoDAsync failed: {:?}", r));
        }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                sys::lib().cuStreamDestroy_v2(self.inner);
            }
        }
    }
}

/// Collection of dedicated CUDA streams for KCMM operations.
///
/// All streams use `CU_STREAM_NON_BLOCKING` so they can overlap
/// with work on the default (inference) stream.
pub struct KcmmStreams {
    /// Stream for GPU→CPU eviction (D2H memcpy, cuMemUnmap).
    pub evict: CudaStream,
    /// Stream for CPU→GPU restoration (cuMemMap, H2D memcpy).
    pub restore: CudaStream,
    /// Stream for speculative prefetch (background H2D).
    pub prefetch: CudaStream,
}

impl KcmmStreams {
    /// Create all three dedicated streams with CU_STREAM_NON_BLOCKING.
    pub fn new() -> Result<Self> {
        let flags = sys::CUstream_flags::CU_STREAM_NON_BLOCKING;
        Ok(Self {
            evict: CudaStream::new(flags)?,
            restore: CudaStream::new(flags)?,
            prefetch: CudaStream::new(flags)?,
        })
    }

    /// Synchronize all three streams.
    pub fn synchronize_all(&self) -> Result<()> {
        self.evict.synchronize()?;
        self.restore.synchronize()?;
        self.prefetch.synchronize()?;
        Ok(())
    }
}
