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

    /// Get the raw CUDA stream handle.
    pub fn as_raw(&self) -> sys::CUstream {
        self.inner
    }

    /// Destroy the stream and clear the handle so `Drop` becomes a no-op.
    pub fn destroy(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                sys::lib().cuStreamDestroy_v2(self.inner);
            }
            self.inner = std::ptr::null_mut();
        }
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
        self.destroy();
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

    /// Explicitly destroy streams while the owning CUDA context is still alive.
    pub fn destroy_all(&mut self) {
        self.evict.destroy();
        self.restore.destroy();
        self.prefetch.destroy();
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;
    use cudarc::driver::CudaSlice;
    use std::sync::Arc;

    /// Allocate zeroed GPU memory (u8) and return both the CudaSlice and raw CUdeviceptr.
    fn alloc_gpu(ctx: &CudaContext, count: usize) -> (CudaSlice<u8>, CUdeviceptr) {
        let slice = ctx.alloc_bytes(count).expect("alloc gpu mem");
        let ptr = CudaContext::device_ptr(&slice) as CUdeviceptr;
        (slice, ptr)
    }

    /// Helper: synchronously copy data from host to GPU (default stream).
    fn h2d_sync(ctx: &CudaContext, host: &[u8], dev: &mut CudaSlice<u8>) {
        ctx.h2d_sync(host, dev).expect("h2d sync");
    }

    /// Helper: synchronously read data from GPU to host (default stream).
    fn d2h_sync(ctx: &CudaContext, dev: &CudaSlice<u8>, host: &mut [u8]) {
        ctx.d2h_sync(dev, host).expect("d2h sync");
    }

    // --- CudaStream lifecycle tests ---

    #[test]
    fn test_stream_new_default() {
        let _ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");
        // Must be able to synchronize an empty stream
        stream.synchronize().expect("sync");
    }

    #[test]
    fn test_stream_new_non_blocking() {
        let _ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_NON_BLOCKING).expect("create stream");
        // Must be able to synchronize
        stream.synchronize().expect("sync");
    }

    #[test]
    fn test_stream_is_done_idle() {
        let _ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");
        // An idle stream should be immediately done
        assert!(stream.is_done(), "idle stream must report done");
    }

    #[test]
    fn test_stream_is_done_after_sync() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        let nbytes = 4096;
        let pattern: Vec<u8> = (0..nbytes).map(|i| (i % 256) as u8).collect();
        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, nbytes);

        // Put data on GPU using default stream
        let mut gpu_mut = gpu_buf;
        h2d_sync(&ctx, &pattern, &mut gpu_mut);

        // Issue async D2H
        let mut host = vec![0u8; nbytes];
        unsafe {
            stream
                .memcpy_d2h_async(host.as_mut_ptr(), gpu_ptr, nbytes)
                .expect("d2h async");
        }

        // Should NOT be done yet (async copy is in-flight or just finished)
        // Note: small copies may finish instantly, so we only check that query doesn't crash
        let _ = stream.is_done();

        stream.synchronize().expect("sync");

        // After synchronize, MUST be done
        assert!(stream.is_done(), "stream must be done after synchronize()");
    }

    #[test]
    fn test_stream_drop_does_not_crash() {
        let _ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");
        drop(stream);
        // If we get here, drop didn't segfault
    }

    #[test]
    fn test_stream_drop_after_async_work() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        let nbytes = 1024;
        let pattern: Vec<u8> = (0..nbytes).map(|i| i as u8).collect();
        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, nbytes);
        let mut gpu_mut = gpu_buf;
        h2d_sync(&ctx, &pattern, &mut gpu_mut);

        let mut host = vec![0u8; nbytes];
        unsafe {
            stream
                .memcpy_d2h_async(host.as_mut_ptr(), gpu_ptr, nbytes)
                .expect("d2h async");
        }

        // Synchronize BEFORE drop — ensures GPU work is complete
        stream.synchronize().expect("sync before drop");
        drop(stream);
    }

    // --- Async memcpy correctness tests ---

    #[test]
    fn test_memcpy_d2h_async_correctness() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        let nbytes = 4096;
        let pattern: Vec<u8> = (0..nbytes)
            .map(|i: usize| (i.wrapping_mul(7).wrapping_add(13)) as u8)
            .collect();

        // Allocate GPU mem and write pattern synchronously (default stream)
        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, nbytes);
        let mut gpu_mut = gpu_buf;
        h2d_sync(&ctx, &pattern, &mut gpu_mut);

        // Async D2H on our test stream
        let mut host = vec![0u8; nbytes];
        unsafe {
            stream
                .memcpy_d2h_async(host.as_mut_ptr(), gpu_ptr, nbytes)
                .expect("d2h async");
        }

        // Synchronize — must complete the copy
        stream.synchronize().expect("sync");

        // Verify data integrity
        assert_eq!(host, pattern, "D2H async memcpy: data mismatch");
    }

    #[test]
    fn test_memcpy_h2d_async_correctness() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        let nbytes = 4096;
        let pattern: Vec<u8> = (0..nbytes)
            .map(|i: usize| (i.wrapping_mul(3).wrapping_add(0xAA)) as u8)
            .collect();

        // Allocate empty GPU buffer
        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, nbytes);

        // Async H2D on our test stream
        unsafe {
            stream
                .memcpy_h2d_async(gpu_ptr, pattern.as_ptr(), nbytes)
                .expect("h2d async");
        }

        // Synchronize
        stream.synchronize().expect("sync");

        // Read back from GPU (sync, default stream)
        let mut readback = vec![0u8; nbytes];
        d2h_sync(&ctx, &gpu_buf, &mut readback);

        assert_eq!(readback, pattern, "H2D async memcpy: data mismatch");
    }

    #[test]
    fn test_memcpy_roundtrip_h2d_then_d2h() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        let nbytes = 8192;
        let pattern: Vec<u8> = (0..nbytes)
            .map(|i| ((i as u64).wrapping_mul(0xDEADBEEF) >> 16) as u8)
            .collect();

        // GPU buffer: write pattern via async H2D
        let (_gpu_a, gpu_ptr_a) = alloc_gpu(&ctx, nbytes);
        unsafe {
            stream
                .memcpy_h2d_async(gpu_ptr_a, pattern.as_ptr(), nbytes)
                .expect("h2d async");
        }
        stream.synchronize().expect("sync after h2d");

        // Read back to host via async D2H
        let mut roundtrip = vec![0u8; nbytes];
        unsafe {
            stream
                .memcpy_d2h_async(roundtrip.as_mut_ptr(), gpu_ptr_a, nbytes)
                .expect("d2h async");
        }
        stream.synchronize().expect("sync after d2h");

        assert_eq!(
            roundtrip, pattern,
            "H2D → D2H roundtrip: data mismatch"
        );
    }

    #[test]
    fn test_memcpy_zero_bytes() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, 1);

        // Zero-byte D2H should succeed without crashing
        let mut host = vec![0u8; 1];
        unsafe {
            stream
                .memcpy_d2h_async(host.as_mut_ptr(), gpu_ptr, 0)
                .expect("d2h async 0 bytes");
        }
        stream.synchronize().expect("sync");

        // Zero-byte H2D should succeed without crashing
        unsafe {
            stream
                .memcpy_h2d_async(gpu_ptr, host.as_ptr(), 0)
                .expect("h2d async 0 bytes");
        }
        stream.synchronize().expect("sync");

        drop(gpu_buf);
    }

    #[test]
    fn test_memcpy_large_buffer() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create stream");

        // 1 MiB — large enough to stress async transfer but not OOM
        let nbytes = 1024 * 1024;
        let pattern: Vec<u8> = (0..nbytes).map(|i| (i % 251) as u8).collect();

        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, nbytes);
        let mut gpu_mut = gpu_buf;
        h2d_sync(&ctx, &pattern, &mut gpu_mut);

        let mut host = vec![0u8; nbytes];
        let start = std::time::Instant::now();
        unsafe {
            stream
                .memcpy_d2h_async(host.as_mut_ptr(), gpu_ptr, nbytes)
                .expect("d2h async large");
        }
        stream.synchronize().expect("sync");
        let elapsed_us = start.elapsed().as_micros();

        assert_eq!(host, pattern, "large D2H async: data mismatch");
        // 1 MiB D2H should complete in well under 1 second
        assert!(
            elapsed_us < 1_000_000,
            "1 MiB D2H took {} µs — expected < 1 s",
            elapsed_us
        );
        println!("1 MiB D2H async latency: {} µs", elapsed_us);
    }

    // --- Multiple-stream non-interference tests ---

    #[test]
    fn test_two_streams_do_not_interfere() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let stream_a =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_NON_BLOCKING).expect("create A");
        let stream_b =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_NON_BLOCKING).expect("create B");

        let nbytes = 4096;

        // Stream A: H2D copy pattern A → gpu_a
        let pattern_a: Vec<u8> = (0..nbytes).map(|_i| 0xAAu8).collect();
        let (gpu_a, gpu_ptr_a) = alloc_gpu(&ctx, nbytes);
        unsafe {
            stream_a
                .memcpy_h2d_async(gpu_ptr_a, pattern_a.as_ptr(), nbytes)
                .expect("h2d A");
        }

        // Stream B: H2D copy pattern B → gpu_b (concurrent with A)
        let pattern_b: Vec<u8> = (0..nbytes).map(|_i| 0xBBu8).collect();
        let (gpu_b, gpu_ptr_b) = alloc_gpu(&ctx, nbytes);
        unsafe {
            stream_b
                .memcpy_h2d_async(gpu_ptr_b, pattern_b.as_ptr(), nbytes)
                .expect("h2d B");
        }

        // Sync both
        stream_a.synchronize().expect("sync A");
        stream_b.synchronize().expect("sync B");

        // Verify A has pattern_a, not pattern_b
        let mut readback_a = vec![0u8; nbytes];
        d2h_sync(&ctx, &gpu_a, &mut readback_a);
        assert_eq!(readback_a, pattern_a, "stream A data corrupted");

        // Verify B has pattern_b, not pattern_a
        let mut readback_b = vec![0u8; nbytes];
        d2h_sync(&ctx, &gpu_b, &mut readback_b);
        assert_eq!(readback_b, pattern_b, "stream B data corrupted");
    }

    // --- KcmmStreams tests ---

    #[test]
    fn test_kcmm_streams_new_creates_all_three() {
        let _ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let streams = KcmmStreams::new().expect("create KcmmStreams");

        // All three streams must be valid (non-null).
        assert!(!streams.evict.inner.is_null(), "evict stream is null");
        assert!(!streams.restore.inner.is_null(), "restore stream is null");
        assert!(!streams.prefetch.inner.is_null(), "prefetch stream is null");

        // All three must be synchronizable individually.
        streams.evict.synchronize().expect("sync evict");
        streams.restore.synchronize().expect("sync restore");
        streams.prefetch.synchronize().expect("sync prefetch");
    }

    #[test]
    fn test_kcmm_streams_synchronize_all() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let streams = KcmmStreams::new().expect("create KcmmStreams");

        let nbytes = 4096;

        // Queue async work on all three streams.
        let pattern: Vec<u8> = (0..nbytes).map(|i| i as u8).collect();
        let (gpu_evict, ptr_evict) = alloc_gpu(&ctx, nbytes);
        let (gpu_restore, ptr_restore) = alloc_gpu(&ctx, nbytes);
        let (gpu_prefetch, ptr_prefetch) = alloc_gpu(&ctx, nbytes);

        let mut gpu_e = gpu_evict;
        let mut gpu_r = gpu_restore;
        let mut gpu_p = gpu_prefetch;
        h2d_sync(&ctx, &pattern, &mut gpu_e);
        h2d_sync(&ctx, &pattern, &mut gpu_r);
        h2d_sync(&ctx, &pattern, &mut gpu_p);

        let mut host_evict = vec![0u8; nbytes];
        let mut host_restore = vec![0u8; nbytes];
        let mut host_prefetch = vec![0u8; nbytes];

        unsafe {
            streams
                .evict
                .memcpy_d2h_async(host_evict.as_mut_ptr(), ptr_evict, nbytes)
                .expect("evict d2h");
            streams
                .restore
                .memcpy_d2h_async(host_restore.as_mut_ptr(), ptr_restore, nbytes)
                .expect("restore d2h");
            streams
                .prefetch
                .memcpy_d2h_async(host_prefetch.as_mut_ptr(), ptr_prefetch, nbytes)
                .expect("prefetch d2h");
        }

        // synchronize_all must wait for all three.
        streams.synchronize_all().expect("synchronize_all");

        assert_eq!(host_evict, pattern, "evict stream data mismatch");
        assert_eq!(host_restore, pattern, "restore stream data mismatch");
        assert_eq!(host_prefetch, pattern, "prefetch stream data mismatch");
    }

    #[test]
    fn test_kcmm_streams_drop_after_work() {
        let ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        let streams = KcmmStreams::new().expect("create KcmmStreams");

        let nbytes = 1024;
        let pattern: Vec<u8> = (0..nbytes).map(|i| i as u8).collect();
        let (gpu_buf, gpu_ptr) = alloc_gpu(&ctx, nbytes);
        let mut gpu_mut = gpu_buf;
        h2d_sync(&ctx, &pattern, &mut gpu_mut);

        let mut host = vec![0u8; nbytes];
        unsafe {
            streams
                .evict
                .memcpy_d2h_async(host.as_mut_ptr(), gpu_ptr, nbytes)
                .expect("d2h");
        }

        // Sync before drop — essential for correctness
        streams.evict.synchronize().expect("sync before drop");
        assert_eq!(host, pattern);
        drop(streams);
    }

    #[test]
    fn test_send_sync_compile_time() {
        let _ctx = Arc::new(CudaContext::new(0).expect("cuda device"));
        // Compile-time assertion: CudaStream and KcmmStreams must be Send + Sync.
        fn assert_send_sync<T: Send + Sync>(_: &T) {}

        let stream =
            CudaStream::new(sys::CUstream_flags::CU_STREAM_DEFAULT).expect("create");
        assert_send_sync(&stream);
        drop(stream);

        let streams = KcmmStreams::new().expect("create streams");
        assert_send_sync(&streams);
        drop(streams);
    }
}
