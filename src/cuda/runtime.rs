use anyhow::{Context, Result};
use cudarc::cublas::{safe::CudaBlas, GemmConfig};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

pub struct Blas {
    inner: CudaBlas,
}

impl Blas {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            inner: CudaBlas::new(device).context("cublas init")?,
        })
    }

    pub fn hgemm(
        &self,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        use cudarc::cublas::safe::Gemm;
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            alpha: half::f16::from_f32(1.0),
            lda: m,
            ldb: k,
            beta: half::f16::from_f32(0.0),
            ldc: m,
        };
        unsafe { self.inner.gemm(cfg, a, b, c)? };
        Ok(())
    }
}
