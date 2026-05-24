use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr};
use std::sync::Arc;

pub mod runtime;

pub struct CudaContext {
    pub device: Arc<CudaDevice>,
}

impl CudaContext {
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal)
            .with_context(|| format!("cuda init dev {ordinal}"))?;
        Ok(Self { device })
    }

    pub fn alloc_bytes(&self, bytes: usize) -> Result<CudaSlice<u8>> {
        Ok(self.device.alloc_zeros::<u8>(bytes)?)
    }

    pub fn h2d_sync<T: DeviceRepr>(&self, host: &[T], dev: &mut CudaSlice<T>) -> Result<()> {
        self.device.htod_sync_copy_into(host, dev)?;
        Ok(())
    }

    pub fn d2h_sync<T: DeviceRepr>(&self, dev: &CudaSlice<T>, host: &mut [T]) -> Result<()> {
        self.device.dtoh_sync_copy_into(dev, host)?;
        Ok(())
    }

    pub fn device_ptr<T>(slice: &CudaSlice<T>) -> u64 {
        *slice.device_ptr()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        Ok(())
    }
}
