use anyhow::{Context, Result};
use bytemuck::cast_slice;
use cudarc::driver::{CudaSlice, DeviceSlice};
use half::f16;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

use crate::config::PipelineConfig;
use crate::cuda::CudaContext;

const HEADER_SIZE: usize = 20;
const MAGIC: u32 = 0xC0FFEE42;

pub struct PipelineStage {
    pub cfg: PipelineConfig,
    pub ctx: Arc<CudaContext>,
    pub hidden_size: usize,
}

fn encode_header(buf: &mut [u8; HEADER_SIZE], batch: u64, hidden: u64) {
    buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    buf[4..12].copy_from_slice(&batch.to_le_bytes());
    buf[12..20].copy_from_slice(&hidden.to_le_bytes());
}

fn decode_header(buf: &[u8; HEADER_SIZE]) -> Result<(u64, u64)> {
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    anyhow::ensure!(magic == MAGIC, "bad magic 0x{:08x}", magic);
    let batch = u64::from_le_bytes(buf[4..12].try_into().unwrap());
    let hidden = u64::from_le_bytes(buf[12..20].try_into().unwrap());
    Ok((batch, hidden))
}

impl PipelineStage {
    pub fn new(cfg: PipelineConfig, ctx: Arc<CudaContext>, hidden_size: usize) -> Self {
        Self { cfg, ctx, hidden_size }
    }

    pub fn recv_activation(&self, listener: &TcpListener) -> Result<CudaSlice<f16>> {
        let (mut sock, _peer) = listener.accept().context("accept")?;
        let mut hdr = [0u8; HEADER_SIZE];
        sock.read_exact(&mut hdr)?;
        let (batch, hidden) = decode_header(&hdr)?;

        let elems = (batch as usize) * (hidden as usize);
        let mut host = vec![0u8; elems * std::mem::size_of::<f16>()];
        sock.read_exact(&mut host)?;

        let mut dev = self.ctx.device.alloc_zeros::<f16>(elems)?;
        self.ctx.h2d_sync::<f16>(cast_slice(&host), &mut dev)?;
        Ok(dev)
    }

    pub fn send_activation(
        &self,
        next_addr: &str,
        activation: &CudaSlice<f16>,
        batch: usize,
    ) -> Result<()> {
        let elems = activation.len();
        let hidden = self.hidden_size;
        anyhow::ensure!(elems == batch * hidden, "shape mismatch");

        let mut host = vec![f16::from_f32(0.0); elems];
        self.ctx.d2h_sync(activation, &mut host)?;

        let t = Instant::now();
        let mut sock = TcpStream::connect(next_addr)?;
        sock.set_nodelay(true)?;
        let mut hdr = [0u8; HEADER_SIZE];
        encode_header(&mut hdr, batch as u64, hidden as u64);
        sock.write_all(&hdr)?;
        sock.write_all(cast_slice(&host))?;
        sock.flush()?;
        tracing::debug!(bytes = elems * 2, ms = t.elapsed().as_secs_f64() * 1e3, "send");
        Ok(())
    }
}
