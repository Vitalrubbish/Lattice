use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::f16;
use std::sync::Arc;
use std::time::Duration;

use crate::cache::KvCache;
use crate::cache::paged_kv::PagedKvCache;
use crate::config::ModelConfig;
use crate::cuda::{runtime::Blas, CudaContext};

use super::weights::ModelWeights;

pub struct NaiveTransformer {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub blas: Blas,
    layers: Vec<CudaSlice<f16>>,
    lm_head: CudaSlice<f16>,
}

impl NaiveTransformer {
    pub fn new(ctx: Arc<CudaContext>, cfg: ModelConfig, _w: &ModelWeights) -> Result<Self> {
        let blas = Blas::new(ctx.device.clone())?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(ctx.device.alloc_zeros::<f16>(cfg.hidden_size * cfg.hidden_size)?);
        }
        let lm_head = ctx.device.alloc_zeros::<f16>(cfg.hidden_size * cfg.vocab_size)?;
        Ok(Self { cfg, ctx, blas, layers, lm_head })
    }

    pub fn forward_step(
        &self,
        hidden: &mut CudaSlice<f16>,
        cache: &mut KvCache,
        slot_ids: &[usize],
        positions: &[usize],
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let batch = slot_ids.len();
        assert_eq!(positions.len(), batch);

        let mut tmp = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
        for (i, w) in self.layers.iter().enumerate() {
            self.blas.hgemm(hidden, w, &mut tmp, batch as i32, h as i32, h as i32)?;
            std::mem::swap(hidden, &mut tmp);
            cache.append_step(i, slot_ids, positions, hidden)?;
        }

        let mut logits = self.ctx.device.alloc_zeros::<f16>(batch * self.cfg.vocab_size)?;
        self.blas.hgemm(
            hidden, &self.lm_head, &mut logits,
            batch as i32, self.cfg.vocab_size as i32, h as i32,
        )?;

        let mut out = vec![f16::from_f32(0.0); batch * self.cfg.vocab_size];
        self.ctx.d2h_sync(&logits, &mut out)?;
        Ok(out.iter().map(|x| x.to_f32()).collect())
    }

    pub fn prefill(
        &self,
        hidden: &mut CudaSlice<f16>,
        cache: &mut KvCache,
        slot_ids: &[usize],
        seq_len: usize,
    ) -> Result<Duration> {
        let t = std::time::Instant::now();
        let batch = slot_ids.len();
        for pos in 0..seq_len {
            let positions = vec![pos; batch];
            self.forward_step(hidden, cache, slot_ids, &positions)?;
        }
        Ok(t.elapsed())
    }

    /// Paged KV cache variant: one forward step for all layers.
    /// `seq_indices` are indices into PagedKvCache.seq_metadata.
    pub fn forward_step_paged(
        &self,
        hidden: &mut CudaSlice<f16>,
        cache: &PagedKvCache,
        seq_indices: &[usize],
        positions: &[usize],
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let batch = seq_indices.len();
        assert_eq!(positions.len(), batch);

        let mut tmp = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
        for (i, w) in self.layers.iter().enumerate() {
            self.blas.hgemm(hidden, w, &mut tmp, batch as i32, h as i32, h as i32)?;
            std::mem::swap(hidden, &mut tmp);
            cache.append_step(i, seq_indices, positions, hidden)?;
        }

        let mut logits = self.ctx.device.alloc_zeros::<f16>(batch * self.cfg.vocab_size)?;
        self.blas.hgemm(
            hidden, &self.lm_head, &mut logits,
            batch as i32, self.cfg.vocab_size as i32, h as i32,
        )?;

        let mut out = vec![f16::from_f32(0.0); batch * self.cfg.vocab_size];
        self.ctx.d2h_sync(&logits, &mut out)?;
        Ok(out.iter().map(|x| x.to_f32()).collect())
    }

    /// Paged KV cache variant: prefill for multiple prompt tokens.
    pub fn prefill_paged(
        &self,
        hidden: &mut CudaSlice<f16>,
        cache: &PagedKvCache,
        seq_indices: &[usize],
        seq_len: usize,
    ) -> Result<Duration> {
        let t = std::time::Instant::now();
        let batch = seq_indices.len();
        for pos in 0..seq_len {
            let positions = vec![pos; batch];
            self.forward_step_paged(hidden, cache, seq_indices, &positions)?;
        }
        Ok(t.elapsed())
    }
}
