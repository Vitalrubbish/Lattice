use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::f16;
use std::sync::Arc;
use std::time::Duration;

use crate::cache::KvCache;
use crate::cache::paged_kv::PagedKvCache;
use crate::config::ModelConfig;
use crate::cuda::kernels::{
    GpuKernels, launch_add, launch_contig_attn_decode, launch_paged_attn_decode, launch_rms_norm,
    launch_rope, launch_silu_mul,
};
use crate::cuda::{runtime::Blas, CudaContext};
use crate::model::weights::{ModelWeights, RawTensor};

use super::transformer::Transformer;

struct TensorRef { ptr: u64, len: usize }

struct LayerWeights {
    input_layernorm: TensorRef,
    q_proj: TensorRef,
    k_proj: TensorRef,
    v_proj: TensorRef,
    o_proj: TensorRef,
    post_attention_layernorm: TensorRef,
    gate_proj: TensorRef,
    up_proj: TensorRef,
    down_proj: TensorRef,
}

pub struct LlamaTransformer {
    pub cfg: ModelConfig,
    pub ctx: Arc<CudaContext>,
    pub blas: Blas,
    kernels: GpuKernels,
    _weights: ModelWeights,
    layers: Vec<LayerWeights>,
    norm: TensorRef,
    lm_head: TensorRef,
    head_dim: usize,
    kv_heads: usize,
    num_heads: usize,
    kv_head_dim: usize,
}

fn check_dtype(dtype: &str) -> Result<()> {
    let lower = dtype.to_lowercase();
    if lower.contains("f16") || lower.contains("float16") || lower.contains("bf16") {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Weight dtype '{dtype}' not supported. Only F16/BF16."))
    }
}

impl LlamaTransformer {
    pub fn new(ctx: Arc<CudaContext>, cfg: ModelConfig, weights: ModelWeights) -> Result<Self> {
        let blas = Blas::new(ctx.device.clone())?;
        let kernels = GpuKernels::compile(&ctx.device)?;

        // Convert BF16 → F16 on CPU
        let mut num_converted = 0usize;
        for (_name, tensor) in weights.tensors.iter() {
            if tensor.dtype.to_lowercase().contains("bf16") {
                let nelems = tensor.num_elements();
                let mut host = vec![0u16; nelems];
                unsafe { cudarc::driver::sys::lib().cuMemcpyDtoHAsync_v2(host.as_mut_ptr() as _, tensor.device_ptr(), nelems * 2, std::ptr::null_mut()); }
                ctx.synchronize()?;
                let f16_host: Vec<u16> = host.iter().map(|&w| half::f16::from_f32(f32::from_bits((w as u32) << 16)).to_bits()).collect();
                unsafe { cudarc::driver::sys::lib().cuMemcpyHtoDAsync_v2(tensor.device_ptr(), f16_host.as_ptr() as _, nelems * 2, std::ptr::null_mut()); }
                num_converted += 1;
            }
        }
        if num_converted > 0 { ctx.synchronize()?; tracing::info!(num_converted, "BF16→F16 conversion complete (CPU)"); }

        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let kv_heads = cfg.kv_heads();
        let kv_head_dim = kv_heads * head_dim;

        check_dtype(&weights.try_get("model.embed_tokens.weight")?.dtype)?;
        check_dtype(&weights.try_get("model.norm.weight")?.dtype)?;
        check_dtype(&weights.try_get("lm_head.weight")?.dtype)?;

        let norm = Self::tref(weights.try_get("model.norm.weight")?);
        let lm_head = Self::tref(weights.try_get("lm_head.weight")?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
                input_layernorm: Self::tref(weights.try_layer(l, "input_layernorm.weight")?),
                q_proj: Self::tref(weights.try_layer(l, "self_attn.q_proj.weight")?),
                k_proj: Self::tref(weights.try_layer(l, "self_attn.k_proj.weight")?),
                v_proj: Self::tref(weights.try_layer(l, "self_attn.v_proj.weight")?),
                o_proj: Self::tref(weights.try_layer(l, "self_attn.o_proj.weight")?),
                post_attention_layernorm: Self::tref(weights.try_layer(l, "post_attention_layernorm.weight")?),
                gate_proj: Self::tref(weights.try_layer(l, "mlp.gate_proj.weight")?),
                up_proj: Self::tref(weights.try_layer(l, "mlp.up_proj.weight")?),
                down_proj: Self::tref(weights.try_layer(l, "mlp.down_proj.weight")?),
            });
        }

        Ok(Self { cfg: cfg.clone(), ctx, blas, kernels, _weights: weights, layers, norm, lm_head, head_dim, kv_heads, num_heads, kv_head_dim })
    }

    fn tref(t: &RawTensor) -> TensorRef { TensorRef { ptr: t.device_ptr(), len: t.num_elements() } }
    fn h(&self) -> usize { self.cfg.hidden_size }
    fn vocab(&self) -> usize { self.cfg.vocab_size }
    fn interm(&self) -> usize { self.cfg.intermediate_size }

    fn w_slice(&self, tr: &TensorRef) -> CudaSlice<f16> {
        unsafe { self.ctx.device.upgrade_device_ptr::<f16>(tr.ptr, tr.len) }
    }

    fn proj(&self, inp: &CudaSlice<f16>, w: &TensorRef, out: &mut CudaSlice<f16>, m: i32, n: i32, k: i32) -> Result<()> {
        let tw = self.w_slice(w);
        let r = self.blas.hgemm(inp, &tw, out, m, n, k);
        tw.leak();
        r
    }

    fn d2d(&self, dst: &mut CudaSlice<f16>, src: &CudaSlice<f16>, elems: usize) -> Result<()> {
        unsafe {
            let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(*dst.device_ptr(), *src.device_ptr(), elems * 2, std::ptr::null_mut());
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(anyhow::anyhow!("d2d: {r:?}")); }
        }
        Ok(())
    }

    fn slice_from(&self, base: u64, off: usize, len: usize) -> CudaSlice<f16> {
        unsafe { self.ctx.device.upgrade_device_ptr::<f16>(base.wrapping_add((off * 2) as u64), len) }
    }

    fn forward_with_cache(&self, hidden: &mut CudaSlice<f16>, batch: usize, slots: &[usize], positions: &[usize],
                          k_layers: &[CudaSlice<f16>], v_layers: &[CudaSlice<f16>], k_stride: usize) -> Result<Vec<f32>> {
        let h = self.h(); let hd = self.head_dim; let nheads = self.num_heads;
        let kvh = self.kv_heads; let kvd = self.kv_head_dim;

        for li in 0..self.cfg.num_hidden_layers {
            let lw = &self.layers[li];
            let k_cache = &k_layers[li];
            let v_cache = &v_layers[li];

            // input layernorm + residual
            let mut residual = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.d2d(&mut residual, hidden, batch * h)?;

            let mut normed = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            { let tw = self.w_slice(&lw.input_layernorm); launch_rms_norm(&self.kernels.rms_norm, hidden, &tw, &mut normed, batch, h, 1e-6)?; tw.leak(); }

            // QKV projections
            let mut q = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.proj(&normed, &lw.q_proj, &mut q, batch as i32, h as i32, h as i32)?;
            let mut k = self.ctx.device.alloc_zeros::<f16>(batch * kvd)?;
            self.proj(&normed, &lw.k_proj, &mut k, batch as i32, kvd as i32, h as i32)?;
            let mut v = self.ctx.device.alloc_zeros::<f16>(batch * kvd)?;
            self.proj(&normed, &lw.v_proj, &mut v, batch as i32, kvd as i32, h as i32)?;

            // RoPE
            for b in 0..batch {
                let pos = positions[b];
                let mut qb = self.slice_from(*q.device_ptr(), b * h, nheads * hd);
                let mut kb = self.slice_from(*k.device_ptr(), b * kvd, kvh * hd);
                launch_rope(&self.kernels.rope, &mut qb, &mut kb, 1, nheads, kvh, hd / 2, pos)?;
                qb.leak(); kb.leak();
            }

            // Write K,V to cache
            for b in 0..batch {
                let slot = slots[b]; let pos = positions[b];
                let dst_off = slot * k_stride + pos * kvd;
                let src_off = b * kvd;
                let kp: u64 = *k.device_ptr(); let vp: u64 = *v.device_ptr();
                let kcp: u64 = *k_cache.device_ptr(); let vcp: u64 = *v_cache.device_ptr();
                unsafe { cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(kcp.wrapping_add((dst_off*2) as u64), kp.wrapping_add((src_off*2) as u64), kvd*2, std::ptr::null_mut()); }
                unsafe { cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(vcp.wrapping_add((dst_off*2) as u64), vp.wrapping_add((src_off*2) as u64), kvd*2, std::ptr::null_mut()); }
            }

            // Attention
            let attn = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            for b in 0..batch {
                let slot = slots[b]; let pos = positions[b];
                if pos == 0 { continue; }
                let seq_len = pos + 1;
                let cache_off = slot * k_stride;
                let cache_len = kvh * seq_len * hd;
                let kcp: u64 = *k_cache.device_ptr(); let vcp: u64 = *v_cache.device_ptr();
                let qb = self.slice_from(*q.device_ptr(), b * h, nheads * hd);
                let kc = self.slice_from(kcp, cache_off, cache_len);
                let vc = self.slice_from(vcp, cache_off, cache_len);
                let mut ab = self.slice_from(*attn.device_ptr(), b * h, nheads * hd);
                launch_contig_attn_decode(&self.kernels.contig_attn_decode, &qb, &kc, &vc, &mut ab, 1, nheads, kvh, hd, seq_len)?;
                qb.leak(); kc.leak(); vc.leak(); ab.leak();
            }

            // O proj + residual
            let mut attn_out = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.proj(&attn, &lw.o_proj, &mut attn_out, batch as i32, h as i32, h as i32)?;
            self.d2d(hidden, &residual, batch * h)?;
            launch_add(&self.kernels.add_kernel, hidden, &attn_out, batch * h)?;

            // post_attention_layernorm
            { let tw = self.w_slice(&lw.post_attention_layernorm); launch_rms_norm(&self.kernels.rms_norm, hidden, &tw, &mut normed, batch, h, 1e-6)?; tw.leak(); }

            // FFN residual save
            self.d2d(&mut residual, hidden, batch * h)?;

            // SwiGLU
            let im = self.interm();
            let mut gate = self.ctx.device.alloc_zeros::<f16>(batch * im)?;
            self.proj(&normed, &lw.gate_proj, &mut gate, batch as i32, im as i32, h as i32)?;
            let mut up = self.ctx.device.alloc_zeros::<f16>(batch * im)?;
            self.proj(&normed, &lw.up_proj, &mut up, batch as i32, im as i32, h as i32)?;
            let mut silu_out = self.ctx.device.alloc_zeros::<f16>(batch * im)?;
            launch_silu_mul(&self.kernels.silu_mul, &gate, &up, &mut silu_out, batch * im)?;
            let mut ffn_out = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.proj(&silu_out, &lw.down_proj, &mut ffn_out, batch as i32, h as i32, im as i32)?;
            self.d2d(hidden, &residual, batch * h)?;
            launch_add(&self.kernels.add_kernel, hidden, &ffn_out, batch * h)?;
        }

        // Final RMS norm + LM head
        let mut final_normed = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
        { let tw = self.w_slice(&self.norm); launch_rms_norm(&self.kernels.rms_norm, hidden, &tw, &mut final_normed, batch, h, 1e-6)?; tw.leak(); }
        let vocab = self.vocab();
        let mut logits = self.ctx.device.alloc_zeros::<f16>(batch * vocab)?;
        self.proj(&final_normed, &self.lm_head, &mut logits, batch as i32, vocab as i32, h as i32)?;
        let mut out = vec![f16::from_f32(0.0); batch * vocab];
        self.ctx.d2h_sync(&logits, &mut out)?;
        Ok(out.iter().map(|x| x.to_f32()).collect())
    }
}

impl Transformer for LlamaTransformer {
    fn forward_step(&self, hidden: &mut CudaSlice<f16>, cache: &mut KvCache, slot_ids: &[usize], token_ids: &[u32], positions: &[usize]) -> Result<Vec<f32>> {
        let h = self.h();
        let embed_ptr = self._weights.try_get("model.embed_tokens.weight").map(|t| t.device_ptr()).unwrap_or(0);
        let hidden_base: u64 = *hidden.device_ptr();
        for b in 0..token_ids.len() {
            let tid = token_ids[b] as usize;
            let src: u64 = embed_ptr.wrapping_add((tid * h * 2) as u64);
            let dst: u64 = hidden_base.wrapping_add((b * h * 2) as u64);
            unsafe {
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(dst, src, h * 2, std::ptr::null_mut());
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(anyhow::anyhow!("embed: {r:?}")); }
            }
        }
        let stride = cache.cfg.kv_heads() * cache.max_seq_len * cache.cfg.head_dim();
        self.forward_with_cache(hidden, token_ids.len(), slot_ids, positions, &cache.k_layers, &cache.v_layers, stride)
    }

    fn prefill(&self, hidden: &mut CudaSlice<f16>, cache: &mut KvCache, slot_ids: &[usize], token_ids: &[u32], seq_len: usize) -> Result<Duration> {
        let t = std::time::Instant::now();
        let batch = slot_ids.len();
        for pos in 0..seq_len {
            let positions = vec![pos; batch];
            self.forward_step(hidden, cache, slot_ids, token_ids, &positions)?;
        }
        Ok(t.elapsed())
    }

    fn forward_step_paged(&self, hidden: &mut CudaSlice<f16>, cache: &PagedKvCache, seq_indices: &[usize], token_ids: &[u32], positions: &[usize]) -> Result<Vec<f32>> {
        // Embed tokens
        let h = self.h();
        let embed_ptr = self._weights.try_get("model.embed_tokens.weight").map(|t| t.device_ptr()).unwrap_or(0);
        let hidden_base: u64 = *hidden.device_ptr();
        for b in 0..token_ids.len() {
            let tid = token_ids[b] as usize;
            let src: u64 = embed_ptr.wrapping_add((tid * h * 2) as u64);
            let dst: u64 = hidden_base.wrapping_add((b * h * 2) as u64);
            unsafe {
                let r = cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(dst, src, h * 2, std::ptr::null_mut());
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(anyhow::anyhow!("embed: {r:?}")); }
            }
        }

        // Build block offsets array
        let block_offsets_f16 = cache.get_all_block_offsets_f16();
        let offsets_dev = self.ctx.device.htod_copy(block_offsets_f16)?;

        let batch = seq_indices.len();
        let hd = self.head_dim;
        let nheads = self.num_heads;
        let kvh = self.kv_heads;
        let kvd = self.kv_head_dim;
        let block_size = cache.block_size;
        let max_bps = cache.max_blocks_per_seq;

        for li in 0..self.cfg.num_hidden_layers {
            let lw = &self.layers[li];

            // input layernorm + residual
            let mut residual = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.d2d(&mut residual, hidden, batch * h)?;

            let mut normed = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            { let tw = self.w_slice(&lw.input_layernorm); launch_rms_norm(&self.kernels.rms_norm, hidden, &tw, &mut normed, batch, h, 1e-6)?; tw.leak(); }

            // QKV projections
            let mut q = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.proj(&normed, &lw.q_proj, &mut q, batch as i32, h as i32, h as i32)?;
            let mut k = self.ctx.device.alloc_zeros::<f16>(batch * kvd)?;
            self.proj(&normed, &lw.k_proj, &mut k, batch as i32, kvd as i32, h as i32)?;
            let mut v = self.ctx.device.alloc_zeros::<f16>(batch * kvd)?;
            self.proj(&normed, &lw.v_proj, &mut v, batch as i32, kvd as i32, h as i32)?;

            // RoPE
            for b in 0..batch {
                let pos = positions[b];
                let mut qb = self.slice_from(*q.device_ptr(), b * h, nheads * hd);
                let mut kb = self.slice_from(*k.device_ptr(), b * kvd, kvh * hd);
                launch_rope(&self.kernels.rope, &mut qb, &mut kb, 1, nheads, kvh, hd / 2, pos)?;
                qb.leak(); kb.leak();
            }

            // Pre-compute block byte offsets for KV write
            let block_byte_offsets: Vec<u64> = {
                let offsets = cache.get_all_block_offsets_f16();
                offsets.iter().map(|&off_f16| off_f16.wrapping_mul(2)).collect()
            };

            // Write K,V to paged cache
            {
                let va_k = cache.va_k(li);
                let va_v = cache.va_v(li);
                let meta = cache.seq_metadata.lock();
                let k_arr_ptr: u64 = *k.device_ptr();
                let v_arr_ptr: u64 = *v.device_ptr();

                for b in 0..batch {
                    let seq_idx = seq_indices[b];
                    let pos = positions[b];
                    let seq = &meta[seq_idx];
                    let lb = pos / block_size;
                    let off = pos % block_size;
                    let block_idx = seq.block_table[lb] as usize;
                    let va_off = block_byte_offsets.get(block_idx).copied().unwrap_or(0);
                    let dst_off = va_off + (off * kvd * 2) as u64;

                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                            va_k.wrapping_add(dst_off as u64),
                            k_arr_ptr.wrapping_add((b * kvd * 2) as u64),
                            kvd * 2,
                            std::ptr::null_mut(),
                        );
                    }
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(
                            va_v.wrapping_add(dst_off as u64),
                            v_arr_ptr.wrapping_add((b * kvd * 2) as u64),
                            kvd * 2,
                            std::ptr::null_mut(),
                        );
                    }
                }
            }

            // Paged attention
            let mut attn = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            {
                // Build per-sequence block tables and seq_lens on GPU
                let meta = cache.seq_metadata.lock();
                let mut block_tables: Vec<i32> = vec![0i32; batch * max_bps];
                let mut seq_lens: Vec<i32> = vec![0i32; batch];
                for b in 0..batch {
                    let seq_idx = seq_indices[b];
                    let seq = &meta[seq_idx];
                    seq_lens[b] = seq.seq_len as i32;
                    let bt_start = b * max_bps;
                    for (j, &blk) in seq.block_table.iter().enumerate() {
                        if j < max_bps { block_tables[bt_start + j] = blk as i32; }
                    }
                }
                drop(meta);
                let bt_dev = self.ctx.device.htod_copy(block_tables)?;
                let sl_dev = self.ctx.device.htod_copy(seq_lens)?;

                // Create temp slices for VA bases
                let va_k_base: u64 = cache.va_k(li);
                let va_v_base: u64 = cache.va_v(li);
                let va_k_slice = self.slice_from(va_k_base, 0, 1); // just need the pointer
                let va_v_slice = self.slice_from(va_v_base, 0, 1);

                launch_paged_attn_decode(
                    &self.kernels.paged_attn_decode,
                    &q, &mut attn, &bt_dev, &sl_dev, &offsets_dev,
                    &va_k_slice, &va_v_slice,
                    batch, nheads, kvh, hd, block_size, max_bps,
                )?;
                va_k_slice.leak(); va_v_slice.leak();
            }

            // O proj + residual
            let mut attn_out = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.proj(&attn, &lw.o_proj, &mut attn_out, batch as i32, h as i32, h as i32)?;
            self.d2d(hidden, &residual, batch * h)?;
            launch_add(&self.kernels.add_kernel, hidden, &attn_out, batch * h)?;

            // post_attention_layernorm
            { let tw = self.w_slice(&lw.post_attention_layernorm); launch_rms_norm(&self.kernels.rms_norm, hidden, &tw, &mut normed, batch, h, 1e-6)?; tw.leak(); }

            // FFN residual save
            self.d2d(&mut residual, hidden, batch * h)?;

            // SwiGLU
            let im = self.interm();
            let mut gate = self.ctx.device.alloc_zeros::<f16>(batch * im)?;
            self.proj(&normed, &lw.gate_proj, &mut gate, batch as i32, im as i32, h as i32)?;
            let mut up = self.ctx.device.alloc_zeros::<f16>(batch * im)?;
            self.proj(&normed, &lw.up_proj, &mut up, batch as i32, im as i32, h as i32)?;
            let mut silu_out = self.ctx.device.alloc_zeros::<f16>(batch * im)?;
            launch_silu_mul(&self.kernels.silu_mul, &gate, &up, &mut silu_out, batch * im)?;
            let mut ffn_out = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
            self.proj(&silu_out, &lw.down_proj, &mut ffn_out, batch as i32, h as i32, im as i32)?;
            self.d2d(hidden, &residual, batch * h)?;
            launch_add(&self.kernels.add_kernel, hidden, &ffn_out, batch * h)?;
        }

        // Final RMS norm + LM head
        let mut final_normed = self.ctx.device.alloc_zeros::<f16>(batch * h)?;
        { let tw = self.w_slice(&self.norm); launch_rms_norm(&self.kernels.rms_norm, hidden, &tw, &mut final_normed, batch, h, 1e-6)?; tw.leak(); }
        let vocab = self.vocab();
        let mut logits = self.ctx.device.alloc_zeros::<f16>(batch * vocab)?;
        self.proj(&final_normed, &self.lm_head, &mut logits, batch as i32, vocab as i32, h as i32)?;
        let mut out = vec![f16::from_f32(0.0); batch * vocab];
        self.ctx.d2h_sync(&logits, &mut out)?;
        Ok(out.iter().map(|x| x.to_f32()).collect())
    }

    fn prefill_paged(&self, hidden: &mut CudaSlice<f16>, cache: &PagedKvCache, seq_indices: &[usize], token_ids: &[u32], seq_len: usize) -> Result<Duration> {
        let t = std::time::Instant::now();
        let batch = seq_indices.len();
        for pos in 0..seq_len {
            let positions = vec![pos; batch];
            self.forward_step_paged(hidden, cache, seq_indices, token_ids, &positions)?;
        }
        Ok(t.elapsed())
    }
}
