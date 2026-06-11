use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use half::f16;
use std::sync::Arc;

pub struct GpuKernels {
    pub rms_norm: CudaFunction,
    pub rope: CudaFunction,
    pub softmax: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add_kernel: CudaFunction,
    pub contig_attn_decode: CudaFunction,
    pub paged_attn_decode: CudaFunction,
    pub bf16_to_f16: CudaFunction,
    pub gather_kv: CudaFunction,
    pub scatter_kv: CudaFunction,
}

impl GpuKernels {
    pub fn compile(device: &Arc<CudaDevice>) -> Result<Self> {
        // Build include paths: always include /usr/include plus CUDA toolkit include
        // from CUDA_HOME or common locations (needed for cuda_fp16.h etc.)
        let mut include_paths = vec!["/usr/include".into()];
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".into());
        include_paths.push(format!("{cuda_home}/include"));
        let opts = cudarc::nvrtc::CompileOptions {
            ftz: Some(true),
            use_fast_math: Some(true),
            include_paths,
            ..Default::default()
        };

        let ptx_data: &[(&str, &str, &[&str])] = &[
            ("rms_norm", include_str!("rms_norm.cu"), &["rms_norm_f16"]),
            ("rope", include_str!("rope.cu"), &["rope_f16"]),
            ("softmax", include_str!("softmax.cu"), &["softmax_f16"]),
            ("silu_mul", include_str!("silu_mul.cu"), &["silu_mul_f16"]),
            ("add", include_str!("add.cu"), &["add_f16"]),
            ("contig_attn_decode", include_str!("contig_attn_decode.cu"), &["contig_attn_decode_f16"]),
            ("paged_attn_decode", include_str!("paged_attn_decode.cu"), &["paged_attn_decode_f16"]),
            ("bf16_to_f16", include_str!("bf16_to_f16.cu"), &["bf16_to_f16"]),
            ("kv_gather", include_str!("kv_gather.cu"), &["gather_kv_layer", "scatter_kv_layer"]),
        ];

        for &(name, src, func_names) in ptx_data {
            let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts.clone())
                .with_context(|| format!("compile NVRTC kernel: {name}"))?;
            device
                .load_ptx(ptx, name, func_names)
                .with_context(|| format!("load kernel module: {name}"))?;
        }

        let get = |mod_name: &str, func_name: &str| -> Result<CudaFunction> {
            device
                .get_func(mod_name, func_name)
                .ok_or_else(|| anyhow::anyhow!("kernel not found: {mod_name}::{func_name}"))
        };

        Ok(Self {
            rms_norm: get("rms_norm", "rms_norm_f16")?,
            rope: get("rope", "rope_f16")?,
            softmax: get("softmax", "softmax_f16")?,
            silu_mul: get("silu_mul", "silu_mul_f16")?,
            add_kernel: get("add", "add_f16")?,
            contig_attn_decode: get("contig_attn_decode", "contig_attn_decode_f16")?,
            paged_attn_decode: get("paged_attn_decode", "paged_attn_decode_f16")?,
            bf16_to_f16: get("bf16_to_f16", "bf16_to_f16")?,
            gather_kv: get("kv_gather", "gather_kv_layer")?,
            scatter_kv: get("kv_gather", "scatter_kv_layer")?,
        })
    }
}

pub fn launch_rms_norm(
    kernel: &CudaFunction,
    x: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let block_dim: u32 = 256;
    let grid_dim = (rows as u32, 1, 1);
    let shared_mem = (block_dim * 4) as u32;
    let cfg = LaunchConfig { grid_dim, block_dim: (block_dim, 1, 1), shared_mem_bytes: shared_mem };
    unsafe { kernel.clone().launch(cfg, (x, weight, out, rows as i32, cols as i32, eps))?; }
    Ok(())
}

pub fn launch_rope(
    kernel: &CudaFunction,
    q: &mut CudaSlice<f16>,
    k: &mut CudaSlice<f16>,
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    half_dim: usize,
    pos: usize,
) -> Result<()> {
    let total_q = batch * num_q_heads * half_dim;
    let total_k = batch * num_kv_heads * half_dim;
    let n = total_q.max(total_k) as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        kernel.clone().launch(cfg, (q, k, batch as i32, num_q_heads as i32, num_kv_heads as i32, half_dim as i32, pos as i32))?;
    }
    Ok(())
}

pub fn launch_softmax(
    kernel: &CudaFunction,
    inp: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let block_dim: u32 = 256;
    let grid_dim = (rows as u32, 1, 1);
    let shared_mem = (block_dim * 4 * 2) as u32;
    let cfg = LaunchConfig { grid_dim, block_dim: (block_dim, 1, 1), shared_mem_bytes: shared_mem };
    unsafe { kernel.clone().launch(cfg, (inp, out, rows as i32, cols as i32))?; }
    Ok(())
}

pub fn launch_silu_mul(
    kernel: &CudaFunction,
    gate: &CudaSlice<f16>,
    up: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    n: usize,
) -> Result<()> {
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { kernel.clone().launch(cfg, (gate, up, out, n as i32))?; }
    Ok(())
}

pub fn launch_add(
    kernel: &CudaFunction,
    a: &mut CudaSlice<f16>,
    b: &CudaSlice<f16>,
    n: usize,
) -> Result<()> {
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { kernel.clone().launch(cfg, (a, b, n as i32))?; }
    Ok(())
}

pub fn launch_contig_attn_decode(
    kernel: &CudaFunction,
    q: &CudaSlice<f16>,
    k: &CudaSlice<f16>,
    v: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    batch: usize,
    num_q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<()> {
    let total = (batch * num_q_heads) as u32;
    let block_dim: u32 = 128;
    let grid_dim = ((total + block_dim - 1) / block_dim, 1, 1);
    let cfg = LaunchConfig { grid_dim, block_dim: (block_dim, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        kernel.clone().launch(cfg, (q, k, v, out, batch as i32, num_q_heads as i32, kv_heads as i32, head_dim as i32, seq_len as i32))?;
    }
    Ok(())
}

pub fn launch_bf16_to_f16(
    kernel: &CudaFunction,
    buf: &mut CudaSlice<f16>,
    n: usize,
) -> Result<()> {
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { kernel.clone().launch(cfg, (buf, n as i32))?; }
    Ok(())
}

pub fn launch_paged_attn_decode(
    kernel: &CudaFunction,
    q: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    block_tables: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    block_offsets_f16: &CudaSlice<u64>,
    va_k: &CudaSlice<f16>,
    va_v: &CudaSlice<f16>,
    batch: usize,
    num_q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    let total = (batch * num_q_heads) as u32;
    let block_dim: u32 = 128;
    let grid_dim = ((total + block_dim - 1) / block_dim, 1, 1);
    let cfg = LaunchConfig { grid_dim, block_dim: (block_dim, 1, 1), shared_mem_bytes: 0 };
    let packed_bs = ((max_blocks_per_seq as i32) << 16) | (block_size as i32);
    unsafe {
        kernel.clone().launch(cfg, (
            q, va_k, va_v, block_tables, seq_lens, block_offsets_f16, out,
            (batch * num_q_heads) as i32, num_q_heads as i32, kv_heads as i32, head_dim as i32, packed_bs,
        ))?;
    }
    Ok(())
}

/// Launch a gather kernel: copy same-layer KV data from N scattered source
/// pointers into a contiguous staging buffer.
///
/// `src_ptrs` is a raw device pointer to an array of `CUdeviceptr` values,
/// each pointing to one block's KV data for a single layer.  `dst` receives
/// the packed data: `[block_0 data][block_1 data]...[block_{N-1} data]`.
/// Accepting a raw `u64` pointer (instead of `&CudaSlice<u64>`) lets
/// callers pass an offset into a larger pre-allocated pointer pool,
/// enabling a single batched H2D for all layers.
pub fn launch_kv_gather(
    kernel: &CudaFunction,
    src_ptrs: u64,
    dst: &CudaSlice<f16>,
    half_count: usize,
    num_blocks: usize,
) -> Result<()> {
    let total = (num_blocks * half_count) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        kernel.clone().launch(
            cfg,
            (src_ptrs, dst, half_count as i32, num_blocks as i32),
        )?;
    }
    Ok(())
}

/// Scatter same-layer KV data from a contiguous source buffer to N scattered
/// destination pointers.  Reverse of [launch_kv_gather].
///
/// `dst_ptrs` is a raw device pointer to an array of `CUdeviceptr` values,
/// each pointing to one block's destination KV slot for a single layer.
pub fn launch_kv_scatter(
    kernel: &CudaFunction,
    src: &CudaSlice<f16>,
    dst_ptrs: u64,
    half_count: usize,
    num_blocks: usize,
) -> Result<()> {
    let total = (num_blocks * half_count) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        kernel.clone().launch(
            cfg,
            (src, dst_ptrs, half_count as i32, num_blocks as i32),
        )?;
    }
    Ok(())
}
