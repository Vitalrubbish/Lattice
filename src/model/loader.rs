use anyhow::{bail, Context, Result};
use safetensors::SafeTensors;
use std::alloc::Layout;
use std::fs::File;
use std::io::Read;
use std::ops::{Deref, DerefMut};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::ModelConfig;
use crate::cuda::CudaContext;

use super::weights::{ModelWeights, RawTensor};

/* ── Loader kind ─────────────────────────────────────────────────── */

#[derive(Debug, Clone, Copy)]
pub enum LoaderKind {
    Read,
    Mmap,
    Direct,
    Gds,
}

impl LoaderKind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "read" => Self::Read,
            "mmap" => Self::Mmap,
            "direct" | "o_direct" => Self::Direct,
            "gds" | "cufile" => Self::Gds,
            other => bail!("unknown loader: {other}"),
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Mmap => "mmap",
            Self::Direct => "direct",
            Self::Gds => "gds",
        }
    }
}

/* ── Load Metrics (for bpftrace cross-validation) ────────────────── */

/// Structured timing breakdown for a single model load.
/// Wall-clock times come from `Instant`; CPU times from `getrusage`.
#[derive(Debug, Clone)]
pub struct LoadMetrics {
    pub loader: String,
    pub total_ms: f64,
    pub read_ms: f64,
    pub parse_ms: f64,
    pub alloc_ms: f64,
    pub h2d_ms: f64,
    pub total_bytes: usize,
    pub cpu_user_ms: f64,
    pub cpu_sys_ms: f64,
}

impl LoadMetrics {
    fn new(kind: LoaderKind) -> Self {
        Self {
            loader: kind.as_str().to_string(),
            total_ms: 0.0,
            read_ms: 0.0,
            parse_ms: 0.0,
            alloc_ms: 0.0,
            h2d_ms: 0.0,
            total_bytes: 0,
            cpu_user_ms: 0.0,
            cpu_sys_ms: 0.0,
        }
    }

    pub fn log(&self) {
        let mbps = if self.total_ms > 0.0 {
            (self.total_bytes as f64 / 1e6) / (self.total_ms / 1e3)
        } else {
            0.0
        };
        tracing::info!(
            loader = %self.loader,
            total_ms = %self.total_ms,
            read_ms = %self.read_ms,
            parse_ms = %self.parse_ms,
            alloc_ms = %self.alloc_ms,
            h2d_ms = %self.h2d_ms,
            cpu_user_ms = %self.cpu_user_ms,
            cpu_sys_ms = %self.cpu_sys_ms,
            total_bytes = %self.total_bytes,
            throughput_mbps = %mbps,
            "load metrics"
        );
    }
}

/// Thin wrapper around `getrusage(2)` for per-thread CPU time.
struct CpuTimer {
    t0: Instant,
    ru0: libc::rusage,
}

impl CpuTimer {
    fn start() -> Self {
        Self {
            t0: Instant::now(),
            ru0: Self::getrusage(),
        }
    }

    /// Returns (wall_ms, user_ms, sys_ms) since `start()`.
    fn elapsed(&self) -> (f64, f64, f64) {
        let wall = self.t0.elapsed().as_secs_f64() * 1e3;
        let now = Self::getrusage();
        let user = Self::ru_delta(&self.ru0, &now, true);
        let sys = Self::ru_delta(&self.ru0, &now, false);
        (wall, user, sys)
    }

    fn getrusage() -> libc::rusage {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut ru) };
        ru
    }

    fn ru_delta(before: &libc::rusage, after: &libc::rusage, user: bool) -> f64 {
        let (a, b) = if user {
            (before.ru_utime, after.ru_utime)
        } else {
            (before.ru_stime, after.ru_stime)
        };
        let a_us = a.tv_sec as f64 * 1e6 + a.tv_usec as f64;
        let b_us = b.tv_sec as f64 * 1e6 + b.tv_usec as f64;
        (b_us - a_us) / 1e3
    }
}

/* ── Aligned buffer for O_DIRECT ─────────────────────────────────── */

/// Heap-allocated buffer aligned to `align` bytes.
/// Uses `std::alloc` with an explicit `Layout` for correct deallocation.
struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    fn new(size: usize, align: usize) -> Self {
        assert!(align.is_power_of_two());
        let layout = Layout::from_size_align(size, align)
            .expect("invalid layout for AlignedBuffer");
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "allocation failed for AlignedBuffer");
        Self { ptr, len: size, layout }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

// SAFETY: AlignedBuffer owns a uniquely-allocated heap block.
unsafe impl Send for AlignedBuffer {}

/* ── Model Loader ────────────────────────────────────────────────── */

pub struct ModelLoader<'a> {
    pub ctx: &'a CudaContext,
    pub cfg: &'a ModelConfig,
    pub kind: LoaderKind,
}

impl<'a> ModelLoader<'a> {
    pub fn new(ctx: &'a CudaContext, cfg: &'a ModelConfig, kind: LoaderKind) -> Self {
        Self { ctx, cfg, kind }
    }

    pub fn load<P: AsRef<Path>>(&self, model_path: P) -> Result<(ModelWeights, LoadMetrics)> {
        let path = model_path.as_ref();
        match self.kind {
            LoaderKind::Read => self.load_with_read(path),
            LoaderKind::Mmap => self.load_with_mmap(path),
            LoaderKind::Direct => self.load_with_direct(path),
            LoaderKind::Gds => self.load_with_gds(path),
        }
    }

    /* ── read(2) + cudaMemcpy baseline ──────────────────────────── */

    fn load_with_read(&self, path: &Path) -> Result<(ModelWeights, LoadMetrics)> {
        let mut metrics = LoadMetrics::new(self.kind);
        let overall = CpuTimer::start();

        let files = enumerate_safetensors(path)?;
        let mut weights = ModelWeights::empty(self.cfg);

        for shard in files {
            /* --- I/O: read whole shard file -------------------- */
            let io = CpuTimer::start();
            let bytes = read_whole_file(&shard)?;
            let (io_wall, io_user, io_sys) = io.elapsed();
            metrics.read_ms += io_wall;
            metrics.cpu_user_ms += io_user;
            metrics.cpu_sys_ms += io_sys;
            metrics.total_bytes += bytes.len();

            tracing::info!(
                file = ?shard,
                n = bytes.len(),
                read_ms = io_wall,
                "shard read"
            );

            /* --- CPU: parse safetensors header ---------------- */
            let parse_timer = CpuTimer::start();
            let st = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("parse {shard:?}"))?;
            let (parse_wall, _parse_user, _parse_sys) = parse_timer.elapsed();
            metrics.parse_ms += parse_wall;

            /* --- GPU: alloc + H→D copy per tensor -------------- */
            for (name, view) in st.tensors() {
                let alloc = CpuTimer::start();
                let mut dev = self.ctx.alloc_bytes(view.data().len())?;
                let (alloc_wall, _au, _as) = alloc.elapsed();
                metrics.alloc_ms += alloc_wall;

                let h2d = CpuTimer::start();
                self.ctx.h2d_sync(view.data(), &mut dev)?;
                let (h2d_wall, _hu, _hs) = h2d.elapsed();
                metrics.h2d_ms += h2d_wall;

                weights.insert(
                    name.to_string(),
                    RawTensor {
                        shape: view.shape().to_vec(),
                        dtype: format!("{:?}", view.dtype()),
                        bytes: dev,
                    },
                );
            }
        }

        self.ctx.synchronize()?;

        let (total_wall, total_user, total_sys) = overall.elapsed();
        metrics.total_ms = total_wall;
        metrics.cpu_user_ms = total_user;
        metrics.cpu_sys_ms = total_sys;

        metrics.log();
        Ok((weights, metrics))
    }

    /* ── mmap(2) + page-fault-driven load ───────────────────────── */

    fn load_with_mmap(&self, path: &Path) -> Result<(ModelWeights, LoadMetrics)> {
        let mut metrics = LoadMetrics::new(self.kind);
        let overall = CpuTimer::start();

        let files = enumerate_safetensors(path)?;
        let mut weights = ModelWeights::empty(self.cfg);

        for shard in files {
            /* --- I/O: mmap the shard file -------------------- */
            let io = CpuTimer::start();
            let file = File::open(&shard)
                .with_context(|| format!("open {shard:?}"))?;
            let file_len = file.metadata()?.len() as usize;

            // SAFETY: the file is kept alive on the stack; Mmap does not
            // outlive it. The mapping is read-only.
            let mmap = unsafe {
                memmap2::Mmap::map(&file)
                    .with_context(|| format!("mmap {shard:?}"))?
            };
            let (io_wall, io_user, io_sys) = io.elapsed();
            metrics.read_ms += io_wall;
            metrics.cpu_user_ms += io_user;
            metrics.cpu_sys_ms += io_sys;
            metrics.total_bytes += file_len;

            tracing::info!(
                file = ?shard,
                n = file_len,
                read_ms = io_wall,
                "shard mmap'd"
            );

            /* --- CPU: parse safetensors header from mmap region -- */
            let parse_timer = CpuTimer::start();
            let st = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("parse {shard:?}"))?;
            let (parse_wall, _pu, _ps) = parse_timer.elapsed();
            metrics.parse_ms += parse_wall;

            /* --- GPU: alloc + H→D copy per tensor -------------- */
            // cudaMemcpy reads from the mmap'd region; each accessed
            // page triggers a major page fault on first touch, which
            // calls filemap_fault → readpage → submit_bio → NVMe.
            for (name, view) in st.tensors() {
                let alloc = CpuTimer::start();
                let mut dev = self.ctx.alloc_bytes(view.data().len())?;
                let (alloc_wall, _au, _as) = alloc.elapsed();
                metrics.alloc_ms += alloc_wall;

                let h2d = CpuTimer::start();
                self.ctx.h2d_sync(view.data(), &mut dev)?;
                let (h2d_wall, _hu, _hs) = h2d.elapsed();
                metrics.h2d_ms += h2d_wall;

                weights.insert(
                    name.to_string(),
                    RawTensor {
                        shape: view.shape().to_vec(),
                        dtype: format!("{:?}", view.dtype()),
                        bytes: dev,
                    },
                );
            }
        }

        self.ctx.synchronize()?;

        let (total_wall, total_user, total_sys) = overall.elapsed();
        metrics.total_ms = total_wall;
        metrics.cpu_user_ms = total_user;
        metrics.cpu_sys_ms = total_sys;

        metrics.log();
        Ok((weights, metrics))
    }

    /* ── O_DIRECT + cudaMemcpy ─────────────────────────────────── */

    fn load_with_direct(&self, path: &Path) -> Result<(ModelWeights, LoadMetrics)> {
        let mut metrics = LoadMetrics::new(self.kind);
        let overall = CpuTimer::start();

        let files = enumerate_safetensors(path)?;
        let mut weights = ModelWeights::empty(self.cfg);
        let align: usize = 4096;

        for shard in files {
            let io = CpuTimer::start();
            let bytes = read_whole_file_direct(&shard, align)
                .with_context(|| format!("O_DIRECT read {shard:?}"))?;
            let (io_wall, io_user, io_sys) = io.elapsed();
            metrics.read_ms += io_wall;
            metrics.cpu_user_ms += io_user;
            metrics.cpu_sys_ms += io_sys;
            metrics.total_bytes += bytes.len();

            tracing::info!(
                file = ?shard,
                n = bytes.len(),
                read_ms = io_wall,
                "shard O_DIRECT read"
            );

            let parse_timer = CpuTimer::start();
            let st = SafeTensors::deserialize(bytes.as_ref())
                .with_context(|| format!("parse {shard:?}"))?;
            let (parse_wall, _pu, _ps) = parse_timer.elapsed();
            metrics.parse_ms += parse_wall;

            for (name, view) in st.tensors() {
                let alloc = CpuTimer::start();
                let mut dev = self.ctx.alloc_bytes(view.data().len())?;
                let (alloc_wall, _au, _as) = alloc.elapsed();
                metrics.alloc_ms += alloc_wall;

                let h2d = CpuTimer::start();
                self.ctx.h2d_sync(view.data(), &mut dev)?;
                let (h2d_wall, _hu, _hs) = h2d.elapsed();
                metrics.h2d_ms += h2d_wall;

                weights.insert(
                    name.to_string(),
                    RawTensor {
                        shape: view.shape().to_vec(),
                        dtype: format!("{:?}", view.dtype()),
                        bytes: dev,
                    },
                );
            }
        }

        self.ctx.synchronize()?;

        let (total_wall, total_user, total_sys) = overall.elapsed();
        metrics.total_ms = total_wall;
        metrics.cpu_user_ms = total_user;
        metrics.cpu_sys_ms = total_sys;

        metrics.log();
        Ok((weights, metrics))
    }

    /* ── GDS (cuFileRead) ──────────────────────────────────────── */

    #[cfg(not(feature = "gds"))]
    fn load_with_gds(&self, _path: &Path) -> Result<(ModelWeights, LoadMetrics)> {
        bail!("GDS loader requires the 'gds' feature. Rebuild with: cargo build --features gds");
    }

    #[cfg(feature = "gds")]
    fn load_with_gds(&self, path: &Path) -> Result<(ModelWeights, LoadMetrics)> {
        use gds_ffi::*;

        let mut metrics = LoadMetrics::new(self.kind);
        let overall = CpuTimer::start();

        // Open the cuFile driver once for all shards.
        let io_init = CpuTimer::start();
        let ret = unsafe { cuFileDriverOpen() };
        if ret != 0 {
            bail!("cuFileDriverOpen failed (check nvidia-fs-dkms): err={ret}");
        }
        let (init_wall, _, _) = io_init.elapsed();
        metrics.read_ms += init_wall;

        let files = enumerate_safetensors(path)?;
        let mut weights = ModelWeights::empty(self.cfg);

        for shard in files {
            let file = File::open(&shard)
                .with_context(|| format!("open {shard:?}"))?;
            let file_len = file.metadata()?.len() as usize;
            metrics.total_bytes += file_len;

            // Register the fd with cuFile.
            let mut desc: CUfileDescr = unsafe { std::mem::zeroed() };
            desc.type_ = CU_FILE_OPEN_FD;
            desc.cookie = file.as_raw_fd() as *mut std::ffi::c_void;
            let mut fh: CUfileHandle = 0;
            let ret = unsafe { cuFileHandleRegister(&mut fh, &mut desc) };
            if ret != 0 {
                bail!("cuFileHandleRegister failed for {shard:?}: err={ret}");
            }

            tracing::info!(file = ?shard, n = file_len, "shard GDS registered");

            // --- 1) Read safetensors header to GPU, then copy back to CPU ---
            // The safetensors header is small (~64 KB).  We do one cuFileRead
            // into a GPU scratch buffer and then D→H copy for parsing.
            let header_io = CpuTimer::start();

            let header_gpu_max = 256 * 1024usize; // 256 KB — generous for large vocab models
            let header_dev = self.ctx.alloc_bytes(header_gpu_max)?;
            let nread = unsafe {
                cuFileRead(
                    fh,
                    header_dev.as_device_ptr(),
                    header_gpu_max, // size to read
                    0,              // file offset
                    0,              // devPtr offset
                )
            };
            if nread < 0 {
                let errno = unsafe { *libc::__errno_location() };
                bail!("cuFileRead header failed for {shard:?}: errno={errno}");
            }
            let actual_header_bytes = (nread as usize).min(header_gpu_max);

            // D→H copy header back for JSON parsing.
            let mut header_cpu = vec![0u8; actual_header_bytes];
            self.ctx.d2h_sync(&header_dev, &mut header_cpu)?;

            let (header_wall, header_user, header_sys) = header_io.elapsed();
            metrics.read_ms += header_wall;
            metrics.cpu_user_ms += header_user;
            metrics.cpu_sys_ms += header_sys;

            // --- 2) Parse safetensors JSON to get tensor offsets ---
            let parse_timer = CpuTimer::start();
            let tensor_metas = parse_safetensors_header(&header_cpu)
                .with_context(|| format!("parse GDS header for {shard:?}"))?;
            let (parse_wall, _, _) = parse_timer.elapsed();
            metrics.parse_ms += parse_wall;

            // --- 3) cuFileRead each tensor directly into GPU VRAM ---
            for meta in &tensor_metas {
                let alloc = CpuTimer::start();
                let dev = self.ctx.alloc_bytes(meta.data_len)?;
                let (alloc_wall, _, _) = alloc.elapsed();
                metrics.alloc_ms += alloc_wall;

                let h2d = CpuTimer::start();
                let nread = unsafe {
                    cuFileRead(
                        fh,
                        dev.as_device_ptr(),
                        meta.data_len,       // size to read
                        meta.data_offset as i64,
                        0,
                    )
                };
                if nread < 0 {
                    let errno = unsafe { *libc::__errno_location() };
                    bail!(
                        "cuFileRead tensor '{}' failed for {shard:?}: errno={errno}",
                        meta.name
                    );
                }
                if nread as usize != meta.data_len {
                    tracing::warn!(
                        tensor = %meta.name,
                        expected = meta.data_len,
                        actual = nread,
                        "GDS short read"
                    );
                }
                let (h2d_wall, _, _) = h2d.elapsed();
                metrics.h2d_ms += h2d_wall;

                weights.insert(
                    meta.name.clone(),
                    RawTensor {
                        shape: meta.shape.clone(),
                        dtype: meta.dtype.clone(),
                        bytes: dev,
                    },
                );
            }

            // Close the cuFile handle for this shard.
            unsafe { cuFileHandleDeregister(fh) };
        }

        self.ctx.synchronize()?;

        unsafe { cuFileDriverClose() };

        let (total_wall, total_user, total_sys) = overall.elapsed();
        metrics.total_ms = total_wall;
        metrics.cpu_user_ms = total_user;
        metrics.cpu_sys_ms = total_sys;

        metrics.log();
        Ok((weights, metrics))
    }
}

/* ── Helpers ─────────────────────────────────────────────────────── */

fn enumerate_safetensors(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("readdir {path:?}"))? {
        let p = entry?.path();
        if p
            .extension()
            .and_then(|s| s.to_str())
            .map_or(false, |s| s.eq_ignore_ascii_case("safetensors"))
        {
            shards.push(p);
        }
    }
    if shards.is_empty() {
        bail!("no .safetensors under {path:?}");
    }
    shards.sort();
    Ok(shards)
}

fn read_whole_file(path: &Path) -> Result<Vec<u8>> {
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut buf = Vec::with_capacity(f.metadata().map(|m| m.len() as usize).unwrap_or(0));
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Read an entire file using O_DIRECT, returning an aligned buffer
/// whose logical length equals the true file size.
///
/// Reads are issued in 2 MiB aligned chunks.  The final partial tail
/// (if the file size is not a multiple of the alignment) is read with
/// a plain `pread` so the caller receives exactly `file_len` valid bytes.
fn read_whole_file_direct(path: &Path, align: usize) -> Result<AlignedBuffer> {
    use std::os::unix::fs::OpenOptionsExt;

    let meta = std::fs::metadata(path)?;
    let file_len = meta.len() as usize;

    // O_DIRECT file handle.
    let fd = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("O_DIRECT open {path:?}"))?;

    // Buffer rounded up to alignment so bulk reads stay aligned.
    let buf_len = (file_len + align - 1) & !(align - 1);
    let mut buf = AlignedBuffer::new(buf_len, align);

    let chunk: usize = 2 * 1024 * 1024; // 2 MiB per read
    let mut written: usize = 0;

    // Bulk: aligned reads via O_DIRECT for all full chunks.
    while written + chunk <= (file_len / align) * align {
        let n = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                buf.as_mut_slice()[written..].as_mut_ptr() as *mut libc::c_void,
                chunk,
                written as i64,
            )
        };
        if n < 0 {
            bail!(
                "O_DIRECT pread error at offset {written}: {}",
                std::io::Error::last_os_error()
            );
        }
        if n == 0 {
            break; // EOF (should not happen since we cap at aligned_len)
        }
        written += n as usize;
    }

    // Tail: if file_len is not a multiple of align, read the last
    // partial block with a plain (non-O_DIRECT) pread so we get the
    // exact remaining bytes.
    let aligned_max = (file_len / align) * align;
    if written < aligned_max {
        // There are still aligned blocks we haven't read — continue.
        let remaining = aligned_max - written;
        let n = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                buf.as_mut_slice()[written..].as_mut_ptr() as *mut libc::c_void,
                remaining,
                written as i64,
            )
        };
        if n < 0 {
            bail!(
                "O_DIRECT pread error at offset {written}: {}",
                std::io::Error::last_os_error()
            );
        }
        written += n as usize;
    }

    // Read any trailing unaligned bytes with a regular fd (no O_DIRECT).
    if file_len > aligned_max {
        let tail_fd = File::open(path)?;
        let tail_off = written;
        let tail_len = file_len - aligned_max;
        let n = unsafe {
            libc::pread(
                tail_fd.as_raw_fd(),
                buf.as_mut_slice()[tail_off..].as_mut_ptr() as *mut libc::c_void,
                tail_len,
                tail_off as i64,
            )
        };
        if n < 0 {
            bail!(
                "tail pread error at offset {tail_off}: {}",
                std::io::Error::last_os_error()
            );
        }
        written += n as usize;
    }

    drop(fd);

    if written != file_len {
        bail!("O_DIRECT read incomplete: got {written}, expected {file_len}");
    }

    // Shrink the logical view to the true file size so safetensors
    // parsing doesn't see garbage in the alignment padding.
    // We allocated buf_len >= file_len, so shrinking the logical len is safe.
    {
        let ptr = buf.ptr;
        let layout = buf.layout;
        std::mem::forget(buf); // don't double-drop
        Ok(AlignedBuffer {
            ptr,
            len: file_len,
            layout,
        })
    }
}

/* ── GDS helpers ────────────────────────────────────────────────── */

#[cfg(feature = "gds")]
use gds_helpers::*;

#[cfg(feature = "gds")]
mod gds_helpers {
    use anyhow::{bail, Context, Result};

    /// Parsed safetensors tensor metadata from the JSON header.
    pub(super) struct GdsTensorMeta {
        pub name: String,
        pub dtype: String,
        pub shape: Vec<usize>,
        pub data_offset: usize,
        pub data_len: usize,
    }

    /// Parse the safetensors JSON header to extract tensor names, dtypes,
    /// shapes, and byte offsets — without needing the full file buffer.
    pub(super) fn parse_safetensors_header(header_bytes: &[u8]) -> Result<Vec<GdsTensorMeta>> {
    if header_bytes.len() < 8 {
        bail!("safetensors header too short: {} bytes", header_bytes.len());
    }

    let header_len = u64::from_le_bytes(header_bytes[..8].try_into().unwrap()) as usize;

    if 8 + header_len > header_bytes.len() {
        bail!(
            "safetensors header truncated: need {} bytes, have {}",
            8 + header_len,
            header_bytes.len()
        );
    }

    let json_bytes = &header_bytes[8..8 + header_len];

    #[derive(serde::Deserialize)]
    struct RawTensorMeta {
        #[serde(default)]
        dtype: Option<String>,
        #[serde(default)]
        shape: Option<Vec<usize>>,
        #[serde(default)]
        data_offsets: Option<[usize; 2]>,
    }

    let map: std::collections::HashMap<String, RawTensorMeta> =
        serde_json::from_slice(json_bytes)
            .with_context(|| "parse safetensors JSON header")?;

    let mut metas = Vec::with_capacity(map.len());
    for (name, raw) in map {
        // Skip metadata entries (e.g. __metadata__) that lack tensor fields.
        let (Some(dtype), Some(shape), Some(data_offsets)) =
            (raw.dtype, raw.shape, raw.data_offsets)
        else {
            continue;
        };
        metas.push(GdsTensorMeta {
            name,
            dtype,
            shape,
            data_offset: data_offsets[0],
            data_len: data_offsets[1] - data_offsets[0],
        });
    }

    // Sort by offset for deterministic iteration order.
    metas.sort_by_key(|m| m.data_offset);
    Ok(metas)
}

} // mod gds_helpers

/* ── GDS FFI bindings ────────────────────────────────────────────── */

#[cfg(feature = "gds")]
mod gds_ffi {
    use cudarc::driver::DevicePtr;
    use std::os::raw::{c_int, c_void};

    pub type CUfileHandle = u64;
    pub type CUfileError = i32;

    pub const CU_FILE_OPEN_FD: c_int = 1;

    #[repr(C)]
    pub struct CUfileDescr {
        pub type_: c_int,
        pub cookie: *mut c_void,
        pub _reserved: [u64; 8],
    }

    // Helpers to get the raw u64 device pointer from a CudaSlice<u8>.
    // cuFileRead expects a CUdeviceptr which is a u64.
    pub trait AsDevicePtr {
        fn as_device_ptr(&self) -> u64;
    }

    impl AsDevicePtr for cudarc::driver::CudaSlice<u8> {
        fn as_device_ptr(&self) -> u64 {
            *self.device_ptr()
        }
    }

    #[link(name = "cufile")]
    extern "C" {
        /// cuFileRead returns ssize_t (bytes read, or -1 on error).
        /// size is an INPUT parameter — how many bytes to read.
        pub fn cuFileRead(
            fh: CUfileHandle,
            devPtr: u64,
            size: usize,
            file_offset: i64,
            devPtr_offset: i64,
        ) -> isize;

        pub fn cuFileHandleRegister(
            fh: *mut CUfileHandle,
            desc: *mut CUfileDescr,
        ) -> CUfileError;

        pub fn cuFileHandleDeregister(fh: CUfileHandle) -> ();

        pub fn cuFileDriverOpen() -> CUfileError;

        pub fn cuFileDriverClose() -> CUfileError;
    }
}
