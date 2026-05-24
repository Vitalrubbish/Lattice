use anyhow::{bail, Context, Result};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::ModelConfig;
use crate::cuda::CudaContext;

use super::weights::{ModelWeights, RawTensor};

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
}

pub struct ModelLoader<'a> {
    pub ctx: &'a CudaContext,
    pub cfg: &'a ModelConfig,
    pub kind: LoaderKind,
}

impl<'a> ModelLoader<'a> {
    pub fn new(ctx: &'a CudaContext, cfg: &'a ModelConfig, kind: LoaderKind) -> Self {
        Self { ctx, cfg, kind }
    }

    pub fn load<P: AsRef<Path>>(&self, model_path: P) -> Result<ModelWeights> {
        let path = model_path.as_ref();
        match self.kind {
            LoaderKind::Read => self.load_with_read(path),
            LoaderKind::Mmap => bail!("mmap loader not implemented"),
            LoaderKind::Direct => bail!("O_DIRECT loader not implemented"),
            LoaderKind::Gds => bail!("GDS loader not implemented"),
        }
    }

    fn load_with_read(&self, path: &Path) -> Result<ModelWeights> {
        let files = enumerate_safetensors(path)?;
        let mut weights = ModelWeights::empty(self.cfg);
        let mut total = 0usize;
        let t0 = Instant::now();

        for shard in files {
            let t = Instant::now();
            let bytes = read_whole_file(&shard)?;
            total += bytes.len();
            tracing::info!(
                file = ?shard, n = bytes.len(),
                read_ms = t.elapsed().as_secs_f64() * 1e3,
                "shard read"
            );

            let st = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("parse {shard:?}"))?;

            for (name, view) in st.tensors() {
                let mut dev = self.ctx.alloc_bytes(view.data().len())?;
                self.ctx.h2d_sync(view.data(), &mut dev)?;
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
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        tracing::info!(total, ms, mbps = (total as f64 / 1e6) / (ms / 1e3), "load done");
        Ok(weights)
    }
}

fn enumerate_safetensors(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("readdir {path:?}"))? {
        let p = entry?.path();
        if p.extension().and_then(|s| s.to_str()).map_or(false, |s| s.eq_ignore_ascii_case("safetensors")) {
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
