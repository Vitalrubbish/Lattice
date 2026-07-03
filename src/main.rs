use anyhow::{bail, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

use baseline_llm_os::batch::{CacheBackend, ContinuousScheduler, InferenceQueue, StaticScheduler, StatsHandle};
use baseline_llm_os::cache::paged_kv::PagedKvCache;
use baseline_llm_os::cache::KvCache;
use baseline_llm_os::config::ModelConfig;
#[cfg(feature = "kcmm")]
use baseline_llm_os::config::KcmmConfig;
use baseline_llm_os::cuda::CudaContext;
#[cfg(feature = "kcmm")]
use baseline_llm_os::kcmm::KcmmPool;
use baseline_llm_os::model::{
    LlamaTransformer, LoaderKind, ModelLoader, ModelWeights, NaiveTransformer, Transformer,
};
use baseline_llm_os::server::serve_http;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "0.0.0.0:8000")]
    listen: String,

    #[arg(long)]
    model_path: PathBuf,

    #[arg(long, default_value_t = 8)]
    max_batch: usize,

    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    #[arg(long, default_value = "read")]
    loader: String,

    #[arg(long, default_value_t = 0)]
    device: usize,

    #[arg(long, default_value = "llama7b")]
    model_type: String,

    /// Use continuous batching with paged KV cache (CUDA VMM).
    #[arg(long)]
    continuous: bool,

    /// Use LlamaTransformer with real attention weights instead of NaiveTransformer.
    #[arg(long)]
    llama: bool,

    /// Use KCMM pool with tiering support instead of the baseline PagedKvCache.
    /// Only meaningful when `--continuous` is also specified.
    /// Requires the `kcmm` feature flag.
    #[cfg(feature = "kcmm")]
    #[arg(long)]
    kcmm: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let ctx = Arc::new(CudaContext::new(cli.device)?);
    let cfg = match cli.model_type.as_str() {
        "tinyllama" => ModelConfig::tiny_llama(),
        _ => ModelConfig::llama_7b_like(),
    };

    let kind = LoaderKind::parse(&cli.loader)?;
    let is_dummy = cli.model_path.to_string_lossy() == "dummy";

    if cli.llama && is_dummy {
        bail!("--llama requires real model weights (--model-path), not 'dummy'");
    }

    let weights = if is_dummy {
        ModelWeights::empty(&cfg)
    } else {
        let (w, metrics) = ModelLoader::new(&ctx, &cfg, kind).load(&cli.model_path)?;
        metrics.log();
        w
    };
    tracing::info!(bytes = weights.total_bytes(), "weights ready");

    let model: Arc<dyn Transformer> = if cli.llama {
        if cli.continuous {
            tracing::info!("using LlamaTransformer with continuous batching (paged KV cache)");
        }
        Arc::new(LlamaTransformer::new(ctx.clone(), cfg.clone(), weights)?)
    } else {
        Arc::new(NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights)?)
    };
    let queue = Arc::new(InferenceQueue::new());

    // Create shared stats handle for server ↔ scheduler communication
    let stats_handle = StatsHandle::new();

    if cli.continuous {
        #[cfg(feature = "kcmm")]
        let backend = if cli.kcmm {
            let kcmm_cfg = KcmmConfig::default();
            let pool = KcmmPool::new(
                ctx.clone(),
                kcmm_cfg,
                cfg.num_hidden_layers,
                cfg.kv_heads(),
                cfg.head_dim(),
                cli.max_batch,
                cli.max_seq_len,
            )?;
            tracing::info!("using KCMM pool backend (tiering={})", pool.tiering.is_some());
            CacheBackend::Kcmm(Arc::new(pool))
        } else {
            let cache = PagedKvCache::new(
                ctx.clone(),
                cfg.clone(),
                cli.max_batch,
                cli.max_seq_len,
                16,
            )?;
            tracing::info!("using baseline PagedKvCache backend");
            CacheBackend::Baseline(Arc::new(cache))
        };
        #[cfg(not(feature = "kcmm"))]
        let backend = {
            let cache = PagedKvCache::new(
                ctx.clone(),
                cfg.clone(),
                cli.max_batch,
                cli.max_seq_len,
                16,
            )?;
            tracing::info!("using baseline PagedKvCache backend");
            CacheBackend::Baseline(Arc::new(cache))
        };
        let sched = ContinuousScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model.clone(),
            cli.max_batch,
            cli.max_seq_len,
            queue.clone(),
            stats_handle.clone(),
            backend,
            false, // disable_swap: keep SwapManager for standalone server mode
        );
        let _h = sched.spawn();
    } else {
        let cache = KvCache::new(ctx.clone(), cfg.clone(), cli.max_batch, cli.max_seq_len)?;
        let sched = StaticScheduler::new(
            cfg.clone(),
            ctx.clone(),
            model.clone(),
            cache,
            cli.max_batch,
            cli.max_seq_len,
            queue.clone(),
        );
        let _h = sched.spawn();
    }

    serve_http(&cli.listen, queue, stats_handle).await?;
    Ok(())
}
