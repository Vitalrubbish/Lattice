use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

use baseline_llm_os::batch::{InferenceQueue, StaticScheduler};
use baseline_llm_os::cache::KvCache;
use baseline_llm_os::config::ModelConfig;
use baseline_llm_os::cuda::CudaContext;
use baseline_llm_os::model::{LoaderKind, ModelLoader, ModelWeights, NaiveTransformer};
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
    let weights = if cli.model_path.to_string_lossy() == "dummy" {
        ModelWeights::empty(&cfg)
    } else {
        let (w, metrics) = ModelLoader::new(&ctx, &cfg, kind).load(&cli.model_path)?;
        metrics.log();
        w
    };
    tracing::info!(bytes = weights.total_bytes(), "weights ready");

    let model = Arc::new(NaiveTransformer::new(ctx.clone(), cfg.clone(), &weights)?);
    let cache = KvCache::new(ctx.clone(), cfg.clone(), cli.max_batch, cli.max_seq_len)?;

    let queue = Arc::new(InferenceQueue::new());
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

    serve_http(&cli.listen, queue).await?;
    Ok(())
}
