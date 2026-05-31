use std::sync::Arc;
use baseline_llm_os::config::ModelConfig;
use baseline_llm_os::cuda::CudaContext;
use baseline_llm_os::model::{LoaderKind, ModelLoader, LlamaTransformer, Transformer};
use baseline_llm_os::cache::KvCache;
use half::f16;

fn model_path() -> String {
    std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("MODEL_PATH env var not set, defaulting to ./models/tinyllama");
        "./models/tinyllama".to_string()
    })
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    
    let ctx = Arc::new(CudaContext::new(0)?);
    let cfg = ModelConfig::tiny_llama();
    
    let (weights, _) = ModelLoader::new(&ctx, &cfg, LoaderKind::Mmap)
        .load(&model_path())?;
    tracing::info!("weights loaded");
    
    let model = LlamaTransformer::new(ctx.clone(), cfg.clone(), weights)?;
    tracing::info!("model ready");
    
    let batch = 1;
    let h = cfg.hidden_size;
    let mut cache = KvCache::new(ctx.clone(), cfg.clone(), 4, 64)?;
    let mut hidden = ctx.device.alloc_zeros::<f16>(batch * h)?;
    
    let logits = model.forward_step(
        &mut hidden, &mut cache,
        &[0], &[1], &[0],
    )?;
    
    let sample = &logits[..10.min(logits.len())];
    tracing::info!(?sample, "logits");
    
    let argmax = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i).unwrap_or(0);
    tracing::info!(argmax, "argmax");
    
    Ok(())
}
