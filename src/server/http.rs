use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use crate::batch::{InferenceQueue, InferenceRequest, StatsHandle};

/// Possible request types in the wire protocol.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WireRequest {
    /// Standard inference request.
    #[serde(rename = "infer")]
    Infer {
        id: u64,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        #[serde(default = "default_eos")]
        eos_token_id: u32,
    },
    /// Stats query — returns current KV cache fragmentation metrics.
    #[serde(rename = "stats")]
    Stats,
    /// Legacy format without a "type" field — treated as inference.
    #[serde(untagged)]
    LegacyInfer {
        id: u64,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        #[serde(default = "default_eos")]
        eos_token_id: u32,
    },
}
fn default_eos() -> u32 {
    2
}

/// Standard inference response.
#[derive(Debug, Serialize)]
pub struct WireResponse {
    pub id: u64,
    pub generated_tokens: Vec<u32>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// Stats response — returned for `{"type":"stats"}` requests.
/// Contains unified fragmentation metrics and raw cache data.
#[derive(Debug, Serialize)]
pub struct WireStatsResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub sample_count: usize,

    // ── Unified Fragmentation Standard metrics ──
    pub internal_frag_rate: f32,
    pub block_utilization: f32,
    pub physical_memory_efficiency: f32,
    pub runtime_frag_index: f32,

    // ── Raw data ──
    pub active_sequences: usize,
    pub blocks_in_use: usize,
    pub total_blocks_allocated: usize,
    pub total_tokens: usize,
    pub ideal_physical_bytes: u64,
    pub actual_physical_bytes: u64,

    // ── Aggregates ──
    pub rfi_avg: f32,
    pub rfi_peak: f32,
    pub rfi_stddev: f32,
}

pub async fn serve_http(addr: &str, queue: Arc<InferenceQueue>, stats: StatsHandle) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");

    loop {
        let (sock, peer) = listener.accept().await?;
        let queue = queue.clone();
        let stats = stats.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(sock, peer.to_string(), queue, stats).await {
                tracing::warn!(%peer, "conn err: {e:?}");
            }
        });
    }
}

async fn handle(
    sock: tokio::net::TcpStream,
    _peer: String,
    queue: Arc<InferenceQueue>,
    stats: StatsHandle,
) -> Result<()> {
    let (read, mut write) = sock.into_split();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }
    let trimmed = line.trim();

    // Try to detect the request type
    // Stats requests have "type":"stats"
    if trimmed.contains("\"stats\"") {
        let snap = stats.snapshot();
        let unified = snap.unified.unwrap_or_default();
        let wire = WireStatsResponse {
            response_type: "stats".into(),
            sample_count: snap.sample_count,
            internal_frag_rate: unified.internal_frag_rate,
            block_utilization: unified.block_utilization,
            physical_memory_efficiency: unified.physical_memory_efficiency,
            runtime_frag_index: unified.runtime_frag_index,
            active_sequences: unified.active_sequences,
            blocks_in_use: unified.blocks_in_use,
            total_blocks_allocated: unified.total_blocks_allocated,
            total_tokens: unified.total_tokens,
            ideal_physical_bytes: unified.ideal_physical_bytes,
            actual_physical_bytes: unified.actual_physical_bytes,
            rfi_avg: snap.rfi_avg,
            rfi_peak: snap.rfi_peak,
            rfi_stddev: snap.rfi_stddev,
        };
        let mut body = serde_json::to_vec(&wire)?;
        body.push(b'\n');
        write.write_all(&body).await?;
        write.shutdown().await?;
        return Ok(());
    }

    // Inference request — parse as legacy format
    let req: WireRequest = serde_json::from_str(trimmed)?;

    let infer_req = match req {
        WireRequest::Infer {
            id,
            prompt_tokens,
            max_new_tokens,
            eos_token_id,
        }
        | WireRequest::LegacyInfer {
            id,
            prompt_tokens,
            max_new_tokens,
            eos_token_id,
        } => InferenceRequest {
            id,
            prompt_tokens,
            max_new_tokens,
            eos_token_id,
        },
        WireRequest::Stats => {
            // Already handled above via string check, but handle gracefully
            let wire = WireStatsResponse {
                response_type: "stats".into(),
                sample_count: 0,
                internal_frag_rate: 0.0,
                block_utilization: 0.0,
                physical_memory_efficiency: 0.0,
                runtime_frag_index: 0.0,
                active_sequences: 0,
                blocks_in_use: 0,
                total_blocks_allocated: 0,
                total_tokens: 0,
                ideal_physical_bytes: 0,
                actual_physical_bytes: 0,
                rfi_avg: 0.0,
                rfi_peak: 0.0,
                rfi_stddev: 0.0,
            };
            let mut body = serde_json::to_vec(&wire)?;
            body.push(b'\n');
            write.write_all(&body).await?;
            write.shutdown().await?;
            return Ok(());
        }
    };

    let resp =
        tokio::task::spawn_blocking(move || queue.submit_blocking(infer_req)).await??;
    let wire = WireResponse {
        id: resp.id,
        generated_tokens: resp.generated_tokens,
        prefill_ms: resp.prefill_ms,
        decode_ms: resp.decode_ms,
    };
    let mut body = serde_json::to_vec(&wire)?;
    body.push(b'\n');
    write.write_all(&body).await?;
    write.shutdown().await?;
    Ok(())
}
