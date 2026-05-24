use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use crate::batch::{InferenceQueue, InferenceRequest};

#[derive(Debug, Deserialize)]
pub struct WireRequest {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
}
fn default_eos() -> u32 {
    2
}

#[derive(Debug, Serialize)]
pub struct WireResponse {
    pub id: u64,
    pub generated_tokens: Vec<u32>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

pub async fn serve_http(addr: &str, queue: Arc<InferenceQueue>) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");

    loop {
        let (sock, peer) = listener.accept().await?;
        let queue = queue.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(sock, peer.to_string(), queue).await {
                tracing::warn!(%peer, "conn err: {e:?}");
            }
        });
    }
}

async fn handle(
    sock: tokio::net::TcpStream,
    _peer: String,
    queue: Arc<InferenceQueue>,
) -> Result<()> {
    let (read, mut write) = sock.into_split();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }
    let req: WireRequest = serde_json::from_str(line.trim())?;
    let infer = InferenceRequest {
        id: req.id,
        prompt_tokens: req.prompt_tokens,
        max_new_tokens: req.max_new_tokens,
        eos_token_id: req.eos_token_id,
    };
    let resp = tokio::task::spawn_blocking(move || queue.submit_blocking(infer)).await??;
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
