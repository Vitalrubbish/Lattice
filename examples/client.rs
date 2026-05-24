use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

#[derive(Parser, Debug, Clone)]
struct Cli {
    #[arg(long, default_value = "127.0.0.1:8000")]
    addr: String,
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
    #[arg(long, default_value_t = 64)]
    prompt_len: usize,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
}

#[derive(Serialize)]
struct Req {
    id: u64,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    eos_token_id: u32,
}

#[derive(Deserialize, Debug)]
struct Resp {
    id: u64,
    generated_tokens: Vec<u32>,
    prefill_ms: f64,
    decode_ms: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let t = Instant::now();
    let mut joins = Vec::new();
    for i in 0..cli.concurrency {
        let cli = cli.clone();
        joins.push(tokio::spawn(async move { run_one(i as u64, &cli).await }));
    }
    let mut total = 0usize;
    for j in joins {
        let r = j.await??;
        total += r.generated_tokens.len();
        println!(
            "id={} n={} prefill={:.2} decode={:.2}",
            r.id, r.generated_tokens.len(), r.prefill_ms, r.decode_ms
        );
    }
    let s = t.elapsed().as_secs_f64();
    println!("{} reqs, {} tok, {:.2}s, {:.2} tok/s", cli.concurrency, total, s, total as f64 / s);
    Ok(())
}

async fn run_one(id: u64, cli: &Cli) -> Result<Resp> {
    let prompt: Vec<u32> = (0..cli.prompt_len as u32).collect();
    let req = Req {
        id,
        prompt_tokens: prompt,
        max_new_tokens: cli.max_new_tokens,
        eos_token_id: 999_999,
    };
    let mut sock = TcpStream::connect(&cli.addr).await?;
    let mut body = serde_json::to_vec(&req)?;
    body.push(b'\n');
    sock.write_all(&body).await?;
    let (read, _w) = sock.into_split();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    Ok(serde_json::from_str(line.trim())?)
}
