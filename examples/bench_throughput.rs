// examples/bench_throughput.rs
//
// Throughput benchmark driver for the baseline LLM inference server.
// Uses a realistic prompt-length distribution derived from Shakespeare's sonnets
// (matching vLLM's sonnet benchmark) with synthetic token content.
//
// Usage:
//   cargo run --release --example bench_throughput -- \
//     --addr 127.0.0.1:8000 \
//     --num-requests 100 \
//     --concurrency 4 \
//     --max-new-tokens 32 \
//     --output-bench bench_results.csv

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Prompt length distribution (derived from sonnet.txt, multiple scales) ──
// 145 samples: 60 short (1-20), 40 medium (21-50), 20 med-long (51-80),
// 7 long (81-150), 18 very-long (151-350)
static SONNET_PROMPT_LENS: [usize; 145] = [
    8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13,
    39, 39, 40, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42,
    42, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45,
    46, 46, 46, 46, 46, 47, 47, 48, 48, 50, 72, 72, 73, 73, 73,
    74, 74, 75, 75, 76, 76, 77, 77, 77, 78, 79, 80, 80, 80, 80,
    106, 122, 126, 128, 135, 145, 146, 152, 152, 152, 153, 155, 155, 156, 157,
    160, 162, 170, 239, 251, 263, 273, 288, 289, 289,
];

// ── CLI ──

#[derive(Parser, Debug, Clone)]
struct Cli {
    #[arg(long, default_value = "127.0.0.1:8000")]
    addr: String,

    #[arg(long, default_value_t = 100)]
    num_requests: usize,

    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 1_000_000)]
    eos_token_id: u32,

    /// CSV output path (prints to stdout if not set)
    #[arg(long)]
    output_csv: Option<String>,
}

// ── JSON-lines protocol ──

#[derive(Serialize)]
struct Req {
    id: u64,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    eos_token_id: u32,
}

#[derive(Deserialize, Debug)]
struct Resp {
    #[allow(dead_code)]
    id: u64,
    generated_tokens: Vec<u32>,
    prefill_ms: f64,
    #[allow(dead_code)]
    decode_ms: f64,
}

/// Per-request measurement.
#[derive(Debug, Clone)]
struct ReqRecord {
    req_id: u64,
    prompt_len: usize,
    status: String,
    ttft_ms: f64,
    total_ms: f64,
    generated: usize,
}

fn sample_prompt_len() -> usize {
    use rand::prelude::*;
    let mut rng = thread_rng();
    SONNET_PROMPT_LENS[rng.gen_range(0..SONNET_PROMPT_LENS.len())]
}

async fn send_one(
    addr: &str,
    id: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    eos_token_id: u32,
) -> Result<ReqRecord> {
    let mut sock = tokio::net::TcpStream::connect(&addr).await
        .with_context(|| format!("connect to {addr}"))?;

    let prompt: Vec<u32> = (0..prompt_len as u32).collect();
    let req = Req {
        id,
        prompt_tokens: prompt,
        max_new_tokens,
        eos_token_id,
    };

    let mut body = serde_json::to_vec(&req)?;
    body.push(b'\n');

    let t0 = Instant::now();
    tokio::io::AsyncWriteExt::write_all(&mut sock, &body).await?;

    use tokio::io::AsyncBufReadExt;
    let (read, _) = sock.into_split();
    let mut reader = tokio::io::BufReader::new(read);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let elapsed = t0.elapsed();

    let resp: Resp = serde_json::from_str(line.trim())
        .with_context(|| format!("parse response: {line:?}"))?;

    Ok(ReqRecord {
        req_id: id,
        prompt_len,
        status: "ok".into(),
        ttft_ms: resp.prefill_ms,
        total_ms: elapsed.as_secs_f64() * 1000.0,
        generated: resp.generated_tokens.len(),
    })
}

// ── Main ──

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!("=== Baseline Throughput Benchmark ===");
    eprintln!("server:       {}", cli.addr);
    eprintln!("num_requests: {}", cli.num_requests);
    eprintln!("concurrency:  {}", cli.concurrency);
    eprintln!("max_new_tok:  {}", cli.max_new_tokens);
    eprintln!("prompt dist:  {} samples, median 42, range 8-289",
        SONNET_PROMPT_LENS.len());
    eprintln!();

    // Generate request configs upfront
    let prompts: Arc<[usize]> = (0..cli.num_requests)
        .map(|_| sample_prompt_len())
        .collect::<Vec<_>>()
        .into();

    let sem = Arc::new(tokio::sync::Semaphore::new(cli.concurrency));
    let counter = Arc::new(AtomicUsize::new(0));
    let total_start = Instant::now();

    let mut handles = Vec::with_capacity(cli.num_requests);
    for (i, &pl) in prompts.iter().enumerate() {
        let addr = cli.addr.clone();
        let mnt = cli.max_new_tokens;
        let eos = cli.eos_token_id;
        let sem = sem.clone();
        let counter = counter.clone();
        let prompts = prompts.clone();
        let id = i as u64;

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let result = send_one(&addr, id, pl, mnt, eos).await;
            drop(_permit);
            match &result {
                Ok(r) => eprintln!("  [{}/{}] id={} pl={} gen={} {:.0}ms",
                    counter.fetch_add(1, Ordering::Relaxed) + 1,
                    prompts.len(), r.req_id, r.prompt_len, r.generated, r.total_ms),
                Err(e) => eprintln!("  [{}/{}] id={} FAIL: {e:?}",
                    counter.fetch_add(1, Ordering::Relaxed) + 1,
                    prompts.len(), id),
            }
            result
        }));
    }

    let mut records: Vec<ReqRecord> = Vec::with_capacity(cli.num_requests);
    for h in handles {
        match h.await? {
            Ok(r) => records.push(r),
            Err(e) => {
                eprintln!("ERROR: {e:?}");
            }
        }
    }

    let total_elapsed = total_start.elapsed();
    let total_s = total_elapsed.as_secs_f64();

    // ── Statistics ──
    let ok: Vec<_> = records.iter().filter(|r| r.status == "ok").collect();
    let total_tok_in: usize = ok.iter().map(|r| r.prompt_len).sum();
    let total_tok_out: usize = ok.iter().map(|r| r.generated).sum();

    let mut ttfts: Vec<f64> = ok.iter().map(|r| r.ttft_ms).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut totals: Vec<f64> = ok.iter().map(|r| r.total_ms).collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p = |v: &[f64], pct: f64| -> f64 {
        if v.is_empty() { 0.0 } else { v[((v.len() as f64) * pct / 100.0) as usize] }
    };

    println!();
    println!("=== Results ===");
    println!("benchmark_duration_s:        {:.2}", total_s);
    println!("requests_completed:          {}", ok.len());
    println!("requests_failed:             {}", records.len() - ok.len());
    println!("total_input_tokens:          {}", total_tok_in);
    println!("total_output_tokens:         {}", total_tok_out);
    println!("request_throughput_req_s:    {:.2}", ok.len() as f64 / total_s);
    println!("output_throughput_tok_s:     {:.2}", total_tok_out as f64 / total_s);
    println!("total_throughput_tok_s:      {:.2}", (total_tok_in + total_tok_out) as f64 / total_s);
    println!("--- latency ---");
    println!("ttft_mean_ms:                {:.2}", ttfts.iter().sum::<f64>() / ok.len().max(1) as f64);
    println!("ttft_p50_ms:                 {:.2}", p(&ttfts, 50.0));
    println!("ttft_p95_ms:                 {:.2}", p(&ttfts, 95.0));
    println!("ttft_p99_ms:                 {:.2}", p(&ttfts, 99.0));
    println!("total_mean_ms:               {:.2}", totals.iter().sum::<f64>() / ok.len().max(1) as f64);
    println!("total_p50_ms:                {:.2}", p(&totals, 50.0));
    println!("total_p95_ms:                {:.2}", p(&totals, 95.0));
    println!("total_p99_ms:                {:.2}", p(&totals, 99.0));

    // ── CSV ──
    if let Some(ref path) = cli.output_csv {
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("create {path}"))?;
        writeln!(f, "req_id,prompt_len,max_new_tokens,status,ttft_ms,total_ms,generated_tokens")?;
        for r in &records {
            writeln!(f, "{},{},{},{},{:.2},{:.2},{}",
                r.req_id, r.prompt_len, cli.max_new_tokens, r.status, r.ttft_ms, r.total_ms, r.generated)?;
        }
        eprintln!("Wrote {} records to {path}", records.len());
    }

    Ok(())
}
