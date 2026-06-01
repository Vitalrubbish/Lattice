// examples/bench_throughput.rs
//
// Throughput benchmark driver for the baseline LLM inference server.
// Uses a realistic prompt-length distribution derived from Shakespeare's sonnets
// (matching vLLM's sonnet benchmark) with synthetic token content.
//
// Now supports Unified Fragmentation Standard (UFS) stats collection
// and a --stress mode for concurrency ramping.
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
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

    /// CSV output path for fragmentation time-series
    #[arg(long)]
    output_frag_csv: Option<String>,

    /// Stress-test mode: ramp concurrency through multiple levels.
    /// Comma-separated list of concurrency levels (e.g. "1,2,4,8,16,32").
    /// When set, --concurrency is ignored.
    #[arg(long)]
    stress_concurrency: Option<String>,

    /// Poll interval for stats collection in milliseconds.
    #[arg(long, default_value_t = 200)]
    stats_poll_ms: u64,
}

// ── JSON-lines protocol ──

#[derive(Serialize)]
struct Req {
    #[serde(rename = "type")]
    req_type: Option<String>,
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

/// Stats response from server.
#[derive(Deserialize, Debug, Clone)]
struct StatsResp {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    response_type: String,
    sample_count: usize,
    internal_frag_rate: f32,
    block_utilization: f32,
    physical_memory_efficiency: f32,
    runtime_frag_index: f32,
    active_sequences: usize,
    blocks_in_use: usize,
    total_blocks_allocated: usize,
    total_tokens: usize,
    rfi_avg: f32,
    rfi_peak: f32,
    rfi_stddev: f32,
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
    let prompt: Vec<u32> = (0..prompt_len as u32).collect();
    let req = Req {
        req_type: Some("infer".into()),
        id,
        prompt_tokens: prompt,
        max_new_tokens,
        eos_token_id,
    };

    let mut sock = tokio::net::TcpStream::connect(&addr).await
        .with_context(|| format!("connect to {addr}"))?;

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

/// Send a stats query and return the parsed response.
async fn query_stats(addr: &str) -> Result<Option<StatsResp>> {
    let req = serde_json::json!({"type": "stats"});
    let mut body = serde_json::to_vec(&req)?;
    body.push(b'\n');

    let mut sock = match tokio::net::TcpStream::connect(&addr).await {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    tokio::io::AsyncWriteExt::write_all(&mut sock, &body).await?;

    use tokio::io::AsyncBufReadExt;
    let (read, _) = sock.into_split();
    let mut reader = tokio::io::BufReader::new(read);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    let stats: StatsResp = serde_json::from_str(line.trim())?;
    Ok(Some(stats))
}

/// Background stats poller. Runs until `running` is set to false.
async fn stats_poller(addr: String, poll_ms: u64, running: Arc<AtomicBool>,
                       samples: Arc<parking_lot::Mutex<Vec<StatsResp>>>) {
    let mut interval = tokio::time::interval(Duration::from_millis(poll_ms));
    while running.load(Ordering::Relaxed) {
        interval.tick().await;
        match query_stats(&addr).await {
            Ok(Some(stats)) => {
                if stats.sample_count > 0 {
                    samples.lock().push(stats);
                }
            }
            _ => {} // Server not ready yet or error — skip
        }
    }
}

// ── Throughput benchmark for one concurrency level ──

#[derive(Debug, Clone)]
struct StressLevelResult {
    concurrency: usize,
    requests_completed: usize,
    requests_failed: usize,
    request_throughput_req_s: f64,
    output_throughput_tok_s: f64,
    total_throughput_tok_s: f64,
    total_mean_ms: f64,
    total_p50_ms: f64,
    total_p95_ms: f64,
    total_p99_ms: f64,
    // UFS fragmentation aggregates
    ifr_avg: f32,
    ifr_peak: f32,
    ifr_stddev: f32,
    bu_avg: f32,
    bu_min: f32,
    bu_stddev: f32,
    pme_avg: f32,
    pme_min: f32,
    pme_stddev: f32,
    rfi_avg: f32,
    rfi_peak: f32,
    rfi_stddev: f32,
    frag_sample_count: usize,
}

/// Fragmentation summary for stress mode (avoids complex tuple destructuring).
#[derive(Debug, Clone, Default)]
struct StressFragSummary {
    sample_count: usize,
    ifr_avg: f32, ifr_peak: f32, ifr_stddev: f32,
    bu_avg: f32, bu_min: f32, bu_stddev: f32,
    pme_avg: f32, pme_min: f32, pme_stddev: f32,
    rfi_avg: f32, rfi_peak: f32, rfi_stddev: f32,
}

async fn run_throughput_bench(
    addr: &str,
    num_requests: usize,
    concurrency: usize,
    max_new_tokens: usize,
    eos_token_id: u32,
    collect_stats: bool,
    stats_poll_ms: u64,
) -> Result<(Vec<ReqRecord>, f64, Option<Vec<StatsResp>>)> {
    let prompts: Arc<[usize]> = (0..num_requests)
        .map(|_| sample_prompt_len())
        .collect::<Vec<_>>()
        .into();

    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let counter = Arc::new(AtomicUsize::new(0));

    // Stats collection
    let stats_running = Arc::new(AtomicBool::new(collect_stats));
    let stats_samples: Arc<parking_lot::Mutex<Vec<StatsResp>>> =
        Arc::new(parking_lot::Mutex::new(Vec::new()));

    let stats_handle = if collect_stats {
        let addr_s = addr.to_string();
        let running = stats_running.clone();
        let samples = stats_samples.clone();
        Some(tokio::spawn(async move {
            stats_poller(addr_s, stats_poll_ms, running, samples).await;
        }))
    } else {
        None
    };

    let total_start = Instant::now();

    let mut handles = Vec::with_capacity(num_requests);
    for (i, &pl) in prompts.iter().enumerate() {
        let addr = addr.to_string();
        let mnt = max_new_tokens;
        let eos = eos_token_id;
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

    let mut records: Vec<ReqRecord> = Vec::with_capacity(num_requests);
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

    // Stop stats collection
    stats_running.store(false, Ordering::Relaxed);
    if let Some(h) = stats_handle {
        let _ = h.await;
    }
    let frag_samples: Option<Vec<StatsResp>> = if collect_stats {
        let samples = stats_samples.lock().clone();
        Some(samples)
    } else {
        None
    };

    Ok((records, total_s, frag_samples))
}

fn compute_summary(records: &[ReqRecord], total_s: f64, _num_requests: usize) {
    let ok: Vec<_> = records.iter().filter(|r| r.status == "ok").collect();
    let total_tok_in: usize = ok.iter().map(|r| r.prompt_len).sum();
    let total_tok_out: usize = ok.iter().map(|r| r.generated).sum();

    let mut ttfts: Vec<f64> = ok.iter().map(|r| r.ttft_ms).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut totals: Vec<f64> = ok.iter().map(|r| r.total_ms).collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p = |v: &[f64], pct: f64| -> f64 {
        if v.is_empty() { 0.0 } else { v[((v.len() as f64 * pct / 100.0) as usize).min(v.len()-1)] }
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
}

fn compute_frag_summary(samples: &[StatsResp]) {
    if samples.is_empty() {
        return;
    }
    let n = samples.len() as f32;

    // IFR
    let ifr_sum: f32 = samples.iter().map(|s| s.internal_frag_rate).sum();
    let ifr_avg = ifr_sum / n;
    let ifr_peak = samples.iter().map(|s| s.internal_frag_rate).fold(0.0f32, f32::max);
    let ifr_var = samples.iter().map(|s| { let d = s.internal_frag_rate - ifr_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
    let ifr_stddev = ifr_var.sqrt();

    // BU
    let bu_sum: f32 = samples.iter().map(|s| s.block_utilization).sum();
    let bu_avg = bu_sum / n;
    let bu_min = samples.iter().map(|s| s.block_utilization).fold(f32::MAX, f32::min);
    let bu_var = samples.iter().map(|s| { let d = s.block_utilization - bu_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
    let bu_stddev = bu_var.sqrt();

    // PME
    let pme_sum: f32 = samples.iter().map(|s| s.physical_memory_efficiency).sum();
    let pme_avg = pme_sum / n;
    let pme_min = samples.iter().map(|s| s.physical_memory_efficiency).fold(f32::MAX, f32::min);
    let pme_var = samples.iter().map(|s| { let d = s.physical_memory_efficiency - pme_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
    let pme_stddev = pme_var.sqrt();

    // RFI
    let rfi_sum: f32 = samples.iter().map(|s| s.runtime_frag_index).sum();
    let rfi_avg = rfi_sum / n;
    let rfi_peak = samples.iter().map(|s| s.runtime_frag_index).fold(0.0f32, f32::max);
    let rfi_var = samples.iter().map(|s| { let d = s.runtime_frag_index - rfi_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
    let rfi_stddev = rfi_var.sqrt();

    println!("--- unified fragmentation (UFS) ---");
    println!("frag_sample_count:          {}", n);
    println!("ifr_avg:                    {:.4}", ifr_avg);
    println!("ifr_peak:                   {:.4}", ifr_peak);
    println!("ifr_stddev:                 {:.4}", ifr_stddev);
    println!("bu_avg:                     {:.4}", bu_avg);
    println!("bu_min:                     {:.4}", bu_min);
    println!("bu_stddev:                  {:.4}", bu_stddev);
    println!("pme_avg:                    {:.4}", pme_avg);
    println!("pme_min:                    {:.4}", pme_min);
    println!("pme_stddev:                 {:.4}", pme_stddev);
    println!("rfi_avg:                    {:.4}", rfi_avg);
    println!("rfi_peak:                   {:.4}", rfi_peak);
    println!("rfi_stddev:                 {:.4}", rfi_stddev);
}

fn write_frag_csv(path: &str, samples: &[StatsResp]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "timestamp,internal_frag_rate,block_utilization,physical_memory_efficiency,runtime_frag_index,active_sequences,blocks_in_use,total_blocks_allocated,total_tokens,rfi_avg,rfi_peak,rfi_stddev,sample_count")?;
    for (i, s) in samples.iter().enumerate() {
        writeln!(f, "{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{:.6},{}",
            i, s.internal_frag_rate, s.block_utilization,
            s.physical_memory_efficiency, s.runtime_frag_index,
            s.active_sequences, s.blocks_in_use, s.total_blocks_allocated,
            s.total_tokens, s.rfi_avg, s.rfi_peak, s.rfi_stddev,
            s.sample_count)?;
    }
    eprintln!("Wrote {} frag samples to {path}", samples.len());
    Ok(())
}

fn write_results_csv(path: &str, records: &[ReqRecord], max_new_tokens: usize) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "req_id,prompt_len,max_new_tokens,status,ttft_ms,total_ms,generated_tokens")?;
    for r in records {
        writeln!(f, "{},{},{},{},{:.2},{:.2},{}",
            r.req_id, r.prompt_len, max_new_tokens, r.status, r.ttft_ms, r.total_ms, r.generated)?;
    }
    eprintln!("Wrote {} records to {path}", records.len());
    Ok(())
}

// ── Main ──

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Parse stress concurrency levels
    let stress_levels: Option<Vec<usize>> = cli.stress_concurrency.as_ref().map(|s| {
        s.split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    });

    if let Some(ref levels) = stress_levels {
        // ── Stress mode: ramp concurrency ──
        eprintln!("=== Baseline Stress Test (Concurrency Ramp) ===");
        eprintln!("server:       {}", cli.addr);
        eprintln!("num_requests: {}", cli.num_requests);
        eprintln!("concurrency:  {:?}", levels);
        eprintln!("max_new_tok:  {}", cli.max_new_tokens);
        eprintln!();

        let mut stress_results: Vec<StressLevelResult> = Vec::new();

        for &concurrency in levels {
            eprintln!("\n>>> Stress level: concurrency={concurrency} <<<");
            let (records, total_s, frag_samples) = run_throughput_bench(
                &cli.addr,
                cli.num_requests,
                concurrency,
                cli.max_new_tokens,
                cli.eos_token_id,
                true,
                cli.stats_poll_ms,
            ).await?;

            compute_summary(&records, total_s, cli.num_requests);

            let frag_summary = if let Some(ref samples) = frag_samples {
                compute_frag_summary(samples);
                // Write per-level frag CSV
                let frag_path = format!(
                    "{}_c{}.frag.csv",
                    cli.output_csv.as_deref().unwrap_or("stress"),
                    concurrency
                );
                write_frag_csv(&frag_path, samples)?;

                let n = samples.len() as f32;

                let ifr_sum: f32 = samples.iter().map(|s| s.internal_frag_rate).sum();
                let ifr_avg = ifr_sum / n.max(1.0);
                let ifr_peak = samples.iter().map(|s| s.internal_frag_rate).fold(0.0f32, f32::max);
                let ifr_var = samples.iter().map(|s| { let d = s.internal_frag_rate - ifr_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
                let ifr_stddev = ifr_var.sqrt();

                let bu_sum: f32 = samples.iter().map(|s| s.block_utilization).sum();
                let bu_avg = bu_sum / n.max(1.0);
                let bu_min = samples.iter().map(|s| s.block_utilization).fold(f32::MAX, f32::min);
                let bu_var = samples.iter().map(|s| { let d = s.block_utilization - bu_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
                let bu_stddev = bu_var.sqrt();

                let pme_sum: f32 = samples.iter().map(|s| s.physical_memory_efficiency).sum();
                let pme_avg = pme_sum / n.max(1.0);
                let pme_min = samples.iter().map(|s| s.physical_memory_efficiency).fold(f32::MAX, f32::min);
                let pme_var = samples.iter().map(|s| { let d = s.physical_memory_efficiency - pme_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
                let pme_stddev = pme_var.sqrt();

                let rfi_sum: f32 = samples.iter().map(|s| s.runtime_frag_index).sum();
                let rfi_avg = rfi_sum / n.max(1.0);
                let rfi_peak = samples.iter().map(|s| s.runtime_frag_index).fold(0.0f32, f32::max);
                let rfi_var = samples.iter().map(|s| { let d = s.runtime_frag_index - rfi_avg; d*d }).sum::<f32>() / (n - 1.0).max(1.0);
                let rfi_stddev = rfi_var.sqrt();

                StressFragSummary {
                    sample_count: samples.len(),
                    ifr_avg, ifr_peak, ifr_stddev,
                    bu_avg, bu_min, bu_stddev,
                    pme_avg, pme_min, pme_stddev,
                    rfi_avg, rfi_peak, rfi_stddev,
                }
            } else {
                StressFragSummary::default()
            };

            let ok: Vec<_> = records.iter().filter(|r| r.status == "ok").collect();
            let total_tok_in: usize = ok.iter().map(|r| r.prompt_len).sum();
            let total_tok_out: usize = ok.iter().map(|r| r.generated).sum();
            let mut totals: Vec<f64> = ok.iter().map(|r| r.total_ms).collect();
            totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |v: &[f64], pct: f64| -> f64 {
                if v.is_empty() { 0.0 } else { v[((v.len() as f64 * pct / 100.0) as usize).min(v.len()-1)] }
            };

            stress_results.push(StressLevelResult {
                concurrency,
                requests_completed: ok.len(),
                requests_failed: records.len() - ok.len(),
                request_throughput_req_s: ok.len() as f64 / total_s.max(0.001),
                output_throughput_tok_s: total_tok_out as f64 / total_s.max(0.001),
                total_throughput_tok_s: (total_tok_in + total_tok_out) as f64 / total_s.max(0.001),
                total_mean_ms: totals.iter().sum::<f64>() / ok.len().max(1) as f64,
                total_p50_ms: p(&totals, 50.0),
                total_p95_ms: p(&totals, 95.0),
                total_p99_ms: p(&totals, 99.0),
                ifr_avg: frag_summary.ifr_avg,
                ifr_peak: frag_summary.ifr_peak,
                ifr_stddev: frag_summary.ifr_stddev,
                bu_avg: frag_summary.bu_avg,
                bu_min: frag_summary.bu_min,
                bu_stddev: frag_summary.bu_stddev,
                pme_avg: frag_summary.pme_avg,
                pme_min: frag_summary.pme_min,
                pme_stddev: frag_summary.pme_stddev,
                rfi_avg: frag_summary.rfi_avg,
                rfi_peak: frag_summary.rfi_peak,
                rfi_stddev: frag_summary.rfi_stddev,
                frag_sample_count: frag_summary.sample_count,
            });
        }

        // Write stress summary CSV
        let stress_csv = cli.output_csv.as_deref().unwrap_or("stress_results").to_string() + "_summary.csv";
        {
            let mut f = std::fs::File::create(&stress_csv)?;
            writeln!(f, "concurrency,req_completed,req_failed,req_s,tok_out_s,total_tok_s,mean_ms,p50_ms,p95_ms,p99_ms,ifr_avg,ifr_peak,ifr_stddev,bu_avg,bu_min,bu_stddev,pme_avg,pme_min,pme_stddev,rfi_avg,rfi_peak,rfi_stddev,frag_samples")?;
            for r in &stress_results {
                writeln!(f, "{},{},{},{:.2},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                    r.concurrency, r.requests_completed, r.requests_failed,
                    r.request_throughput_req_s, r.output_throughput_tok_s, r.total_throughput_tok_s,
                    r.total_mean_ms, r.total_p50_ms, r.total_p95_ms, r.total_p99_ms,
                    r.ifr_avg, r.ifr_peak, r.ifr_stddev,
                    r.bu_avg, r.bu_min, r.bu_stddev,
                    r.pme_avg, r.pme_min, r.pme_stddev,
                    r.rfi_avg, r.rfi_peak, r.rfi_stddev,
                    r.frag_sample_count,
                )?;
            }
        }
        eprintln!("Wrote stress summary to {stress_csv}");

        // Print final stress comparison table
        println!();
        println!("=== Stress Test UFS Comparison ===");
        println!("{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "conc", "ifr_avg", "ifr_pk", "bu_avg", "bu_min", "pme_avg", "pme_min", "rfi_avg", "rfi_pk");
        for r in &stress_results {
            println!("{:>4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                r.concurrency, r.ifr_avg, r.ifr_peak, r.bu_avg, r.bu_min,
                r.pme_avg, r.pme_min, r.rfi_avg, r.rfi_peak);
        }

    } else {
        // ── Single-level benchmark ──
        eprintln!("=== Baseline Throughput Benchmark ===");
        eprintln!("server:       {}", cli.addr);
        eprintln!("num_requests: {}", cli.num_requests);
        eprintln!("concurrency:  {}", cli.concurrency);
        eprintln!("max_new_tok:  {}", cli.max_new_tokens);
        eprintln!("prompt dist:  {} samples, median 42, range 8-289",
            SONNET_PROMPT_LENS.len());
        eprintln!();

        let (records, total_s, frag_samples) = run_throughput_bench(
            &cli.addr,
            cli.num_requests,
            cli.concurrency,
            cli.max_new_tokens,
            cli.eos_token_id,
            true,
            cli.stats_poll_ms,
        ).await?;

        compute_summary(&records, total_s, cli.num_requests);

        // Fragmentation report
        if let Some(ref samples) = frag_samples {
            compute_frag_summary(samples);

            // Write frag CSV
            if let Some(ref path) = cli.output_frag_csv {
                write_frag_csv(path, samples)?;
            } else if let Some(ref bench_csv) = cli.output_csv {
                let frag_path = bench_csv.replace(".csv", ".frag.csv");
                write_frag_csv(&frag_path, samples)?;
            }
        }

        // Write per-request CSV
        if let Some(ref path) = cli.output_csv {
            write_results_csv(path, &records, cli.max_new_tokens)?;
        }
    }

    Ok(())
}
