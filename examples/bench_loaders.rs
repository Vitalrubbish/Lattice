use anyhow::{Context, Result};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use baseline_llm_os::config::ModelConfig;
use baseline_llm_os::cuda::CudaContext;
use baseline_llm_os::model::{LoaderKind, ModelLoader};

fn model_path() -> String {
    std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("MODEL_PATH env var not set, defaulting to ./models/tinyllama");
        "./models/tinyllama".to_string()
    })
}
const RUNS_PER_METHOD: usize = 3;

fn drop_caches() {
    let pass = std::env::var("SUDO_PASS").unwrap_or_default();
    let mut child = Command::new("sudo")
        .args(["-S", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sudo");

    if let Some(mut stdin) = child.stdin.take() {
        let _ = writeln!(stdin, "{pass}");
    }
    let out = child.wait_with_output();
    match out {
        Ok(o) if o.status.success() => println!("  [ok] page cache, dentries, inodes dropped"),
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            println!("  [warn] drop_caches failed: {}", stderr.trim());
        }
        Err(e) => println!("  [warn] drop_caches failed: {e}"),
    }
}

fn run_one(kind: LoaderKind, label: &str) -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let cfg = ModelConfig::tiny_llama();
    let t0 = Instant::now();
    let (_weights, metrics) = ModelLoader::new(&ctx, &cfg, kind)
        .load(&model_path())
        .with_context(|| format!("load with {}", kind.as_str()))?;
    let wall = t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "  [{label}] total={total_ms:.0?}ms read={read_ms:.0?}ms parse={parse_ms:.0?}ms \
         alloc={alloc_ms:.0?}ms h2d={h2d_ms:.0?}ms cpu_user={cpu_user_ms:.0?}ms \
         cpu_sys={cpu_sys_ms:.0?}ms bytes={total_bytes} throughput={mbps:.0?} MB/s",
        total_ms = metrics.total_ms,
        read_ms = metrics.read_ms,
        parse_ms = metrics.parse_ms,
        alloc_ms = metrics.alloc_ms,
        h2d_ms = metrics.h2d_ms,
        cpu_user_ms = metrics.cpu_user_ms,
        cpu_sys_ms = metrics.cpu_sys_ms,
        total_bytes = metrics.total_bytes,
        mbps = if metrics.total_ms > 0.0 {
            (metrics.total_bytes as f64 / 1e6) / (metrics.total_ms / 1e3)
        } else {
            0.0
        },
    );
    let _ = wall;
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let methods: &[(LoaderKind, &str)] = &[
        (LoaderKind::Read, "read(2)"),
        (LoaderKind::Mmap, "mmap(2)"),
        (LoaderKind::Direct, "O_DIRECT"),
    ];

    for &(kind, name) in methods {
        println!("\n=== {name} ===");

        // Cold: drop caches, then load
        drop_caches();
        for i in 0..RUNS_PER_METHOD {
            println!("  run {}/{}:", i + 1, RUNS_PER_METHOD);
            run_one(kind, "cold")?;
        }

        // Warm: load again without dropping caches
        println!("  --- warm ---");
        for i in 0..RUNS_PER_METHOD {
            println!("  run {}/{}:", i + 1, RUNS_PER_METHOD);
            run_one(kind, "warm")?;
        }
    }

    println!("\nDone.");
    Ok(())
}
