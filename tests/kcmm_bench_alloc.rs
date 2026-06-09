// tests/kcmm_bench_alloc.rs
//
// KCMM Phase E — Benchmark 1: Block allocation / free throughput.
//
// Measures the latency of single-block alloc_sequence + free_sequence
// operations across varying block sizes and pool capacities.
//
// Success criteria (§E.2):
//   - KCMM alloc/free throughput regression < 5% vs baseline (vLLM-equivalent
//     PagedKvCache).
//
// These tests require a CUDA device.

use baseline_llm_os::config::KcmmConfig;
use baseline_llm_os::cuda::CudaContext;
use baseline_llm_os::kcmm::pool::KcmmPool;
use std::sync::Arc;
use std::time::Instant;

mod bench_utils;
use bench_utils::*;

/// Run `num_ops` alloc(1-block) → free cycles, returning raw latency samples
/// in nanoseconds.
fn run_alloc_free_cycle(pool: &KcmmPool, num_ops: usize) -> (Vec<u64>, Vec<u64>) {
    let mut alloc_lat = Vec::with_capacity(num_ops);
    let mut free_lat = Vec::with_capacity(num_ops);

    // Warmup: 100 ops to stabilise CUDA driver caches.
    for _ in 0..100 {
        let table = pool.alloc_sequence(1).expect("warmup alloc");
        pool.free_sequence(&table);
    }

    for _ in 0..num_ops {
        let t0 = Instant::now();
        let table = pool.alloc_sequence(1).expect("bench alloc");
        alloc_lat.push(t0.elapsed().as_nanos() as u64);

        let t0 = Instant::now();
        pool.free_sequence(&table);
        free_lat.push(t0.elapsed().as_nanos() as u64);
    }

    (alloc_lat, free_lat)
}

/// Helper to build a `KcmmConfig` with tiering disabled for raw allocation
/// benchmarking.
fn bench_config(block_size: usize, max_blocks: usize) -> KcmmConfig {
    KcmmConfig {
        block_size,
        max_blocks,
        tiering: false, // no tiering overhead
        cpu_cache_path: String::new(), // not used when tiering is false
        eviction_policy: "lru".to_string(),
        prefetch_window: 0,
        max_batch_blocks: 0,
    }
}

// --- Single-point benchmark (used by test runner) ---

#[test]
fn kcmm_bench_alloc_throughput() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));

    // Model parameters (tiny_llama-like, small layer count for fast init).
    let num_layers = 2;
    let kv_heads = 4;
    let head_dim = 64;
    // block_bytes = kv_heads * block_size * head_dim * 2 (f16)
    //
    //   block_size=64  → block_bytes = 4*64*64*2  =  32 768 B = 32 KiB
    //   block_size=128 → block_bytes = 4*128*64*2 =  65 536 B = 64 KiB
    //   block_size=256 → block_bytes = 4*256*64*2 = 131 072 B = 128 KiB
    let block_sizes: &[usize] = &[64, 128, 256];

    println!("\n=== KCMM Benchmark 1: Allocation / Free Throughput ===");
    println!(
        "model: kv_heads={kv_heads}, head_dim={head_dim}, num_layers={num_layers}"
    );
    println!("{:>10} {:>14} {:>12} {:>12} {:>12} {:>12}",
             "blk_bytes", "pool_blocks", "alloc_p50", "alloc_p99", "free_p50", "free_p99");
    println!("{}", "-".repeat(76));

    for &block_size in block_sizes {
        let cfg = bench_config(block_size, 4096);
        let pool = KcmmPool::new(
            ctx.clone(),
            cfg,
            num_layers,
            kv_heads,
            head_dim,
            256, // max_batch
            256, // max_seq_len
        )
        .expect("create KcmmPool");

        let block_bytes = pool.block_bytes;
        let num_ops = 500;
        let (mut alloc_lat, mut free_lat) = run_alloc_free_cycle(&pool, num_ops);

        let alloc_p50 = percentile(&mut alloc_lat, 50.0);
        let alloc_p99 = percentile(&mut alloc_lat, 99.0);
        let free_p50 = percentile(&mut free_lat, 50.0);
        let free_p99 = percentile(&mut free_lat, 99.0);

        println!(
            "{:>10} {:>14} {:>9} ns {:>9} ns {:>9} ns {:>9} ns",
            block_bytes, 4096, alloc_p50, alloc_p99, free_p50, free_p99
        );

        // Sanity: latency should be well under 1 ms per operation.
        assert!(
            alloc_p50 < 1_000_000,
            "alloc p50={alloc_p50} ns exceeds 1 ms — something is wrong"
        );
        assert!(
            free_p50 < 1_000_000,
            "free p50={free_p50} ns exceeds 1 ms — something is wrong"
        );
    }

    println!("=== End Benchmark 1 ===\n");
}

// --- Pool-size sweep ---

#[test]
fn kcmm_bench_alloc_pool_size_sweep() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));

    let num_layers = 2;
    let kv_heads = 4;
    let head_dim = 64;
    let block_size = 128; // 64 KiB blocks

    println!("\n=== KCMM Benchmark 1b: Pool-Size Sweep ===");
    println!(
        "block_size={block_size} tokens ({blk_bytes} bytes/block)",
        blk_bytes = kv_heads * block_size * head_dim * 2
    );
    println!(
        "{:>14} {:>12} {:>12} {:>12} {:>12}",
        "pool_blocks", "alloc_p50", "alloc_p99", "free_p50", "free_p99"
    );
    println!("{}", "-".repeat(62));

    for &max_blocks in &[1024usize, 4096, 16384] {
        let cfg = bench_config(block_size, max_blocks);
        let pool = KcmmPool::new(
            ctx.clone(),
            cfg,
            num_layers,
            kv_heads,
            head_dim,
            (max_blocks / 16).max(1), // max_batch
            256,                        // max_seq_len
        )
        .expect("create KcmmPool");

        let num_ops = 300;
        let (mut alloc_lat, mut free_lat) = run_alloc_free_cycle(&pool, num_ops);

        println!(
            "{:>14} {:>9} ns {:>9} ns {:>9} ns {:>9} ns",
            max_blocks,
            percentile(&mut alloc_lat, 50.0),
            percentile(&mut alloc_lat, 99.0),
            percentile(&mut free_lat, 50.0),
            percentile(&mut free_lat, 99.0),
        );
    }

    println!("=== End Pool-Size Sweep ===\n");
}

// --- Multi-sequence concurrent allocation stress ---

#[test]
fn kcmm_bench_alloc_concurrent_sequences() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));

    let num_layers = 2;
    let kv_heads = 4;
    let head_dim = 64;
    let block_size = 128;

    let cfg = bench_config(block_size, 4096);
    let pool = KcmmPool::new(
        ctx.clone(),
        cfg,
        num_layers,
        kv_heads,
        head_dim,
        256,
        256,
    )
    .expect("create KcmmPool");

    println!("\n=== KCMM Benchmark 1c: Multi-Sequence Allocation ===");

    // Allocate many concurrent sequences, each with 4 blocks (simulates
    // multi-user workload), then free all.  Run multiple rounds for
    // statistically meaningful results.
    let concurrency = 64usize;
    let blocks_per_seq = 4usize;
    let total_blocks = concurrency * blocks_per_seq;
    let rounds = 16;
    let mut alloc_per_block_ns: Vec<u64> = Vec::with_capacity(rounds);
    let mut free_per_block_ns: Vec<u64> = Vec::with_capacity(rounds);

    // Warmup
    for _ in 0..16 {
        let table = pool.alloc_sequence(blocks_per_seq).expect("warmup");
        pool.free_sequence(&table);
    }

    for _ in 0..rounds {
        let t0 = Instant::now();
        let mut all_tables = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let table = pool.alloc_sequence(blocks_per_seq).expect("multi-seq alloc");
            all_tables.push(table);
        }
        let total_ns = t0.elapsed().as_nanos() as u64;
        alloc_per_block_ns.push(total_ns / total_blocks as u64);

        let t0 = Instant::now();
        for table in &all_tables {
            pool.free_sequence(table);
        }
        let total_ns = t0.elapsed().as_nanos() as u64;
        free_per_block_ns.push(total_ns / total_blocks as u64);
    }

    println!("  concurrency:          {concurrency} sequences");
    println!("  blocks per sequence:  {blocks_per_seq}");
    println!("  total blocks:         {total_blocks}");
    print_latency_stats("alloc_per_block", &mut alloc_per_block_ns, "ns");
    print_latency_stats("free_per_block", &mut free_per_block_ns, "ns");

    // Sanity: per-block overhead should be reasonable.
    // With N layers, each logical block requires 2N physical allocations
    // (K+V per layer), plus cuMemMap.  200 µs/block is a generous upper bound.
    let alloc_ns_per_block = mean(&alloc_per_block_ns);
    assert!(
        alloc_ns_per_block < 200_000.0,
        "alloc per block {alloc_ns_per_block:.0} ns exceeds 200 µs — suspicious"
    );

    println!("=== End Multi-Sequence Allocation ===\n");
}
