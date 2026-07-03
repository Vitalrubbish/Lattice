// tests/kcmm_bench_memory_pressure.rs
//
// KCMM Phase 1c — Memory Pressure Integration Benchmark.
//
// Measures KCMM's capacity improvement under memory pressure by comparing
// how many sequences can be concurrently active with tiering ON vs. OFF.
// Uses a dynamic workload: sequences complete at different times, new
// sequences keep arriving — simulating continuous batching in a real
// inference server.
//
// Success criterion from implementation analysis §5.3 BM5:
//   KCMM with tiering admits ≥ 1.3× concurrent sequences vs. baseline
//   (no swap) at the same GPU memory budget.

use baseline_llm_os::cache::paged_kv::PagedKvCache;
use baseline_llm_os::config::{KcmmConfig, ModelConfig};
use baseline_llm_os::cuda::CudaContext;
use baseline_llm_os::kcmm::pool::{BlockLocation, KcmmPool};
use baseline_llm_os::kcmm::superblock::BlockHandle;
use baseline_llm_os::kcmm::tiering::TieringEngine;
use std::sync::Arc;
use std::time::Instant;

// --- Shared geometry ---
//
// TinyLlama: 22 layers, 4 kv_heads, 64 head_dim.
// Block_bytes = block_size_tokens × kv_heads × head_dim × 2 (f16) × num_layers
const NUM_LAYERS: usize = 22;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;

// --- Workload configuration ---

#[derive(Debug, Clone)]
struct WorkloadConfig {
    /// Tokens per block.
    block_size_tokens: usize,
    /// Prompt lengths for arriving sequences (cycled).
    prompt_lens: Vec<usize>,
    /// Decode steps before a sequence completes.
    max_new_tokens: usize,
    /// Max concurrent sequences the pool is sized for (VA reservation).
    max_batch: usize,
    /// Max sequence length in tokens (controls VA reservation size).
    max_seq_len: usize,
    /// Number of new sequences that arrive during the workload.
    /// If this exceeds max_batch, sequences must complete before new ones fit.
    total_arrivals: usize,
    /// How many decode steps between new sequence arrivals.
    arrival_interval: usize,
}

impl WorkloadConfig {
    fn label(&self) -> String {
        let prompts: Vec<String> = self.prompt_lens.iter().map(|l| l.to_string()).collect();
        format!(
            "bs{}_mb{}_msl{}_pl[{}]_mnt{}_arr{}",
            self.block_size_tokens,
            self.max_batch,
            self.max_seq_len,
            prompts.join(","),
            self.max_new_tokens,
            self.total_arrivals,
        )
    }

    fn block_bytes(&self) -> usize {
        self.block_size_tokens * KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS
    }
}

// --- Result types ---

#[derive(Debug, Clone)]
struct CapacityResult {
    /// Total sequences that completed successfully (grew to full length).
    completed: usize,
    /// Sequences that were capped (couldn't allocate during decode).
    capped: usize,
    /// Sequences that couldn't even be admitted (prefill OOM).
    rejected: usize,
    /// Peak concurrent sequences.
    peak_concurrent: usize,
    /// Peak physical blocks in use.
    peak_blocks: usize,
    /// Total blocks allocated over the entire workload.
    total_blocks_allocated: usize,
    /// Eviction count (KCMM only).
    eviction_count: usize,
    /// CPU swap peak bytes (KCMM only).
    cpu_swap_peak_bytes: u64,
    elapsed_ms: u64,
}

// --- Tracked sequence ---

struct TrackedSeq {
    seq_idx: usize,
    block_indices: Vec<u32>,
    current_len_tokens: usize,
    target_len_tokens: usize,
    is_active: bool,
}

// ============================================================================
// Baseline (PagedKvCache — no tiering)
// ============================================================================

fn run_baseline_workload(cache: &PagedKvCache, cfg: &WorkloadConfig) -> CapacityResult {
    let t0 = Instant::now();
    let mut active_seqs: Vec<TrackedSeq> = Vec::new();
    let mut completed = 0usize;
    let mut capped = 0usize;
    let mut rejected = 0usize;
    let mut total_blocks_allocated = 0usize;
    let mut seq_counter = 0usize;

    // Pre-fill: admit sequences until we hit capacity (OOM).
    // This establishes the baseline pool saturation point.
    loop {
        let prompt_len = cfg.prompt_lens[seq_counter % cfg.prompt_lens.len()];
        let initial_blocks = (prompt_len + cfg.block_size_tokens - 1) / cfg.block_size_tokens;
        let target_len = prompt_len + cfg.max_new_tokens;

        let block_table = match cache.alloc_sequence(initial_blocks) {
            Ok(bt) => bt,
            Err(_) => break, // Pool is full — pre-fill complete.
        };

        total_blocks_allocated += initial_blocks;
        let seq_idx = cache.register_sequence(block_table.clone());
        active_seqs.push(TrackedSeq {
            seq_idx,
            block_indices: block_table,
            current_len_tokens: prompt_len,
            target_len_tokens: target_len,
            is_active: true,
        });
        seq_counter += 1;

        // Don't pre-fill beyond 80% of max_batch to leave room for the dynamic phase.
        if active_seqs.len() >= cfg.max_batch * 8 / 10 {
            break;
        }
    }

    let mut peak_concurrent = active_seqs.len();

    // Dynamic phase: decode growth + new arrivals.
    let total_steps = cfg.max_new_tokens * 3; // Long enough for arrivals + completions.
    let mut next_arrival_at = cfg.arrival_interval;

    for step in 0..total_steps {
        // 1. Grow existing active sequences.
        let mut still_active: Vec<TrackedSeq> = Vec::new();

        for mut seq in active_seqs.drain(..) {
            if seq.current_len_tokens >= seq.target_len_tokens {
                // Complete.
                if let Some(bt) = cache.get_block_table(seq.seq_idx) {
                    cache.free_sequence(&bt);
                }
                completed += 1;
                continue;
            }

            seq.current_len_tokens += 1;
            let blocks_needed =
                (seq.current_len_tokens + cfg.block_size_tokens - 1) / cfg.block_size_tokens;

            if blocks_needed > seq.block_indices.len() {
                match cache.alloc_block() {
                    Ok(block_idx) => {
                        total_blocks_allocated += 1;
                        cache.append_block_to_sequence(seq.seq_idx, block_idx);
                        seq.block_indices.push(block_idx);
                        still_active.push(seq);
                    }
                    Err(_) => {
                        // OOM — cap this sequence.
                        capped += 1;
                        if let Some(bt) = cache.get_block_table(seq.seq_idx) {
                            cache.free_sequence(&bt);
                        }
                    }
                }
            } else {
                still_active.push(seq);
            }
        }

        active_seqs = still_active;
        peak_concurrent = peak_concurrent.max(active_seqs.len());

        // 2. New arrivals (simulates continuous batching).
        if step >= next_arrival_at && seq_counter < cfg.total_arrivals {
            next_arrival_at = step + cfg.arrival_interval;

            let prompt_len = cfg.prompt_lens[seq_counter % cfg.prompt_lens.len()];
            let initial_blocks = (prompt_len + cfg.block_size_tokens - 1) / cfg.block_size_tokens;
            let target_len = prompt_len + cfg.max_new_tokens;

            match cache.alloc_sequence(initial_blocks) {
                Ok(block_table) => {
                    total_blocks_allocated += initial_blocks;
                    let seq_idx = cache.register_sequence(block_table.clone());
                    active_seqs.push(TrackedSeq {
                        seq_idx,
                        block_indices: block_table,
                        current_len_tokens: prompt_len,
                        target_len_tokens: target_len,
                        is_active: true,
                    });
                    seq_counter += 1;
                }
                Err(_) => {
                    rejected += 1;
                }
            }
        }

        if active_seqs.is_empty() && seq_counter >= cfg.total_arrivals {
            break;
        }

        peak_concurrent = peak_concurrent.max(active_seqs.len());
    }

    // Clean up.
    for seq in &active_seqs {
        if let Some(bt) = cache.get_block_table(seq.seq_idx) {
            cache.free_sequence(&bt);
        }
    }

    CapacityResult {
        completed,
        capped,
        rejected,
        peak_concurrent,
        peak_blocks: 0, // not tracked for baseline
        total_blocks_allocated,
        eviction_count: 0,
        cpu_swap_peak_bytes: 0,
        elapsed_ms: t0.elapsed().as_millis() as u64,
    }
}

// ============================================================================
// KCMM (KcmmPool — tiering ON)
// ============================================================================

fn evict_coldest_blocks(
    pool: &KcmmPool,
    tiering: &TieringEngine,
    seqs: &[TrackedSeq],
    min_count: usize,
) -> usize {
    // Always collect at least MIN_BATCH_FOR_GATHER (8) candidates so the
    // batched eviction path (gather kernel) is used — one synchronise for
    // all victims instead of one per victim.  This cuts synchronise calls
    // from O(victims) to O(1) per eviction batch.
    const TARGET_BATCH: usize = 8;
    let target = min_count.max(TARGET_BATCH);

    let t_scan = Instant::now();
    let mut candidates: Vec<(u32, BlockHandle)> = Vec::with_capacity(target);

    // Select blocks from inactive (cooled) sequences first.
    for seq in seqs.iter().filter(|s| !s.is_active) {
        for &block_idx in &seq.block_indices {
            if let Some(handle) = pool.get_block_handle(block_idx) {
                candidates.push((block_idx, handle));
                if candidates.len() >= target {
                    break;
                }
            }
        }
        if candidates.len() >= target {
            break;
        }
    }

    // If not enough inactive, take from active sequences too.
    if candidates.len() < target {
        for seq in seqs.iter().filter(|s| s.is_active) {
            for &block_idx in &seq.block_indices {
                if let Some(handle) = pool.get_block_handle(block_idx) {
                    candidates.push((block_idx, handle));
                    if candidates.len() >= target {
                        break;
                    }
                }
            }
            if candidates.len() >= target {
                break;
            }
        }
    }

    let scan_us = t_scan.elapsed().as_micros() as u64;

    if candidates.is_empty() {
        return 0;
    }

    let handles: Vec<BlockHandle> = candidates.iter().map(|(_, h)| *h).collect();
    let batch_size = handles.len();

    let t_evict = Instant::now();
    let evicted_count = match tiering.evict_blocks(pool, &handles, batch_size) {
        Ok(evicted) => evicted.len(),
        Err(_) => 0,
    };
    let evict_us = t_evict.elapsed().as_micros() as u64;

    if scan_us + evict_us > 5000 {
        println!(
            "    [evict detail] batch={} evicted={} scan={}µs  evict={}µs  total={}µs",
            batch_size,
            evicted_count,
            scan_us,
            evict_us,
            scan_us + evict_us,
        );
    }

    evicted_count
}

fn count_cpu_resident_blocks(pool: &KcmmPool, block_indices: &[u32]) -> usize {
    block_indices
        .iter()
        .filter(|&&idx| {
            matches!(
                pool.get_block_location(idx),
                Some(BlockLocation::CpuResident(_))
            )
        })
        .count()
}

fn free_sequence_and_account(
    pool: &KcmmPool,
    block_table: &[u32],
    cpu_swap_live_bytes: &mut u64,
    block_bytes: u64,
) {
    let cpu_blocks = count_cpu_resident_blocks(pool, block_table);
    *cpu_swap_live_bytes = cpu_swap_live_bytes.saturating_sub(cpu_blocks as u64 * block_bytes);
    pool.free_sequence(block_table);
}

fn try_alloc_with_eviction(
    pool: &KcmmPool,
    tiering: &TieringEngine,
    active_seqs: &[TrackedSeq],
    num_blocks: usize,
    eviction_count: &mut usize,
    cpu_swap_live_bytes: &mut u64,
    peak_cpu_swap: &mut u64,
    block_bytes: u64,
) -> Option<Vec<u32>> {
    // Try direct allocation first.
    if let Ok(bt) = pool.alloc_sequence(num_blocks) {
        return Some(bt);
    }

    // Evict to make room, then retry.
    let evicted = evict_coldest_blocks(pool, tiering, active_seqs, num_blocks.max(4));
    if evicted > 0 {
        *eviction_count += 1;
        *cpu_swap_live_bytes = cpu_swap_live_bytes.saturating_add(evicted as u64 * block_bytes);
        *peak_cpu_swap = (*peak_cpu_swap).max(*cpu_swap_live_bytes);
    }

    pool.alloc_sequence(num_blocks).ok()
}

fn run_kcmm_workload(pool: &KcmmPool, cfg: &WorkloadConfig) -> CapacityResult {
    let t0 = Instant::now();
    let block_bytes = cfg.block_bytes() as u64;
    let tiering = pool.tiering.as_ref().expect("tiering enabled");

    let mut active_seqs: Vec<TrackedSeq> = Vec::new();
    let mut completed = 0usize;
    let mut capped = 0usize;
    let mut rejected = 0usize;
    let mut peak_blocks = 0usize;
    let mut total_blocks_allocated = 0usize;
    let mut eviction_count = 0usize;
    let mut peak_cpu_swap: u64 = 0;
    let mut cpu_swap_live_bytes: u64 = 0;
    let mut seq_counter = 0usize;
    let mut eviction_time_us: u64 = 0;
    let mut alloc_time_us: u64 = 0;
    let mut cool_touch_time_us: u64 = 0;

    // Pre-fill: admit sequences until the pool is saturated (~80% of max_batch).
    loop {
        let prompt_len = cfg.prompt_lens[seq_counter % cfg.prompt_lens.len()];
        let initial_blocks = (prompt_len + cfg.block_size_tokens - 1) / cfg.block_size_tokens;
        let target_len = prompt_len + cfg.max_new_tokens;

        let t_alloc = Instant::now();
        let block_table = match try_alloc_with_eviction(
            pool,
            tiering,
            &active_seqs,
            initial_blocks,
            &mut eviction_count,
            &mut cpu_swap_live_bytes,
            &mut peak_cpu_swap,
            block_bytes,
        ) {
            Some(bt) => bt,
            None => break, // Can't admit — pre-fill complete.
        };
        alloc_time_us += t_alloc.elapsed().as_micros() as u64;

        total_blocks_allocated += initial_blocks;
        let seq_idx = pool.register_sequence(block_table.clone());
        pool.update_seq_len(seq_idx, prompt_len);
        pool.touch(seq_idx);

        active_seqs.push(TrackedSeq {
            seq_idx,
            block_indices: block_table,
            current_len_tokens: prompt_len,
            target_len_tokens: target_len,
            is_active: true,
        });
        seq_counter += 1;

        if active_seqs.len() >= cfg.max_batch * 8 / 10 {
            break;
        }
    }

    let mut peak_concurrent = active_seqs.len();

    // Dynamic phase.
    let total_steps = cfg.max_new_tokens * 3;
    let mut next_arrival_at = cfg.arrival_interval;

    for step in 0..total_steps {
        // Periodically cool some active sequences to create eviction candidates.
        if step % 8 == 0 && step > 0 {
            let t_cool = Instant::now();
            let cool_count = (active_seqs.len() / 4).max(1);
            let mut cooled = 0;
            for seq in active_seqs.iter_mut() {
                if seq.is_active && cooled < cool_count {
                    pool.cool(seq.seq_idx);
                    seq.is_active = false;
                    cooled += 1;
                }
            }
            // Re-touch remaining active sequences.
            for seq in active_seqs.iter_mut().filter(|s| s.is_active) {
                pool.touch(seq.seq_idx);
            }
            cool_touch_time_us += t_cool.elapsed().as_micros() as u64;
        }

        // 1. Grow active sequences.
        let mut still_active: Vec<TrackedSeq> = Vec::new();

        for mut seq in active_seqs.drain(..) {
            let seq_idx = seq.seq_idx;

            if seq.current_len_tokens >= seq.target_len_tokens {
                // Complete.
                if let Some(bt) = pool.get_block_table(seq_idx) {
                    free_sequence_and_account(pool, &bt, &mut cpu_swap_live_bytes, block_bytes);
                }
                completed += 1;
                continue;
            }

            seq.current_len_tokens += 1;
            let blocks_needed =
                (seq.current_len_tokens + cfg.block_size_tokens - 1) / cfg.block_size_tokens;

            if blocks_needed > seq.block_indices.len() {
                match pool.alloc_block() {
                    Ok(block_idx) => {
                        total_blocks_allocated += 1;
                        pool.append_block_to_sequence(seq_idx, block_idx);
                        seq.block_indices.push(block_idx);
                        seq.is_active = true;
                        pool.touch(seq_idx);
                        still_active.push(seq);
                    }
                    Err(_) => {
                        // Try eviction, then retry once.
                        let t_evict = Instant::now();
                        let evicted = evict_coldest_blocks(pool, tiering, &still_active, 4);
                        if evicted > 0 {
                            eviction_count += 1;
                            cpu_swap_live_bytes =
                                cpu_swap_live_bytes.saturating_add(evicted as u64 * block_bytes);
                            peak_cpu_swap = peak_cpu_swap.max(cpu_swap_live_bytes);
                        }
                        eviction_time_us += t_evict.elapsed().as_micros() as u64;
                        match pool.alloc_block() {
                            Ok(block_idx) => {
                                total_blocks_allocated += 1;
                                pool.append_block_to_sequence(seq_idx, block_idx);
                                seq.block_indices.push(block_idx);
                                pool.touch(seq_idx);
                                still_active.push(seq);
                            }
                            Err(_) => {
                                capped += 1;
                                if let Some(bt) = pool.get_block_table(seq_idx) {
                                    free_sequence_and_account(
                                        pool,
                                        &bt,
                                        &mut cpu_swap_live_bytes,
                                        block_bytes,
                                    );
                                }
                            }
                        }
                    }
                }
            } else {
                pool.touch(seq_idx);
                still_active.push(seq);
            }
        }

        active_seqs = still_active;
        peak_concurrent = peak_concurrent.max(active_seqs.len());

        // 2. New arrivals.
        if step >= next_arrival_at && seq_counter < cfg.total_arrivals {
            next_arrival_at = step + cfg.arrival_interval;

            let prompt_len = cfg.prompt_lens[seq_counter % cfg.prompt_lens.len()];
            let initial_blocks = (prompt_len + cfg.block_size_tokens - 1) / cfg.block_size_tokens;
            let target_len = prompt_len + cfg.max_new_tokens;

            match pool.alloc_sequence(initial_blocks) {
                Ok(block_table) => {
                    total_blocks_allocated += initial_blocks;
                    let seq_idx = pool.register_sequence(block_table.clone());
                    pool.update_seq_len(seq_idx, prompt_len);
                    pool.touch(seq_idx);
                    active_seqs.push(TrackedSeq {
                        seq_idx,
                        block_indices: block_table,
                        current_len_tokens: prompt_len,
                        target_len_tokens: target_len,
                        is_active: true,
                    });
                    seq_counter += 1;
                }
                Err(_) => {
                    rejected += 1;
                }
            }
        }

        if active_seqs.is_empty() && seq_counter >= cfg.total_arrivals {
            break;
        }

        // Sample peak usage.
        peak_concurrent = peak_concurrent.max(active_seqs.len());
        let used = pool
            .total_physical_blocks()
            .saturating_sub(pool.free_physical_blocks());
        peak_blocks = peak_blocks.max(used);
    }

    // Clean up.
    for seq in &active_seqs {
        if let Some(bt) = pool.get_block_table(seq.seq_idx) {
            free_sequence_and_account(pool, &bt, &mut cpu_swap_live_bytes, block_bytes);
        }
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    if eviction_count > 0 {
        println!(
            "    [kcmm timing] total={}ms  eviction={}ms  alloc={}ms  cool/touch={}ms",
            elapsed_ms,
            eviction_time_us / 1000,
            alloc_time_us / 1000,
            cool_touch_time_us / 1000,
        );
    }

    CapacityResult {
        completed,
        capped,
        rejected,
        peak_concurrent,
        peak_blocks,
        total_blocks_allocated,
        eviction_count,
        cpu_swap_peak_bytes: peak_cpu_swap,
        elapsed_ms,
    }
}

// ============================================================================
// Main benchmarks
// ============================================================================

fn tiny_llama_cfg() -> ModelConfig {
    ModelConfig {
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: 32,
        num_key_value_heads: Some(KV_HEADS),
        vocab_size: 32000,
        max_position_embeddings: 2048,
        rope_theta: 10000.0,
        torch_dtype: "float16".to_string(),
    }
}

fn run_capacity_comparison(cfg: &WorkloadConfig) -> (CapacityResult, CapacityResult) {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));

    // --- Baseline ---
    let baseline = PagedKvCache::new(
        ctx.clone(),
        tiny_llama_cfg(),
        cfg.max_batch,
        cfg.max_seq_len,
        cfg.block_size_tokens,
    )
    .expect("create PagedKvCache");

    let baseline_result = run_baseline_workload(&baseline, cfg);

    // --- KCMM ---
    let dir = tempfile::tempdir().expect("create temp dir");
    let cpu_path = dir
        .path()
        .join("kcmm_mempressure")
        .to_str()
        .expect("valid UTF-8 path")
        .to_string();

    let kcmm_config = KcmmConfig {
        block_size: cfg.block_size_tokens,
        max_blocks: cfg.max_batch
            * ((cfg.max_seq_len + cfg.block_size_tokens - 1) / cfg.block_size_tokens),
        cpu_cache_path: cpu_path,
        tiering: true,
        eviction_policy: "lru".to_string(),
        prefetch_window: 4,
        max_batch_blocks: 64,
            low_watermark_threshold: 0.2,
            background_evict_interval_ms: 100,
            attention_sink_blocks: 1,
            recent_window_blocks: 4,
    };

    let pool = KcmmPool::new(
        ctx.clone(),
        kcmm_config,
        NUM_LAYERS,
        KV_HEADS,
        HEAD_DIM,
        cfg.max_batch,
        cfg.max_seq_len,
    )
    .expect("create KcmmPool with tiering");

    let kcmm_result = run_kcmm_workload(&pool, cfg);

    (baseline_result, kcmm_result)
}

fn completed_count_ratio(baseline: &CapacityResult, kcmm: &CapacityResult) -> f64 {
    if baseline.completed > 0 {
        kcmm.completed as f64 / baseline.completed as f64
    } else {
        f64::NAN
    }
}

fn completed_per_second(result: &CapacityResult) -> f64 {
    if result.elapsed_ms > 0 {
        result.completed as f64 * 1000.0 / result.elapsed_ms as f64
    } else {
        f64::NAN
    }
}

// --- Single config ---

#[test]
fn kcmm_bench_memory_pressure_single() {
    println!("\n=== KCMM Benchmark 5: Memory Pressure — Single Config ===");
    println!("Model: TinyLlama (L={NUM_LAYERS}, kv_heads={KV_HEADS}, head_dim={HEAD_DIM})");

    // Tight config: VA reservation barely fits peak demand, so new arrivals
    // face OOM unless eviction frees space.
    // block_size=16, block_bytes=176 KiB
    // VA reservation: max_batch * max_blocks_per_seq = 16 * ceil(640/16) = 16 * 40 = 640 blocks (~110 MiB)
    // Each seq: prompt=256 → 16 prefill blocks + up to 24 decode blocks = 40 blocks max
    // 16 concurrent seqs × avg 36 blocks = ~576 blocks → close to 640 limit
    // With 32 total arrivals and arrival_interval=12, new seqs arrive while pool is near capacity.
    let cfg = WorkloadConfig {
        block_size_tokens: 16,
        prompt_lens: vec![128, 256],
        max_new_tokens: 384, // long decode → sequences stay active longer
        max_batch: 16,
        max_seq_len: 640,   // 256 + 384
        total_arrivals: 32, // 2× max_batch → guaranteed churn
        arrival_interval: 12,
    };

    let max_blocks_per_seq = (cfg.max_seq_len + cfg.block_size_tokens - 1) / cfg.block_size_tokens;
    let max_blocks_total = cfg.max_batch * max_blocks_per_seq;
    let gpu_budget_mb = (max_blocks_total * cfg.block_bytes()) as f64 / (1024.0 * 1024.0);

    println!("Config: {}", cfg.label());
    println!(
        "block_bytes={} ({} KiB), VA blocks={} (~{:.0} MiB), total_arrivals={}",
        cfg.block_bytes(),
        cfg.block_bytes() / 1024,
        max_blocks_total,
        gpu_budget_mb,
        cfg.total_arrivals,
    );

    let (baseline, kcmm) = run_capacity_comparison(&cfg);

    let completion_ratio = completed_count_ratio(&baseline, &kcmm);
    let baseline_completed_per_sec = completed_per_second(&baseline);
    let kcmm_completed_per_sec = completed_per_second(&kcmm);

    println!("\n  --- Results ---");
    println!("  Baseline (PagedKvCache, no tiering):");
    println!(
        "    completed={}, capped={}, rejected={}, peak_concurrent={}, total_alloc={}",
        baseline.completed,
        baseline.capped,
        baseline.rejected,
        baseline.peak_concurrent,
        baseline.total_blocks_allocated,
    );
    println!(
        "    elapsed={}ms, elapsed_throughput={:.2} completed sequences/s",
        baseline.elapsed_ms, baseline_completed_per_sec,
    );

    println!("  KCMM (KcmmPool, tiering ON):");
    println!(
        "    completed={}, capped={}, rejected={}, peak_concurrent={}, total_alloc={}",
        kcmm.completed,
        kcmm.capped,
        kcmm.rejected,
        kcmm.peak_concurrent,
        kcmm.total_blocks_allocated,
    );
    println!(
        "    evictions={}, cpu_swap_peak={} B, peak_blocks={}",
        kcmm.eviction_count, kcmm.cpu_swap_peak_bytes, kcmm.peak_blocks,
    );
    println!(
        "    elapsed={}ms, elapsed_throughput={:.2} completed sequences/s",
        kcmm.elapsed_ms, kcmm_completed_per_sec,
    );

    println!(
        "\n  completion_ratio = KCMM / Baseline = {} / {} = {:.2}×",
        kcmm.completed, baseline.completed, completion_ratio,
    );
    println!(
        "  elapsed_throughput is reported separately: baseline={:.2} completed sequences/s, kcmm={:.2} completed sequences/s",
        baseline_completed_per_sec, kcmm_completed_per_sec,
    );

    // Primary metric: completed sequences ratio.
    if completion_ratio >= 1.3 {
        println!("  ✅ PASS: completion_ratio ≥ 1.3×");
    } else if completion_ratio >= 1.0 {
        println!(
            "  ⚡ Marginal: completion_ratio {:.2}× (below 1.3× target)",
            completion_ratio
        );
    } else {
        println!("  ❌ FAIL: KCMM completed fewer sequences than baseline");
    }

    // Secondary metrics.
    if kcmm.rejected < baseline.rejected {
        println!(
            "  ✅ KCMM rejected fewer arrivals ({}/{})",
            kcmm.rejected, baseline.rejected
        );
    }
    if kcmm.peak_concurrent > baseline.peak_concurrent {
        println!(
            "  ✅ KCMM supported higher peak concurrency ({}/{})",
            kcmm.peak_concurrent, baseline.peak_concurrent
        );
    }
    if kcmm.eviction_count > 0 {
        println!(
            "  ℹ️  Evictions triggered: {} (tiering is active)",
            kcmm.eviction_count
        );
    } else {
        println!("  ⚠️  No evictions triggered — workload may not be creating memory pressure");
    }
}

// --- Parameter sweep ---

#[test]
fn kcmm_bench_memory_pressure_sweep() {
    println!("\n=== KCMM Benchmark 5: Memory Pressure Sweep ===");
    println!("Model: TinyLlama (L={NUM_LAYERS}, kv_heads={KV_HEADS}, head_dim={HEAD_DIM})");

    // Sweep configurations — all designed to create memory pressure.
    let configs = vec![
        // Tight VA, long decode — sustained concurrency pressure.
        WorkloadConfig {
            block_size_tokens: 16,
            prompt_lens: vec![128, 256],
            max_new_tokens: 384,
            max_batch: 16,
            max_seq_len: 640,
            total_arrivals: 32,
            arrival_interval: 12,
        },
        // Smaller VA, more churn.
        WorkloadConfig {
            block_size_tokens: 16,
            prompt_lens: vec![128, 256],
            max_new_tokens: 256,
            max_batch: 12,
            max_seq_len: 512,
            total_arrivals: 36,
            arrival_interval: 8,
        },
        // Larger block size, different pressure profile.
        WorkloadConfig {
            block_size_tokens: 32,
            prompt_lens: vec![128, 256],
            max_new_tokens: 256,
            max_batch: 16,
            max_seq_len: 512,
            total_arrivals: 32,
            arrival_interval: 12,
        },
        // Very tight, high churn.
        WorkloadConfig {
            block_size_tokens: 16,
            prompt_lens: vec![64, 128, 256],
            max_new_tokens: 128,
            max_batch: 10,
            max_seq_len: 384,
            total_arrivals: 40,
            arrival_interval: 4,
        },
    ];

    println!();
    println!(
        "  {:<50} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "Config",
        "BaseDone",
        "KcmmDone",
        "CompRatio",
        "RejB",
        "RejK",
        "CappedB",
        "CappedK",
        "Evict",
        "BaseMs",
        "KcmmMs",
        "ComplB/s",
        "ComplK/s"
    );
    println!("  {}", "-".repeat(183));

    let mut max_ratio = 0.0f64;
    let mut best_label = String::new();

    for cfg in &configs {
        let (baseline, kcmm) = run_capacity_comparison(cfg);

        let ratio = completed_count_ratio(&baseline, &kcmm);
        let baseline_completed_per_sec = completed_per_second(&baseline);
        let kcmm_completed_per_sec = completed_per_second(&kcmm);

        let status = if ratio >= 1.3 {
            "✅"
        } else if ratio >= 1.0 {
            "⚡"
        } else {
            "❌"
        };

        println!(
            "  {:<50} {:>8} {:>8} {:>10.2}× {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.2} {:>9.2} {:<4}",
            cfg.label(),
            baseline.completed,
            kcmm.completed,
            ratio,
            baseline.rejected,
            kcmm.rejected,
            baseline.capped,
            kcmm.capped,
            kcmm.eviction_count,
            baseline.elapsed_ms,
            kcmm.elapsed_ms,
            baseline_completed_per_sec,
            kcmm_completed_per_sec,
            status,
        );

        if ratio > max_ratio && ratio.is_finite() {
            max_ratio = ratio;
            best_label = cfg.label();
        }
    }

    println!("\n  Best completion_ratio: {max_ratio:.2}×  ({best_label})");

    let any_pass = max_ratio >= 1.3;

    if any_pass {
        println!("  ✅ At least one configuration meets the 1.3× target.");
    } else {
        println!("  ⚠️  No configuration reached 1.3×.");
        if max_ratio >= 1.0 {
            println!("     KCMM shows improvement but below threshold.");
            println!("     Tuning suggestions:");
            println!("     - Decrease arrival_interval to create more memory pressure");
            println!("     - Increase total_arrivals relative to max_batch");
            println!("     - Use smaller block_size for finer-grained eviction");
        }
    }
}
