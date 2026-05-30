// tests/step3_benchmarks.rs
//
// Step 3 GPU integration benchmarks: fragmentation rate, max concurrent requests,
// cuMemMap/cuMemUnmap overhead, and runtime fragmentation tracking.
//
// These tests require a CUDA device.

use baseline_llm_os::cache::fragmentation_tracker::{
    format_bytes, RuntimeFragmentationTracker,
};
use baseline_llm_os::cache::paged_kv::PagedKvCache;
use baseline_llm_os::config::ModelConfig;
use baseline_llm_os::cuda::CudaContext;
use std::sync::Arc;

const SUPERBLOCK_SIZE: usize = 2 * 1024 * 1024; // 2 MiB

// --- Step 3: Maximum Concurrent Requests ---

#[test]
fn step3_max_concurrent_requests() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
    let cfg = ModelConfig::tiny_llama();

    let max_batch = 256;
    let max_seq_len = 256;
    let block_size = 16;
    let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;

    let cache = PagedKvCache::new(ctx, cfg.clone(), max_batch, max_seq_len, block_size)
        .expect("create PagedKvCache");

    println!("\n=== Step 3: Maximum Concurrent Requests ===");
    println!(
        "model: tiny_llama (kv_heads={}, head_dim={}, layers={})",
        cfg.kv_heads(),
        cfg.head_dim(),
        cfg.num_hidden_layers
    );
    println!(
        "block_size={}, max_seq_len={}, blocks_per_seq={}",
        block_size, max_seq_len, max_blocks_per_seq
    );
    println!(
        "block_bytes={}, blocks_per_superblock={}",
        cache.block_bytes, cache.blocks_per_superblock()
    );

    // Allocate until OOM
    let mut allocated = 0usize;
    for _ in 0..max_batch {
        match cache.alloc_sequence(max_blocks_per_seq) {
            Ok(table) => {
                cache.register_sequence(table);
                allocated += 1;
            }
            Err(e) => {
                println!("alloc_sequence failed at seq {}: {:?}", allocated, e);
                break;
            }
        }
    }

    let stats = cache.stats();
    println!("\nResults:");
    println!("  max concurrent requests:  {}", allocated);
    println!("  total blocks allocated:   {}", stats.total_blocks_allocated);
    println!("  blocks in use:            {}", stats.blocks_in_use);
    println!("  free blocks in pool:      {}", stats.free_blocks_in_pool);
    println!("  superblocks allocated:    {}", stats.superblocks_allocated);
    println!(
        "  physical memory:          {:.2} MiB",
        stats.physical_memory_mib
    );
    println!(
        "  physical blocks / request: {}",
        stats.blocks_in_use as f32 / allocated.max(1) as f32
    );

    // Check how maps are batched
    let num_layers = cfg.num_hidden_layers;
    let maps_per_superblock = num_layers * 2;
    let total_map_calls = stats.superblocks_allocated * maps_per_superblock;
    println!(
        "  total cuMemMap calls:     {} ({} per logical superblock position)",
        total_map_calls, maps_per_superblock
    );

    // Free all
    for i in 0..allocated {
        cache.unregister_sequence(i);
    }
    let after = cache.stats();
    println!("\nAfter freeing all:");
    println!("  blocks in use:            {}", after.blocks_in_use);
    println!("  free blocks in pool:      {}", after.free_blocks_in_pool);
    println!(
        "  physical memory waste ratio: {:.4}",
        cache.fragmentation_ratio()
    );

    assert!(allocated > 0, "should allocate at least some sequences");
    println!("=== End Max Concurrent Requests ===\n");
}

// --- Step 3: Fragmentation Rate ---

#[test]
fn step3_fragmentation_rate() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
    let cfg = ModelConfig::tiny_llama();

    let max_batch = 64;
    let max_seq_len = 256;
    let block_size = 16;
    let max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;

    let cache = PagedKvCache::new(ctx, cfg.clone(), max_batch, max_seq_len, block_size)
        .expect("create PagedKvCache");

    println!("\n=== Step 3: Fragmentation Rate ===");

    // Phase 1: allocate all sequences
    let mut tables: Vec<Vec<u32>> = Vec::new();
    let mut seq_indices: Vec<usize> = Vec::new();
    for _i in 0..max_batch {
        if let Ok(table) = cache.alloc_sequence(max_blocks_per_seq) {
            let si = cache.register_sequence(table.clone());
            cache.update_seq_len(si, max_seq_len);
            tables.push(table);
            seq_indices.push(si);
        }
    }

    let stats = cache.stats();
    println!("After allocating {} sequences:", tables.len());
    println!(
        "  internal_fragmentation: {:.4}",
        stats.internal_fragmentation
    );
    println!("  blocks in use: {}", stats.blocks_in_use);
    println!(
        "  physical memory: {:.2} MiB",
        stats.physical_memory_mib
    );
    println!("  superblocks: {}", stats.superblocks_allocated);

    // Phase 2: free 50% to create holes
    for (idx, table) in tables.iter().enumerate() {
        if idx % 2 != 0 {
            cache.free_sequence(table);
        }
    }

    let mid_stats = cache.stats();
    println!("\nAfter freeing 50% of sequences:");
    println!("  blocks in use:         {}", mid_stats.blocks_in_use);
    println!(
        "  free blocks in pool:   {}",
        mid_stats.free_blocks_in_pool
    );
    println!(
        "  physical memory waste ratio: {:.4}",
        cache.fragmentation_ratio()
    );

    // Phase 3: re-allocate with shorter sequences
    let half_blocks = max_blocks_per_seq / 2;
    let half_seq_len = half_blocks * block_size / 2;
    let mut new_count = 0usize;
    for _ in 0..(max_batch / 2) {
        if let Ok(table) = cache.alloc_sequence(half_blocks) {
            let si = cache.register_sequence(table);
            cache.update_seq_len(si, half_seq_len);
            new_count += 1;
        } else {
            break;
        }
    }

    let final_stats = cache.stats();
    println!(
        "\nAfter re-allocating {} shorter sequences:",
        new_count
    );
    println!(
        "  active sequences:     {}",
        final_stats.active_sequences
    );
    println!("  blocks in use:        {}", final_stats.blocks_in_use);
    println!(
        "  free blocks in pool:  {}",
        final_stats.free_blocks_in_pool
    );
    println!(
        "  total blocks:         {}",
        final_stats.total_blocks_allocated
    );
    println!(
        "  superblocks:          {}",
        final_stats.superblocks_allocated
    );

    let wasted_tokens_in_last_block =
        new_count * (block_size - (half_seq_len % block_size));
    println!("\nInternal fragmentation:");
    println!(
        "  total tokens stored:     {}",
        final_stats.total_tokens_stored
    );
    println!(
        "  total slots allocated:   {}",
        final_stats.blocks_in_use * block_size
    );
    println!(
        "  internal_fragmentation:  {:.4} ({:.2}%)",
        final_stats.internal_fragmentation,
        final_stats.internal_fragmentation * 100.0
    );
    println!(
        "  wasted slots in last blocks: ~{}",
        wasted_tokens_in_last_block
    );
    println!("=== End Fragmentation Rate ===\n");

    assert!(!tables.is_empty());
}

// --- Step 3: cuMemMap/cuMemUnmap Overhead ---

#[test]
fn step3_cumemmap_overhead() {
    use baseline_llm_os::cache::cuda_vmm::CudaVmm;

    let vmm = CudaVmm::new(0).expect("cuda device 0");
    let cfg = ModelConfig::tiny_llama();
    let num_layers = cfg.num_hidden_layers;

    println!("\n=== Step 3: cuMemMap/cuMemUnmap Overhead ===");
    println!("GPU map granularity: {} bytes", vmm.map_granularity);
    println!(
        "num_layers={}, maps per superblock = {} (K+V per layer)",
        num_layers,
        num_layers * 2
    );

    // Setup: one 2MB VA region per layer, one 2MB physical handle
    let va_k: Vec<u64> = (0..num_layers)
        .map(|_| vmm.reserve_address(SUPERBLOCK_SIZE).expect("reserve K"))
        .collect();
    let va_v: Vec<u64> = (0..num_layers)
        .map(|_| vmm.reserve_address(SUPERBLOCK_SIZE).expect("reserve V"))
        .collect();

    let iters = 16;

    // --- Per-layer mapping benchmark (mimics per-block approach) ---
    let per_layer_sizes = [
        8192, 16384, 32768, 65536, 131072, 262144, 524288, SUPERBLOCK_SIZE,
    ];

    println!("\nPer-call latency vs. mapping size:");
    println!("  {:>8}  {:>12}  {:>12}", "size", "map (µs)", "unmap (µs)");

    for &size in &per_layer_sizes {
        if size > SUPERBLOCK_SIZE || size < vmm.map_granularity {
            continue;
        }
        let phys = vmm.create_physical(size).expect("create phys");

        // Warmup
        for _ in 0..2 {
            for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                vmm.map(vk, 0, phys, 0, size).unwrap();
                vmm.unmap(vk, 0, size).unwrap();
                vmm.map(vv, 0, phys, 0, size).unwrap();
                vmm.unmap(vv, 0, size).unwrap();
            }
        }

        let start = std::time::Instant::now();
        for _ in 0..iters {
            for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
                vmm.map(vk, 0, phys, 0, size).unwrap();
                vmm.map(vv, 0, phys, 0, size).unwrap();
                vmm.unmap(vk, 0, size).unwrap();
                vmm.unmap(vv, 0, size).unwrap();
            }
        }
        let elapsed = start.elapsed();
        let total_ops = iters * num_layers * 2 * 2; // map+unmap × K+V
        let avg_us = elapsed.as_micros() as f64 / total_ops as f64;

        println!("  {:>8}  {:>12.2}  {:>12.2}", size, avg_us, avg_us);

        vmm.release_physical(phys).expect("release");
    }

    // --- Full superblock (2MB) mapping per layer ---
    println!("\nFull-superblock (2MB) mapping per layer:");
    let phys = vmm
        .create_physical(SUPERBLOCK_SIZE)
        .expect("create phys");

    // Warmup
    for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
        vmm.map(vk, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
        vmm.map(vv, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
        vmm.unmap(vk, 0, SUPERBLOCK_SIZE).unwrap();
        vmm.unmap(vv, 0, SUPERBLOCK_SIZE).unwrap();
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
            vmm.map(vk, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.map(vv, 0, phys, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.unmap(vk, 0, SUPERBLOCK_SIZE).unwrap();
            vmm.unmap(vv, 0, SUPERBLOCK_SIZE).unwrap();
        }
    }
    let elapsed = start.elapsed();
    let total_ops = iters * num_layers * 2 * 2;
    let avg_us = elapsed.as_micros() as f64 / total_ops as f64;
    println!("  avg per 2MB map/unmap:  {:.2} µs", avg_us);
    println!(
        "  total for {} layers:    {:.2} µs",
        num_layers,
        avg_us * num_layers as f64 * 2.0
    );

    // Cleanup
    for (&vk, &vv) in va_k.iter().zip(va_v.iter()) {
        vmm.unmap(vk, 0, SUPERBLOCK_SIZE).unwrap();
        vmm.unmap(vv, 0, SUPERBLOCK_SIZE).unwrap();
    }
    vmm.release_physical(phys).expect("release phys");
    for v in va_k.iter().chain(va_v.iter()) {
        vmm.free_address(*v, SUPERBLOCK_SIZE).expect("free va");
    }

    println!("=== End cuMemMap/cuMemUnmap Overhead ===\n");
}

// --- Step 3: Internal Fragmentation Analysis ---

#[test]
fn step3_internal_fragmentation_analysis() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
    let cfg = ModelConfig::tiny_llama();

    let max_batch = 16;
    let max_seq_len = 128;
    let block_size = 16;
    let cache =
        PagedKvCache::new(ctx, cfg, max_batch, max_seq_len, block_size)
            .expect("create PagedKvCache");

    println!("\n=== Step 3: Internal Fragmentation Analysis ===");
    println!("block_size={} tokens", block_size);

    let seq_lengths = [
        1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 100, 127, 128, 7,
    ];
    let mut total_wasted = 0usize;
    let mut total_slots = 0usize;

    for &sl in &seq_lengths {
        let blocks_needed = (sl + block_size - 1) / block_size;
        let table = cache.alloc_sequence(blocks_needed).expect("alloc");
        let seq_idx = cache.register_sequence(table);
        cache.update_seq_len(seq_idx, sl);

        let slots = blocks_needed * block_size;
        let waste = slots - sl;
        let frag = waste as f32 / slots as f32;
        total_wasted += waste;
        total_slots += slots;
        println!(
            "  seq_len={:3}  blocks={:2}  slots={:3}  waste={:2}  frag={:.3}",
            sl, blocks_needed, slots, waste, frag
        );
    }

    let overall_frag = total_wasted as f32 / total_slots as f32;
    println!("\nSummary:");
    println!("  total sequences:         {}", seq_lengths.len());
    println!("  total slots allocated:   {}", total_slots);
    println!(
        "  total tokens stored:     {}",
        total_slots - total_wasted
    );
    println!("  total wasted slots:      {}", total_wasted);
    println!(
        "  overall internal frag:   {:.4} ({:.2}%)",
        overall_frag,
        overall_frag * 100.0
    );
    println!(
        "  average waste per seq:   {:.1} tokens",
        total_wasted as f32 / seq_lengths.len() as f32
    );

    let stats = cache.stats();
    println!(
        "\n  cache.internal_fragmentation() = {:.4}",
        stats.internal_fragmentation
    );
    println!("=== End Internal Fragmentation Analysis ===\n");

    assert_eq!(
        total_slots - total_wasted,
        seq_lengths.iter().sum::<usize>()
    );
}

// --- Step 3: Runtime Fragmentation ---

#[test]
fn step3_runtime_fragmentation() {
    let ctx = Arc::new(CudaContext::new(0).expect("cuda device 0"));
    let cfg = ModelConfig::tiny_llama();

    // Bimodal prompt length distribution (200 entries).
    // ~60% short queries (10-60 tokens), ~30% medium (100-250),
    // ~10% long (260-500).  This ensures peak block usage exceeds one
    // 256-block superblock so that memory_allocated_not_free varies
    // across time steps.
    static PROMPT_LENS: [usize; 200] = [
        // Short prompts — simple queries (120 entries, 10-60 tokens)
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        14, 14, 14, 14, 14, 14, 14, 14,
        16, 16, 16, 16, 16, 16,
        18, 18, 18, 18, 18, 18,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        22, 22, 22, 22, 22, 22,
        24, 24, 24, 24, 24,
        26, 26, 26, 26, 26,
        28, 28, 28, 28,
        30, 30, 30, 30, 30,
        32, 32, 32, 32,
        34, 34, 34, 34,
        36, 36, 36,
        38, 38, 38,
        40, 40, 40, 40, 40, 40, 40, 40,
        44, 44, 44, 44,
        48, 48, 48,
        52, 52,
        56, 56,
        60, 60,
        // Medium prompts — context-heavy queries (62 entries, 100-250)
        100, 100, 100, 100, 100, 100,
        110, 110, 110, 110, 110,
        120, 120, 120, 120, 120, 120,
        130, 130, 130, 130,
        140, 140, 140, 140,
        150, 150, 150, 150, 150, 150,
        160, 160, 160,
        170, 170, 170,
        180, 180, 180, 180,
        190, 190,
        200, 200, 200, 200, 200, 200, 200,
        210, 210, 210,
        220, 220, 220, 220,
        230, 230,
        240, 240,
        250, 250,
        // Long prompts — long-context documents (24 entries, 260-500)
        260, 270, 280, 290,
        300, 300, 310, 320, 330,
        340, 350, 350, 360, 370,
        390, 400, 410, 420, 430,
        440, 450, 460, 470,
        480, 490, 500,
    ];

    const BLOCK_SIZE: usize = 16;
    const MAX_BATCH: usize = 32;
    const MAX_SEQ_LEN: usize = 512;
    const MAX_NEW_TOKENS: usize = 128;
    const TOTAL_REQUESTS: usize = 200;
    const STEPS_PER_ROUND: usize = 4;

    let cache = PagedKvCache::new(
        ctx,
        cfg.clone(),
        MAX_BATCH,
        MAX_SEQ_LEN,
        BLOCK_SIZE,
    )
    .expect("create PagedKvCache");

    println!("\n=== Step 3: Runtime Fragmentation Rate ===");
    println!(
        "model: tiny_llama (kv_heads={}, head_dim={}, layers={})",
        cfg.kv_heads(),
        cfg.head_dim(),
        cfg.num_hidden_layers
    );
    println!(
        "block_size={}, max_seq_len={}, max_batch={}",
        BLOCK_SIZE, MAX_SEQ_LEN, MAX_BATCH
    );
    println!(
        "block_bytes={}, blocks_per_superblock={}",
        cache.block_bytes,
        cache.blocks_per_superblock()
    );
    println!(
        "total_requests={}, max_new_tokens={}, steps_per_round={}",
        TOTAL_REQUESTS, MAX_NEW_TOKENS, STEPS_PER_ROUND
    );

    // Print prompt length distribution summary
    let mut sorted_lens = PROMPT_LENS.to_vec();
    sorted_lens.sort();
    let p50 = sorted_lens[sorted_lens.len() / 2];
    let p95 = sorted_lens[(sorted_lens.len() as f64 * 0.95) as usize];
    let p99 = sorted_lens[(sorted_lens.len() as f64 * 0.99) as usize];
    println!(
        "\nPrompt length distribution ({} samples):",
        sorted_lens.len()
    );
    println!(
        "  min={}, p50={}, p95={}, p99={}, max={}",
        sorted_lens[0],
        p50,
        p95,
        p99,
        sorted_lens.last().unwrap()
    );
    println!(
        "  mean={:.1}",
        sorted_lens.iter().sum::<usize>() as f32 / sorted_lens.len() as f32
    );

    // Prepare all requests upfront
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let requests: Vec<(usize, usize)> = (0..TOTAL_REQUESTS)
        .map(|_| {
            let pl = PROMPT_LENS[rng.gen_range(0..PROMPT_LENS.len())];
            (pl, pl + MAX_NEW_TOKENS)
        })
        .collect();

    let mut tracker =
        RuntimeFragmentationTracker::new(cfg.kv_heads() * cfg.head_dim() * 2);

    struct SimRequest {
        seq_idx: usize,
        #[allow(dead_code)]
        prompt_len: usize,
        target_len: usize,
        position: usize,
        num_blocks: usize,
        done: bool,
    }

    let mut running: Vec<SimRequest> = Vec::new();
    let mut next_req = 0usize;
    let mut total_completed = 0usize;
    let mut admission_failures = 0usize;

    // Simulation loop
    let mut round = 0usize;
    loop {
        round += 1;

        // 1. Admit new requests while capacity allows
        while next_req < TOTAL_REQUESTS && running.len() < MAX_BATCH {
            let (prompt_len, target_len) = requests[next_req];
            let blocks_needed = (prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            match cache.alloc_sequence(blocks_needed) {
                Ok(table) => {
                    let seq_idx = cache.register_sequence(table);
                    cache.update_seq_len(seq_idx, prompt_len);
                    running.push(SimRequest {
                        seq_idx,
                        prompt_len,
                        target_len,
                        position: prompt_len,
                        num_blocks: blocks_needed,
                        done: false,
                    });
                    next_req += 1;
                }
                Err(_) => {
                    admission_failures += 1;
                    break;
                }
            }
        }

        // 2. Snapshot fragmentation after admission
        tracker.record(&cache);

        // 3. Simulate decode steps
        for req in running.iter_mut() {
            if req.done {
                continue;
            }
            let advance = STEPS_PER_ROUND
                .min(req.target_len.saturating_sub(req.position));
            req.position += advance;

            let blocks_needed = (req.position + BLOCK_SIZE - 1) / BLOCK_SIZE;
            while blocks_needed > req.num_blocks {
                match cache.alloc_block() {
                    Ok(block_idx) => {
                        cache.append_block_to_sequence(req.seq_idx, block_idx);
                        req.num_blocks += 1;
                    }
                    Err(_) => {
                        req.position = req.num_blocks * BLOCK_SIZE;
                        break;
                    }
                }
            }
            cache.update_seq_len(req.seq_idx, req.position);

            if req.position >= req.target_len {
                req.done = true;
            }
        }

        // 4. Snapshot after decode step
        tracker.record(&cache);

        // 5. Remove completed requests
        let mut i = 0;
        while i < running.len() {
            if running[i].done {
                let req = running.remove(i);
                cache.unregister_sequence(req.seq_idx);
                total_completed += 1;
            } else {
                i += 1;
            }
        }

        // 6. Check termination
        if total_completed >= TOTAL_REQUESTS
            || (running.is_empty() && next_req >= TOTAL_REQUESTS)
        {
            break;
        }

        if round > 2000 {
            eprintln!("WARNING: simulation exceeded 2000 rounds, breaking");
            break;
        }
    }

    // --- Report ---
    let avg_ratio = tracker.average_ratio();
    let stddev = tracker.ratio_stddev();
    let peak = tracker.peak_ratio();
    let min_r = tracker.min_ratio();
    let samples = tracker.samples();

    println!("\n--- Runtime Fragmentation Results ---");
    println!("  total requests simulated:  {}", TOTAL_REQUESTS);
    println!("  total completed:           {}", total_completed);
    println!("  admission failures (OOM):  {}", admission_failures);
    println!("  simulation rounds:         {}", round);
    println!(
        "  fragmentation samples:     {}",
        tracker.sample_count()
    );
    println!();
    println!(
        "  avg runtime fragmentation ratio:  {:.4}",
        avg_ratio
    );
    println!("  stddev:                            {:.4}", stddev);
    println!("  peak (worst):                      {:.4}", peak);
    println!("  min (best):                        {:.4}", min_r);
    println!();

    // Show how the ratio varies with load
    if !samples.is_empty() {
        let active_seqs: Vec<usize> =
            samples.iter().map(|s| s.active_sequences).collect();
        let max_active = active_seqs.iter().max().copied().unwrap_or(0);
        let avg_active =
            active_seqs.iter().sum::<usize>() as f32 / active_seqs.len() as f32;

        println!("  max concurrent sequences:  {}", max_active);
        println!("  avg concurrent sequences:  {:.1}", avg_active);

        let buckets = 5;
        let bucket_size = (max_active.max(1) + buckets - 1) / buckets;
        println!();
        println!("  Fragmentation vs. load:");
        println!(
            "  {:>20}  {:>8}  {:>12}",
            "active seqs range", "samples", "avg ratio"
        );
        for b in 0..buckets {
            let lo = b * bucket_size;
            let hi = ((b + 1) * bucket_size).min(max_active + 1);
            let bucket_samples: Vec<_> = samples
                .iter()
                .filter(|s| s.active_sequences >= lo && s.active_sequences < hi)
                .collect();
            if bucket_samples.is_empty() {
                continue;
            }
            let bucket_avg = bucket_samples
                .iter()
                .map(|s| s.ratio)
                .sum::<f32>()
                / bucket_samples.len() as f32;
            println!(
                "  {:>20}  {:>8}  {:>12.4}",
                format!("[{}, {})", lo, hi),
                bucket_samples.len(),
                bucket_avg
            );
        }
    }

    // --- mem_not_free variation ---
    // This is the key metric: it should vary when block usage crosses
    // superblock boundaries (256, 512, ... blocks).
    if !samples.is_empty() {
        let mem_values: Vec<usize> = samples
            .iter()
            .map(|s| s.memory_allocated_not_free)
            .collect();
        let mem_min = mem_values.iter().min().copied().unwrap_or(0);
        let mem_max = mem_values.iter().max().copied().unwrap_or(0);
        let mut mem_unique: Vec<usize> = mem_values.clone();
        mem_unique.sort();
        mem_unique.dedup();

        println!();
        println!("  --- mem_not_free variation ---");
        println!(
            "  min:  {}  max:  {}",
            format_bytes(mem_min),
            format_bytes(mem_max)
        );
        println!("  unique values observed:  {}", mem_unique.len());
        if mem_unique.len() <= 8 {
            println!(
                "  all values: {}",
                mem_unique
                    .iter()
                    .map(|&v| format_bytes(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        } else {
            println!(
                "  values: {}",
                mem_unique
                    .iter()
                    .map(|&v| format_bytes(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    // Print evenly-spaced sample snapshots to show variation over time
    if samples.len() >= 6 {
        println!();
        println!(
            "  Sample snapshots ({} of {} samples, evenly spaced):",
            (6usize).min(samples.len()),
            samples.len()
        );
        println!(
            "  {:>6}  {:>10}  {:>14}  {:>14}  {:>10}",
            "step", "active", "mem_not_free", "mem_tokens", "ratio"
        );
        let n = samples.len();
        let indices: Vec<usize> = if n <= 10 {
            (0..n).collect()
        } else {
            let step = n as f32 / 7.0;
            (0..6).map(|i| (i as f32 * step) as usize).collect()
        };
        for &i in &indices {
            let s = &samples[i];
            println!(
                "  {:>6}  {:>10}  {:>14}  {:>14}  {:>10.4}",
                i,
                s.active_sequences,
                format_bytes(s.memory_allocated_not_free),
                format_bytes(s.memory_active_tokens),
                s.ratio
            );
        }
        // Always show the last sample
        if !indices.contains(&(n - 1)) {
            let s = &samples[n - 1];
            println!(
                "  {:>6}  {:>10}  {:>14}  {:>14}  {:>10.4}",
                n - 1,
                s.active_sequences,
                format_bytes(s.memory_allocated_not_free),
                format_bytes(s.memory_active_tokens),
                s.ratio
            );
        }
    }

    // Fragmentation breakdown by mem_not_free level
    if !samples.is_empty() {
        let mut mem_buckets: std::collections::BTreeMap<usize, Vec<f32>> =
            std::collections::BTreeMap::new();
        for s in samples {
            mem_buckets
                .entry(s.memory_allocated_not_free)
                .or_default()
                .push(s.ratio);
        }
        if mem_buckets.len() > 1 {
            println!();
            println!("  Fragmentation vs. mem_not_free:");
            println!(
                "  {:>16}  {:>8}  {:>10}",
                "mem_not_free", "samples", "avg ratio"
            );
            for (mem, ratios) in &mem_buckets {
                let avg = ratios.iter().sum::<f32>() / ratios.len() as f32;
                println!(
                    "  {:>16}  {:>8}  {:>10.4}",
                    format_bytes(*mem),
                    ratios.len(),
                    avg
                );
            }
        }
    }

    // Compare with static fragmentation
    let stats = cache.stats();
    println!();
    println!("  --- Final cache state ---");
    println!(
        "  internal fragmentation:    {:.4}",
        stats.internal_fragmentation
    );
    println!(
        "  superblocks:               {}",
        stats.superblocks_allocated
    );
    println!(
        "  physical memory:           {:.2} MiB",
        stats.physical_memory_mib
    );

    let bytes_per_token = cfg.kv_heads()
        * cfg.head_dim()
        * 2 // f16 = 2 bytes
        * cfg.num_hidden_layers
        * 2;
    println!(
        "  bytes per token (K+V all layers): {}",
        bytes_per_token
    );
    println!();

    println!("=== End Runtime Fragmentation Rate ===\n");

    assert!(
        tracker.sample_count() > 0,
        "should have recorded fragmentation samples"
    );
    assert!(
        avg_ratio >= 0.0 && avg_ratio <= 1.0,
        "fragmentation ratio should be in [0.0, 1.0], got {:.4}",
        avg_ratio
    );
    assert!(total_completed > 0, "should have completed some requests");
}
