// tests/step3_benchmarks.rs
//
// Step 3 GPU integration benchmarks: fragmentation rate, max concurrent requests,
// cuMemMap/cuMemUnmap overhead, and runtime fragmentation tracking.
//
// These tests require a CUDA device.

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

    // Use a wider range than the vLLM bench to exercise the allocator.
    // Every sequence starts with a small prompt, then grows via alloc_block
    // during simulated decode.  This measures *capacity at workload*, not
    // worst-case pre-allocation.
    let max_batch = 1024;
    let max_seq_len = 512;
    let block_size = 16;
    let max_new_tokens = 64;

    // Prompt-length distribution matching vLLM's max-concurrency benchmark:
    // short prompts (8, 16, 32 tokens) to maximise the number of admitted
    // sequences.  Cycle through deterministically so the test is reproducible.
    let prompt_lens: &[usize] = &[8, 16, 32];

    let cache = PagedKvCache::new(ctx, cfg.clone(), max_batch, max_seq_len, block_size)
        .expect("create PagedKvCache");

    println!("\n=== Step 3: Capacity at Workload ===");
    println!(
        "model: tiny_llama (kv_heads={}, head_dim={}, layers={})",
        cfg.kv_heads(),
        cfg.head_dim(),
        cfg.num_hidden_layers
    );
    println!(
        "block_size={}, max_seq_len={}, max_new_tokens={}",
        block_size, max_seq_len, max_new_tokens
    );
    println!(
        "block_bytes={}, blocks_per_superblock={}",
        cache.block_bytes, cache.blocks_per_superblock()
    );
    println!(
        "prompt lens (cycle): {:?}",
        prompt_lens
    );

    // ── Phase 1: admit sequences with initial prompt blocks ──
    let mut admitted = 0usize;
    for i in 0..max_batch {
        let pl = prompt_lens[i % prompt_lens.len()];
        let initial_blocks = (pl + block_size - 1) / block_size;

        match cache.alloc_sequence(initial_blocks) {
            Ok(table) => {
                cache.register_sequence(table);
                cache.update_seq_len(admitted, pl);
                admitted += 1;
            }
            Err(e) => {
                println!(
                    "admission stopped at seq {} (prompt_len={}): {:?}",
                    admitted, pl, e
                );
                break;
            }
        }
    }
    println!("\nPhase 1 (admission): {} sequences admitted", admitted);

    // ── Phase 2: simulate decode growth ──
    let mut capped_seqs = 0usize;
    for seq_idx in 0..admitted {
        let mut current_len = cache.get_seq_len(seq_idx);
        let target_len = (current_len + max_new_tokens).min(max_seq_len);

        while current_len < target_len {
            current_len += 1;
            let blocks_needed = (current_len + block_size - 1) / block_size;
            if blocks_needed > cache.seq_block_count(seq_idx) {
                match cache.alloc_block() {
                    Ok(block_idx) => {
                        cache.append_block_to_sequence(seq_idx, block_idx);
                    }
                    Err(_) => {
                        // OOM during decode — cap here
                        current_len -= 1;
                        capped_seqs += 1;
                        break;
                    }
                }
            }
            cache.update_seq_len(seq_idx, current_len);
        }
    }
    println!(
        "Phase 2 (decode): {} sequences grew to max_new_tokens, {} capped (OOM)",
        admitted.saturating_sub(capped_seqs),
        capped_seqs,
    );

    let stats = cache.stats();
    println!("\nResults:");
    println!("  capacity at workload:     {}", admitted);
    println!("  total blocks allocated:   {}", stats.total_blocks_allocated);
    println!("  blocks in use:            {}", stats.blocks_in_use);
    println!("  free blocks in pool:      {}", stats.free_blocks_in_pool);
    println!("  superblocks allocated:    {}", stats.superblocks_allocated);
    println!(
        "  physical memory:          {:.2} MiB",
        stats.physical_memory_mib
    );
    println!(
        "  avg blocks / request:     {:.2}",
        stats.blocks_in_use as f32 / admitted.max(1) as f32
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
    for i in 0..admitted {
        cache.unregister_sequence(i);
    }
    let after = cache.stats();
    println!("\nAfter freeing all:");
    println!("  blocks in use:            {}", after.blocks_in_use);
    println!("  free blocks in pool:      {}", after.free_blocks_in_pool);
    println!(
        "  physical idle ratio:      {:.4}",
        cache.physical_idle_ratio()
    );

    assert!(admitted > 0, "should admit at least some sequences");
    println!("=== End Capacity at Workload ===\n");
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

    let iters = 64;
    

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
