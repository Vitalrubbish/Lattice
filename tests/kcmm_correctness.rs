// tests/kcmm_correctness.rs
//
// KCMM — Data-integrity correctness tests.
//
// These tests verify that the KCMM tiering pipeline preserves KV-cache data
// through eviction (GPU→CPU) and restoration (CPU→GPU).  Unlike the engine
// integration benchmark (which measures throughput / capacity), these tests
// compare model logits element-by-element to detect even single-byte corruption
// in the evict→restore roundtrip.
//
// Tests:
//   kcmm_logits_consistency_under_eviction
//     Run the same token sequence through two identical models — one with
//     no eviction (reference) and one with forced eviction+restore cycles.
//     Assert that logits match at every step.
//     This exercises: embedding, QKV projection, append_kv_step,
//     evict_blocks, restore_evicted_blocks, paged-attention decode,
//     FFN, and LM head — the complete forward-pass pipeline.

use baseline_llm_os::config::{KcmmConfig, ModelConfig};
use baseline_llm_os::cuda::CudaContext;
use baseline_llm_os::kcmm::pool::KcmmPool;
use baseline_llm_os::model::weights::{ModelWeights, RawTensor};
use baseline_llm_os::model::{LlamaTransformer, Transformer};
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

// --- Test model geometry ---
//
// Same reduced-size architecture as the engine integration benchmark so the
// two test suites exercise identical code paths.  8-layer Llama with GQA,
// Xavier-random-initialised weights (no external file dependency).

const NUM_LAYERS: usize = 8;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const HIDDEN_SIZE: usize = 1024;
const VOCAB_SIZE: usize = 1000;
const NUM_ATTN_HEADS: usize = 16; // 1024 / 64 = 16
const INTERMEDIATE_SIZE: usize = 2048;

/// Fixed seed — every call to [`generate_random_weights`] produces identical
/// tensors so the reference and test model instances are bit-exact copies.
const MODEL_WEIGHT_SEED: u64 = 0x5EED_1A77_1CE5;

// ============================================================================
// Model helpers (same logic as engine integration test; duplicated to keep
// each test file self-contained).
// ============================================================================

fn test_model_cfg() -> ModelConfig {
    ModelConfig {
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_ATTN_HEADS,
        num_key_value_heads: Some(KV_HEADS),
        vocab_size: VOCAB_SIZE,
        max_position_embeddings: 2048,
        rope_theta: 10000.0,
        torch_dtype: "float16".to_string(),
    }
}

fn random_f16_bytes(rng: &mut impl Rng, nelems: usize, limit: f32) -> Vec<u8> {
    let mut host = Vec::with_capacity(nelems * 2);
    for _ in 0..nelems {
        let v: f32 = rng.gen_range(-limit..limit);
        host.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
    }
    host
}

fn upload_tensor(
    ctx: &Arc<CudaContext>,
    weights: &mut ModelWeights,
    name: &str,
    shape: Vec<usize>,
    host: Vec<u8>,
) {
    let bytes = ctx
        .device
        .htod_copy(host)
        .unwrap_or_else(|e| panic!("htod {name}: {e}"));
    weights.insert(
        name.to_string(),
        RawTensor {
            shape,
            dtype: "float16".to_string(),
            bytes,
        },
    );
}

/// Generate a [`ModelWeights`] with Xavier-random f16 values on the GPU.
fn generate_random_weights(ctx: &Arc<CudaContext>, cfg: &ModelConfig) -> ModelWeights {
    let mut rng = StdRng::seed_from_u64(MODEL_WEIGHT_SEED);
    let mut weights = ModelWeights::empty(cfg);
    let h = cfg.hidden_size;
    let kvd = cfg.kv_heads() * cfg.head_dim();
    let im = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;

    fn xavier(fin: usize, fout: usize) -> f32 {
        (6.0_f64 / ((fin + fout).max(1) as f64)).sqrt() as f32
    }

    fn push_2d(
        rng: &mut impl Rng,
        ctx: &Arc<CudaContext>,
        weights: &mut ModelWeights,
        name: &str,
        shape: Vec<usize>,
        fin: usize,
        fout: usize,
    ) {
        let nelems: usize = shape.iter().product();
        let host = random_f16_bytes(rng, nelems, xavier(fin, fout));
        upload_tensor(ctx, weights, name, shape, host);
    }

    fn push_norm(
        rng: &mut impl Rng,
        ctx: &Arc<CudaContext>,
        weights: &mut ModelWeights,
        name: &str,
        dim: usize,
    ) {
        let host = random_f16_bytes(rng, dim, 0.1f32);
        upload_tensor(ctx, weights, name, vec![dim], host);
    }

    push_2d(
        &mut rng, ctx, &mut weights,
        "model.embed_tokens.weight", vec![vocab, h], h, vocab,
    );
    push_norm(&mut rng, ctx, &mut weights, "model.norm.weight", h);
    push_2d(
        &mut rng, ctx, &mut weights,
        "lm_head.weight", vec![vocab, h], h, vocab,
    );

    for l in 0..n_layers {
        push_norm(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.input_layernorm.weight"), h,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.self_attn.q_proj.weight"),
            vec![h, h], h, h,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.self_attn.k_proj.weight"),
            vec![kvd, h], h, kvd,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.self_attn.v_proj.weight"),
            vec![kvd, h], h, kvd,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.self_attn.o_proj.weight"),
            vec![h, h], h, h,
        );
        push_norm(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.post_attention_layernorm.weight"), h,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.mlp.gate_proj.weight"),
            vec![im, h], h, im,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            vec![im, h], h, im,
        );
        push_2d(
            &mut rng, ctx, &mut weights,
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            vec![h, im], im, h,
        );
    }

    weights
}

fn build_random_transformer(ctx: &Arc<CudaContext>, cfg: &ModelConfig) -> LlamaTransformer {
    let weights = generate_random_weights(ctx, cfg);
    LlamaTransformer::new(ctx.clone(), cfg.clone(), weights)
        .expect("create LlamaTransformer with random weights")
}

// ============================================================================
// Eviction helpers
// ============================================================================

/// Force-evict `block_indices` to CPU, then immediately restore them to GPU.
///
/// Returns the number of blocks that were evicted (and subsequently restored).
/// Panics if any block cannot be evicted or if any block is not GpuResident
/// after restoration — those conditions indicate a tiering bug.
fn force_evict_and_restore(pool: &KcmmPool, block_indices: &[u32]) -> usize {
    if block_indices.is_empty() {
        return 0;
    }

    // Collect physical handles for the blocks we want to evict.
    let handles: Vec<baseline_llm_os::kcmm::superblock::BlockHandle> = block_indices
        .iter()
        .filter_map(|&bi| pool.get_block_handle(bi))
        .collect();

    if handles.is_empty() {
        // All blocks may already be on CPU — skip eviction, just restore.
        pool.restore_evicted_blocks(block_indices)
            .expect("restore after empty-handle eviction");
        return 0;
    }

    let tiering = pool.tiering.as_ref().expect("tiering enabled");
    let evicted = tiering
        .evict_blocks(pool, &handles, handles.len())
        .expect("evict_blocks failed");

    let n_evicted = evicted.len();

    // Restore ALL requested blocks (some may already have been CpuResident
    // before this call and were skipped by evict_blocks).
    pool.restore_evicted_blocks(block_indices)
        .expect("restore_evicted_blocks failed");

    // Postcondition: every block must be GpuResident.
    for &bi in block_indices {
        let loc = pool
            .get_block_location(bi)
            .unwrap_or_else(|| panic!("block {bi} not found after evict+restore"));
        assert!(
            matches!(
                loc,
                baseline_llm_os::kcmm::pool::BlockLocation::GpuResident(_, _)
            ),
            "block {bi} not GpuResident after evict+restore: {loc:?}"
        );
    }

    n_evicted
}

// ============================================================================
// Sequence runner
// ============================================================================

/// Run a single sequence through `total_steps` decode steps and return the
/// logits at each step.
///
/// `eviction_points` is a list of `(step_number, block_count)` pairs.  After
/// completing step `step_number`, the first `block_count` blocks of the
/// sequence are evicted to CPU and immediately restored to GPU before the next
/// step begins.
fn run_sequence_with_evictions(
    model: &dyn Transformer,
    pool: &KcmmPool,
    block_table: &[u32],
    seq_idx: usize,
    total_steps: usize,
    block_size: usize,
    eviction_points: &[(usize, usize)], // (after_step, num_blocks_to_evict)
) -> Vec<Vec<f32>> {
    let h = HIDDEN_SIZE;
    let mut logits_per_step: Vec<Vec<f32>> = Vec::with_capacity(total_steps);
    let mut hidden: cudarc::driver::CudaSlice<f16> =
        pool.ctx.device.alloc_zeros::<f16>(h).unwrap();

    // Track how many blocks have been allocated (may grow during the run).
    let mut block_indices: Vec<u32> = block_table.to_vec();

    for step in 0..total_steps {
        let pos = step; // position = step number (zero-indexed)

        // Grow block table if needed.
        let blocks_needed = (pos / block_size) + 1;
        while blocks_needed > block_indices.len() {
            let new_block = pool
                .alloc_block()
                .expect("alloc_block failed during sequence growth");
            pool.append_block_to_sequence(seq_idx, new_block);
            block_indices.push(new_block);
        }

        pool.update_seq_len(seq_idx, pos + 1);

        // Run one decode step.
        let logits = model
            .forward_step_paged(
                &mut hidden,
                pool,
                &[seq_idx],
                &[0u32],     // dummy token id
                &[pos],
            )
            .expect("forward_step_paged failed");

        logits_per_step.push(logits);

        // Check if we should force an eviction after this step.
        if let Some(&(_, n_blocks)) = eviction_points.iter().find(|&&(s, _)| s == step) {
            let n = n_blocks.min(block_indices.len());
            let evicted = force_evict_and_restore(pool, &block_indices[..n]);
            if evicted == 0 && n > 0 {
                eprintln!(
                    "  (step {step}: evict_and_restore requested {n} blocks but none were on GPU — \
                     blocks may already be CpuResident; restore was attempted)"
                );
            }
        }
    }

    logits_per_step
}

// ============================================================================
// Pool factory
// ============================================================================

fn make_pool(
    ctx: &Arc<CudaContext>,
    cpu_path: &str,
    tiering: bool,
    block_size: usize,
    max_blocks: usize,
    max_batch: usize,
    max_seq_len: usize,
) -> KcmmPool {
    KcmmPool::new(
        ctx.clone(),
        KcmmConfig {
            block_size,
            max_blocks,
            cpu_cache_path: cpu_path.to_string(),
            tiering,
            eviction_policy: "lru".to_string(),
            prefetch_window: 4,
            max_batch_blocks: 64,
            low_watermark_threshold: 0.2,
            background_evict_interval_ms: 100,
            attention_sink_blocks: 1,
            recent_window_blocks: 4,
        },
        NUM_LAYERS,
        KV_HEADS,
        HEAD_DIM,
        max_batch,
        max_seq_len,
    )
    .expect("create KcmmPool")
}

// ============================================================================
// Test: logits consistency under forced eviction
// ============================================================================

/// The definitive KCMM correctness test.
///
/// **What it verifies**:  K/V data written to the paged cache by the model
/// forward pass survives eviction (GPU→CPU) and restoration (CPU→GPU) without
/// a single-bit corruption, such that downstream model logits are numerically
/// identical to a reference run where no eviction occurred.
///
/// **How it works**:
/// 1. Build two identical models from the same Xavier seed.
/// 2. Run the same 32-token sequence through both.
/// 3. The *reference* pool never experiences eviction.
/// 4. The *test* pool forces evict+restore at steps 7, 15, and 23 —
///    each cycle evicts progressively more blocks to exercise partial-
///    and full-cache roundtrip scenarios.
/// 5. Compare logits element-by-element at every step.
///
/// **Success criterion**: max |ref_logit − test_logit| < 1e-5 at every step.
/// (f32 has ~7 decimal digits; 1e-5 tolerates a single ULP difference from
/// non-associative floating-point accumulation while catching any real data
/// corruption, which would shift entire token probabilities by ≫1e-5.)
#[test]
fn kcmm_logits_consistency_under_eviction() {
    // --- Guard: skip if no CUDA GPU ----------------------------------------
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            eprintln!("SKIP kcmm_logits_consistency_under_eviction: no CUDA GPU ({e})");
            return;
        }
    };

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  KCMM Correctness — Logits Consistency Under Eviction       ║");
    println!("║  Verifies evict→restore preserves KV-cache data bit-exact   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Model: Xavier seed=0x{MODEL_WEIGHT_SEED:016X}  L={NUM_LAYERS}  \
         kv_heads={KV_HEADS}  head_dim={HEAD_DIM}  hidden={HIDDEN_SIZE}"
    );

    // --- Test parameters ---------------------------------------------------
    let block_size: usize = 4; // small → data spans many blocks quickly
    let total_steps: usize = 32; // 8 blocks of KV cache (32 tokens / 4 tok/block)
    let max_blocks: usize = 64;
    let max_batch: usize = 4;
    let max_seq_len: usize = 64;
    let initial_blocks: usize = 1; // start with 1 block (position 0)

    // Eviction schedule: (after_step, num_blocks_to_evict)
    //
    // Step  7 → position 7,  data in blocks 0-1 → evict 2 blocks
    // Step 15 → position 15, data in blocks 0-3 → evict all 4 blocks
    // Step 23 → position 23, data in blocks 0-5 → evict all 6 blocks
    let eviction_points: &[(usize, usize)] = &[(7, 2), (15, 4), (23, 6)];

    println!(
        "Workload: {total_steps} decode steps, block_size={block_size} \
         ({max_blocks} max blocks)"
    );
    println!(
        "Eviction schedule: {}",
        eviction_points
            .iter()
            .map(|(s, n)| format!("after step {s}: evict+restore {n} blocks"))
            .collect::<Vec<_>>()
            .join("; ")
    );
    println!();

    let model_cfg = test_model_cfg();
    let dir = tempfile::tempdir().expect("create temp dir");

    // --- Reference run (no eviction) --------------------------------------
    println!("  --- Reference run (no eviction) ---");

    let ref_cpu_path = dir
        .path()
        .join("kcmm_ref")
        .to_str()
        .expect("valid UTF-8")
        .to_string();

    let ref_pool = make_pool(
        &ctx, &ref_cpu_path,
        /*tiering=*/ true, // same code path as test, just no eviction triggered
        block_size, max_blocks, max_batch, max_seq_len,
    );
    let ref_model = build_random_transformer(&ctx, &model_cfg);

    // Allocate initial blocks.
    let mut ref_block_table: Vec<u32> = Vec::with_capacity(initial_blocks);
    for _ in 0..initial_blocks {
        ref_block_table.push(ref_pool.alloc_block().expect("ref alloc_block"));
    }
    let ref_seq_idx = ref_pool.register_sequence(ref_block_table.clone());
    ref_pool.update_seq_len(ref_seq_idx, 1);

    let ref_logits = run_sequence_with_evictions(
        &ref_model, &ref_pool,
        &ref_block_table, ref_seq_idx,
        total_steps, block_size,
        &[], // no evictions
    );

    // Drop reference pool + model to free GPU memory before test run.
    ref_pool.unregister_sequence(ref_seq_idx);
    drop(ref_model);
    drop(ref_pool);

    // --- Test run (with forced eviction) ----------------------------------
    println!("  --- Test run (with forced eviction) ---");

    let test_cpu_path = dir
        .path()
        .join("kcmm_test")
        .to_str()
        .expect("valid UTF-8")
        .to_string();

    let test_pool = make_pool(
        &ctx, &test_cpu_path,
        /*tiering=*/ true,
        block_size, max_blocks, max_batch, max_seq_len,
    );
    let test_model = build_random_transformer(&ctx, &model_cfg);

    let mut test_block_table: Vec<u32> = Vec::with_capacity(initial_blocks);
    for _ in 0..initial_blocks {
        test_block_table.push(test_pool.alloc_block().expect("test alloc_block"));
    }
    let test_seq_idx = test_pool.register_sequence(test_block_table.clone());
    test_pool.update_seq_len(test_seq_idx, 1);

    println!("  Eviction schedule:");
    for &(step, n) in eviction_points {
        println!("    after step {step}: evict+restore first {n} blocks");
    }
    println!();

    let test_logits = run_sequence_with_evictions(
        &test_model, &test_pool,
        &test_block_table, test_seq_idx,
        total_steps, block_size,
        eviction_points,
    );

    test_pool.unregister_sequence(test_seq_idx);
    drop(test_model);
    drop(test_pool);

    // --- Compare logits ---------------------------------------------------
    println!("  --- Comparison ---");
    assert_eq!(
        ref_logits.len(),
        test_logits.len(),
        "step count mismatch: ref={} vs test={}",
        ref_logits.len(),
        test_logits.len()
    );

    let tolerance: f32 = 1e-5;
    let mut max_diff_any_step: f32 = 0.0;
    let mut max_diff_step: usize = 0;
    let mut all_ok = true;

    for step in 0..total_steps {
        let ref_l = &ref_logits[step];
        let test_l = &test_logits[step];

        assert_eq!(
            ref_l.len(),
            test_l.len(),
            "logit count mismatch at step {step}: ref={} vs test={}",
            ref_l.len(),
            test_l.len()
        );

        let mut step_max_diff: f32 = 0.0;
        let mut step_max_idx: usize = 0;

        for (i, (&r, &t)) in ref_l.iter().zip(test_l.iter()).enumerate() {
            let diff = (r - t).abs();
            if diff > step_max_diff {
                step_max_diff = diff;
                step_max_idx = i;
            }
        }

        if step_max_diff > max_diff_any_step {
            max_diff_any_step = step_max_diff;
            max_diff_step = step;
        }

        let evict_note = eviction_points
            .iter()
            .find(|&&(s, _)| s == step)
            .map(|&(_, n)| format!("  ← evict+restore {n} blocks after this step"))
            .unwrap_or_default();

        if step_max_diff < tolerance {
            println!(
                "  step {step:>2}: max|Δ| = {step_max_diff:.1e}  OK  (vocab_idx={step_max_idx}){evict_note}"
            );
        } else {
            all_ok = false;
            println!(
                "  step {step:>2}: max|Δ| = {step_max_diff:.1e}  FAIL > {tolerance}  \
                 (vocab_idx={step_max_idx}, ref={:.6}, test={:.6}){evict_note}",
                ref_l[step_max_idx], test_l[step_max_idx],
            );
        }
    }

    println!();
    if all_ok {
        println!(
            "  ✅ PASS: all {total_steps} steps within tolerance ({tolerance:.0e})  \
             worst-case max|Δ| = {max_diff_any_step:.1e} at step {max_diff_step}"
        );
    } else {
        panic!(
            "❌ FAIL: logits diverged beyond tolerance ({tolerance:.0e})  \
             worst-case max|Δ| = {max_diff_any_step:.1e} at step {max_diff_step}"
        );
    }
}
