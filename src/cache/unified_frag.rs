// src/cache/unified_frag.rs
//
// Unified Fragmentation Standard (UFS) — a set of KV-cache fragmentation
// metrics that can be computed identically on both the baseline system
// (CUDA VMM with 2 MiB superblocks) and vLLM (PyTorch CUDA allocator).
//
// ## Core Metrics
//
//  1. IFR — Internal Fragmentation Rate
//     (total_slots - total_tokens) / total_slots
//     Waste within partially-filled blocks (last-block problem).
//     Range: [0, 1).  Directly comparable across systems.
//
//  2. BU  — Block Utilization
//     blocks_in_use / total_blocks_allocated
//     How efficiently the block pool is utilized.
//     Range: [0, 1].  Directly comparable across systems.
//
//  3. PME — Physical Memory Efficiency
//     ideal_bytes / actual_physical_bytes
//     Physical memory waste from allocator granularity.
//     Range: (0, 1].  System-specific actual_physical_bytes formula.
//
//  4. RFI — Runtime Fragmentation Index
//     1 - (total_tokens × BPT / actual_active_bytes)
//     Combined internal+external waste during active processing.
//     Range: [0, 1).  System-specific actual_active_bytes formula.
//
// ## All-layer accounting
//
// All byte values account for ALL layers (K+V).
//   BPT (bytes per token, all layers) = kv_heads × head_dim × 2 × num_layers × 2
//   ideal_bytes = blocks_in_use × block_bytes × num_layers × 2
//
// ## System-specific actual bytes
//
//   Baseline (CUDA VMM):
//     actual_physical_bytes = superblock_count × 2 MiB × num_layers × 2
//     actual_active_bytes   = ⌈blocks_in_use / blocks_per_sb⌉ × 2 MiB × num_layers × 2
//
//   vLLM (PyTorch):
//     actual_physical_bytes = total_blocks_allocated × block_bytes × num_layers × 2
//     actual_active_bytes   = blocks_in_use × block_bytes × num_layers × 2

use serde::Serialize;

use super::paged_kv::PagedKvCache;

/// Bytes per token-element for one layer of K (f16):
/// kv_heads × head_dim × sizeof(f16).
#[inline]
fn bytes_per_token_k(cache: &PagedKvCache) -> usize {
    cache.cfg.kv_heads() * cache.cfg.head_dim() * 2
}

/// Bytes per token for ALL layers (K+V):
/// kv_heads × head_dim × sizeof(f16) × num_layers × 2.
#[inline]
fn bytes_per_token_all_layers(cache: &PagedKvCache) -> usize {
    bytes_per_token_k(cache) * cache.cfg.num_hidden_layers * 2
}

/// Round `n` up to the nearest multiple of `m`.
#[inline]
fn round_up(n: usize, m: usize) -> usize {
    (n + m - 1) / m * m
}

/// A single snapshot of unified fragmentation metrics at one point in time.
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct UnifiedFragMetrics {
    // ── Directly comparable (formula identical across systems) ──
    /// Internal Fragmentation Rate: (total_slots - total_tokens) / total_slots.
    /// Waste within partially-filled blocks.
    pub internal_frag_rate: f32,

    /// Block Utilization: blocks_in_use / total_blocks_allocated.
    pub block_utilization: f32,

    // ── System-specific (formula documented, same metric name) ──
    /// Physical Memory Efficiency: ideal_bytes / actual_physical_bytes.
    pub physical_memory_efficiency: f32,

    /// Runtime Fragmentation Index: 1 - (total_tokens × BPT / actual_active_bytes).
    pub runtime_frag_index: f32,

    // ── Raw counts for verification / debugging ──
    pub active_sequences: usize,
    pub blocks_in_use: usize,
    pub total_blocks_allocated: usize,
    pub total_tokens: usize,
    pub ideal_physical_bytes: u64,
    pub actual_physical_bytes: u64,
}

impl UnifiedFragMetrics {
    /// Compute unified fragmentation metrics from the PagedKvCache state
    /// (baseline system, CUDA VMM with 2 MiB superblocks).
    pub fn from_cache(cache: &PagedKvCache) -> Self {
        let meta = cache.seq_metadata.lock();

        let active_sequences = meta
            .iter()
            .filter(|s| !s.block_table.is_empty())
            .count();

        let total_blocks_used: usize = meta.iter().map(|s| s.block_table.len()).sum();
        let total_tokens: usize = meta.iter().map(|s| s.seq_len).sum();
        drop(meta);

        let block_size = cache.block_size;
        let total_slots = total_blocks_used * block_size;

        // IFR: internal fragmentation
        let internal_frag_rate = if total_slots > 0 {
            (total_slots - total_tokens) as f32 / total_slots as f32
        } else {
            0.0
        };

        let blocks_in_use = cache.blocks_in_use();
        let total_blocks_allocated = cache.total_physical_blocks();
        let block_bytes = cache.block_bytes;
        let num_layers = cache.cfg.num_hidden_layers;
        let bpt_all = bytes_per_token_all_layers(cache);

        // BU: block utilization
        let block_utilization = if total_blocks_allocated > 0 {
            blocks_in_use as f32 / total_blocks_allocated as f32
        } else {
            0.0
        };

        // PME + RFI: system-specific physical memory
        let superblock_count = cache.superblock_count();
        let superblock_size = 2usize * 1024 * 1024; // 2 MiB
        let actual_physical_bytes =
            (superblock_count * superblock_size * num_layers * 2) as u64;

        let ideal_physical_bytes =
            (blocks_in_use * block_bytes * num_layers * 2) as u64;

        let physical_memory_efficiency = if actual_physical_bytes > 0 {
            ideal_physical_bytes as f32 / actual_physical_bytes as f32
        } else {
            1.0
        };

        // RFI: actual_active_bytes rounds blocks_in_use up to superblock boundaries
        let blocks_per_sb = cache.blocks_per_superblock();
        let active_superblocks = round_up(blocks_in_use, blocks_per_sb) / blocks_per_sb;
        let actual_active_bytes =
            (active_superblocks * superblock_size * num_layers * 2) as u64;

        let ideal_active_bytes = (total_tokens * bpt_all) as u64;

        let runtime_frag_index = if actual_active_bytes > 0 {
            1.0 - (ideal_active_bytes as f32 / actual_active_bytes as f32)
        } else {
            0.0
        };

        Self {
            internal_frag_rate,
            block_utilization,
            physical_memory_efficiency,
            runtime_frag_index,
            active_sequences,
            blocks_in_use,
            total_blocks_allocated,
            total_tokens,
            ideal_physical_bytes,
            actual_physical_bytes,
        }
    }

    /// Compute unified fragmentation metrics from raw values.
    /// Used by external systems (vLLM, Python) that can supply the
    /// same base counts with their own `actual_physical_bytes` formula.
    ///
    /// For vLLM (PyTorch allocator — no superblocks):
    ///   actual_physical_bytes = total_blocks_allocated × block_bytes × num_layers × 2
    ///   actual_active_bytes   = blocks_in_use × block_bytes × num_layers × 2
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        block_size: usize,
        blocks_in_use: usize,
        total_blocks_allocated: usize,
        total_blocks_used_by_seqs: usize,
        total_tokens: usize,
        block_bytes: usize,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        actual_physical_bytes: u64,
        actual_active_bytes: u64,
    ) -> Self {
        let total_slots = total_blocks_used_by_seqs * block_size;

        let internal_frag_rate = if total_slots > 0 {
            (total_slots - total_tokens) as f32 / total_slots as f32
        } else {
            0.0
        };

        let block_utilization = if total_blocks_allocated > 0 {
            blocks_in_use as f32 / total_blocks_allocated as f32
        } else {
            0.0
        };

        let ideal_physical_bytes =
            (blocks_in_use * block_bytes * num_layers * 2) as u64;

        let physical_memory_efficiency = if actual_physical_bytes > 0 {
            ideal_physical_bytes as f32 / actual_physical_bytes as f32
        } else {
            1.0
        };

        // Bytes per token for all layers (K+V)
        let bpt_all = kv_heads * head_dim * 2 * num_layers * 2;
        let ideal_active_bytes = (total_tokens * bpt_all) as u64;

        let runtime_frag_index = if actual_active_bytes > 0 {
            1.0 - (ideal_active_bytes as f32 / actual_active_bytes as f32)
        } else {
            0.0
        };

        Self {
            internal_frag_rate,
            block_utilization,
            physical_memory_efficiency,
            runtime_frag_index,
            active_sequences: 0, // caller should set if known
            blocks_in_use,
            total_blocks_allocated,
            total_tokens,
            ideal_physical_bytes,
            actual_physical_bytes,
        }
    }
}

/// Time-series aggregation of unified fragmentation metrics.
#[derive(Debug, Clone, Serialize)]
pub struct UnifiedFragSummary {
    /// Number of samples recorded.
    pub sample_count: usize,

    // ── IFR ──
    pub ifr_avg: f32,
    pub ifr_peak: f32,
    pub ifr_stddev: f32,

    // ── BU ──
    pub bu_avg: f32,
    pub bu_min: f32,
    pub bu_stddev: f32,

    // ── PME ──
    pub pme_avg: f32,
    pub pme_min: f32,
    pub pme_stddev: f32,

    // ── RFI ──
    pub rfi_avg: f32,
    pub rfi_peak: f32,
    pub rfi_stddev: f32,
}

impl UnifiedFragSummary {
    /// Compute summary statistics from a time series of UnifiedFragMetrics samples.
    pub fn from_samples(samples: &[UnifiedFragMetrics]) -> Self {
        if samples.is_empty() {
            return Self {
                sample_count: 0,
                ifr_avg: 0.0, ifr_peak: 0.0, ifr_stddev: 0.0,
                bu_avg: 0.0, bu_min: 0.0, bu_stddev: 0.0,
                pme_avg: 0.0, pme_min: 0.0, pme_stddev: 0.0,
                rfi_avg: 0.0, rfi_peak: 0.0, rfi_stddev: 0.0,
            };
        }

        let n = samples.len();
        let sample_count = n;
        let nf = n as f32;

        // IFR
        let ifr_sum: f32 = samples.iter().map(|s| s.internal_frag_rate).sum();
        let ifr_avg = ifr_sum / nf;
        let ifr_peak = samples.iter().map(|s| s.internal_frag_rate).fold(0.0f32, f32::max);
        let ifr_var = samples.iter().map(|s| {
            let d = s.internal_frag_rate - ifr_avg;
            d * d
        }).sum::<f32>() / if n > 1 { (n - 1) as f32 } else { 1.0 };
        let ifr_stddev = ifr_var.sqrt();

        // BU
        let bu_sum: f32 = samples.iter().map(|s| s.block_utilization).sum();
        let bu_avg = bu_sum / nf;
        let bu_min = samples.iter().map(|s| s.block_utilization).fold(f32::MAX, f32::min);
        let bu_var = samples.iter().map(|s| {
            let d = s.block_utilization - bu_avg;
            d * d
        }).sum::<f32>() / if n > 1 { (n - 1) as f32 } else { 1.0 };
        let bu_stddev = bu_var.sqrt();

        // PME
        let pme_sum: f32 = samples.iter().map(|s| s.physical_memory_efficiency).sum();
        let pme_avg = pme_sum / nf;
        let pme_min = samples.iter().map(|s| s.physical_memory_efficiency).fold(f32::MAX, f32::min);
        let pme_var = samples.iter().map(|s| {
            let d = s.physical_memory_efficiency - pme_avg;
            d * d
        }).sum::<f32>() / if n > 1 { (n - 1) as f32 } else { 1.0 };
        let pme_stddev = pme_var.sqrt();

        // RFI
        let rfi_sum: f32 = samples.iter().map(|s| s.runtime_frag_index).sum();
        let rfi_avg = rfi_sum / nf;
        let rfi_peak = samples.iter().map(|s| s.runtime_frag_index).fold(0.0f32, f32::max);
        let rfi_var = samples.iter().map(|s| {
            let d = s.runtime_frag_index - rfi_avg;
            d * d
        }).sum::<f32>() / if n > 1 { (n - 1) as f32 } else { 1.0 };
        let rfi_stddev = rfi_var.sqrt();

        Self {
            sample_count,
            ifr_avg, ifr_peak, ifr_stddev,
            bu_avg, bu_min, bu_stddev,
            pme_avg, pme_min, pme_stddev,
            rfi_avg, rfi_peak, rfi_stddev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_from_raw_no_fragmentation() {
        // Perfect packing: all blocks in use, all slots filled
        let m = UnifiedFragMetrics::from_raw(
            16,     // block_size
            8,      // blocks_in_use
            8,      // total_blocks_allocated
            8,      // total_blocks_used_by_seqs
            128,    // total_tokens (8 * 16 = 128 — perfect)
            8192,   // block_bytes
            22,     // num_layers
            4,      // kv_heads
            64,     // head_dim
            8 * 8192 * 22 * 2,  // actual_physical = ideal
            8 * 8192 * 22 * 2,  // actual_active = ideal
        );

        assert!((m.internal_frag_rate - 0.0).abs() < 0.001,
            "perfect packing should have IFR=0, got {}", m.internal_frag_rate);
        assert!((m.block_utilization - 1.0).abs() < 0.001,
            "all blocks in use should have BU=1, got {}", m.block_utilization);
        assert!((m.physical_memory_efficiency - 1.0).abs() < 0.001,
            "ideal=actual should have PME=1, got {}", m.physical_memory_efficiency);
        assert!((m.runtime_frag_index - 0.0).abs() < 0.001,
            "no waste should have RFI=0, got {}", m.runtime_frag_index);
    }

    #[test]
    fn test_unified_from_raw_with_fragmentation() {
        // 10 blocks in use, 16 allocated, 128 slots, 100 tokens
        let m = UnifiedFragMetrics::from_raw(
            16,     // block_size
            10,     // blocks_in_use
            16,     // total_blocks_allocated
            10,     // total_blocks_used_by_seqs
            100,    // total_tokens (10*16=160 slots, 60 wasted)
            8192,   // block_bytes
            22,     // num_layers
            4,      // kv_heads
            64,     // head_dim
            16 * 8192 * 22 * 2,  // actual_physical (all allocated blocks)
            16 * 8192 * 22 * 2,  // actual_active (superblock rounds up)
        );

        // IFR: (160 - 100) / 160 = 0.375
        assert!((m.internal_frag_rate - 0.375).abs() < 0.001,
            "IFR should be 0.375, got {}", m.internal_frag_rate);

        // BU: 10/16 = 0.625
        assert!((m.block_utilization - 0.625).abs() < 0.001,
            "BU should be 0.625, got {}", m.block_utilization);

        // PME: (10 * 8192 * 22 * 2) / (16 * 8192 * 22 * 2) = 10/16 = 0.625
        assert!((m.physical_memory_efficiency - 0.625).abs() < 0.001,
            "PME should be 0.625, got {}", m.physical_memory_efficiency);

        // RFI > 0 (waste present)
        assert!(m.runtime_frag_index > 0.0,
            "RFI should be >0 with fragmentation, got {}", m.runtime_frag_index);
    }

    #[test]
    fn test_unified_summary_empty() {
        let s = UnifiedFragSummary::from_samples(&[]);
        assert_eq!(s.sample_count, 0);
    }

    #[test]
    fn test_unified_summary_constant() {
        let samples: Vec<UnifiedFragMetrics> = (0..100).map(|_| {
            UnifiedFragMetrics::from_raw(
                16, 8, 8, 8, 128, 8192, 22, 4, 64,
                8 * 8192 * 22 * 2, 8 * 8192 * 22 * 2,
            )
        }).collect();
        let s = UnifiedFragSummary::from_samples(&samples);
        assert_eq!(s.sample_count, 100);
        assert!((s.ifr_stddev - 0.0).abs() < 0.001, "constant samples should have stddev=0");
        assert!((s.bu_stddev - 0.0).abs() < 0.001);
        assert!((s.pme_stddev - 0.0).abs() < 0.001);
        assert!((s.rfi_stddev - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_unified_summary_bounds() {
        // All metrics should be in [0, 1]
        let samples: Vec<UnifiedFragMetrics> = vec![
            UnifiedFragMetrics::from_raw(
                16, 5, 10, 5, 60, 8192, 22, 4, 64,
                10 * 8192 * 22 * 2, 10 * 8192 * 22 * 2,
            ),
            UnifiedFragMetrics::from_raw(
                16, 8, 10, 8, 120, 8192, 22, 4, 64,
                10 * 8192 * 22 * 2, 10 * 8192 * 22 * 2,
            ),
        ];
        let s = UnifiedFragSummary::from_samples(&samples);

        assert!(s.ifr_avg >= 0.0 && s.ifr_avg <= 1.0);
        assert!(s.bu_avg >= 0.0 && s.bu_avg <= 1.0);
        assert!(s.pme_avg >= 0.0 && s.pme_avg <= 1.0);
        assert!(s.rfi_avg >= 0.0 && s.rfi_avg <= 1.0);
    }
}
