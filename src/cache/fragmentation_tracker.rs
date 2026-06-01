// src/cache/fragmentation_tracker.rs
//
// Runtime fragmentation tracking for paged KV cache.
// Captures snapshots at each scheduler step to compute time-averaged
// fragmentation ratios during multi-request inference.
//
// Now implements the Unified Fragmentation Standard (UFS) alongside
// the original raw metrics for backward compatibility.

use super::paged_kv::PagedKvCache;
use super::unified_frag::{UnifiedFragMetrics, UnifiedFragSummary};

/// A single snapshot of the fragmentation state at one point in time
/// during multi-request inference.
#[derive(Debug, Clone, Copy)]
pub struct FragmentationSample {
    /// Physical memory for non-free blocks, rounded up to the 2 MiB superblock
    /// boundary (CUDA VMM alignment).  = active_superblocks × 2 MiB.
    /// Now correctly computed using superblock rounding.
    pub memory_allocated_not_free: usize,
    /// Ideal contiguous memory for active token KV data.
    /// = total_tokens × bytes_per_token_elem (one layer of K).
    pub memory_active_tokens: usize,
    /// Number of active sequences at this moment.
    pub active_sequences: usize,
    /// Fragmentation ratio: 1 - (memory_active_tokens / memory_allocated_not_free).
    /// 0.0 = no waste; approaches 1.0 as waste increases (internal frag + superblock alignment).
    pub ratio: f32,
}

/// Tracks fragmentation across time steps during multi-request inference.
#[derive(Debug, Clone)]
pub struct RuntimeFragmentationTracker {
    samples: Vec<FragmentationSample>,
    /// Unified fragmentation samples (UFS standard).
    unified_samples: Vec<UnifiedFragMetrics>,
    /// Bytes per token element for one layer of K data:
    /// kv_heads × head_dim × 2 (f16).
    bytes_per_token_elem: usize,
}

impl RuntimeFragmentationTracker {
    /// Create a new tracker.
    ///
    /// `bytes_per_token_elem` = kv_heads × head_dim × 2
    /// (one f16 element = 2 bytes, one layer of K data per token).
    pub fn new(bytes_per_token_elem: usize) -> Self {
        Self {
            samples: Vec::new(),
            unified_samples: Vec::new(),
            bytes_per_token_elem,
        }
    }

    /// Record a fragmentation snapshot from the current cache state.
    ///
    /// The ratio is computed as:
    ///
    /// ```text
    ///                     total_tokens × bytes_per_token_elem
    ///     ratio = 1.0 - ----------------------------------------
    ///                   active_superblocks × SUPERBLOCK_SIZE
    /// ```
    ///
    /// where `active_superblocks = ⌈blocks_not_free / blocks_per_sb⌉`.
    /// This accounts for CUDA VMM alignment: physical memory is allocated in
    /// 2 MiB superblocks, so even a single used block forces the entire
    /// superblock to be allocated.
    ///
    /// **Fixed**: Now correctly rounds blocks_not_free up to the next
    /// superblock boundary, using the cache's actual block_bytes (not
    /// the hardcoded BLOCK_BYTES constant).
    ///
    /// A ratio of 0.0 means no fragmentation (all allocated memory is used by
    /// active tokens); approaching 1.0 means most allocated memory is wasted.
    fn record(&mut self, cache: &PagedKvCache) {
        let sb_count = cache.superblock_count();
        if sb_count == 0 {
            return;
        }

        let total_blocks = cache.total_physical_blocks();
        let free_blocks = cache.free_physical_blocks();
        let blocks_per_sb = cache.blocks_per_superblock();

        let blocks_not_free = total_blocks - free_blocks;

        // Round up to superblock boundary — the key fix.
        // Physical memory is allocated in 2 MiB superblocks, so
        // even a single used block forces the entire superblock
        // to be resident.
        let active_superblocks = (blocks_not_free + blocks_per_sb - 1) / blocks_per_sb;
        let superblock_size = 2usize * 1024 * 1024; // 2 MiB
        let memory_allocated_not_free = active_superblocks * superblock_size;

        // Memory occupied by active token KV data (one layer of K).
        let meta = cache.seq_metadata.lock();
        let total_tokens: usize = meta.iter().map(|s| s.seq_len).sum();
        let active_seqs = meta
            .iter()
            .filter(|s| !s.block_table.is_empty())
            .count();
        drop(meta);

        let memory_active_tokens = total_tokens * self.bytes_per_token_elem;

        // Ratio = 1 - (active_tokens / allocated_not_free)
        // 0.0 = no fragmentation; approaching 1.0 = severe fragmentation.
        let ratio = if memory_allocated_not_free > 0 {
            1.0 - (memory_active_tokens as f32 / memory_allocated_not_free as f32)
        } else {
            0.0
        };

        self.samples.push(FragmentationSample {
            memory_allocated_not_free,
            memory_active_tokens,
            active_sequences: active_seqs,
            ratio,
        });
    }

    /// Record both legacy and unified fragmentation snapshots from the
    /// current cache state.  Call this instead of `record()` to get
    /// Unified Fragmentation Standard (UFS) metrics.
    pub fn record_unified(&mut self, cache: &PagedKvCache) {
        // Record the legacy sample first (backward compat)
        self.record(cache);

        // Record unified metrics
        let unified = UnifiedFragMetrics::from_cache(cache);
        self.unified_samples.push(unified);
    }

    /// Number of recorded legacy samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Number of recorded unified samples.
    pub fn unified_sample_count(&self) -> usize {
        self.unified_samples.len()
    }

    /// Average fragmentation ratio across all recorded time steps.
    /// Returns 0.0 if no samples were recorded.
    pub fn average_ratio(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.ratio).sum::<f32>() / self.samples.len() as f32
    }

    /// Standard deviation of the fragmentation ratio.
    pub fn ratio_stddev(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.average_ratio();
        let variance = self
            .samples
            .iter()
            .map(|s| {
                let diff = s.ratio - mean;
                diff * diff
            })
            .sum::<f32>()
            / (self.samples.len() - 1) as f32;
        variance.sqrt()
    }

    /// Peak (worst) fragmentation ratio observed.
    pub fn peak_ratio(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.ratio)
            .fold(0.0f32, f32::max)
    }

    /// Minimum (best) fragmentation ratio observed.
    pub fn min_ratio(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.ratio)
            .fold(f32::MAX, f32::min)
    }

    /// Return all recorded legacy samples for analysis.
    pub fn samples(&self) -> &[FragmentationSample] {
        &self.samples
    }

    // ── Unified metrics accessors ──

    /// Return all recorded unified fragmentation samples.
    pub fn unified_samples(&self) -> &[UnifiedFragMetrics] {
        &self.unified_samples
    }

    /// Compute summary statistics from the unified samples time series.
    pub fn unified_summary(&self) -> UnifiedFragSummary {
        UnifiedFragSummary::from_samples(&self.unified_samples)
    }

    /// Legacy interface: average of the runtime_frag_index (RFI) across time.
    pub fn average_rfi(&self) -> f32 {
        self.unified_summary().rfi_avg
    }

    /// Legacy interface: peak of the runtime_frag_index (RFI) across time.
    pub fn peak_rfi(&self) -> f32 {
        self.unified_summary().rfi_peak
    }

    /// Legacy interface: stddev of the runtime_frag_index (RFI) across time.
    pub fn stddev_rfi(&self) -> f32 {
        self.unified_summary().rfi_stddev
    }
}

#[allow(dead_code)]
pub fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f32 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f32 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
