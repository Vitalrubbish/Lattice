// KCMM metrics — UFS-compatible fragmentation indicators.
//
// Collects and exposes standardized GPU memory fragmentation metrics
// (IFR, PME, BU, RFI) to the C API and monitoring infrastructure.
//
// Also tracks eviction/restoration counters, per-policy statistics,
// and operation latency histograms for observability.

use crate::cache::unified_frag::{UnifiedFragMetrics, UnifiedFragSummary};
use std::collections::HashMap;

/// Per-eviction-policy statistics.
#[derive(Debug, Clone, Default)]
pub struct PolicyStats {
    /// Total eviction operations under this policy.
    pub eviction_count: u64,
    /// Total restoration operations under this policy.
    pub restoration_count: u64,
    /// Cumulative blocks evicted under this policy.
    pub evicted_blocks: u64,
    /// Cumulative blocks restored under this policy.
    pub restored_blocks: u64,
    /// Average blocks per eviction batch (running sum / count).
    pub avg_evict_batch_size: f64,
}

/// Simple histogram bucket for operation latencies.
#[derive(Debug, Clone, Default)]
pub struct LatencyHistogram {
    /// Count of samples recorded.
    pub count: u64,
    /// Sum of all latencies in microseconds.
    pub sum_us: u64,
    /// Minimum latency in microseconds.
    pub min_us: u64,
    /// Maximum latency in microseconds.
    pub max_us: u64,
    /// Bucket boundaries in microseconds: [0, 100, 250, 500, 1000, 2500, 5000, 10000, inf].
    pub buckets: [u64; 9],
}

impl LatencyHistogram {
    /// Record a latency sample in microseconds.
    pub fn record(&mut self, latency_us: u64) {
        self.count += 1;
        self.sum_us += latency_us;
        if self.count == 1 {
            self.min_us = latency_us;
            self.max_us = latency_us;
        } else {
            self.min_us = self.min_us.min(latency_us);
            self.max_us = self.max_us.max(latency_us);
        }
        // Bucket boundaries: 0, 100, 250, 500, 1000, 2500, 5000, 10000, inf
        const BOUNDS: [u64; 8] = [100, 250, 500, 1000, 2500, 5000, 10000, u64::MAX];
        for (i, &bound) in BOUNDS.iter().enumerate() {
            if latency_us < bound {
                self.buckets[i] += 1;
                return;
            }
        }
        self.buckets[8] += 1;
    }

    /// Average latency in microseconds, or 0 if no samples.
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_us as f64 / self.count as f64
        }
    }
}

/// KCMM metrics snapshot — mirrors the C API `kcmm_metrics_t`.
#[derive(Debug, Clone)]
pub struct KcmmMetrics {
    /// Internal fragmentation ratio (wasted space within allocated blocks).
    pub ifr: f64,
    /// Physical memory efficiency (ratio of useful data to total physical pages).
    pub pme: f64,
    /// Block utilization (ratio of allocated blocks holding active sequence data).
    pub bu: f64,
    /// Runtime fragmentation index (time-series accumulation of fragmentation).
    pub rfi: f64,
    /// Number of blocks currently in GPU.
    pub gpu_blocks: u64,
    /// Number of blocks currently in CPU swap.
    pub cpu_blocks: u64,
    /// Number of blocks currently on NVMe.
    pub nvme_blocks: u64,
    /// Total eviction operations since pool creation.
    pub eviction_count: u64,
    /// Total restoration operations since pool creation.
    pub restoration_count: u64,

    // --- New observability fields (step 3 eviction improvements) ---
    /// Cumulative blocks evicted (GPU→CPU).
    pub evicted_blocks_total: u64,
    /// Cumulative blocks restored (CPU→GPU).
    pub restored_blocks_total: u64,
    /// Eviction operations that failed (e.g. CPU slot exhaustion).
    pub eviction_failures: u64,
    /// Restoration operations that failed.
    pub restoration_failures: u64,
    /// Background eviction operations triggered by low-watermark.
    pub background_eviction_count: u64,
    /// Per-policy statistics keyed by policy name.
    pub policy_stats: HashMap<String, PolicyStats>,
    /// Eviction latency histogram (GPU→CPU, microseconds).
    pub eviction_latency: LatencyHistogram,
    /// Restoration latency histogram (CPU→GPU, microseconds).
    pub restoration_latency: LatencyHistogram,
}

impl Default for KcmmMetrics {
    fn default() -> Self {
        Self {
            ifr: 0.0,
            pme: 1.0,
            bu: 0.0,
            rfi: 0.0,
            gpu_blocks: 0,
            cpu_blocks: 0,
            nvme_blocks: 0,
            eviction_count: 0,
            restoration_count: 0,
            evicted_blocks_total: 0,
            restored_blocks_total: 0,
            eviction_failures: 0,
            restoration_failures: 0,
            background_eviction_count: 0,
            policy_stats: HashMap::new(),
            eviction_latency: LatencyHistogram::default(),
            restoration_latency: LatencyHistogram::default(),
        }
    }
}

impl KcmmMetrics {
    /// Create a KcmmMetrics snapshot from UFS metrics.
    pub fn from_ufs(ufs: &UnifiedFragMetrics) -> Self {
        Self {
            ifr: ufs.internal_frag_rate as f64,
            pme: ufs.physical_memory_efficiency as f64,
            bu: ufs.block_utilization as f64,
            rfi: ufs.runtime_frag_index as f64,
            gpu_blocks: ufs.blocks_in_use as u64,
            cpu_blocks: 0,
            nvme_blocks: 0,
            eviction_count: 0,
            restoration_count: 0,
            evicted_blocks_total: 0,
            restored_blocks_total: 0,
            eviction_failures: 0,
            restoration_failures: 0,
            background_eviction_count: 0,
            policy_stats: HashMap::new(),
            eviction_latency: LatencyHistogram::default(),
            restoration_latency: LatencyHistogram::default(),
        }
    }

    /// Create a UFS summary from these metrics (single-sample).
    pub fn to_ufs_summary(&self) -> UnifiedFragSummary {
        UnifiedFragSummary {
            sample_count: 1,
            ifr_avg: self.ifr as f32,
            ifr_peak: self.ifr as f32,
            ifr_stddev: 0.0,
            bu_avg: self.bu as f32,
            bu_min: self.bu as f32,
            bu_stddev: 0.0,
            pme_avg: self.pme as f32,
            pme_min: self.pme as f32,
            pme_stddev: 0.0,
            rfi_avg: self.rfi as f32,
            rfi_peak: self.rfi as f32,
            rfi_stddev: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ufs_metrics() -> UnifiedFragMetrics {
        UnifiedFragMetrics {
            internal_frag_rate: 0.25,
            block_utilization: 0.75,
            physical_memory_efficiency: 0.5,
            runtime_frag_index: 0.125,
            active_sequences: 3,
            blocks_in_use: 12,
            total_blocks_allocated: 16,
            total_tokens: 150,
            ideal_physical_bytes: 1024 * 1024,
            actual_physical_bytes: 2 * 1024 * 1024,
        }
    }

    #[test]
    fn test_kcmm_metrics_default() {
        let m = KcmmMetrics::default();
        assert_eq!(m.ifr, 0.0);
        assert_eq!(m.pme, 1.0);
        assert_eq!(m.bu, 0.0);
        assert_eq!(m.rfi, 0.0);
        assert_eq!(m.gpu_blocks, 0);
        assert_eq!(m.cpu_blocks, 0);
        assert_eq!(m.nvme_blocks, 0);
        assert_eq!(m.eviction_count, 0);
        assert_eq!(m.restoration_count, 0);
        assert_eq!(m.evicted_blocks_total, 0);
        assert_eq!(m.restored_blocks_total, 0);
        assert_eq!(m.eviction_failures, 0);
        assert_eq!(m.restoration_failures, 0);
        assert_eq!(m.background_eviction_count, 0);
        assert!(m.policy_stats.is_empty());
        assert_eq!(m.eviction_latency.count, 0);
        assert_eq!(m.restoration_latency.count, 0);
    }

    #[test]
    fn test_from_ufs_maps_fields() {
        let ufs = make_ufs_metrics();
        let m = KcmmMetrics::from_ufs(&ufs);

        assert!((m.ifr - 0.25).abs() < 0.001);
        assert!((m.pme - 0.5).abs() < 0.001);
        assert!((m.bu - 0.75).abs() < 0.001);
        assert!((m.rfi - 0.125).abs() < 0.001);
        assert_eq!(m.gpu_blocks, 12); // blocks_in_use
        assert_eq!(m.cpu_blocks, 0); // not populated yet
        assert_eq!(m.nvme_blocks, 0);
        assert_eq!(m.eviction_count, 0);
        assert_eq!(m.restoration_count, 0);
    }

    #[test]
    fn test_from_ufs_perfect_packing() {
        let ufs = UnifiedFragMetrics {
            internal_frag_rate: 0.0,
            block_utilization: 1.0,
            physical_memory_efficiency: 1.0,
            runtime_frag_index: 0.0,
            active_sequences: 1,
            blocks_in_use: 8,
            total_blocks_allocated: 8,
            total_tokens: 128,
            ideal_physical_bytes: 8192,
            actual_physical_bytes: 8192,
        };
        let m = KcmmMetrics::from_ufs(&ufs);
        assert_eq!(m.ifr, 0.0);
        assert_eq!(m.pme, 1.0);
        assert_eq!(m.bu, 1.0);
        assert_eq!(m.rfi, 0.0);
    }

    #[test]
    fn test_to_ufs_summary_single_sample() {
        let m = KcmmMetrics {
            ifr: 0.1,
            pme: 0.8,
            bu: 0.6,
            rfi: 0.05,
            gpu_blocks: 5,
            cpu_blocks: 2,
            nvme_blocks: 0,
            eviction_count: 3,
            restoration_count: 1,
            evicted_blocks_total: 12,
            restored_blocks_total: 8,
            eviction_failures: 0,
            restoration_failures: 0,
            background_eviction_count: 2,
            policy_stats: HashMap::new(),
            eviction_latency: LatencyHistogram::default(),
            restoration_latency: LatencyHistogram::default(),
        };
        let summary = m.to_ufs_summary();
        assert_eq!(summary.sample_count, 1);
        assert!((summary.ifr_avg - 0.1).abs() < 0.001);
        assert!((summary.ifr_peak - 0.1).abs() < 0.001);
        assert_eq!(summary.ifr_stddev, 0.0);
        assert!((summary.bu_avg - 0.6).abs() < 0.001);
        assert!((summary.bu_min - 0.6).abs() < 0.001);
        assert_eq!(summary.bu_stddev, 0.0);
        assert!((summary.pme_avg - 0.8).abs() < 0.001);
        assert!((summary.pme_min - 0.8).abs() < 0.001);
        assert_eq!(summary.pme_stddev, 0.0);
        assert!((summary.rfi_avg - 0.05).abs() < 0.001);
        assert!((summary.rfi_peak - 0.05).abs() < 0.001);
        assert_eq!(summary.rfi_stddev, 0.0);
    }

    #[test]
    fn test_from_ufs_to_summary_roundtrip() {
        let ufs = make_ufs_metrics();
        let m = KcmmMetrics::from_ufs(&ufs);
        let summary = m.to_ufs_summary();

        // Summary averages should match single-sample values
        assert!((summary.ifr_avg - ufs.internal_frag_rate).abs() < 0.001);
        assert!((summary.bu_avg - ufs.block_utilization).abs() < 0.001);
        assert!((summary.pme_avg - ufs.physical_memory_efficiency).abs() < 0.001);
        assert!((summary.rfi_avg - ufs.runtime_frag_index).abs() < 0.001);
    }

    #[test]
    fn test_kcmm_metrics_clone() {
        let m1 = KcmmMetrics {
            ifr: 0.42,
            pme: 0.88,
            bu: 0.55,
            rfi: 0.12,
            gpu_blocks: 100,
            cpu_blocks: 20,
            nvme_blocks: 5,
            eviction_count: 10,
            restoration_count: 7,
            evicted_blocks_total: 40,
            restored_blocks_total: 30,
            eviction_failures: 1,
            restoration_failures: 0,
            background_eviction_count: 3,
            policy_stats: HashMap::new(),
            eviction_latency: LatencyHistogram::default(),
            restoration_latency: LatencyHistogram::default(),
        };
        let m2 = m1.clone();
        assert_eq!(m2.ifr, m1.ifr);
        assert_eq!(m2.pme, m1.pme);
        assert_eq!(m2.bu, m1.bu);
        assert_eq!(m2.rfi, m1.rfi);
        assert_eq!(m2.gpu_blocks, m1.gpu_blocks);
        assert_eq!(m2.eviction_count, m1.eviction_count);
        assert_eq!(m2.evicted_blocks_total, m1.evicted_blocks_total);
        assert_eq!(m2.background_eviction_count, m1.background_eviction_count);
    }

    // --- LatencyHistogram tests ---

    #[test]
    fn test_latency_histogram_default() {
        let h = LatencyHistogram::default();
        assert_eq!(h.count, 0);
        assert_eq!(h.sum_us, 0);
        assert_eq!(h.min_us, 0);
        assert_eq!(h.max_us, 0);
        assert_eq!(h.avg_us(), 0.0);
        for b in &h.buckets {
            assert_eq!(*b, 0);
        }
    }

    #[test]
    fn test_latency_histogram_single_record() {
        let mut h = LatencyHistogram::default();
        h.record(150);
        assert_eq!(h.count, 1);
        assert_eq!(h.min_us, 150);
        assert_eq!(h.max_us, 150);
        assert_eq!(h.avg_us(), 150.0);
        // 150 falls in bucket [100, 250)
        assert_eq!(h.buckets[1], 1);
        assert_eq!(h.buckets[0], 0);
        assert_eq!(h.buckets[2], 0);
    }

    #[test]
    fn test_latency_histogram_multiple_records() {
        let mut h = LatencyHistogram::default();
        // BOUNDS: [100, 250, 500, 1000, 2500, 5000, 10000, u64::MAX]
        // bucket 0: [0, 100)
        // bucket 1: [100, 250)
        // bucket 2: [250, 500)
        // bucket 3: [500, 1000)
        // bucket 4: [1000, 2500)
        // bucket 5: [2500, 5000)
        // bucket 6: [5000, 10000)
        // bucket 7: [10000, u64::MAX)
        // bucket 8: overflow (>= u64::MAX, effectively unreachable)
        h.record(50);   // bucket 0
        h.record(200);  // bucket 1
        h.record(400);  // bucket 2
        h.record(800);  // bucket 3
        h.record(2000); // bucket 4
        h.record(4000); // bucket 5
        h.record(8000); // bucket 6
        h.record(15000);// bucket 7 (15000 < u64::MAX)

        assert_eq!(h.count, 8);
        assert_eq!(h.min_us, 50);
        assert_eq!(h.max_us, 15000);
        assert_eq!(h.buckets[0], 1);
        assert_eq!(h.buckets[1], 1);
        assert_eq!(h.buckets[2], 1);
        assert_eq!(h.buckets[3], 1);
        assert_eq!(h.buckets[4], 1);
        assert_eq!(h.buckets[5], 1);
        assert_eq!(h.buckets[6], 1);
        assert_eq!(h.buckets[7], 1);
        assert_eq!(h.buckets[8], 0);
    }

    #[test]
    fn test_latency_histogram_avg() {
        let mut h = LatencyHistogram::default();
        h.record(100);
        h.record(200);
        h.record(300);
        assert_eq!(h.count, 3);
        assert!((h.avg_us() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_histogram_min_max() {
        let mut h = LatencyHistogram::default();
        h.record(500);
        h.record(100);
        h.record(1000);
        assert_eq!(h.min_us, 100);
        assert_eq!(h.max_us, 1000);
    }

    #[test]
    fn test_latency_histogram_boundary() {
        let mut h = LatencyHistogram::default();
        h.record(99);   // bucket 0
        h.record(100);  // bucket 1 (boundary: 100 is in [100, 250))
        h.record(249);  // bucket 1
        h.record(250);  // bucket 2
        assert_eq!(h.buckets[0], 1);
        assert_eq!(h.buckets[1], 2);
        assert_eq!(h.buckets[2], 1);
    }

    // --- PolicyStats tests ---

    #[test]
    fn test_policy_stats_default() {
        let s = PolicyStats::default();
        assert_eq!(s.eviction_count, 0);
        assert_eq!(s.restoration_count, 0);
        assert_eq!(s.evicted_blocks, 0);
        assert_eq!(s.restored_blocks, 0);
        assert_eq!(s.avg_evict_batch_size, 0.0);
    }

    #[test]
    fn test_policy_stats_running_average() {
        let mut s = PolicyStats::default();
        // Simulate the record_eviction formula:
        //   let n = entry.eviction_count as f64;  // count AFTER increment
        //   entry.avg = (entry.avg * (n - 1.0) + blocks as f64) / n;

        // First evict: 4 blocks, n=1 after increment
        s.eviction_count = 1;
        s.evicted_blocks = 4;
        s.avg_evict_batch_size = (0.0 * 0.0 + 4.0) / 1.0;
        assert!((s.avg_evict_batch_size - 4.0).abs() < 0.001);

        // Second evict: 8 blocks, n=2 after increment
        s.eviction_count = 2;
        s.evicted_blocks += 8;
        s.avg_evict_batch_size = (s.avg_evict_batch_size * 1.0 + 8.0) / 2.0;
        assert!((s.avg_evict_batch_size - 6.0).abs() < 0.001);

        // Third evict: 12 blocks, n=3 after increment
        s.eviction_count = 3;
        s.evicted_blocks += 12;
        s.avg_evict_batch_size = (s.avg_evict_batch_size * 2.0 + 12.0) / 3.0;
        assert!((s.avg_evict_batch_size - 8.0).abs() < 0.001);
    }
}
