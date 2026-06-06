// KCMM metrics — UFS-compatible fragmentation indicators.
//
// Collects and exposes standardized GPU memory fragmentation metrics
// (IFR, PME, BU, RFI) to the C API and monitoring infrastructure.
//
// Full implementation in step 3 weeks 15-16.

use crate::cache::unified_frag::{UnifiedFragMetrics, UnifiedFragSummary};

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
        };
        let m2 = m1.clone();
        assert_eq!(m2.ifr, m1.ifr);
        assert_eq!(m2.pme, m1.pme);
        assert_eq!(m2.bu, m1.bu);
        assert_eq!(m2.rfi, m1.rfi);
        assert_eq!(m2.gpu_blocks, m1.gpu_blocks);
        assert_eq!(m2.eviction_count, m1.eviction_count);
    }
}
