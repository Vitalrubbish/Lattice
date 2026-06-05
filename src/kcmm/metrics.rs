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
