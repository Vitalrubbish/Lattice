// src/batch/stats.rs
//
// Shared cache-stats handle for the server to query fragmentation metrics
// from the continuous scheduler thread without direct cache access.

use parking_lot::Mutex;
use std::sync::Arc;

use crate::cache::unified_frag::UnifiedFragMetrics;

/// A snapshot of the latest cache stats, updated by the scheduler
/// and read by the server for `{"type":"stats"}` requests.
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    /// Latest unified fragmentation metrics, if any samples recorded.
    pub unified: Option<UnifiedFragMetrics>,
    /// Sample count (how many time steps recorded).
    pub sample_count: usize,
    /// Average runtime fragmentation index across all samples.
    pub rfi_avg: f32,
    /// Peak runtime fragmentation index.
    pub rfi_peak: f32,
    /// Standard deviation of runtime fragmentation index.
    pub rfi_stddev: f32,
    /// Legacy ratio average (for backward-compatible reporting).
    pub legacy_ratio_avg: f32,
    /// Legacy ratio peak.
    pub legacy_ratio_peak: f32,
    /// Legacy ratio stddev.
    pub legacy_ratio_stddev: f32,
}

impl Default for StatsSnapshot {
    fn default() -> Self {
        Self {
            unified: None,
            sample_count: 0,
            rfi_avg: 0.0,
            rfi_peak: 0.0,
            rfi_stddev: 0.0,
            legacy_ratio_avg: 0.0,
            legacy_ratio_peak: 0.0,
            legacy_ratio_stddev: 0.0,
        }
    }
}

/// Thread-safe handle for the scheduler to publish stats and the
/// server to read them.
#[derive(Clone)]
pub struct StatsHandle {
    inner: Arc<Mutex<StatsSnapshot>>,
}

impl StatsHandle {
    /// Create a new stats handle with an empty initial snapshot.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(StatsSnapshot::default())),
        }
    }

    /// Update the snapshot from the fragmentation tracker.
    /// Called by the scheduler after each step.
    pub fn update_from_tracker(
        &self,
        unified: Option<UnifiedFragMetrics>,
        sample_count: usize,
        rfi_avg: f32,
        rfi_peak: f32,
        rfi_stddev: f32,
        legacy_avg: f32,
        legacy_peak: f32,
        legacy_stddev: f32,
    ) {
        let mut snap = self.inner.lock();
        snap.unified = unified;
        snap.sample_count = sample_count;
        snap.rfi_avg = rfi_avg;
        snap.rfi_peak = rfi_peak;
        snap.rfi_stddev = rfi_stddev;
        snap.legacy_ratio_avg = legacy_avg;
        snap.legacy_ratio_peak = legacy_peak;
        snap.legacy_ratio_stddev = legacy_stddev;
    }

    /// Get the current stats snapshot.
    pub fn snapshot(&self) -> StatsSnapshot {
        self.inner.lock().clone()
    }
}
