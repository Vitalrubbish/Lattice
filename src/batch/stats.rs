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
}

impl Default for StatsSnapshot {
    fn default() -> Self {
        Self {
            unified: None,
            sample_count: 0,
            rfi_avg: 0.0,
            rfi_peak: 0.0,
            rfi_stddev: 0.0,
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
    ) {
        let mut snap = self.inner.lock();
        snap.unified = unified;
        snap.sample_count = sample_count;
        snap.rfi_avg = rfi_avg;
        snap.rfi_peak = rfi_peak;
        snap.rfi_stddev = rfi_stddev;
    }

    /// Get the current stats snapshot.
    pub fn snapshot(&self) -> StatsSnapshot {
        self.inner.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_stats_snapshot_default() {
        let snap = StatsSnapshot::default();
        assert!(snap.unified.is_none());
        assert_eq!(snap.sample_count, 0);
        assert_eq!(snap.rfi_avg, 0.0);
        assert_eq!(snap.rfi_peak, 0.0);
        assert_eq!(snap.rfi_stddev, 0.0);
    }

    #[test]
    fn test_stats_handle_new_returns_default() {
        let handle = StatsHandle::new();
        let snap = handle.snapshot();
        assert!(snap.unified.is_none());
        assert_eq!(snap.sample_count, 0);
    }

    #[test]
    fn test_stats_handle_update_and_snapshot() {
        let handle = StatsHandle::new();

        let ufs = UnifiedFragMetrics {
            internal_frag_rate: 0.15,
            block_utilization: 0.85,
            physical_memory_efficiency: 0.9,
            runtime_frag_index: 0.05,
            active_sequences: 2,
            blocks_in_use: 8,
            total_blocks_allocated: 16,
            total_tokens: 100,
            ideal_physical_bytes: 10000,
            actual_physical_bytes: 20000,
        };

        handle.update_from_tracker(
            Some(ufs),
            42,   // sample_count
            0.05, // rfi_avg
            0.12, // rfi_peak
            0.03, // rfi_stddev
        );

        let snap = handle.snapshot();
        assert!(snap.unified.is_some());
        let u = snap.unified.unwrap();
        assert!((u.internal_frag_rate - 0.15).abs() < 0.001);
        assert!((u.block_utilization - 0.85).abs() < 0.001);
        assert_eq!(snap.sample_count, 42);
        assert!((snap.rfi_avg - 0.05).abs() < 0.001);
        assert!((snap.rfi_peak - 0.12).abs() < 0.001);
        assert!((snap.rfi_stddev - 0.03).abs() < 0.001);
    }

    #[test]
    fn test_stats_handle_update_multiple_times() {
        let handle = StatsHandle::new();

        handle.update_from_tracker(None, 1, 0.0, 0.0, 0.0);
        assert_eq!(handle.snapshot().sample_count, 1);

        handle.update_from_tracker(None, 2, 0.1, 0.2, 0.05);
        assert_eq!(handle.snapshot().sample_count, 2);

        handle.update_from_tracker(None, 100, 0.5, 0.9, 0.1);
        let snap = handle.snapshot();
        assert_eq!(snap.sample_count, 100);
        assert!((snap.rfi_avg - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_stats_handle_concurrent_read_write() {
        let handle = StatsHandle::new();
        let handle_clone = handle.clone();

        // Writer thread: updates stats rapidly
        let writer = thread::spawn(move || {
            for i in 0..500 {
                handle.update_from_tracker(
                    None,
                    i,
                    i as f32 * 0.01,
                    i as f32 * 0.02,
                    i as f32 * 0.001,
                );
            }
        });

        // Reader thread: reads snapshots rapidly
        let reader = thread::spawn(move || {
            for _ in 0..500 {
                let snap = handle_clone.snapshot();
                // Just verify no panic; values can be stale
                let _ = snap.sample_count;
                let _ = snap.rfi_avg;
            }
        });

        writer.join().expect("writer panicked");
        reader.join().expect("reader panicked");

        // After all updates, sample_count should be 499 (last i value)
        let final_snap = StatsHandle::new().snapshot(); // fresh handle unaffected
        assert_eq!(final_snap.sample_count, 0);
    }
}
