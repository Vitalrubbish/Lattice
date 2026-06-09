// tests/bench_utils.rs — shared statistics helpers for KCMM benchmarks.
//
// Provides percentile, mean, standard deviation, standard error, and a
// unified latency-stats printer used across all KCMM integration tests.

// --- Statistical functions ---

/// Compute the `p`-th percentile (0..100).  Sorts in-place.
pub fn percentile(data: &mut [u64], p: f64) -> u64 {
    assert!(!data.is_empty(), "percentile: empty data");
    assert!((0.0..=100.0).contains(&p), "percentile p out of range");
    data.sort_unstable();
    let idx = ((data.len() as f64 * p / 100.0).ceil() as usize).saturating_sub(1);
    data[idx.min(data.len() - 1)]
}

/// Compute arithmetic mean of a slice.
pub fn mean(data: &[u64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<u64>() as f64 / data.len() as f64
}

/// Compute sample standard deviation (Bessel-corrected: divide by n-1).
pub fn stddev(data: &[u64], m: f64) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }
    let variance = data
        .iter()
        .map(|&x| {
            let diff = x as f64 - m;
            diff * diff
        })
        .sum::<f64>()
        / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Standard error of the mean: stddev / sqrt(n).
pub fn std_error(sd: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    sd / (n as f64).sqrt()
}

// --- Unified output ---

/// Print a standardised latency summary line for one metric.
///
/// `label` is the left-aligned name (e.g. "alloc_p50").  `data` is
/// consumed (sorted in-place).  `unit` is "ns" or "µs"; values are
/// already in the given unit.
///
/// Output format: `  {label:<24} {mean:>8.1} ± {stddev:>6.1} {unit}  [{min}, {p50}, {p99}, {max}]  (n={n}, SE=±{se:.1})`
pub fn print_latency_stats(label: &str, data: &mut [u64], unit: &str) {
    let n = data.len();
    if n == 0 {
        println!("  {label:<24} (no data)");
        return;
    }
    let m = mean(data);
    let sd = stddev(data, m);
    let se = std_error(sd, n);
    // Percentile sorts in-place; make a copy so we don't destroy order
    // for later percentile calls on the same slice.
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let min = sorted.first().copied().unwrap_or(0);
    let p50 = percentile(&mut sorted, 50.0);
    let p99 = percentile(&mut sorted, 99.0);
    let max = sorted.last().copied().unwrap_or(0);

    println!(
        "  {label:<24} {m:>8.1} ± {sd:>6.1} {unit}  [{min}, {p50}, {p99}, {max}]  (n={n}, SE=±{se:.1})",
    );
}
