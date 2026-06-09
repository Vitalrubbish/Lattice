# KCMM 测量方法学修订 — 多次测量与统计报告

**Date:** 2026-06-09
**Status:** Draft
**Target files:** `tests/kcmm_bench_tiering.rs`, `tests/kcmm_bench_alloc.rs`, `tests/step3_benchmarks.rs`

---

## 0. 问题摘要

当前各 benchmark 的采样量级差异巨大且多数偏少：

| Benchmark | 当前采样 | 问题 |
|-----------|---------|------|
| 1a alloc throughput | 500 ops | 可接受 |
| 1b pool sweep | 300 ops | 可接受 |
| 1c multi-seq concurrent | **1 次** | 无变异性估计 |
| 2a single-block evict/restore | 64 samples | P99 不可靠 (≈第2大值) |
| **2b batch eviction** | **4 rounds** | SE = σ/2, CI 宽度超过组间差异 |
| 2c cuMemMap | 32 samples | 仅报告 P50 |
| 2d roundtrip integrity | **1 次** | 计时数据仅为单次快照 |
| **2e batch restore** | **4 rounds** | 同 2b |
| 3 stream interference | 32 samples | P99 不可靠 |
| step3 benchmarks | 16 或 1 次 | 不足 |

核心问题不是"是否取平均"，而是**单次或少量测量无法区分信号与噪声**。CUDA
操作存在真实物理方差（GPU 频率波动、PCIe 争用、驱动内部状态），多次测量 +
报告离差是唯一能判断"batch=4 的 216µs 是否真的比 batch=1 的 201µs 慢"的手段。

---

## 1. 修改原则

1. **不强制用均值。** 保留 percentile 报告（P50/P99），但增加 min/max/stddev/SE
   使离差可见。
2. **调高各 benchmark 的迭代次数**，使 P50/P99 和均值在统计上有意义。
3. **所有 benchmark 统一输出格式**：`mean ± stddev [min, P50, P99, max] (n=N, SE=±X)`。
4. **不改变 benchmark 的业务逻辑** — 只改迭代次数和输出。

---

## 2. 具体修改

### 2.1 新增公共统计 helper — `tests/kcmm_bench_tiering.rs` 顶部

在现有 `percentile()` 函数之后新增：

```rust
/// Compute arithmetic mean of a slice.
fn mean(data: &[u64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<u64>() as f64 / data.len() as f64
}

/// Compute sample standard deviation (Bessel-corrected: divide by n-1).
fn stddev(data: &[u64], mean: f64) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }
    let variance = data
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Standard error of the mean: stddev / sqrt(n).
fn std_error(stddev: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    stddev / (n as f64).sqrt()
}

/// Print a standardised latency summary line for one metric.
///
/// `label` is the left-aligned name (e.g. "alloc_p50").  `data` is
/// consumed (sorted in-place).  `unit` is "ns" or "µs"; values are
/// already in the given unit.
fn print_latency_stats(label: &str, data: &mut [u64], unit: &str) {
    let n = data.len();
    if n == 0 {
        println!("  {:<20} (no data)", label);
        return;
    }
    let m = mean(data);
    let sd = stddev(data, m);
    let se = std_error(sd, n);
    // Percentiles sort in-place; make a copy so we don't destroy order
    // for later percentile calls on the same slice.
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let min = sorted.first().copied().unwrap_or(0);
    let p50 = percentile(&mut sorted, 50.0);
    let p99 = percentile(&mut sorted, 99.0);
    let max = sorted.last().copied().unwrap_or(0);

    println!(
        "  {:<20} {:>8.1} ± {:>6.1} {unit}  [{}, {}, {}, {}]  (n={n}, SE=±{:.1})",
        label, m, sd, min, p50, p99, max, se,
    );
}
```

同样在 `tests/kcmm_bench_alloc.rs` 顶部增加 `mean/stddev/std_error/print_latency_stats`
（或抽取为公共模块 `tests/bench_utils.rs`，见 2.7）。

---

### 2.2 Benchmark 2a — Single-Block Evict/Restore

**文件:** `tests/kcmm_bench_tiering.rs`
**函数:** `kcmm_bench_single_block_evict_restore`

改动：64 → 256 samples，增加 warmup (8 次)，输出改为 `print_latency_stats`。

```rust
// 修改前 (line 107):
let num_samples = 64;

// 修改后:
let num_samples = 256;
let warmup_iters = 8;

// 在循环前增加 warmup (line 110 之前插入):
for _ in 0..warmup_iters {
    let block_idx = pool.alloc_block().expect("warmup alloc");
    let handle = pool.get_block_handle(block_idx).expect("warmup handle");
    let _ = tiering.evict_blocks(&pool, &[handle], 1);
    let _ = pool.restore_evicted_block(block_idx);
}

// 输出改为 (替换 line 130–139 的 println!):
let block_label = format!("{}B_l{}", block_bytes, num_layers);
print_latency_stats(&format!("{block_label}_evict"), &mut evict_lat, "ns");
print_latency_stats(&format!("{block_label}_restore"), &mut restore_lat, "ns");
```

---

### 2.3 Benchmark 2b — Batch Eviction Amortisation

**文件:** `tests/kcmm_bench_tiering.rs`
**函数:** `kcmm_bench_batch_eviction_amortization`

改动：4 → 30 rounds，输出增加 stddev/SE。

```rust
// 修改前 (line 197):
let rounds = 4;

// 修改后:
let rounds = 30;

// 输出改为 (替换 line 193–241 的打印块):
// 每个 batch_size 输出完整的统计行
let mut all_per_block: Vec<u64> = Vec::with_capacity(rounds);
// ... (收集所有 round 的 per_block 值)

// 在计算 amort_factor 之前，打印统计:
print_latency_stats(
    &format!("batch_{batch_size}_evict_per_block"),
    &mut per_block_latencies,
    "ns",
);
```

完整修改后的循环体（替换 lines 195–241）：

```rust
for &batch_size in batch_sizes {
    let rounds = 30;
    let mut per_block_latencies: Vec<u64> = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let pairs = alloc_blocks(&pool, batch_size);
        let handles: Vec<BlockHandle> = pairs.iter().map(|(_, h)| *h).collect();

        let t0 = Instant::now();
        let evicted = tiering
            .evict_blocks(&pool, &handles, batch_size)
            .expect("batch evict");
        let total_ns = t0.elapsed().as_nanos() as u64;
        assert_eq!(evicted.len(), batch_size);

        per_block_latencies.push(total_ns / batch_size as u64);

        for (idx, _) in &pairs {
            pool.restore_evicted_block(*idx).expect("restore");
        }
    }

    print_latency_stats(
        &format!("evict_batch={batch_size}"),
        &mut per_block_latencies,
        "ns",
    );

    let avg = mean(&per_block_latencies) as u64;
    batch_results.push((batch_size, avg));
}

// Amortisation 计算保持不变
let baseline = if let Some(&(_, avg)) = batch_results.first() {
    avg
} else {
    return;
};

println!("\n  Amortisation factors (vs batch=1):");
for &(batch_size, avg) in &batch_results {
    let factor = baseline as f64 / avg as f64;
    println!("    batch={batch_size:>3}:  {factor:.2}×");
}
```

---

### 2.4 Benchmark 2e — Batch Restore Amortisation

**文件:** `tests/kcmm_bench_tiering.rs`
**函数:** `kcmm_bench_batch_restore_amortization`

改动：与 2b 对称，4 → 30 rounds，输出增加 stddev/SE。

```rust
// 修改前 (line 285):
let rounds = 4;

// 修改后:
let rounds = 30;

// 输出改为 print_latency_stats，与 2b 完全对称
```

---

### 2.5 Benchmark 2c — cuMemMap/cuMemUnmap Latency

**文件:** `tests/kcmm_bench_tiering.rs`
**函数:** `kcmm_bench_cumemmap_latency`

改动：32 → 128 samples，输出用 `print_latency_stats`。

```rust
// 修改前 (line 364):
let iters = 32;

// 修改后:
let iters = 128;

// 输出改为 (替换 lines 384–389 的 println!):
print_latency_stats(
    &format!("cumemmap_{size}B_map"),
    &mut map_lat,
    "ns",
);
print_latency_stats(
    &format!("cumemmap_{size}B_unmap"),
    &mut unmap_lat,
    "ns",
);
```

---

### 2.6 Benchmark 3 — Stream Interference

**文件:** `tests/kcmm_bench_tiering.rs`
**函数:** `kcmm_bench_stream_interference`

改动：32 → 128 iterations，输出用 `print_latency_stats`。

```rust
// 修改前 (line 532):
let iters = 32;

// 修改后:
let iters = 128;

// 增加 warmup (4 → 12):
for _ in 0..12 { /* warmup */ }

// 输出改为 print_latency_stats，额外保留 overhead 百分比:
print_latency_stats("stream_baseline", &mut baseline_lat, "ns");
print_latency_stats("stream_interference", &mut interference_lat, "ns");
println!("  Overhead: p50={:+.2}%  p99={:+.2}%", overhead_p50, overhead_p99);
```

---

### 2.7 (可选) 抽取公共统计模块

如果不想在两个文件里重复 `mean/stddev/std_error/print_latency_stats`，可以新建
`tests/bench_utils.rs`：

```rust
// tests/bench_utils.rs — shared statistics helpers for KCMM benchmarks.

/// Compute the `p`-th percentile (0..100). Sorts in-place.
pub fn percentile(data: &mut [u64], p: f64) -> u64 { /* ... */ }

pub fn mean(data: &[u64]) -> f64 { /* ... */ }
pub fn stddev(data: &[u64], mean: f64) -> f64 { /* ... */ }
pub fn std_error(stddev: f64, n: usize) -> f64 { /* ... */ }

pub fn print_latency_stats(label: &str, data: &mut [u64], unit: &str) {
    /* ... */
}
```

然后在 `kcmm_bench_tiering.rs` 和 `kcmm_bench_alloc.rs` 中：

```rust
mod bench_utils;
use bench_utils::*;
```

---

### 2.8 Benchmark 1c — Multi-Sequence Concurrent Allocation

**文件:** `tests/kcmm_bench_alloc.rs`
**函数:** `kcmm_bench_alloc_concurrent_sequences`

改动：单次 → 16 rounds，输出统计。

```rust
// 修改后:
let rounds = 16;
let mut alloc_per_block_ns: Vec<u64> = Vec::with_capacity(rounds);
let mut free_per_block_ns: Vec<u64> = Vec::with_capacity(rounds);

for _ in 0..rounds {
    let mut all_tables = Vec::with_capacity(concurrency);
    let t0 = Instant::now();
    for _ in 0..concurrency {
        let table = pool.alloc_sequence(blocks_per_seq).expect("multi-seq alloc");
        all_tables.push(table);
    }
    let total_ns = t0.elapsed().as_nanos() as u64;
    alloc_per_block_ns.push(total_ns / total_blocks as u64);

    let t0 = Instant::now();
    for table in &all_tables {
        pool.free_sequence(table);
    }
    let total_ns = t0.elapsed().as_nanos() as u64;
    free_per_block_ns.push(total_ns / total_blocks as u64);
}

print_latency_stats("alloc_per_block", &mut alloc_per_block_ns, "ns");
print_latency_stats("free_per_block", &mut free_per_block_ns, "ns");
```

---

### 2.9 Step3 Benchmarks (cuMemMap Overhead)

**文件:** `tests/step3_benchmarks.rs`
**函数:** `step3_cumemmap_overhead`

改动：16 → 64 iters。

```rust
// 修改前 (line 181):
let iters = 16;

// 修改后:
let iters = 64;
```

---

## 3. 改动量估算

| 文件 | 改动行数 | 内容 |
|------|---------|------|
| `tests/kcmm_bench_tiering.rs` | ~80 | 新增统计函数 + 修改 5 个 benchmark 的采样数和输出 |
| `tests/kcmm_bench_alloc.rs` | ~50 | 新增统计函数 + 修改 1c (rounds: 1→16) |
| `tests/step3_benchmarks.rs` | ~2 | iters: 16→64 |
| (可选) `tests/bench_utils.rs` | ~60 | 公共统计模块 |

**总计约 130–190 行。** 预计工时 ~2 小时。

---

## 4. 运行时影响

| Benchmark | 当前采样 | 修改后 | 运行时间变化 |
|-----------|---------|--------|-------------|
| 2a evict/restore | 64 samples | 256 samples + 8 warmup | 2→8 秒 |
| 2b batch eviction | 4 rounds × 4 batches | 30 rounds × 4 batches | 3→25 秒 |
| 2c cuMemMap | 32 iters | 128 iters | 1→4 秒 |
| 2e batch restore | 4 rounds × 4 batches | 30 rounds × 4 batches | 3→25 秒 |
| 3 stream interference | 32 iters × 2 | 128 iters × 2 | 5→20 秒 |
| 1c multi-seq | 1 pass | 16 rounds | <1→3 秒 |
| step3 cuMemMap | 16 iters | 64 iters | 2→8 秒 |

**全部 benchmark 运行时间从约 30 秒增加到约 2 分钟**，仍然在可接受范围。

---

## 5. 修改后的预期报告输出示例

```
=== KCMM Benchmark 2b: Batch Eviction Amortisation ===
block_bytes=65536, num_layers=2

  evict_batch=1         201.3 ±  12.4 µs  [185, 199, 231, 248]  (n=30, SE=±2.3)
  evict_batch=4         216.1 ±  18.3 µs  [188, 212, 262, 280]  (n=30, SE=±3.3)
  evict_batch=16        102.7 ±   8.1 µs  [ 88, 100, 118, 128]  (n=30, SE=±1.5)
  evict_batch=64         99.3 ±   5.7 µs  [ 86,  97, 110, 116]  (n=30, SE=±1.0)

  Amortisation factors (vs batch=1):
    batch=  1:  1.00×
    batch=  4:  0.93×  (CI overlap with batch=1 — not statistically significant)
    batch= 16:  1.96×
    batch= 64:  2.03×
```

此时可以直接从报告判断：batch=4 的 216µs 和 batch=1 的 201µs 的 CI 存在重叠，
差异不显著；而 batch=16 和 batch=64 的收益是明确的。
