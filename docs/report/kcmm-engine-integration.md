# KCMM §1.6 Engine Integration Benchmark — Verification Report

**生成时间:** 2026-06-13 08:48 UTC
**分支:** `kcmm`
**GPU:** NVIDIA A30，VRAM 24576 MiB
**Profile:** release，`--features kcmm`
**测试文件:** `tests/kcmm_bench_engine_integration.rs`

## 1. 概述

本文档是对 KCMM engine integration benchmark 两个 targeted verification run
的独立报告：

| 结果目录 | 测试 | 状态 |
|---|---:|---|
| `results/kcmm_engine_integration_20260613_032231` | `kcmm_engine_integration_single` | 1/1 passed (218.5s) |
| `results/kcmm_engine_integration_20260613_032650` | `kcmm_engine_integration_sweep` | 1/1 passed (597.7s) |

这两个 run 均在 issue 17（`Peak GPU blocks` 不再超过 superblock-aligned physical
ceiling）的修复生效后执行。

### 1.1 测试模型与工作负载设计

- **模型:** LlamaTransformer，L=8，kv_heads=4，head_dim=64，hidden=1024
- **权重:** deterministic Xavier-init，seed `0x00005EED1A771CE5`，OFF/ON 使用相同权重
- **Admission policy:** Tiering ON 最多重试 4 轮 cold-block eviction 后才 reject
- **Thrashing 阈值:** evictions/full-completion > 3.0 标记为 `THRASH`
- **GPU ballast:** Tiering OFF 时预留 TieringEngine staging-buffer 等效 GPU 内存，保证 OFF/ON 公平对比

工作负载模拟 continuous batching 场景：prefill → dynamic arrivals → decode，包含
cooling/re-heating 循环以触发 evict→restore 路径。

## 2. Single Config 详细结果

**来源:** `results/kcmm_engine_integration_20260613_032231/kcmm_engine_integration_single.log`

**配置:**

```text
bs16_mb16_msl640_pl[128,256]_mnt384_reqs32_ari12
block_bytes=65536 (64 KiB), VA blocks=640 (~40 MiB), total_requests=32
max_blocks_total=640, blocks_per_superblock=256, aligned_physical_ceiling=768
```

**方法:** 5 次 alternating OFF/ON run（even runs OFF-first，odd runs ON-first），
取平均值以消除首轮 bias。

### 2.1 主要指标（5 次运行平均）

| Metric | Tiering OFF | Tiering ON |
|---|---:|---:|
| Full completions | 30.0 | 32.0 |
| Capped | 2.0 | 0.0 |
| Rejected | 0.0 | 0.0 |
| Leftover at end | 0.0 | 0.0 |
| Total tokens | 18,112.0 | 18,432.0 |
| Decode tokens | 11,968.0 | 12,288.0 |
| Elapsed (ms) | 20,457.6 | 20,455.2 |
| Tokens/sec | 885.3 | 901.1 |
| Peak concurrent | 32 | 32 |
| Step P50 (µs) | 24,976 | 24,951 |
| Step P90 (µs) | 31,783 | 31,568 |
| Step P95 (µs) | 32,215 | 31,895 |
| Step P99 (µs) | 32,974 | 33,143 |
| Evictions | 0.0 | 160.0 |
| Restores | 0.0 | 10.0 |
| Evict/full completion | 0.0 | 5.0 |
| Peak GPU blocks | 766 | 768 |

### 2.2 派生指标

| Metric | Value |
|---|---:|
| Throughput ratio (ON/OFF) | 1.02 ± 0.00× |
| Capacity ratio (ON/OFF) | 1.07 ± 0.00× |
| Step P50 mean ± std | OFF = 24,986 ± 26 µs，ON = 24,953 ± 12 µs |
| Step P99 mean ± std | OFF = 32,955 ± 56 µs，ON = 33,131 ± 19 µs |
| P50 latency overhead | −0.1% |
| P99 latency overhead | +0.5% |
| Avg batch size | OFF = 19.5，ON = 19.8 |
| Status | **THRASH** |

### 2.3 解读

- **容量提升：** Tiering ON 将 full completions 从 30 提高到 32（+6.7%），capped
  从 2 降到 0。capacity ratio 1.07×，未达 1.3× 目标。
- **吞吐量基本持平：** throughput ratio 1.02×，P50 latency 甚至有 −0.1% 的微小优势。
- **Latency 开销极小：** P50 −0.1%，P99 +0.5%。Tiering 的 per-step 开销在当前
  配置下可忽略不计，说明 eviction/restore 不在 critical path 上产生明显延迟。
- **THRASH 警告：** 5.0 evictions/full-completion 超过 3.0 阈值。160 次 evictions
  只换回 2 个额外 completion——eviction 效率不高。
- **Peak GPU blocks 合规：** OFF peak 766、ON peak 768，均在 aligned physical ceiling
  (768) 以下。issue 17 修复确认有效。

## 3. Sweep 结果

**来源:** `results/kcmm_engine_integration_20260613_032650/kcmm_engine_integration_sweep.log`

4 个配置 × 5 次 alternating OFF/ON repeat = 20 组对比。

### 3.1 Sweep 汇总表

| Config | OFF F/C/R/L | ON F/C/R/L | TpRatio | CapRatio | Ev/Full | Evict | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `bs16_mb16_msl640_pl[128,256]_mnt384_reqs32_ari12` | 30/2/0/0 | 32/0/0/0 | 1.02±0.00× | 1.07±0.00× | 5.0 | 160 | THRASH |
| `bs16_mb12_msl512_pl[128,256]_mnt256_reqs36_ari8` | 29/7/0/0 | 36/0/0/0 | 1.08±0.00× | 1.24±0.00× | 8.0 | 288 | THRASH |
| `bs32_mb16_msl512_pl[128,256]_mnt256_reqs32_ari12` | 27/5/0/0 | 32/0/0/0 | 1.03±0.00× | 1.19±0.00× | 9.2 | 296 | THRASH |
| `bs16_mb10_msl384_pl[64,128,256]_mnt128_reqs40_ari4` | 20/8/12/0 | 40/0/0/0 | 1.38±0.00× | 2.00±0.00× | 6.4 | 255 | THRASH |

> F/C/R/L = Full completions / Capped / Rejected / Leftover at end。每个值为 5
> 次运行平均。

### 3.2 Physical Ceiling 确认

每个配置在下都打印了 `max_blocks_total`、`blocks_per_superblock` 和
`aligned_physical_ceiling`，且没有任何配置触发 `Peak GPU blocks ... exceeded
aligned` warning。

| Config | max_blocks_total | blocks_per_superblock | aligned_physical_ceiling |
|---|---:|---:|---:|
| `bs16_mb16_msl640...` | 640 | 256 | 768 |
| `bs16_mb12_msl512...` | 384 | 256 | 512 |
| `bs32_mb16_msl512...` | 256 | 128 | 256 |
| `bs16_mb10_msl384...` | 240 | 256 | 256 |

### 3.3 Per-Config Step Latency

| Config | OFF P50±σ (µs) | OFF P99±σ (µs) | ON P50±σ (µs) | ON P99±σ (µs) |
|---|---:|---:|---:|---:|
| `bs16_mb16_msl640...` | 24,886 ± 34 | 32,896 ± 20 | 24,897 ± 31 | 33,199 ± 18 |
| `bs16_mb12_msl512...` | 20,615 ± 14 | 26,192 ± 7 | 20,609 ± 12 | 25,819 ± 10 |
| `bs32_mb16_msl512...` | 21,112 ± 10 | 26,588 ± 46 | 21,050 ± 12 | 26,783 ± 25 |
| `bs16_mb10_msl384...` | 13,677 ± 13 | 19,621 ± 20 | 14,454 ± 11 | 19,081 ± 16 |

### 3.4 关键观察

**容量维度：**

- Tiering ON 在所有 4 个 sweep 配置中都实现了更高的 full completions，且每个配置
  都把 capped/rejected/leftover 降到了 0。
- 容量提升幅度从 1.07× 到 2.00×。配置越紧（更小的 max_batch、更频繁的 arrival、
  更多样的 prompt lengths），容量收益越大。
- 配置 2（CapRatio 1.24×）已接近 1.3× 容量目标；配置 4 达到 2.00×。

**吞吐量维度：**

- 最佳 throughput ratio 来自最紧的配置 4：1.38×。这是唯一达到 ≥1.3× 吞吐量目标的
  配置。
- 配置 1–3 的 throughput ratio 在 1.02–1.08×，说明 tiering 的吞吐量收益在较宽松
  的 VA 约束下不显著。
- 注意：配置 4 的 step P50 从 13,677 µs 上升到 14,454 µs（+5.7%），但 P99 从
  19,621 µs 降到 19,081 µs（−2.8%）。ON 的 P50 略高但 tail latency 更好，说明
  tiering 平滑了 tail。

**THRASH 问题：**

- 所有 4 个配置均被标记为 `THRASH`。evictions/full-completion 范围从 5.0 到 9.2，
  远高于 3.0 阈值。
- 配置 3 的 eviction pressure 最高（9.2 evictions/full-completion），对应 296 次
  evictions 换取 32 个 full completions。
- Tiering ON 在压力 workload 下有效恢复了容量，但当前 eviction policy 以大量
  eviction 为代价——这是真实问题，不是 metric artifact。

**Latency 影响：**

- 3/4 配置中 ON 的 P50 latency 与 OFF 基本持平（差异在 ±0.5% 以内）。
- 配置 4 中 ON P50 高出 5.7%（+777 µs），但 P99 反而低 2.8%（−540 µs）。
- P99 在所有配置中差异均在 ±3% 以内，说明 tiering 的 tail latency 影响可控。
- 所有配置的 step-to-step variance（σ）都很小（7–46 µs），说明测试的确定性良好。

**Physical ceiling（issue 17 验证）：**

- 所有配置的 aligned physical ceiling 均被正确执行，无任何 `Peak GPU blocks
  exceeded` warning。
- 配置 2 的 aligned physical ceiling 为 512，而 max_blocks_total 仅 384——
  superblock 粒度（2 MiB）下的小池子会被向上取整较多，这是预期的。

## 4. 总体结论

### 4.1 正面发现

1. **Tiering 在所有配置下都提升了容量。** 无一例外。capacity ratio 范围 1.07–2.00×。
2. **Latency 开销极小。** P50 overhead 在 −0.1% 到 +5.7% 之间；P99 overhead 在
   −2.8% 到 +0.8% 之间。Tiering 不在 critical path 上产生显著延迟。
3. **最佳配置达到 1.38× throughput，满足 ≥1.3× 目标。** 这是 tight/churny workload
   （小块、小 batch、频繁 arrival、多样 prompt）下的结果。
4. **Physical ceiling 合规。** issue 17 修复后，所有配置的 Peak GPU blocks 均未超过
   aligned physical capacity。
5. **结果高度一致。** 5 次 alternating run 的 std 极小（tp_ratio/cap_ratio std ≈ 0，
   latency std < 50 µs），说明测试方法学具有足够的统计功效。

### 4.2 需要关注的问题

1. **所有配置均为 THRASH。** evictions/full-completion 在 5.0–9.2 范围内，远超 3.0
   阈值。当前 tiering policy 在 tight memory 下靠大量 eviction 换取容量。
2. **Throughput 收益不均衡。** 宽松配置（1.02–1.08×）的吞吐量提升远低于紧配置
   （1.38×）。eviction 的固定开销在吞吐量较小时占比更高。
3. **容量提升与 eviction cost 不成比例。** 配置 1 中 160 次 evictions 只换回 2 个
   额外 completion。需要评估 eviction targeting policy 的选择效率。

### 4.3 下一步建议

1. **Eviction path 性能诊断：** 进入 `pool.rs` 和 `tiering.rs` 中的 eviction
   hot path，解释为什么 batch eviction 在 memory pressure 中总时间占比极高。
2. **Eviction policy 改进：** 降低 THRASH 配置中的 evictions/full-completion。
   可能方向：更智能的 victim selection（当前 prefer cooled → fallback active）、
   预取窗口调优、admission retry limit 动态调整。
3. **扩大 sweep 覆盖面：** 添加 block_size=64 的大块配置、更多样的 prompt-length
   分布（如 Zipf），以更全面地表征 tiering 在不同 workload 下的行为。
4. **与 memory pressure benchmark 交叉验证：** 两个 benchmark 都在 tight memory
   下看到 THRASH；交叉分析可以定位 eviction overhead 是 tiering 层面还是 engine
   integration 层面的问题。

## 5. 数据源

| 路径 | 内容 |
|---|---|
| `results/kcmm_engine_integration_20260613_032231/kcmm_engine_integration_single.log` | Single config 完整 stdout |
| `results/kcmm_engine_integration_20260613_032231/summary.txt` | Single config 元数据 |
| `results/kcmm_engine_integration_20260613_032650/kcmm_engine_integration_sweep.log` | Sweep 完整 stdout |
| `results/kcmm_engine_integration_20260613_032650/summary.txt` | Sweep 元数据 |
| `tests/kcmm_bench_engine_integration.rs` | 测试源码 |

---

*本报告基于 issue 17 修复后的 targeted verification run。完整 benchmark 批次
（含 allocation、tiering microbenchmarks、memory pressure、step3 等）的最新结果
参见 `docs/report/kcmm-benchmark-review-followups.md`。*
