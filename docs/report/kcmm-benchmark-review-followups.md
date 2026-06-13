# KCMM Benchmark Review Followups - Results 复核报告

**生成时间:** 2026-06-13 08:37 UTC
**分支:** `kcmm` @ `2e490a1`，当前工作区仍有未提交改动
**主要结果集:** `results/kcmm_bench_20260613_012500`
**GPU/Profile:** NVIDIA A30，release，`--features kcmm`
**主要结果状态:** 15 passed, 0 failed

## 1. 数据范围和可信度

当前 `results/` 下有多个结果目录，但它们的含义不同：

| 结果目录 | 状态 | 本报告如何使用 |
|---|---:|---|
| `results/kcmm_bench_20260613_012500` | 15/15 passed | 最新完整 benchmark 批次。当前性能结论以它为准。 |
| `results/kcmm_engine_integration_20260613_021915` | 0/1 passed | 中间状态编译失败产生的 stale run。不能作为性能结论。 |
| `results/kcmm_bench_20260613_022104` | 1/1 passed | 只重跑了 alloc throughput。只用于确认 alloc benchmark 的命名/范围修正。 |
| `results/kcmm_engine_integration_20260613_025519` | 1/1 passed | issue 17 修复前的 engine integration single targeted run；已被 `032231` 取代。 |
| `results/kcmm_engine_integration_20260613_025945` | 1/1 passed | issue 17 修复前的 engine integration sweep targeted run；已被 `032650` 取代。 |
| `results/kcmm_engine_integration_20260613_032231` | 1/1 passed | issue 17 修复后的 engine integration single targeted verification。 |
| `results/kcmm_engine_integration_20260613_032650` | 1/1 passed | issue 17 修复后的 engine integration sweep targeted verification。 |

重要前提：`results/kcmm_bench_20260613_012500` 仍是最新完整结果批次；当前工作区
的 issue 09-17 改动已经跑过 targeted verification，但还没有重新跑完整
`scripts/run_kcmm_benches.sh`。当前代码已经通过：

```text
cargo test --features kcmm --tests --no-run
```

因此，本报告的整体性能结论仍以完整批次 `012500` 为主；issue 09-17 的完成状态
以 targeted verification 为准。

## 2. 总体结论

最新完整结果是自洽的：15 个 benchmark 全部通过。它支持以下结论：

1. Tiering ON 在 engine integration 和 memory pressure 场景里都提升了
   Capacity-at-Workload 或 completion count。
2. 这个提升不是免费的。Memory pressure 中 KCMM 的 elapsed throughput 明显更差，
   主要时间被 eviction 吃掉。
3. Engine integration 中最强的收益来自 tight/churny 压力配置，但这些配置也被
   标记为 `THRASH`，说明当前 tiering policy 的 eviction pressure 偏高。
4. Batch eviction 现在不能再说“明显退化”。当前数据更准确的说法是：per-block
   eviction latency 基本持平，没有明显 batch amortization，batch=64 略差。
5. issue 17 的 targeted verification 已确认 `Peak GPU blocks` 不再超过
   superblock-aligned physical ceiling；之前的超限是 `ensure_capacity()` 未限制
   物理 superblock 数导致的真实 over-allocation。
6. `step3_cumemmap_overhead.log` 里的 map/unmap 分列仍是旧 bug 结果，不能用于
   map vs unmap 的独立结论；要看独立 map/unmap，请用
   `kcmm_bench_cumemmap_latency.log`。

下一步不应该继续只做指标命名清理。真正需要进入的是 eviction path 的性能诊断：
解释为什么单 block evict/restore 很快，但 memory pressure 里的 batch eviction
总时间占比极高，并降低 `THRASH` 配置中的 evictions/full-completion。

## 3. 完整结果批次元数据

来源：`results/kcmm_bench_20260613_012500/summary.txt`

| 字段 | 值 |
|---|---|
| Date | Sat Jun 13 01:29:36 AM CDT 2026 |
| GPU | NVIDIA A30 |
| VRAM | 24576 MiB |
| Profile | release |
| Features | `--features kcmm` |
| Passed | 15 |
| Failed | 0 |

该批次包含 allocation、tiering microbenchmarks、Step 3 benchmarks、memory
pressure、engine integration single/sweep。

## 4. Engine Integration

### 4.1 Single Config

来源：`results/kcmm_engine_integration_20260613_032231/kcmm_engine_integration_single.log`

配置：

```text
bs16_mb16_msl640_pl[128,256]_mnt384_reqs32_ari12
block_bytes=65536, VA blocks=640 (~40 MiB), total_requests=32
model=LlamaTransformer, L=8, kv_heads=4, head_dim=64, hidden=1024
```

该测试使用 deterministic Xavier-init seed `0x00005EED1A771CE5`，所以 Tiering
OFF/ON 对比使用相同权重。Tiering ON 在 admission 时最多重试 4 轮 cold-block
eviction。

结果为 5 次 alternating OFF/ON 运行平均：

| Metric | Tiering OFF | Tiering ON |
|---|---:|---:|
| Full completions | 30 | 32 |
| Capped | 2 | 0 |
| Rejected | 0 | 0 |
| Leftover at end | 0 | 0 |
| Total tokens | 18,112 | 18,432 |
| Decode tokens | 11,968 | 12,288 |
| Elapsed | 20,457.6 ms | 20,455.2 ms |
| Tokens/sec | 885.3 | 901.1 |
| Peak concurrent | 32 | 32 |
| Step P50 | 24,976 us | 24,951 us |
| Step P90 | 31,783 us | 31,568 us |
| Step P95 | 32,215 us | 31,895 us |
| Step P99 | 32,974 us | 33,143 us |
| Evictions | 0 | 160 |
| Restores | 0 | 10 |
| Evict/full completion | 0.0 | 5.0 |
| Peak GPU blocks | 766 | 768 |

派生指标：

| Metric | Value |
|---|---:|
| Throughput ratio | 1.02x |
| Capacity ratio | 1.07x |
| P50 overhead | -0.1% |
| P99 overhead | +0.5% |
| Status | `THRASH` |

解读：

- Tiering ON 将 full completions 从 30 提高到 32，capacity ratio 为 1.07x。
- latency/throughput 基本持平：P50 还略低，P99 只高 0.5%。
- 这个配置没有达到 1.3x capacity 目标；同时 evictions/full-completion 为 5.0，
  高于当前 `THRASH` 阈值 3.0。
- issue 17 修复后，`Peak GPU blocks` 最高为 768，等于
  `ceil(max_blocks_total=640 / blocks_per_superblock=256) * 256`，没有再超过
  aligned physical ceiling。

注意：`Peak GPU blocks` 不能直接当作 Capacity-at-Workload。它是代表性 per-layer
physical blocks in use。它可以超过逻辑 `max_blocks_total`，但不能超过按
`blocks_per_superblock` 向上取整后的 physical ceiling；当前 targeted run 已验证
这个上限被正确执行。

### 4.2 Sweep

来源：`results/kcmm_engine_integration_20260613_032650/kcmm_engine_integration_sweep.log`

| Config | OFF F/C/R/L | ON F/C/R/L | TpRatio | CapRatio | Ev/Full | Evict | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `bs16_mb16_msl640_pl[128,256]_mnt384_reqs32_ari12` | 30/2/0/0 | 32/0/0/0 | 1.02x | 1.07x | 5.0 | 160 | `THRASH` |
| `bs16_mb12_msl512_pl[128,256]_mnt256_reqs36_ari8` | 29/7/0/0 | 36/0/0/0 | 1.08x | 1.24x | 8.0 | 288 | `THRASH` |
| `bs32_mb16_msl512_pl[128,256]_mnt256_reqs32_ari12` | 27/5/0/0 | 32/0/0/0 | 1.03x | 1.19x | 9.2 | 296 | `THRASH` |
| `bs16_mb10_msl384_pl[64,128,256]_mnt128_reqs40_ari4` | 20/8/12/0 | 40/0/0/0 | 1.38x | 2.00x | 6.4 | 255 | `THRASH` |

关键观察：

- Tiering ON 在所有 sweep 配置中都提高了 full completions。
- Tiering ON 在这些配置里把 capped/rejected/leftover 都降到 0。
- 唯一达到 1.3x throughput target 的是最紧的 `bs16_mb10...`，但它同时也是
  明显的 `THRASH`：255 次 evictions，6.4 evictions/full-completion。
- issue 17 修复后，所有 sweep 配置都打印 `max_blocks_total`、
  `blocks_per_superblock` 和 `aligned_physical_ceiling`，且没有出现
  `Peak GPU blocks ... exceeded aligned` warning。
- 修复后 sweep 里的 4 个配置全部被 `THRASH` 标记；容量收益仍在，但 eviction
  pressure 比旧完整批次中看到的更突出。

结论：engine integration 结果说明 tiering 可以在压力下恢复容量，但当前策略在
tight/churny workload 下靠大量 eviction 换容量，`THRASH` 是真实问题。

## 5. Memory Pressure

### 5.1 Single Config

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_memory_pressure_single.log`

配置：

```text
TinyLlama, L=22, kv_heads=4, head_dim=64
bs16_mb16_msl640_pl[128,256]_mnt384_arr32
block_bytes=180224 (176 KiB), VA blocks=640 (~110 MiB), total_arrivals=32
```

| Metric | Baseline | KCMM Tiering ON |
|---|---:|---:|
| Completed | 24 | 32 |
| Capped | 8 | 0 |
| Rejected | 0 | 0 |
| Peak concurrent | 32 | 32 |
| Total alloc | 1079 | 1152 |
| Evictions | 0 | 30 |
| CPU swap peak | n/a | 5,406,720 B |
| Peak blocks | n/a | 795 |
| Elapsed | 12 ms | 201 ms |
| Elapsed throughput | 2000.00 completed/s | 159.20 completed/s |

派生指标：

| Metric | Value |
|---|---:|
| Completion ratio | 1.33x |
| KCMM elapsed slowdown | 16.75x |
| KCMM eviction timing | 188 ms / 201 ms |

解读：

- capacity/completion 结果是正面的：KCMM 完成 32 个，baseline 完成 24 个。
- throughput 结果不是正面的：KCMM elapsed throughput 从 2000.00 completed/s
  降到 159.20 completed/s。
- KCMM 总时间 201 ms，其中 eviction 是 188 ms，说明 bottleneck 非常集中。

`cpu_swap_peak=5,406,720 B` 是 issue 10 修复前的完整结果。当前工作区已经有
live CPU swap peak 的修正，但还没有新的完整结果验证，所以这个字段暂时只能当
pre-fix artifact 看。

### 5.2 Sweep

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_memory_pressure_sweep.log`

| Config | BaseDone | KcmmDone | CompRatio | RejB | RejK | CappedB | CappedK | Evict | BaseMs | KcmmMs | ThrB/s | ThrK/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs16_mb16_msl640_pl[128,256]_mnt384_arr32` | 24 | 32 | 1.33x | 0 | 0 | 8 | 0 | 30 | 12 | 200 | 2000.00 | 160.00 |
| `bs16_mb12_msl512_pl[128,256]_mnt256_arr36` | 23 | 36 | 1.57x | 13 | 13 | 13 | 0 | 32 | 8 | 207 | 2875.00 | 173.91 |
| `bs32_mb16_msl512_pl[128,256]_mnt256_arr32` | 21 | 32 | 1.52x | 2 | 8 | 11 | 0 | 13 | 7 | 163 | 3000.00 | 196.32 |
| `bs16_mb10_msl384_pl[64,128,256]_mnt128_arr40` | 19 | 40 | 2.11x | 11 | 17 | 21 | 0 | 22 | 4 | 142 | 4750.00 | 281.69 |

关键观察：

- KCMM 所有配置 completion ratio 都不低于 1.33x。
- KCMM 所有配置都消除了 capping：`CappedK = 0`。
- KCMM elapsed throughput 在所有配置里都显著低于 baseline。
- KCMM 总时间 142-207 ms，其中 eviction time 为 137-198 ms。

结论：memory pressure 支持的是 capacity story，不是 throughput story。下一步
需要解释和降低 eviction time，而不是继续只改 metric 名称。

## 6. Batch Eviction 和 Batch Restore

### 6.1 Batch Eviction

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_batch_eviction_amortization.log`

配置：64 KiB blocks，2 layers，30 rounds。

| Batch | Mean per block | P50 | P99 | P50 factor | Mean factor |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.3 us | 60 us | 79 us | 1.00x | 1.00x |
| 4 | 59.0 us | 59 us | 61 us | 1.02x | 1.04x |
| 16 | 59.3 us | 59 us | 61 us | 1.02x | 1.03x |
| 64 | 63.5 us | 63 us | 69 us | 0.95x | 0.96x |

当前结果不能支持“batch eviction 明显退化”的结论。更准确的结论是：

- batch=4 和 batch=16 与 batch=1 基本持平；
- batch=64 略差；
- 没有明显 batch amortization；
- 如果期望 batch 更大 per-block 更低，当前路径没有实现这个收益。

可能原因是 per-block copy/sync 仍占主导，batch 层面的调度或 bookkeeping 开销抵消
了 amortization。

### 6.2 Batch Restore

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_batch_restore_amortization.log`

| Batch | Mean per block | P50 | P99 | Factor vs batch=1 |
|---:|---:|---:|---:|---:|
| 1 | 39.7 us | 41 us | 49 us | 1.00x |
| 4 | 35.4 us | 35 us | 39 us | 1.12x |
| 16 | 26.2 us | 25 us | 38 us | 1.51x |
| 64 | 46.5 us | 47 us | 47 us | 0.85x |

Restore 有一个比较明确的甜点：batch=16。batch=64 反而退化。

## 7. Tiering Microbenchmarks

### 7.1 Single-Block Evict/Restore

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_single_block_evict_restore.log`

| Block bytes | Evict mean | Evict P50 | Evict P99 | Restore mean | Restore P50 | Restore P99 |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 47.8 us | 47 us | 53 us | 25.3 us | 24 us | 33 us |
| 65,536 | 61.6 us | 61 us | 68 us | 36.1 us | 35 us | 46 us |
| 131,072 | 88.6 us | 88 us | 97 us | 59.9 us | 59 us | 65 us |

单 block 路径的 evict/restore 成本随 block size 增长，数值本身不高。它和
memory pressure 中 6-12 ms 的 batch eviction detail 差距很大，这正是下一步
需要诊断的地方：memory pressure 里的慢不只是单 block copy 慢。

### 7.2 Roundtrip Data Integrity

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_tiering_roundtrip_data_integrity.log`

| Metric | Value |
|---|---:|
| Evict 16 blocks | 10,251 us |
| Evict per block | 640.7 us/block |
| Restore 16 blocks | 516 us |
| Restore per block | 32.2 us/block |
| Coverage | 16 blocks x 2 layers x K+V = 64 cache payloads |
| Data integrity | 64/64 payloads OK |

数据完整性结果是正面的：扩展到 all-layer K/V coverage 后，64/64 cache payloads
均通过 evict -> restore roundtrip 校验。

### 7.3 CUDA Stream Interference

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_stream_interference.log`

| Metric | Baseline | Interference |
|---|---:|---:|
| Mean | 1718.0 us | 1720.5 us |
| P50 | 1716 us | 1713 us |
| P99 | 1736 us | 1892 us |
| Max | 1744 us | 2115 us |

派生 overhead：

| Percentile | Overhead |
|---|---:|
| P50 | -0.21% |
| P99 | +9.00% |

P50 基本无影响；P99 有 9% overhead，但与 memory pressure 中 eviction 占 137-198 ms
相比，这不是当前最大的瓶颈。

## 8. Allocation Benchmarks

### 8.1 完整批次中的 allocation 结果

来源：

- `results/kcmm_bench_20260613_012500/kcmm_bench_alloc_throughput.log`
- `results/kcmm_bench_20260613_012500/kcmm_bench_alloc_pool_size_sweep.log`
- `results/kcmm_bench_20260613_012500/kcmm_bench_alloc_concurrent_sequences.log`

单 block alloc/free：

| Block bytes | Pool blocks | Alloc P50 | Alloc P99 | Free P50 | Free P99 |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 4096 | 140 ns | 141 ns | 130 ns | 131 ns |
| 65,536 | 4096 | 140 ns | 141 ns | 130 ns | 131 ns |
| 131,072 | 4096 | 140 ns | 141 ns | 130 ns | 131 ns |

Pool-size sweep，65,536 bytes/block：

| Pool blocks | Alloc P50 | Alloc P99 | Free P50 | Free P99 |
|---:|---:|---:|---:|---:|
| 1024 | 140 ns | 151 ns | 130 ns | 131 ns |
| 4096 | 140 ns | 151 ns | 130 ns | 131 ns |
| 16384 | 140 ns | 151 ns | 130 ns | 131 ns |

Multi-sequence allocation：

| Metric | Value |
|---|---:|
| Concurrency | 64 sequences |
| Blocks per sequence | 4 |
| Total blocks | 256 |
| Alloc per block mean | 668.4 ns |
| Alloc per block P50 | 114 ns |
| Alloc per block P99 | 8947 ns |
| Free per block mean | 74.6 ns |
| Free per block P50 | 68 ns |
| Free per block P99 | 174 ns |

### 8.2 后续 alloc-only rerun

来源：`results/kcmm_bench_20260613_022104/kcmm_bench_alloc_throughput.log`

该 rerun 只覆盖 alloc throughput，不更新 integration 或 memory pressure 结论。

它的重要意义是 benchmark 标题和说明已经修正：

```text
Pool Allocator Metadata / Free-List Path Latency
(pool is pre-provisioned; this measures CPU bookkeeping, not CUDA physical allocation)
```

结果仍是同一数量级：

| Block bytes | Pool blocks | Alloc P50 | Alloc P99 | Free P50 | Free P99 |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 4096 | 130 ns | 141 ns | 130 ns | 140 ns |
| 65,536 | 4096 | 140 ns | 151 ns | 130 ns | 131 ns |
| 131,072 | 4096 | 140 ns | 151 ns | 130 ns | 131 ns |

因此这个 benchmark 应描述为 allocator metadata/free-list path latency，而不是
CUDA physical allocation latency。

## 9. cuMemMap 和 Step 3

### 9.1 Standalone cuMemMap latency

来源：`results/kcmm_bench_20260613_012500/kcmm_bench_cumemmap_latency.log`

| Operation | Size | Mean | P50 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `cuMemMap` | 2,097,152 B | 580.2 us | 601 us | 624 us | 779 us |
| `cuMemUnmap` | 2,097,152 B | 390.9 us | 416 us | 426 us | 427 us |

单独判断 map/unmap latency 时，应使用这个 benchmark。

### 9.2 Step 3 cuMemMap overhead

来源：`results/kcmm_bench_20260613_012500/step3_cumemmap_overhead.log`

该 log 中 map 和 unmap 数值完全相同：

| Size | Map | Unmap |
|---:|---:|---:|
| 2,097,152 B | 233.39 us | 233.39 us |

这是这个结果集中的已知 measurement/reporting 问题。当前工作区已经有修改让
Step 3 分开测 map/unmap，但还没有新的完整结果。因此，在新结果出来前，不应
用这个 Step 3 log 得出 map vs unmap 的独立结论。

同一个 Step 3 log 还给出：

| Metric | Value |
|---|---:|
| Layers | 22 |
| Maps per superblock | 44 |
| Avg per 2 MiB map/unmap | 231.11 us |
| Total for 22 layers | 10,168.90 us |

### 9.3 Step 3 Capacity-at-Workload

来源：`results/kcmm_bench_20260613_012500/step3_max_concurrent_requests.log`

| Metric | Value |
|---|---:|
| Model | tiny_llama |
| Block size | 16 tokens |
| Max seq len | 512 |
| Max new tokens | 64 |
| Prompt lens cycle | [8, 16, 32] |
| Capacity at workload | 1024 |
| Total blocks allocated | 5632 |
| Blocks in use | 5461 |
| Free blocks in pool | 171 |
| Superblocks allocated | 22 |
| Physical memory | 1936.00 MiB |
| Avg blocks/request | 5.33 |
| Total cuMemMap calls | 968 |
| Physical idle ratio after freeing | 1.0000 |

这个结果符合 `CONTEXT.md` 里的 Capacity-at-Workload 定义：它是给定 prompt
length 分布和 max_new_tokens 下的容量，不是 worst-case max sequence capacity。

## 10. Stale failed integration run

来源：

- `results/kcmm_engine_integration_20260613_021915/summary.txt`
- `results/kcmm_engine_integration_20260613_021915/kcmm_engine_integration_single.log`

这个后续 run 是编译失败，不是 benchmark 性能失败。错误来自中间状态里把
`IntegrationResult` 字段改为 `f64` 后，aggregate 仍按 `usize` 求和：

```text
error[E0277]: a value of type `usize` cannot be made by summing an iterator over elements of type `f64`
```

当前代码已通过 `cargo test --features kcmm --tests --no-run`，所以 `021915`
目录只应作为历史记录，不应算作当前 benchmark failure。

## 11. Issue 覆盖状态

| Issue group | 当前 results 能说明什么 |
|---|---|
| 01: `free_sequence` CpuResident safety | benchmark logs 不直接单独证明该行为；之前的单元测试覆盖该回归，当前完整 benchmark 没暴露相关失败。 |
| 02-05: integration counters、admission retry、deterministic weights、THRASH metric | integration single/sweep logs 已验证主要指标行为。 |
| 06: memory pressure metric names | `completion_ratio` 和 elapsed throughput 已分开报告，语义比旧结果清楚。 |
| 07: batch eviction amortization statistic | 当前 log 同时报告 P50 factor 和 mean factor，且结果修正了旧报告中的错误结论。 |
| 08: roundtrip integrity coverage | 64/64 payloads OK，覆盖 16 blocks x 2 layers x K+V。 |
| 09-17 | 当前工作区实现已完成并通过 targeted verification；仍缺一次新的完整 release benchmark 批次。 |
| 17 细节 | 根因是 `KcmmPool::ensure_capacity()` 没有限制 aligned physical ceiling，导致真实 over-allocation；修复后 single 的 `Peak GPU blocks` 为 766/768，ON 正好等于 ceiling，sweep 无 aligned-capacity warning。 |

## 12. 当前结论表

| 领域 | 结论 |
|---|---|
| Build/test health | 最新完整结果 15/15 passed；当前工作区 no-run compile 和 09-17 targeted verification 通过，但还需完整 rerun。 |
| Capacity-at-Workload | Tiering 在 engine 和 memory pressure 中都提高完成数。 |
| Throughput | 尚不能说整体提升。Engine single 基本持平；memory pressure 明显更差。 |
| THRASH | issue 17 修复后的 engine integration sweep 4 个配置全部是 THRASH；容量收益和高 eviction pressure 同时出现。 |
| Batch eviction | 不是明显退化；但也没有有效 amortization。 |
| Batch restore | batch=16 最好，batch=64 退化。 |
| Data integrity | all-layer K/V roundtrip integrity 通过。 |
| Physical capacity accounting | `Peak GPU blocks` 是 per-layer physical blocks in use；它可以超过逻辑 `max_blocks_total`，但修复后不会超过 superblock-aligned ceiling。 |
| cuMemMap | 最新完整批次中的 Step 3 split map/unmap 结果是 stale；当前 targeted run 已验证 Step 3 会分开输出 map/unmap/combined。 |

## 13. 建议下一步

1. 基于当前工作区重新跑完整 release benchmark：

   ```text
   scripts/run_kcmm_benches.sh
   ```

2. 新结果要重点对比 `results/kcmm_bench_20260613_012500` 和 targeted verification：

   - integration single/sweep 的 5-run statistical output；
   - issue 10 后 memory-pressure live CPU swap peak；
   - issue 11 后 Step 3 map/unmap 分列；
   - issue 16 后脚本默认 release profile 是否统一；
   - issue 17 后 integration 的 aligned physical ceiling diagnostics 和无超限 warning。

3. 如果新结果形态不变，下一步应进入 performance diagnosis：

   - 在 engine integration 里也加入 eviction phase timing；
   - 解释 single-block eviction 与 memory-pressure batch eviction 的巨大时间差；
   - 调整 admission/eviction policy，降低 `THRASH` config 的
     evictions/full-completion。

4. 优先用 `bs16_mb10_msl384_pl[64,128,256]_mnt128_reqs40_ari4` 做 stress case。
   它同时是收益最强和 eviction pressure 最明显的配置。
