# KCMM 交换策略基准测试文档

> 对应脚本：`scripts/run_kcmm_benches.sh`
>
> 测试源文件：`tests/kcmm_bench_alloc.rs`、`tests/kcmm_bench_tiering.rs`、`tests/step3_benchmarks.rs`、`tests/kcmm_bench_memory_pressure.rs`

---

## 概述

`run_kcmm_benches.sh` 是 KCMM（Kernel-managed Cache Memory Management）交换策略的综合性基准测试套件。它涵盖了从底层分配/释放吞吐量、逐出/恢复延迟、CUDA 流干扰，到顶层内存压力下容量收益的全方位性能评估。

### 运行方式

```bash
# 运行所有基准测试（release 模式）
./scripts/run_kcmm_benches.sh

# Debug 模式（编译更快，运行更慢）
./scripts/run_kcmm_benches.sh --debug

# 按名称过滤运行特定测试
./scripts/run_kcmm_benches.sh --filter alloc
./scripts/run_kcmm_benches.sh --filter tiering
./scripts/run_kcmm_benches.sh --filter batch_eviction
```

### 输出

结果保存至 `results/kcmm_bench_<timestamp>/` 目录，包含每个测试的独立日志文件（`<test_name>.log`）以及汇总文件（`summary.txt`）。

### 前置条件

- 需要 NVIDIA GPU 及 CUDA 驱动（通过 `nvidia-smi` 检测）
- 需要 `--features kcmm` 编译特性

---

## 测试内容与方法

### 基准测试 1：块分配/释放吞吐量

**测试源文件**：`tests/kcmm_bench_alloc.rs`

**目的**：测量 KCMM 池（KcmmPool）在单块分配和释放操作上的原始延迟，验证 KCMM 的分配/释放吞吐量相比基线没有显著退化。

**成功标准**：KCMM 分配/释放吞吐量退化 < 5%（相比 vLLM 等价的 PagedKvCache 基线）。

#### 1a — 块大小扫描（`kcmm_bench_alloc_throughput`）

- **模型参数**：2 层、4 个 KV 头、头维度 64（模拟 TinyLlama 小规模配置）
- **方法**：对不同块大小（64、128、256 tokens）分别执行 500 次 `alloc_sequence(1)` → `free_sequence` 循环。每次操作前进行 100 次预热以稳定 CUDA 驱动缓存。
- **测量指标**：分配和释放的 P50、P99 延迟（纳秒）
- **健全性检查**：P50 延迟应 < 1 ms

#### 1b — 池容量扫描（`kcmm_bench_alloc_pool_size_sweep`）

- **方法**：固定块大小（128 tokens，64 KiB/块），在三种池容量（1024、4096、16384 块）下分别执行 300 次分配/释放循环。
- **目的**：评估池规模对分配器性能的影响（例如内部数据结构查找开销是否随容量增长）。

#### 1c — 多序列并发分配（`kcmm_bench_alloc_concurrent_sequences`）

- **方法**：模拟多用户并发场景：64 个并发序列，每个序列分配 4 个块（共 256 块），运行 16 轮。测量每轮的总耗时，计算每块平均分配/释放时间。
- **健全性检查**：每块分配时间 < 200 µs（含 N 层 K+V 物理分配及 cuMemMap 开销）。

---

### 基准测试 2：逐出/恢复延迟

**测试源文件**：`tests/kcmm_bench_tiering.rs`

**目的**：测量块粒度的 GPU↔CPU 数据迁移端到端延迟，包括已知瓶颈 cuMemMap/cuMemUnmap 的开销。

**成功标准**：
- 单块恢复 P50 < 1000 µs（实际约束，因 WSL2 抖动放宽了原始 200 µs 目标）
- 批量逐出展现摊销效益（每块延迟随批量增大而降低）

#### 2a — 单块逐出/恢复（`kcmm_bench_single_block_evict_restore`）

- **方法**：对三种块大小（64、128、256 tokens）各执行 256 次逐出→恢复循环。每次迭代：分配新块 → 测量逐出耗时 → 测量恢复耗时。
- **启用了分级存储**（tiering=true），CPU 缓存使用临时文件支持。
- **预热**：使用单个块进行 8 次预热迭代。
- **输出**：每种块大小的逐出和恢复 P50/P99/P999 延迟（µs）。

#### 2b — 批量逐出摊销（`kcmm_bench_batch_eviction_amortization`）

- **方法**：固定块大小 128 tokens，对批量大小 [1, 4, 16, 64] 各运行 30 轮。每轮分配 batch_size 个块，测量逐出总耗时，计算每块平均延迟。
- **摊销因子计算**：`baseline（batch=1）/ batch=N 的每块平均延迟`，> 1.0 表示批量化带来提升。

#### 2c — cuMemMap/cuMemUnmap 独立延迟（`kcmm_bench_cumemmap_latency`）

- **方法**：使用 `CudaVmm` 直接测量 CUDA 虚拟内存映射/解除映射的原始延迟。对不同映射大小（64 KiB 到 2 MiB）各执行 128 次映射→解除映射循环。
- **输出**：每种大小下的 map/unmap P50/P99/P999 延迟（µs）。
- **目的**：隔离 cuMemMap 开销，作为逐出延迟的基线参考。

#### 2d — 逐出→恢复数据完整性（`kcmm_bench_tiering_roundtrip_data_integrity`）

- **方法**：分配 16 个块，每块写入唯一数据模式（基于块索引的 XOR 模式），然后逐出所有块，再逐个恢复，最后读回 GPU 数据与原始模式比对。
- **目的**：验证 GPU→CPU→GPU 往返过程中数据完整性不丢失。
- **成功标准**：16/16 块数据完整性校验通过。

#### 2e — 批量恢复摊销（`kcmm_bench_batch_restore_amortization`）

- **方法**：与 2b 对称，但测量的是恢复方向的批量摊销效果。对批量大小 [1, 4, 16, 64] 各运行 30 轮。每轮：分配 → 逐出所有块 → 测量恢复总耗时 → 计算每块平均延迟。
- **输出**：每种批量大小的恢复 P50/P99/P999 延迟及摊销因子。

---

### 基准测试 3：CUDA 流干扰

**测试源文件**：`tests/kcmm_bench_tiering.rs`（`kcmm_bench_stream_interference`）

**目的**：验证 KCMM 专用逐出流（CU_STREAM_NON_BLOCKING）不会显著干扰默认流上的推理计算。

**方法**：
1. **基线阶段**：在默认流上执行 128 次 32 MiB H2D memcpy（模拟推理工作负载），测量每次的完成延迟。
2. **干扰阶段**：在默认流执行 H2D 的同时，在 KCMM 逐出流上并发执行 32 MiB D2H memcpy（模拟后台逐出活动占用 PCIe 带宽）。同样执行 128 次迭代。
3. **对比**：计算基线与干扰场景下 H2D 延迟的 P50/P99 百分比差异。

**成功标准**：推理内核干扰 < 25%（考虑到 WSL2 / 笔记本 GPU 的准虚拟化抖动；裸金属目标 < 1%）。

---

### 基准测试 4：每层 cuMemMap/cuMemUnmap 开销

**测试源文件**：`tests/step3_benchmarks.rs`（`step3_cumemmap_overhead`）

**目的**：测量在完整 TinyLlama 模型（22 层）的 K+V 映射场景下，cuMemMap/cuMemUnmap 的累积开销。

**方法**：
1. **按大小扫描**：对 8 KiB 到 2 MiB 的各种映射大小，对每层 K 和 V 分别执行 map+unmap（共 22 层 × 2（K+V）× 2（map+unmap）= 88 次操作），运行 64 轮，计算每次操作的平均耗时。
2. **全超级块映射**：使用 2 MiB 完整超级块大小进行 64 轮测试，输出每层平均 map/unmap 耗时和全部 22 层的总耗时。

---

### 基准测试 5：内存压力 — 分级存储容量收益

**测试源文件**：`tests/kcmm_bench_memory_pressure.rs`

**目的**：通过对比分级存储开启（KCMM）和关闭（基线 PagedKvCache）时能同时活跃的序列数量，量化 KCMM 在内存压力下的容量提升。

**成功标准**：KCMM 分级存储相比基线支持 ≥ 1.3× 并发序列（相同 GPU 内存预算）。

**工作负载模型**：
- **预填充阶段**：持续接纳序列直到池容量达到 ~80% 饱和度。
- **动态阶段**：模拟连续批处理（continuous batching）：
  - 活跃序列逐 token 增长（decode），需要时分配新块。
  - 序列达到目标长度后完成并释放资源。
  - 新序列按 `arrival_interval` 间隔到达。
  - KCMM 模式下周期性地将部分序列标记为"冷却"（cool），创建逐出候选。
  - 当分配失败（OOM）时，KCMM 触发批量逐出（每次至少 8 个候选块），释放 GPU 内存后重试。

**测量指标**：
- 完成的序列数、被截断（capped）序列数、被拒绝（rejected）序列数
- 峰值并发序列数、峰值物理块使用量
- KCMM 特有：逐出次数、CPU 交换峰值字节数
- 吞吐量比率 = KCMM 完成数 / 基线完成数

#### 5a — 单配置测试（`kcmm_bench_memory_pressure_single`）

- **配置**：block_size=16, max_batch=16, max_seq_len=640, max_new_tokens=384, 32 个请求, arrival_interval=12
- **GPU 预算**：~110 MiB（640 块 × 176 KiB/块）
- **输出**：基线和 KCMM 的详细结果对比，含通过/失败/边缘判定。

#### 5b — 参数扫描（`kcmm_bench_memory_pressure_sweep`）

- **4 种配置**，变化维度：块大小（16/32）、最大批次（10/12/16）、最大序列长度（384/512/640）、提示分布（2-3 种长度）、总到达数（32-40）、到达间隔（4-12）。
- **输出**：表格形式的全部配置对比，含最佳吞吐量比率。

---

### 基准测试 6：最大并发请求数

**测试源文件**：`tests/step3_benchmarks.rs`（`step3_max_concurrent_requests`）

**目的**：测量 PagedKvCache 在工作负载下的容量上限（不涉及 KCMM 分级存储）。

**方法**：
1. **接纳阶段**：使用 TinyLlama 配置（block_size=16, max_batch=1024, max_seq_len=512），循环使用短提示长度 [8, 16, 32] 持续接纳序列，直到 OOM。
2. **解码增长阶段**：每个已接纳的序列最多增长 64 个 token，按需分配新块，OOM 时截断。
3. **指标**：接纳序列数、总分配块数、超级块分配数、物理内存使用量、cuMemMap 调用总数。

---

## 测试框架结构

| 基准测试 | 测试文件 | 测试函数 |
|---------|---------|---------|
| 1a 块大小扫描 | `kcmm_bench_alloc` | `kcmm_bench_alloc_throughput` |
| 1b 池容量扫描 | `kcmm_bench_alloc` | `kcmm_bench_alloc_pool_size_sweep` |
| 1c 并发分配 | `kcmm_bench_alloc` | `kcmm_bench_alloc_concurrent_sequences` |
| 2a 单块逐出/恢复 | `kcmm_bench_tiering` | `kcmm_bench_single_block_evict_restore` |
| 2b 批量逐出摊销 | `kcmm_bench_tiering` | `kcmm_bench_batch_eviction_amortization` |
| 2c cuMemMap 延迟 | `kcmm_bench_tiering` | `kcmm_bench_cumemmap_latency` |
| 2d 往返完整性 | `kcmm_bench_tiering` | `kcmm_bench_tiering_roundtrip_data_integrity` |
| 2e 批量恢复摊销 | `kcmm_bench_tiering` | `kcmm_bench_batch_restore_amortization` |
| 3 流干扰 | `kcmm_bench_tiering` | `kcmm_bench_stream_interference` |
| 4 cuMemMap 每层开销 | `step3_benchmarks` | `step3_cumemmap_overhead` |
| 5a 内存压力单配置 | `kcmm_bench_memory_pressure` | `kcmm_bench_memory_pressure_single` |
| 5b 内存压力扫描 | `kcmm_bench_memory_pressure` | `kcmm_bench_memory_pressure_sweep` |
| 6 最大并发请求 | `step3_benchmarks` | `step3_max_concurrent_requests` |

---

## 输出示例说明

每个测试的标准输出包含：
- 分隔线和测试名称
- 表格形式的延迟统计（P50/P99/P999）
- 健全性断言结果（PASSED/FAILED）
- 失败时自动打印最后 20 行日志

最终汇总文件包含：日期、GPU 型号、VRAM、编译模式、全部测试通过/失败统计以及各测试日志文件名。
