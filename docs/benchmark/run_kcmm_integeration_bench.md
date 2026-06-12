# KCMM 引擎集成基准测试文档

> 对应脚本：`scripts/run_kcmm_integration_bench.sh`
>
> 测试源文件：`tests/kcmm_bench_engine_integration.rs`

---

## 概述

`run_kcmm_integration_bench.sh` 是 KCMM §1.6 引擎集成基准测试的运行脚本。该测试将 `NaiveTransformer` 模型与 `KcmmPool` 缓存后端结合，在模拟的连续批处理（continuous batching）工作负载下运行，对比 **Tiering OFF（基线，无分级存储）** 和 **Tiering ON（分级存储开启）** 两种配置的性能表现。两者使用完全相同的 `KcmmPool` 结构体，仅 `KcmmConfig.tiering` 开关不同，实现控制变量法比较。

### 运行方式

```bash
# 运行全部测试（debug 模式，默认）
./scripts/run_kcmm_integration_bench.sh

# Release 优化模式
./scripts/run_kcmm_integration_bench.sh --release

# 仅运行单配置测试（约 30-60 秒，使用 dummy 权重）
./scripts/run_kcmm_integration_bench.sh --single

# 仅运行参数扫描（约 2 分钟，使用 dummy 权重）
./scripts/run_kcmm_integration_bench.sh --sweep

# 按名称过滤
./scripts/run_kcmm_integration_bench.sh --filter single
```

### 输出

结果保存至 `results/kcmm_engine_integration_<timestamp>/` 目录，包含每个测试的独立日志文件及汇总文件。

### 前置条件

- 需要 NVIDIA GPU 及 CUDA 驱动（通过 `nvidia-smi` 检测）
- 需要 `--features kcmm` 编译特性

---

## 测试内容与方法

### 核心测量指标

| 指标 | 说明 |
|------|------|
| **吞吐量（tokens/sec）** | 每秒处理的 token 总数（prompt + decode） |
| **容量比率（capacity ratio）** | Tiering ON 完成的请求数 / Tiering OFF 完成的请求数 |
| **吞吐量比率（throughput ratio）** | Tiering ON 吞吐量 / Tiering OFF 吞吐量 |
| **每步延迟分布** | 每个 forward step 的 P50/P90/P95/P99 延迟（µs） |
| **每步延迟开销** | Tiering ON 相比 Tiering OFF 的 P50/P99 延迟增加百分比 |
| **逐出/恢复次数** | KCMM 分级存储触发的 GPU→CPU 逐出和 CPU→GPU 恢复操作计数 |
| **峰值并发数** | 运行期间同时活跃的最大序列数 |
| **峰值 GPU 块使用量** | KCMM 模式下物理 GPU 块的峰值占用数 |
| **平均批次大小** | 每个 step 的平均活跃序列数 |

### 成功标准

分级存储在相同 GPU 内存预算下，在内存压力场景中实现 ≥ 1.3× 的吞吐量/容量提升（相比 Tiering OFF 配置）。

---

## 工作负载设计

### 模型配置

使用 **零初始化 dummy 权重** 搭配压缩模型结构（无外部文件依赖）：

| 参数 | 值 |
|------|-----|
| 层数（num_hidden_layers） | 8 |
| KV 头数（num_key_value_heads） | 4 |
| 注意力头数（num_attention_heads）| 16 |
| 头维度（head_dim） | 64 |
| 隐藏维度（hidden_size） | 1024 |
| 词表大小（vocab_size） | 1000 |
| block 大小（16 tokens） | 64 KiB（16 × 4 × 64 × 2 × 8） |

### 模拟请求与连续批处理

测试通过 `SimRequest` 结构体模拟推理请求，每个请求包含：
- `prompt_tokens`：提示词 token 序列
- `target_len`：目标总长度（prompt + max_new_tokens）
- `seq_idx`：在缓存后端中的序列索引
- `block_indices`：已分配的物理块索引列表
- `position`：当前处理位置

工作负载分两个阶段执行：

#### 阶段一：预填充（Pre-fill）

持续接纳新请求，每个请求的提示长度从 `prompt_lens` 配置中循环选取。当达到 `max_batch` 的 80% 或 OOM 时停止。此阶段使 GPU 内存接近饱和。

#### 阶段二：动态批处理（Dynamic Batching）

模拟真实推理服务器的连续批处理行为，循环执行以下步骤：

1. **冷却周期**：每 8 步将约 1/4 活跃序列标记为"冷却"（cool），为逐出创建候选；其余活跃序列重新 touch 保持热度。Tiering OFF 时此操作无副作用（仅修改 `SequenceState`，无后续逐出）。
2. **序列增长**：遍历所有活跃序列，为需要扩展的序列分配新块。若分配失败：
   - **Tiering ON**：触发批量逐出（每次至少 8 个候选块），然后重试分配。若逐出前需检查块是否处于 `CpuResident` 状态，若是则批量恢复。
   - **Tiering OFF**：`evict_coldest_blocks()` 检测到 `tiering` 为 `None` 直接返回 `false`，序列被截断。
3. **前向推理**：调用 `NaiveTransformer::forward_step_paged()` 执行一步推理，测量本步耗时（µs）。更新序列位置（prefill 阶段消耗 prompt token，decode 阶段产生新 token）。
4. **新请求到达**：按 `arrival_interval` 间隔尝试接纳新请求，OOM 时拒绝。
5. **序列完成**：达到目标长度的序列释放所有块并注销。

---

## 测试配置

### 单配置测试（`kcmm_engine_integration_single`）

**约 30-60 秒**，使用零初始化 dummy 权重和压缩模型结构（8 层 × 1024 hidden）快速验证 KCMM tiering 行为。

| 参数 | 值 |
|------|-----|
| block_size_tokens | 16 |
| prompt_lens | [128, 256] |
| max_new_tokens | 384 |
| max_batch | 16 |
| max_seq_len | 640 |
| total_requests | 32 |
| arrival_interval | 12 |
| 模型 | 8×1024 dummy（零值权重），block_bytes=64 KiB |
| GPU 内存预算 | ~40 MiB（640 块 × 64 KiB/块） |

**输出详情**：
- Tiering OFF 与 Tiering ON 的完整指标对比表格（含完成数、总 token 数、decode token 数、耗时、吞吐量、峰值并发数）
- 每步延迟 P50/P90/P95/P99 对比
- 逐出/恢复次数、峰值 GPU 块使用量
- 吞吐量比率、容量比率分析
- 每步延迟开销百分比
- 分级存储活跃度评估
- 通过/失败/边缘判定及优化建议

### 参数扫描测试（`kcmm_engine_integration_sweep`）

**约 4 分钟**，遍历 4 种不同的工作负载配置，探索不同内存压力条件下的表现：

| 配置 | block_size | max_batch | max_seq_len | max_new_tokens | total_requests | arrival_interval | 特点 |
|------|-----------|-----------|-------------|----------------|----------------|------------------|------|
| 1 | 16 | 16 | 640 | 384 | 32 | 12 | 紧 VA，长解码，持续并发压力 |
| 2 | 16 | 12 | 512 | 256 | 36 | 8 | 更小 VA，更高周转率 |
| 3 | 32 | 16 | 512 | 256 | 32 | 12 | 更大块尺寸，不同压力剖面 |
| 4 | 16 | 10 | 384 | 128 | 40 | 4 | 极紧 VA，高周转率，三种提示长度 |

**输出**：表格形式汇总全部配置的吞吐量比率、容量比率、逐出/恢复次数，标注通过/边缘/失败状态，输出最佳吞吐量比率及其对应配置。

---

## 关键实现细节

### 分级存储交互

Tiering ON 模式下的逐出和恢复流程（Tiering OFF 时 `TieringEngine` 为 `None`，所有相关操作退化为无操作）：

1. **逐出触发**：`alloc_block()` 返回 OOM 时触发。调用 `evict_coldest_blocks()` 从冷却序列中选择至少 8 个候选块，通过 `TieringEngine::evict_blocks()` 执行批量 GPU→CPU 数据传输。
2. **恢复触发**：在序列增长阶段，每次分配新块后检查序列的已有块是否处于 `CpuResident` 状态（即已被逐出到 CPU），若是则调用 `restore_evicted_blocks()` 进行批量 CPU→GPU 恢复。
3. **LRU 策略**：通过 `touch()` / `cool()` 调用维护访问热度，冷却的序列优先被选为逐出候选。

### 与 run_kcmm_benches.sh 的关系

| 维度 | `run_kcmm_benches.sh` | `run_kcmm_integration_bench.sh` |
|------|----------------------|--------------------------------|
| 测试粒度 | 微基准（单操作级别） | 集成基准（端到端推理） |
| 测试对象 | KcmmPool 底层 API | NaiveTransformer + KcmmPool (tiering ON vs OFF) |
| 是否执行推理 | 否 | 是（调用 forward_step_paged） |
| 典型运行时间 | 较长（多项测试） | 较短（2 项测试，共约 6 分钟） |
| 评估维度 | 分配/逐出/恢复延迟、cuMemMap 开销、流干扰 | 吞吐量、容量收益、每步延迟开销 |

两者互补：`run_kcmm_benches.sh` 验证 KCMM 各组件的微观性能，`run_kcmm_integration_bench.sh` 验证 KCMM 在真实推理流水线中的端到端收益。
