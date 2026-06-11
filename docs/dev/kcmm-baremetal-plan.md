# KCMM Bare-Metal 验证与优化计划

**Date:** 2026-06-11
**Status:** Living document
**Target Hardware:** d7525 — AMD EPYC 7302 (32 core), 128GB RAM, NVIDIA A30 24GB, NVMe 1.6TB, Mellanox ConnectX-6 DX 100Gb
**Related:** [`kcmm-ffi-roadmap.md`](kcmm-ffi-roadmap.md), [`docs/benchmark/run_kcmm_benches.md`](../benchmark/run_kcmm_benches.md), [`docs/benchmark/run_kcmm_integeration_bench.md`](../benchmark/run_kcmm_integeration_bench.md)

---

## 1. 背景与目标

### 1.1 为什么 Bare-Metal 验证至关重要

WSL2 的 GPU 准虚拟化（GPU-PV）层引入显著性能失真：

| 操作 | WSL2 典型延迟 | Bare-Metal 典型延迟 | 失真倍数 |
|------|-------------|-------------------|---------|
| `cuMemAlloc_v2` P99 | ~12.96 ms | ~10-30 µs | **400-1000×** |
| `cuMemMap` (2 MiB) | 数百 µs | ~165 µs | 2-5× |
| 批量逐出 per-block | 被 PV 层抖动主导 | 真实的摊销曲线 | 无法比较 |

当前所有 Benchmark 数据（Benchmark 1-6）均在 WSL2 上采集，只能得出**相对结论**（KCMM on vs off 的比率），无法提供论文所需的**绝对性能数据**。

### 1.2 Bare-Metal 阶段目标

1. **获取可发表的绝对性能数据**：所有 Benchmark 在 A30 上的 bare-metal 延迟绝对值
2. **验证批量优化收益**：gather/scatter kernel、批量 D2H/H2D 在真实硬件上的摊销曲线
3. **换出策略全面对比**：LRU/LFU/FIFO 在不同访问模式下的命中率与延迟
4. **高级换出策略实现与评估**：分层温控、自适应水位线、Hint API 等 P0 优化的落地
5. **完善文档与可观测性**：bpftrace 追踪、API 指南、时间序列指标、策略配置文档
6. **Benchmark 增强**：UFS 对比、per-sequence latency、真实 trace 回放、LLaMA-7B 规模测试

---

## 2. 阶段一：基准测试完整复现（预计 2-3 天）

### 2.1 环境准备

```bash
# 1. 确认 GPU 状态
nvidia-smi
# 预期: NVIDIA A30, 24258 MiB

# 2. 确认 NVMe 可用
lsblk -d -o name,size,rota /dev/nvme*
# 预期: /dev/nvme0n1 1.6T

# 3. 编译 KCMM 完整测试套件（release 模式）
cd /path/to/llm-rust-ebpf
cargo test --features kcmm --release --no-run

# 4. 确认所有测试二进制可执行
ls -lh target/release/deps/kcmm_bench_* step3_benchmarks*
```

### 2.2 运行 `run_kcmm_benches.sh`（微基准测试套件）

```bash
# 完整运行（预计 20-40 分钟）
./scripts/run_kcmm_benches.sh --release

# 结果目录: results/kcmm_bench_<timestamp>/
```

#### 2.2.1 重点关注指标

| Benchmark | 关键指标 | WSL2 参考值 | Bare-Metal 预期 |
|-----------|---------|------------|----------------|
| **1a** 分配吞吐量 | alloc/free P50 (ns) | 受 PV 层放大 | < 10 µs |
| **1b** 池容量扫描 | 1024/4096/16384 块的延迟曲线 | — | 近常数 O(1) |
| **1c** 并发分配 | 每块平均分配/释放时间 | — | < 10 µs/block |
| **2a** 单块逐出/恢复 | restore P50 (µs) | < 1000 µs（已放宽） | **< 200 µs**（原始目标） |
| **2b** 批量逐出摊销 | batch=64 的摊销因子 | — | > 4× |
| **2c** cuMemMap 独立延迟 | 2 MiB map P50 (µs) | 数百 µs | **~165 µs** |
| **2d** 往返完整性 | 16/16 校验通过 | ✅ 通过 | ✅ 应继续通过 |
| **2e** 批量恢复摊销 | batch=64 的摊销因子 | — | > 4× |
| **3** 流干扰 | P50 overhead | < 25%（放宽） | **< 1%**（原始目标） |
| **4** 每层 cuMemMap 开销 | 22 层总延迟 (µs) | — | 论文关键数据 |
| **5a** 内存压力单配置 | throughput_ratio | ≥ 1.0× | ≥ 1.3× |
| **5b** 内存压力扫描 | 最佳 capacity_ratio | — | ≥ 1.3× |
| **6** 最大并发请求 | 接纳序列数 | — | 容量上限参考 |

#### 2.2.2 数据采集脚本

```bash
#!/bin/bash
# 提取所有 benchmark 的关键数据，生成 CSV 供论文使用
for log in results/kcmm_bench_*/kcmm_bench_*.log; do
    test_name=$(basename "$log" .log)
    echo "=== $test_name ==="
    # 提取 P50/P99 数据
    grep -E '(p50|p99|P50|P99|throughput_ratio|capacity_ratio)' "$log"
    echo ""
done
```

### 2.3 运行 `run_kcmm_integration_bench.sh`（引擎集成测试）

```bash
# 完整运行（单配置 + 参数扫描，预计 15-30 分钟）
./scripts/run_kcmm_integration_bench.sh --release

# 结果目录: results/kcmm_engine_integration_<timestamp>/
```

#### 2.3.1 重点关注指标

| 指标 | WSL2 参考值 | Bare-Metal 预期 |
|------|------------|----------------|
| Throughput ratio (KCMM/Baseline) | ≥ 1.0× | ≥ 1.3× |
| Capacity ratio (KCMM/Baseline) | ≥ 1.0× | ≥ 1.3× |
| Per-step P50 overhead | 可能为负（tiering 开销 < PV 抖动） | 应有小额正值（真实 tiering 开销） |
| Per-step P99 overhead | PV 主导 | 真实尾延迟特征 |
| Eviction count | — | 活跃逐出次数 |
| Avg batch size | — | 负载饱和度指标 |

#### 2.3.2 多次 Trial 采集

Bare-metal 上抖动远小于 WSL2，但为论文统计显著性仍需多次运行：

```bash
# 每个配置至少运行 5 次
for i in $(seq 1 5); do
    echo "=== Trial $i ==="
    ./scripts/run_kcmm_integration_bench.sh --release --single
    mv results/kcmm_engine_integration_* results/trial_${i}/
done

# 聚合分析
python3 scripts/aggregate_bench_results.py results/trial_*/
```

---

## 3. 阶段二：换出策略对比测试（预计 2-3 天）

### 3.1 现有策略回顾

KCMM 当前已实现三种可插拔策略（`src/kcmm/tiering.rs`）：

| 策略 | 决策依据 | 状态 | 适用场景 |
|------|---------|------|---------|
| **LRU** | `last_access` 时间戳，最久未访问优先 | ✅ 已实现，默认策略 | 通用场景，时间局部性好 |
| **LFU** | 访问频率计数器，访问次数最少优先 | ✅ 已实现 | 热点数据保护 |
| **FIFO** | 分配时间戳，最早分配优先 | ✅ 已实现 | 简单、低开销 |

### 3.2 新增：换出策略对比 Benchmark

当前测试套件中所有 benchmark 均默认使用 LRU（`eviction_policy: "lru"`），没有跨策略对比。需要新增 **Benchmark 7：换出策略对比**。

#### 3.2.1 测试文件：`tests/kcmm_bench_policy_comparison.rs`

```rust
// 测试框架设计：
//
// 1. 合成访问模式生成器
//    - Uniform: 均匀随机访问，无局部性
//    - Zipf (α=0.8, 1.0, 1.2): 幂律分布，模拟真实热点
//    - Hot-Cold交替: 80% 请求命中 20% 块，周期性切换热点
//    - Sequential scan: 顺序扫描，FIFO 最优
//    - Multi-turn conversation: 模拟多轮对话的间歇访问
//
// 2. 每种模式 × 每种策略 × 多次 trial → 命中率矩阵
//    - 测量：命中率（hit rate）、逐出次数、平均逐出延迟
//    - 固定内存预算（如 GPU pool 128 blocks），请求数 10× pool 容量
//
// 3. Oracle 基线
//    - Belady's MIN 算法（理论最优，需要已知未来访问序列）
//    - 提供命中率上界作为参考
```

#### 3.2.2 测量指标

| 指标 | 说明 |
|------|------|
| **命中率（Hit Rate）** | 请求的 block 已在 GPU 中的比例 |
| **逐出次数** | 触发的总逐出操作数 |
| **逐出延迟分布** | 每次逐出的 P50/P99 耗时 |
| **策略开销** | 策略自身计算开销（`select_victims` 耗时） |
| **容量利用比** | 实际并发数 / 理论最大并发数 |

#### 3.2.3 合成访问模式详解

```
1. Uniform Random
   └─ 每个 block 被访问概率相等
   └─ 预期：三种策略命中率接近（无法利用局部性）

2. Zipf Distribution (α=1.0)
   └─ 80% 请求命中 20% 块
   └─ 预期：LRU ≈ LFU > FIFO（LRU/LFU 能保留热点）

3. Hot-Cold Phase Change
   └─ Phase A (0-5000 reqs): 热点 = blocks [0, 20)
   └─ Phase B (5000-10000 reqs): 热点切换为 blocks [80, 100)
   └─ 预期：LRU 自适应最快，FIFO 无法适应

4. Sequential Scan
   └─ 请求顺序: 0,1,2,...,127,0,1,2,...
   └─ 池容量 < 128 时任何策略都命中率为 0
   └─ 池容量 ≥ 128 时 FIFO 最优（完美预知顺序）

5. Multi-Turn Conversation
   └─ Turn 1: 快速分配 32 blocks → 30s 静默 → Turn 2: 重新访问同样 32 blocks
   └─ 预期：LRU 在静默期逐出全部（命中率 0），需要 Hint API 优化
```

#### 3.2.4 运行方式

```bash
# 新增脚本: scripts/run_kcmm_policy_bench.sh
./scripts/run_kcmm_policy_bench.sh --release

# 选项
./scripts/run_kcmm_policy_bench.sh --release --filter lru   # 仅 LRU
./scripts/run_kcmm_policy_bench.sh --release --filter zipf  # 仅 Zipf 分布
```

---

## 4. 阶段三：高级换出策略实现与测试（预计 2-3 周）

基于 `kcmm-implementation-analysis.md` §1.6 的分析，按优先级分三批实现。

### 4.1 P0 第一批：分层温控 + 自适应水位线（预计 1 周）

#### 4.1.1 分层温控（Temperature-Tiered Eviction）

**改动范围**：`src/kcmm/tiering.rs` — 新增 `TemperatureTieredPolicy`

```
三层温度模型：
┌────────────────────────────────────────────┐
│ HOT 层  — 序列尾部 4 block                 │  ← 永不换出（正在被 attention 使用）
│ WARM 层 — 中间 block，正常 LRU 管理         │  ← 标准 LRU 选择受害者
│ COLD 层 — 已完成序列的 block，显式标记      │  ← 首选受害者，优先换出
└────────────────────────────────────────────┘
```

**实现要点**：
- 扩展 `EvictionPolicy` trait 添加 `fn set_temperature(&self, block: BlockHandle, temp: Temperature)`
- `Temperature::Hot`：`select_victims` 中跳过（即使用户标记为冷也不逐出）
- `Temperature::Cold`：始终排在 `select_victims` 结果的最前面（即使 LRU 时间戳较新）
- `Temperature::Warm`：正常 LRU/LFU/FIFO 排序
- 通过 `KcmmPool::cool(seq_idx)` 自动将完成序列的全部 block 标记为 COLD

**测试计划**：
- 单元测试：验证三种温度的 victim 选择顺序
- 集成测试：在 `memory_pressure` benchmark 中对比温控开启 vs 关闭的逐出效率
- 指标：COLD block 是否确实被优先逐出，HOT block 是否被保护

#### 4.1.2 自适应水位线（Adaptive Watermark）

**改动范围**：`src/kcmm/pool.rs` — 新增 `AdaptiveWatermark` 结构体

```
核心算法：
  alloc_rate_ewma  = α × current_alloc_rate + (1-α) × alloc_rate_ewma
  free_rate_ewma   = α × current_free_rate  + (1-α) × free_rate_ewma
  net_drain_rate   = alloc_rate_ewma - free_rate_ewma

  if net_drain_rate > 0:
      time_to_oom = free_blocks / net_drain_rate
      if time_to_oom < avg_eviction_latency × 2.0:
          trigger_preemptive_eviction()  // 提前触发换出
```

**实现要点**：
- EWMA 系数 α = 0.2（每 100ms 采样一次）
- 追踪 `avg_eviction_latency` 的 EWMA（从 `TieringEngine` 的每次 evict 耗时获取）
- 在 `alloc_block()` 路径中插桩：每次分配后检查水位线
- 提前换出量 = `(time_to_oom / avg_eviction_latency) × net_drain_rate` 个 block

**测试计划**：
- 模拟突发负载：快速分配 100 blocks → 观察是否在 OOM 前触发换出
- 对比固定水位线 vs 自适应水位线的 OOM 发生次数
- 压力测试：极端分配速率下的稳定性

### 4.2 P1 第二批：Hint API + 写入缓冲区 + 策略对比增强（预计 1 周）

#### 4.2.1 Hint API 基础框架

**改动范围**：`src/kcmm/ffi.rs`（类型已定义）+ `src/kcmm/tiering.rs`（策略层）

**已定义的 FFI 类型**（`ffi.rs:38`）：
```c
kcmm_hint_t {
    KCMM_HINT_MULTI_TURN,     // 多轮对话，延迟换出
    KCMM_HINT_NEAR_END,       // 即将结束，优先受害者
    KCMM_HINT_SYSTEM_PROMPT,  // 系统提示词，高缓存价值
    KCMM_HINT_HIGH_PRIORITY,  // SLO 关键请求
    KCMM_HINT_LOW_PRIORITY,   // 后台批处理
    KCMM_HINT_ATTENTION_SINK, // 注意力沉没 token
    KCMM_HINT_HEAVY_HITTER,   // 高注意力 token
    KCMM_HINT_EVICTABLE,      // 可丢弃
}
```

**实现任务**：
1. Rust trait `HintProvider`：`fn apply_hint(&self, seq_idx: usize, hint: Hint)`
2. `LruPolicy` 扩展：HINT_MULTI_TURN 给 `last_access` 加偏移（相当于假造"最近访问过"）
3. `TemperatureTieredPolicy` 扩展：HINT_NEAR_END 标记整个序列为 COLD
4. C FFI 函数体实现（当前为注释骨架）：
   - `kcmm_hint(pool, seq_id, hint_type)` — 应用 hint
   - `kcmm_protect(pool, seq_id, block_ids, level)` — 设置保护级别

**测试计划**：
- 多轮对话模拟：Turn1 → HINT_MULTI_TURN → 30s 静默 → Turn2 检查 block 是否仍在 GPU
- C 测试程序：通过 `libkcmm.so` 调用 FFI 验证 ABI

#### 4.2.2 写入缓冲区（Write Buffer）

**改动范围**：`src/kcmm/tiering.rs` — 新增 `WriteBuffer` 结构体

```
设计：
  WriteBuffer {
      pending: VecDeque<(BlockHandle, CpuOffset)>,  // 待持久化队列
      total_bytes: usize,                             // 累计待写入字节
      flush_threshold: usize,                         // 如 2 MiB 或 16 blocks
  }

  触发 flush 条件：
  - pending.len() >= 16 blocks
  - total_bytes >= 2 MiB
  - 显式 flush 调用（pool OOM 时）
  - 定时器超时（100ms 无新 block 加入）
```

**收益分析**：
- NVMe 场景：顺序写 vs 随机写吞吐量差异 ≥ 10×
- PCIe 场景：批量 D2H 减少 DMA 引擎 setup/teardown 次数
- 延迟换出（HINT_MULTI_TURN）期间 block 留在 write buffer 中，可以零成本取消

**测试计划**：
- 对比 write buffer 开启 vs 关闭的 NVMe 写入吞吐量
- 测量 write buffer flush 对推理延迟的干扰（应在专用 stream 上）
- 验证 write buffer 中 block 的取消机制（`cancel_pending(block)`）

#### 4.2.3 换出策略对比 Benchmark 完善

在阶段二的基础上，增加以下测试：

| 测试 | 说明 |
|------|------|
| **真实 trace 回放** | 录制真实推理负载的 block 访问 trace，回放对比策略 |
| **策略稳定性** | 长时间运行（≥ 100K 请求）的策略行为一致性 |
| **多策略混合** | 同一池中 HOT 层用 FIFO、WARM 层用 LRU 的混合策略 |
| **Oracle 对比** | Belady 最优算法作为理论上界 |

### 4.3 P2 第三批：有损换出 + 流式恢复 + 跨引擎优化（预计 1 周）

#### 4.3.1 有损换出（Quantized Eviction）

**实现任务**：
1. 在 evict CUDA stream 上插入量化 kernel（FP16 → INT8/INT4）
2. 在 restore CUDA stream 上插入反量化 kernel
3. 量化精度选择策略：COLD block → INT4，WARM block → INT8，HOT block → 无损（FP16）
4. 精度验证：量化-反量化往返后的 KV Cache 余弦相似度

**测试计划**：
- 精度损失测量：使用 TinyLlama + 标准 prompt 对比量化/无损换出的 token 序列一致性
- 带宽收益：量化后的实际 D2H/H2D 带宽对比

#### 4.3.2 流式恢复（Streaming Restore）

**实现任务**：
1. 将单次大块 H2D 拆分为 4 个流水线阶段
2. 第一批 25% 数据到位后立即返回给引擎（不等全部完成）
3. 剩余 75% 在后台继续传输
4. 引擎侧配合：paged attention kernel 只访问已到位的 KV Cache 区域

**测试计划**：
- 长上下文场景（block_size=256, 128 KiB/block）的 latency hiding 效果
- 对比标准恢复 vs 流式恢复的首 token 延迟（TTFT）

#### 4.3.3 全局压力平衡（多引擎场景）

**适用条件**：系统中有多个推理引擎实例共享 KCMM

**实现任务**：
1. `TieringEngine` 支持多个 `KcmmPool` 注册
2. 全局 victim 优先级队列（跨池排序）
3. 跨池物理块借用（Block Lending）：
   - 池 A 空闲块临时转移给池 B
   - 比换出到 CPU 快 ~3 个数量级（纯映射表更新 vs D2H + H2D）

**测试计划**：
- 模拟双引擎场景：引擎 A 80% 利用率（冷块为主）、引擎 B 95% 利用率（热块为主）
- 验证 KCMM 是否优先从引擎 A 换出
- 块借用延迟对比（ns 级映射更新 vs µs 级 D2H）

---

## 5. 阶段四：文档与可观测性（可并行于阶段二/三）

**目标：** 完善开发和用户文档，准备论文材料。大部分工作可在 WSL2 上预先完成，仅 bpftrace 脚本需在 bare-metal 上验证。

### 5.1 bpftrace 追踪脚本（`src/trace/kcmm_events.bt`）

- 追踪 USDT probe：evict 开始/完成、restore 开始/完成、prefetch 事件
- 用于生成论文中的时间线图（eviction pipeline 各阶段耗时瀑布图）
- **Bare-metal 专属**：bpftrace 需要 Linux host 内核支持，WSL2 不可用

```
预期事件流（单次 eviction batch）：
  evict:begin → evict:submit_d2h → evict:sync → evict:finalize → evict:end
  每个 probe 附带：pool_id, batch_size, block_count, timestamp_ns
```

### 5.2 API 使用指南

- 面向引擎集成者的 `kcmm.h` 使用文档（`docs/dev/kcmm-api-guide.md`）
- 策略配置指南：何时选择 LRU vs LFU vs FIFO（`docs/dev/kcmm-policy-guide.md`）
  - 基于阶段二的策略命中率矩阵给出推荐
- C API 示例程序（`examples/kcmm_c_integration.c`）

### 5.3 时间序列指标采样（G3）

**改动范围**：`src/kcmm/metrics.rs`

- `KcmmMetrics` 目前只有快照（`from_ufs` / `to_ufs_summary`）
- 新增环形缓冲区（如保留最近 1000 个采样点，每秒一次）：
  - `free_blocks_history: CircularBuffer<u32>`
  - `eviction_rate_history: CircularBuffer<f64>`
  - `alloc_latency_history: CircularBuffer<Duration>`
- 导出为 CSV/JSON 供离线分析（Python 脚本 `scripts/plot_metrics.py` 生成论文图表）

### 5.4 开发文档更新

- 补充周实现文档（`docs/dev/kcmm-week14-impl.md` 等）
- 更新 `docs/task/kcmm-implementation-analysis.md` 标记各优化项的实际完成状态
- 编写 `docs/dev/kcmm-advanced-strategies.md`：高级策略的实现总结与设计决策记录

---

## 6. 阶段五：Benchmark 完善（可并行于阶段二/三/四）

**目标：** 产出完整的、可发表的 benchmark 数据。WSL2 部分侧重相对结论，bare-metal 部分产出绝对值。

### 6.1 Benchmark 6：UFS 指标对比

- **内容**：KCMM vs PagedKvCache 在**无分层模式**（tiering off）下的 UFS 指标对比
  - IFR（Internal Fragmentation Rate）、PME（Physical Memory Efficiency）、BU（Block Utilization）、RFI（Reserved-but-Free Index）
- **环境**：WSL2 + TinyLlama 小规模推理
- **目的**：验证 KCMM 的 CUDA VMM 分配器本身不引入额外碎片（与 vLLM 的 PagedKvCache 基线一致）

### 6.2 Benchmark 4 增强：换出策略命中率

- **合成模式扩展**（纯 CPU 模拟，不需要 GPU）：
  - Zipf 分布（α=0.8, 1.0, 1.2）— 模拟真实热点
  - Hot-Cold 相位交替 — 测试策略自适应速度
  - Sequential scan — FIFO 最优场景
  - Multi-turn conversation — 测试 Hint API 收益
- **指标**：命中率、逐出次数、策略开销（`select_victims` 耗时）
- **产出**：策略命中率对比矩阵（LRU/LFU/FIFO/Oracle × 5 种模式）
- **与 baremetal-plan 阶段二的关系**：本节是 WSL2 上的纯模拟版本（更快、可复现），阶段二在 bare-metal 上用真实 GPU 数据验证

### 6.3 Benchmark 5 增强（内存压力）

- **Per-sequence latency 分布**：增加 decode step 的 P50/P99 延迟（当前只有 aggregate throughput）
- **多次 trial（≥ 5）**：确保统计显著性，汇报均值 ± 标准差
- **预热阶段**：消除 CUDA 首次启动偏差（warmup iterations）
- **更大规模**：在 bare-metal A30 上使用 LLaMA-7B（而非仅 TinyLlama）跑压力测试
- **产出**：KCMM 在不同模型规模下的 capacity scaling 曲线

### 6.4 Benchmark 8：换出策略对比（真实负载）

- 使用真实推理 trace 回放（录制 ContinuousScheduler 的 block 访问序列）
- 对比 LRU/LFU/FIFO/温控 四种策略在相同负载下的容量利用率
- **Bare-metal 专属**：需要足够 GPU 内存运行 LLaMA-7B + 高并发负载

---

## 7. NVMe GDS 路径评估（可选，预计 3-5 天）

### 5.1 前提条件

- 需要 `nvidia-fs.ko` 内核模块
- 需要 NVIDIA GPUDirect Storage SDK
- A30 + NVMe 1.6TB 满足硬件要求

### 5.2 评估内容

1. **GDS 延迟基准**：GPU ↔ NVMe 直接 DMA 延迟 vs GPU → CPU → NVMe 路径
2. **适用性判断**：KCMM 三级存储（GPU HBM ↔ CPU DRAM ↔ NVMe）中 NVMe 层的可行性
3. **写入缓冲区整合**：write buffer flush 到 NVMe 的顺序写优化

---

## 8. 完整任务清单

### 阶段一：基准测试复现（2-3 天）

- [ ] 环境准备（nvidia-smi, NVMe 确认, 编译）
- [ ] 运行 `run_kcmm_benches.sh --release`，采集全部 13 项微基准数据
- [ ] 运行 `run_kcmm_integration_bench.sh --release`，采集集成基准数据
- [ ] 每个配置 ≥ 5 次 trial，确保统计显著性
- [ ] 数据汇总：生成 WSL2 vs Bare-Metal 对比表
- [ ] 关键发现记录（哪些 WSL2 假设在 bare-metal 上成立/不成立）

### 阶段二：换出策略对比（2-3 天）

- [ ] 创建 `tests/kcmm_bench_policy_comparison.rs`
- [ ] 实现 5 种合成访问模式生成器
- [ ] 实现 Belady Oracle 基线（离线计算最优命中率）
- [ ] 实现策略对比脚本 `scripts/run_kcmm_policy_bench.sh`
- [ ] 运行全部策略 × 全部模式的对比测试
- [ ] 产出策略命中率矩阵 + 推荐指南

### 阶段三：高级策略（2-3 周）

**第一批（P0，1 周）：**
- [ ] 实现 `TemperatureTieredPolicy`
- [ ] 实现 `AdaptiveWatermark`
- [ ] 单元测试 + 集成测试

**第二批（P1，1 周）：**
- [ ] 实现 Hint API 完整 FFI + Rust trait
- [ ] 实现 `WriteBuffer`
- [ ] 换出策略对比 Benchmark 完善（真实 trace、长时间运行、混合策略）
- [ ] C 测试程序验证 ABI

**第三批（P2，1 周）：**
- [ ] 实现有损换出（量化/反量化 kernel）
- [ ] 实现流式恢复
- [ ] 实现全局压力平衡 + 跨池块借用
- [ ] 精度与性能评估

### 阶段四：文档与可观测性（可并行）

- [ ] 编写 bpftrace 追踪脚本 `src/trace/kcmm_events.bt`（需 bare-metal 验证）
- [ ] 编写 `kcmm.h` API 使用指南（`docs/dev/kcmm-api-guide.md`）
- [ ] 编写策略配置指南（`docs/dev/kcmm-policy-guide.md`）
- [ ] 实现 `KcmmMetrics` 环形缓冲区（`src/kcmm/metrics.rs`）
- [ ] 编写 `scripts/plot_metrics.py` 图表生成脚本
- [ ] 补充周实现文档（`docs/dev/kcmm-week*-impl.md`）
- [ ] 更新 `kcmm-implementation-analysis.md` 标记完成状态
- [ ] 编写 C API 示例程序（`examples/kcmm_c_integration.c`）

### 阶段五：Benchmark 完善（可并行）

- [ ] Benchmark 6：UFS 指标对比（WSL2 + TinyLlama）
- [ ] Benchmark 4 增强：合成模式策略命中率（纯 CPU 模拟）
- [ ] Benchmark 5 增强：per-sequence latency、多次 trial、bare-metal LLaMA-7B
- [ ] Benchmark 8：真实 trace 回放 + 四种策略对比（bare-metal）
- [ ] 数据汇总：产出策略命中率矩阵 + 论文图表

### 可选（可并行于阶段四/五）

- [ ] NVMe GDS 路径评估
- [ ] CUDA Graph 优化（消除 API launch 开销）
- [ ] 参考 kvcached 方案做 vLLM 底层拦截

---

## 9. 预期产出

### 7.1 数据产出

| 产出 | 内容 | 用途 |
|------|------|------|
| **Bare-Metal 基准数据** | 全部 13 项微基准 + 2 项集成基准的绝对值 | 论文核心性能数据 |
| **WSL2 vs Bare-Metal 对比** | 同配置下两环境的延迟/吞吐对比表 | 论证 bare-metal 必要性 |
| **策略命中率矩阵** | 5 种访问模式 × 3 种策略 × Oracle | 论文策略分析部分 |
| **高级策略收益** | 温控/水位线/Hint API 的 incremental benefit | 论文创新性论证 |

### 7.2 代码产出

| 产出 | 文件 |
|------|------|
| `TemperatureTieredPolicy` | `src/kcmm/tiering.rs` |
| `AdaptiveWatermark` | `src/kcmm/pool.rs` |
| `WriteBuffer` | `src/kcmm/tiering.rs` |
| Hint API FFI 实现 | `src/kcmm/ffi.rs` |
| 策略对比 Benchmark | `tests/kcmm_bench_policy_comparison.rs` |
| 策略对比运行脚本 | `scripts/run_kcmm_policy_bench.sh` |
| 量化/反量化 kernel | `src/kcmm/quantize.cu` |

### 7.3 文档产出

| 产出 | 文件 |
|------|------|
| Bare-Metal 基准测试报告 | `docs/report/baremetal/kcmm-baremetal-benchmark-report.md` |
| 换出策略对比分析 | `docs/report/baremetal/kcmm-policy-comparison.md` |
| 高级策略实现总结 | `docs/dev/kcmm-advanced-strategies.md` |
