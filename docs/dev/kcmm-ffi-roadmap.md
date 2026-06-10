# KCMM 务实开发路线：在无 Bare-Metal 条件下可完成的工作

**Date:** 2026-06-10
**Status:** Living document
**Related:** `docs/task/kcmm-implementation-analysis.md`, `docs/task/kcmm-related-research.md`

---

## 1. 已完成的工作（远超分析文档计划）

对照 `kcmm-implementation-analysis.md` 第 13-15 周计划，实际实现大幅超前：

| 功能模块 | 计划周 | 状态 | 关键实现 |
|---------|--------|------|---------|
| A. KcmmPool + 超级块管理 | 13 | ✅ 完成 | `pool.rs`: 完整生命周期, alloc/free, touch/cool, BlockLocation 5 状态机, 低水位线检测 |
| B. 分层存储引擎 | 14-15 | ✅ 完成 | `tiering.rs`: GPU↔CPU evict/restore, 批量 gather/scatter CUDA kernel, CpuSlotAllocator, 三阶段异步流水线, 完整错误回滚 |
| C. 可插拔换出策略 | 14 | ✅ 完成 | `EvictionPolicy` trait + LRU + LFU + FIFO 实现, 运行时策略切换 |
| D. CUDA Stream 管理 | 13 | ✅ 完成 | `streams.rs`: evict/restore/prefetch 三流, `CU_STREAM_NON_BLOCKING`, 异步 memcpy |
| G. UFS 指标 | 15-16 | ✅ 完成 | `metrics.rs`: KcmmMetrics, from_ufs, to_ufs_summary |
| H. 构建配置 | 全阶段 | ✅ 完成 | `Cargo.toml`: `cdylib` target, `kcmm` feature flag |
| Benchmark 1-5 | 13-17 | ✅ 完成 | `tests/kcmm_bench_*.rs`: alloc, tiering, stream, memory_pressure, sweep |
| 批量优化 | 未计划 | ✅ 完成 | ptrs_dev 复用, mmap prefault, gather/scatter kernel, auto-dispatch 阈值 |

## 2. 核心约束分析

### 2.1 WSL2 的能力边界

| 可以在 WSL2 做 | 不能在 WSL2 做 |
|---------------|---------------|
| ✅ 所有纯 Rust 代码开发 | ❌ 获得有意义的性能绝对值（GPU-PV 层 `cuMemAlloc` P99=12.96ms vs bare-metal ~10-30µs，差距 400-1000×） |
| ✅ 正确性验证（token 精确匹配、状态机转换） | ❌ Benchmark 5/7 的性能结论 |
| ✅ 单元测试 + 集成测试（项目已有完整 GPU 测试套件） | ❌ vLLM 端到端性能对比 |
| ✅ C FFI 开发 + 符号导出验证 | ❌ NVMe GDS 路径（需要 `nvidia-fs.ko` + 特定硬件） |
| ✅ CUDA kernel 编译验证（NVRTC 路径） | ❌ CUDA Graph 优化（driver 行为差异） |
| ✅ 策略算法验证（LRU/LFU/FIFO 行为正确性） | ❌ 多 GPU / MIG 测试 |

### 2.2 为什么不能简单地换 vLLM 做集成

vLLM 的内存模型与 KCMM 存在根本性不匹配：

- **vLLM**: 初始化时 `torch.zeros()` 预分配全部 GPU 显存 → 推理中只做 free list bookkeeping →底层走 PyTorch caching allocator → `cudaMalloc`
- **KCMM**: 使用 `cuMemCreate/cuMemMap/cuMemUnmap` 做细粒度物理页管理

`cuMemUnmap` 不能作用于 `cudaMalloc` 分配的内存——两套 API 完全隔离。vLLM 推理过程中不调用任何分配 API，KCMM 无法介入。

**结论：** 项目的 Rust 推理引擎是 KCMM 的**天然集成目标**——两者共享同一套 CUDA VMM 基础设施（`cuda_vmm.rs`、`PhysicalBlockAllocator`、`BlockHandle`），内存模型完全一致。

### 2.3 KCMM vs vAttention 的差异化定位

vAttention (ASPLOS 2025) 解决"如何用 CUDA VMM 让 FlashAttention 更快"——追求的是 **速度**（消除 block table 查表，decode +97%）。
KCMM 解决"物理内存不够时如何决策换出"——追求的是 **容量**（分层存储，并发 +30%）。

两者互补而非竞争。论文应 frame 为"CUDA VMM 在推理场景下的两个正交优化方向"。

## 3. 开发路线

### 阶段一：Rust 引擎集成（第 1-2 周）

**目标：** 让 KCMM 的 tiering 在真实推理路径上端到端跑通，生成 A/B 对比数据。

**具体任务：**

1. **`main.rs` 添加 KCMM 路径**
   - 在 `--continuous` 模式下，通过 `--features kcmm` 或配置项选择 `KcmmPool` 替代 `PagedKvCache`
   - `ContinuousScheduler` 适配：插入 `touch()`/`cool()` 调用，处理 `CpuResident` 块的自动恢复

2. **正确性验证**
   - 固定 seed + 相同 prompt，对比 KCMM-on vs KCMM-off 的生成 token 序列
   - 验证 tiering 触发后的 block 数据完整性（GPU↔CPU roundtrip）

3. **Benchmark 套件完善**
   - 在 WSL2 上跑完整的 Benchmark 1-5 + 6（UFS 对比）
   - 产出 KCMM-on vs KCMM-off 的相对收益数据（throughput_ratio, capacity_ratio）

### 阶段二：C FFI API 实现（第 2-3 周）

**目标：** 交付 `libkcmm.so` 作为可被外部引擎调用的独立共享库。

**具体任务：**

1. **实现 `extern "C"` 函数体**（`ffi.rs` 中类型已定义，函数体尚为注释）
   - `kcmm_pool_create` / `kcmm_pool_destroy`
   - `kcmm_alloc_blocks` / `kcmm_free_blocks`
   - `kcmm_touch` / `kcmm_cool`
   - `kcmm_get_metrics` / `kcmm_get_pool_stats`
   - `kcmm_set_eviction_policy`

2. **C 头文件与测试**
   - 编写 `include/kcmm.h`
   - 用纯 C 测试程序验证 ABI 正确性
   - 验证 `libkcmm.so` 符号导出

### 阶段三：策略进阶优化（第 3-5 周）

**目标：** 实现分析文档 1.6 节中标记为 P0 的优化，增强论文的创新性。

**具体任务：**

1. **分层温控 + 显式回收**（1.6.1）
   - 为 `EvictionPolicy` 扩展温度分级：灼热（序列尾部 4 block，永不换出）/ 温（正常 LRU）/ 冷（已完成序列，首选受害者）
   - `kcmm_cool(seq)` 直接将已完成序列的 block 标记为首选受害者——这些块永远不会再被访问
   - 改动范围：`tiering.rs` 内 `select_victims` + `LruPolicy` 扩展

2. **自适应水位线**（1.6.3）
   - 追踪 alloc/free rate 的指数加权移动平均（EWMA）
   - 预测 GPU 池耗尽时间，在 OOM 前提前触发换出
   - 改动范围：`pool.rs` 纯内部逻辑

3. **写入缓冲区**（1.6.4）
   - 将小粒度换出延迟合并为批量操作
   - 利用已有 `evict_blocks_batched` 基础设施
   - 对 NVMe 层尤其重要（顺序写 vs 随机写吞吐量差 10×+）

4. **Hint API 基础框架**（1.6.6）
   - `ffi.rs` 已定义 `kcmm_hint_t` 枚举
   - 实现 Rust trait 方法 + C API 存根
   - 策略层预留扩展点（`HINT_MULTI_TURN`、`HINT_NEAR_END`、`HINT_SYSTEM_PROMPT`）

### 阶段四：文档与可观测性（第 4-6 周，可并行）

**目标：** 完善开发和用户文档，准备论文材料。

**具体任务：**

1. **bpftrace 追踪脚本**（`src/trace/kcmm_events.bt`）
   - 追踪 USDT probe：evict 开始/完成、restore 开始/完成、prefetch 事件
   - 用于生成论文中的时间线图

2. **API 使用指南**
   - 面向引擎集成者的 `kcmm.h` 使用文档
   - 策略配置指南（何时选择 LRU vs LFU vs FIFO）

3. **时间序列指标采样**（G3）
   - `metrics.rs` 目前只有快照，添加环形缓冲区保留历史
   - 支持离线分析和论文图表生成

4. **开发文档更新**
   - 补充周实现文档（`kcmm-week14-impl.md` 等）
   - 更新 `kcmm-implementation-analysis.md` 标记实际完成状态

### 阶段五：Benchmark 完善（第 6-8 周）

**目标：** 产出完整的、可发表的 benchmark 数据（相对值）。

**具体任务：**

1. **Benchmark 6：UFS 指标对比**
   - KCMM vs PagedKvCache 在无分层模式下的 IFR/PME/BU/RFI
   - 在 WSL2 上用 TinyLLaMA 做小规模推理

2. **Benchmark 4 增强：换出策略命中率**
   - 增加合成模式：Zipf 分布、热冷交替
   - 完全模拟（不需要 GPU）可得出策略命中率数据

3. **Benchmark 5 增强**
   - 增加 per-sequence latency (P50/P99 decode step time)
   - 多次 trial（≥5）的统计显著性
   - 预热阶段消除 CUDA 首次启动偏差

## 4. Bare-Metal 恢复后优先做的事

| 优先级 | 任务 | 预计耗时 |
|--------|------|---------|
| P0 | 在 A30 上跑 Benchmark 5（内存压力端到端）获取绝对值 | 1-2 天 |
| P0 | 在 A30 上跑 Benchmark 2（换出/恢复延迟）获取 bare-metal p50/p99 | 半天 |
| P1 | 评估 NVMe GDS 路径（需要 `nvidia-fs.ko`） | 3-5 天 |
| P1 | Benchmark 8（换出策略对比，真实负载） | 2-3 天 |
| P2 | CUDA Graph 优化（消除 44 次 API launch 开销） | 1 周 |
| P2 | 参考 kvcached autopatch 方案做 vLLM 底层拦截 | 2-4 周 |

## 5. 里程碑检查点

```
Week 2:  Rust 引擎 KCMM-on 端到端跑通，token 正确性验证通过
Week 3:  libkcmm.so 符号导出，C 测试程序通过
Week 5:  分层温控 + 自适应水位线实现，单元测试覆盖
Week 6:  bpftrace 脚本 + API 文档初稿
Week 8:  完整 benchmark suite 可运行，数据可复现
```

## 6. 论文叙述建议

基于以上路线，论文评估部分可采用**双轨策略**：

**轨道一（WSL2，现在可做）：**
- Rust 引擎 KCMM-on vs KCMM-off 的相对收益（capacity_ratio, throughput_ratio）
- 换出策略命中率对比（合成 trace + LRU/LFU/FIFO/Oracle）
- UFS 指标一致性（KCMM vs PagedKvCache 无分层）
- CUDA Stream 隔离的非干扰性验证

**轨道二（bare-metal，将来补充）：**
- A30 + LLaMA-7B 的绝对性能数据
- 与 vLLM baseline 的对比（需先完成 vLLM 集成）
- NVMe 层的延迟/吞吐数据
- 大规模并发压测（128+ 并发）

审稿人更关心 **delta 而非绝对值**——"KCMM 让同一个引擎多接纳了 30% 并发"比"KCMM 的绝对延迟是 200µs"更有说服力。因此 WSL2 上的数据已经可以构成论文的核心论据。
