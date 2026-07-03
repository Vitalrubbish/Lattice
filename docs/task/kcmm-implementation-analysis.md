# 步骤 3 KCMM 实现分析：现有项目修改与 Benchmark 设计

## 目录

0. [KCMM 实现目标与所需功能](#0-kcmm-实现目标与所需功能)
   - [0.1 KCMM 的定位](#01-kcmm-的定位)
   - [0.2 实现目标](#02-实现目标)
   - [0.3 所需功能清单](#03-所需功能清单)
   - [0.4 功能依赖图](#04-功能依赖图)
   - [0.5 向后兼容性约束](#05-向后兼容性约束)
   - [0.6 关键风险与缓解措施](#06-关键风险与缓解措施)
1. [现有代码能力 vs KCMM 差距分析](#1-现有代码能力-vs-kcmm-差距分析)
   - [1.1 `cuda_vmm.rs` — GPU 物理页管理](#11-srccachecuda_vmmrs--gpu-物理页管理)
   - [1.2 `paged_kv.rs` — PagedKvCache](#12-srccachepaged_kvrs--pagedkvcache)
   - [1.3 `swap.rs` — SwapManager](#13-srccacheswaprs--swapmanager)
   - [1.4 `fragmentation_tracker.rs` — 碎片追踪](#14-srccachefragmentation_trackerrs--碎片追踪)
   - [1.5 `unified_frag.rs` — UFS 指标](#15-srccacheunified_fragrs--ufs-指标)
   - [1.6 KCMM 置换策略的进阶优化手段](#16-kcmm-置换策略的进阶优化手段)
     - [1.6.1 利用 KV Cache 独有的语义信号](#161-利用-kv-cache-独有的语义信号)
     - [1.6.2 注意力信号指导的差异化置换](#162-注意力信号指导的差异化置换)
     - [1.6.3 预测性操作](#163-预测性操作)
     - [1.6.4 传输层面的优化](#164-传输层面的优化)
     - [1.6.5 跨引擎的全局优化](#165-跨引擎的全局优化)
     - [1.6.6 引擎提示接口（Hint API）](#166-引擎提示接口hint-api)
     - [1.6.7 优化手段汇总与优先级](#167-优化手段汇总与优先级)
2. [需要新建的文件](#2-需要新建的文件)
3. [需要修改的现有文件](#3-需要修改的现有文件)
4. [实现顺序](#4-实现顺序)
5. [Benchmark 设计方案](#5-benchmark-设计方案)
6. [Benchmark 清单与优先级](#6-benchmark-清单与优先级)

---

## 0. KCMM 实现目标与所需功能

### 0.1 KCMM 的定位

KCMM（KV Cache Memory Manager）是一个**独立的、轻量级的用户态 OS 服务**，将操作系统的虚拟内存管理理念引入 GPU KV Cache 领域。其核心设计理念是将 GPU KV Cache 视为 OS 管理的系统资源，而非单个推理引擎私有的资源池。

**KCMM 在学术/工业界的独特生态位：**

| 特性 | 现有系统覆盖 | KCMM |
|------|------------|------|
| CUDA VMM 虚拟内存管理 | vAttention、kvcached、GMLake | ✅ |
| GPU + CPU + NVMe 三层分层存储 | KVBM、LMCache、Mooncake | ✅ |
| 独立 OS 服务/守护进程 | kvcached（仅 GPU）、WombatKV（非 VMM） | ✅ |
| 跨引擎通用 API（非引擎特定插件） | kvcached（autopatch 机制） | ✅ |
| 可插拔换出/预取策略 | 无 | ✅ |
| UFS 兼容的碎片化指标 | 无 | ✅ |

**KCMM 占据的独特生态位：** 一个使用 CUDA VMM 提供 OS 风格 GPU KV Cache 虚拟内存管理、支持完整 GPU→CPU→NVMe 三层分层存储、以引擎无关的服务 API 暴露给任意推理引擎的用户态守护进程。这是现有文献和工业系统中**首个**将 CUDA VMM 的细粒度虚拟内存管理与多级存储分层相结合的系统。

### 0.2 实现目标

#### 目标 1：独立的 OS 风格内存管理服务（核心架构目标）

将 KV Cache 内存管理从推理引擎内部提取为独立的外部服务（`libkcmm.so` + 协调守护进程），类似于 Linux 内核 `kswapd` 在虚拟内存管理中的角色。

```
传统模型：每个引擎内嵌自己的 KV Cache 管理器
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ vLLM     │  │ SGLang   │  │ TRT-LLM  │
  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │
  │ │BM    │ │  │ │BM    │ │  │ │BM    │ │
  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │
  └──────────┘  └──────────┘  └──────────┘

KCMM 模型：统一的外部内存管理服务
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ vLLM     │  │ SGLang   │  │ TRT-LLM  │
  │ (client) │  │ (client) │  │ (client) │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      │ libkcmm.so API
              ┌───────┴───────┐
              │  KCMM 服务    │
              │  (OS 风格)    │
              └───────────────┘
```

**实现要点：**
- `libkcmm.so` 以 C ABI 共享库形式发布，任何语言均可绑定
- 支持多引擎并发注册池，内存压力感知提升为**系统级**（所有注册池汇总决策）
- 换出策略在**所有引擎间统一决策**，避免各自为政

#### 目标 2：CUDA VMM + 块粒度分层存储（核心技术目标）

将 CUDA VMM 物理页管理（2MB 超级块）与以**块（Block）为粒度的分层存储**结合。这是现有系统中首个实现此组合的方案。

**三层延迟/容量梯度：**

| 层级 | 存储介质 | 典型延迟 | 容量上限 | 角色 |
|------|---------|---------|---------|------|
| G1 | GPU HBM | ~ns | 24GB (A30) | 活跃推理块，热数据 |
| G2 | CPU DRAM（`/dev/shm`） | ~μs | 128GB (系统) | 温数据，首级换出目标 |
| G3 | NVMe SSD（可选 GDS） | ~ms | 1.6TB (本地) | 冷数据，大容量持久化 |

**关键设计选择：**
1. **超级块**负责 GPU 物理内存管理（`cuMemCreate`/`cuMemMap`），2MB 对齐
2. **换出以单个冷块为粒度**（而非整个 2MB 超级块）：先将块数据拷贝到 CPU 缓冲区（`cudaMemcpy D2H`），再释放物理页
3. **`BlockLocation` 枚举**追踪每个块的状态：

```rust
enum BlockLocation {
    GpuResident(BlockHandle, u64),  // GPU 常驻（物理句柄 + VA 偏移）
    CpuResident(usize),              // CPU 交换缓冲区中的槽位
    NvmeResident(u64),               // NVMe 交换文件中的偏移
    Evicting,                         // 传输中（GPU → CPU/NVMe）
    Restoring,                        // 传输中（CPU/NVMe → GPU）
}
```

#### 目标 3：可插拔换出策略 + 启发式预取（策略目标）

将**策略与机制分离**——KCMM 提供通用的分层存储机制（块迁移、状态追踪、CUDA Stream 管理），换出/预取策略通过可插拔的 trait 接口独立演进。

**可插拔换出策略框架：**

```rust
trait EvictionPolicy {
    /// 选择要换出的受害者块（按优先级排序）
    fn select_victims(&self, candidates: &[BlockHandle], count: usize)
        -> Vec<BlockHandle>;
    /// 块被访问后更新策略状态
    fn on_access(&mut self, block: BlockHandle);
    /// 块被换出后更新策略状态
    fn on_evict(&mut self, block: BlockHandle);
}
```

- **LRU（默认）：** 选择 `last_access` 最旧的块，适合大多数工作负载
- **LFU：** 选择访问频率最低的块，适合有明显冷热区分的负载
- **FIFO：** 选择分配时间最早的块，开销最低

**启发式预取（可选优化）：**
- 后台线程对每个活跃序列预测接下来需要的块（逻辑块 K 活跃 → 预取 K+1、K+2）
- 若预取候选为 `CpuResident`，启动异步 `cudaMemcpy H2D`
- 使用专用 `prefetch` CUDA Stream，不与推理计算竞争

#### 目标 4：引擎无关的通用服务接口（集成目标）

通过 C ABI 共享库暴露稳定的服务 API，任何推理引擎（vLLM、SGLang、TRT-LLM 等）均可通过 FFI 绑定使用。

**核心 C API：**

```c
kcmm_pool_t* kcmm_pool_create(size_t block_size, size_t max_blocks,
                               const char* cpu_cache_path);
void kcmm_pool_destroy(kcmm_pool_t* pool);
int kcmm_alloc_blocks(kcmm_pool_t* pool, uint64_t seq_id,
                      size_t num_blocks, uint32_t* out_blocks);
void kcmm_free_blocks(kcmm_pool_t* pool, uint64_t seq_id,
                      const uint32_t* blocks, size_t num);
void kcmm_touch(kcmm_pool_t* pool, uint64_t seq_id);
void kcmm_cool(kcmm_pool_t* pool, uint64_t seq_id);
void kcmm_get_metrics(kcmm_pool_t* pool, kcmm_metrics_t* out);
int kcmm_share_prefix(kcmm_pool_t* pool, uint64_t src_seq,
                      uint64_t dst_seq, size_t num_blocks, uint32_t* out);
```

**集成方式：**
- Python 绑定 + vLLM monkey-patch（步骤 3 主要集成目标）
- 纯 C/FFI 绑定，任何语言均可调用
- 可选：通过 Unix Domain Socket 与 KCMM 协调守护进程通信实现跨进程共享

#### 目标 5：UFS 兼容的跨引擎指标体系（观测性目标）

内置碎片化追踪，将标准化的 GPU 内存碎片化指标（IFR、PME、BU、RFI）与 KCMM 内存管理服务深度集成。

**指标维度：**

| 指标 | 缩写 | 描述 | 用途 |
|------|------|------|------|
| 内碎片率 | IFR | 超级块内部因块大小不对齐导致的浪费 | 指导块大小和超级块配置 |
| 物理内存效率 | PME | GPU 物理页中实际存放有效数据的比例 | 评估超级块利用率 |
| 块利用率 | BU | 已分配块中存放活跃序列数据的比例 | 决策换出阈值 |
| 运行时碎片指数 | RFI | 时间序列上的碎片化累积程度 | 触发块整理操作 |

**观测性闭环：** 指标收集 → 压力感知（低水位线检测）→ 策略决策（换出/整理触发）→ 指标验证

#### 目标 6：从项目现有代码演进（工程目标）

KCMM 并非从零开始，而是从项目前序步骤的代码库直接演进：

| 当前文件 | KCMM 中的新角色 | 关键演进 |
|--------|-------------|---------|
| `src/cache/cuda_vmm.rs` | GPU 物理页管理（超级块、cuMemMap） | 新增 CUDA Stream 封装 |
| `src/cache/paged_kv.rs` | 块分配 + 序列追踪 + BlockLocation | 提取 `PhysicalBlockAllocator`，扩展 `BlockInfo` → `BlockLocation` |
| `src/cache/swap.rs` | TieringEngine（GPU↔CPU↔NVMe 迁移） | 序列粒度 → 块粒度，新增可插拔策略 |
| `src/cache/fragmentation_tracker.rs` | UFS 指标收集（IFR、PME、RFI） | 暴露给 KCMM C API |
| `src/cache/unified_frag.rs` | 跨引擎对比的标准化指标 | 泛化 `from_cache` → `  from_kcmm_pool` |

#### 性能目标

| 指标 | 目标值 | 测量基准 |
|------|--------|---------|
| 块分配延迟（无分层） | 回退 < 5% vs vLLM 内部分配器 | Benchmark 1 |
| 单块恢复延迟 (GPU←CPU) | p50 < 200μs | Benchmark 2 |
| 推理 kernel 干扰（专用流） | 时间增长 < 1% | Benchmark 3 |
| 最大可接纳并发（有分层） | ≥ 1.3× vs 无 swap | Benchmark 5 |
| UFS 指标一致性（无分层） | 偏差 < 2% vs vLLM | Benchmark 6 |
| Token 输出正确性 | 精确匹配 vLLM（有 KCMM 时） | Benchmark 7 |

---

### 0.3 所需功能清单

以下按功能模块列出 KCMM 步骤 3 需要实现的全部功能，标注实现优先级和依赖关系。

#### 功能模块 A：块分配与管理（优先级 P0 — 第 13 周）

**这是 KCMM 最基础的功能，所有其他功能依赖于此。**

| # | 功能 | 描述 | 来源文件 |
|---|------|------|---------|
| A1 | `KcmmPool` 创建与销毁 | 池生命周期管理：预分配超级块、初始化 free list、创建 CUDA Stream | 新建 `kcmm/pool.rs` |
| A2 | 块分配（`alloc_blocks`） | 分配 N 个物理块给指定序列；自动检测 BlockLocation 并触发恢复 | 从 `paged_kv.rs` 提取并泛化 |
| A3 | 块释放（`free_blocks`） | 归还块到 free list；若为 `CpuResident` 则清理 CPU 缓冲区 | 从 `paged_kv.rs` 提取并改进 |
| A4 | 序列注册与追踪 | `SequenceState` 管理：`is_active`、`last_access`、`shared_prefix_len` | 扩展 `SeqMetadata` |
| A5 | `touch` / `cool` 操作 | `touch` 标记序列活跃（更新 `last_access`，LRU 晋升）；`cool` 标记序列可换出 | 新增功能 |
| A6 | 低水位线检测 | `free_blocks < low_watermark` 时触发 TieringEngine 换出 | 新增逻辑 |
| A7 | `PhysicalBlockAllocator` 提取 | 从 `paged_kv.rs` 提取为独立模块，支持超级块生命周期管理 | 提取到 `kcmm/superblock.rs` |

#### 功能模块 B：分层存储引擎（优先级 P0 — 第 14–15 周）

**KCMM 的核心差异化功能，GPU↔CPU↔NVMe 三层数据迁移。**

| # | 功能 | 描述 | 来源/依赖 |
|---|------|------|----------|
| B1 | `BlockLocation` 状态机 | 5 状态枚举 + 状态转换验证；`GpuResident ↔ Evicting ↔ CpuResident ↔ Restoring ↔ GpuResident` | 替换 `BlockInfo.in_use: bool` |
| B2 | 块粒度 GPU→CPU 换出 | 选择受害者块 → `cudaMemcpy D2H` → `cuMemUnmap` → 标记 `CpuResident` → 归还物理块 | 改造 `swap.rs` 序列粒度换出 |
| B3 | 块粒度 CPU→GPU 恢复 | 分配物理块 → `cuMemMap` → `cudaMemcpy H2D` → 标记 `GpuResident` | 改造 `swap.rs` 序列粒度恢复 |
| B4 | CPU 缓冲区管理 | `mmap` 的 `/dev/shm/kcmm_swap` 区域；槽位分配/释放；碎片管理 | 从 `swap.rs` 演进 |
| B5 | 批量换出/恢复优化 | 批量 `cudaMemcpy` + 批量 `cuMemMap`/`cuMemUnmap`，摊销单块延迟 | 新增优化 |
| B6 | NVMe 层（可选） | GPU↔NVMe 直接传输（GDS `cuFileRead`/`cuFileWrite`，标准 I/O 回退） | 新增功能 |
| B7 | 超级块碎片整理 | 当超级块利用率 < 25% 时，将活跃块迁移到其他超级块并释放空超级块 | 新增功能（低优先级） |
| B8 | 全局 epoch / 时间戳 LRU | 使用 `Instant` 替代全局 epoch 计数器，更精确的 LRU 排序 | 改进现有逻辑 |

#### 功能模块 C：可插拔换出策略（优先级 P1 — 第 14 周）

| # | 功能 | 描述 |
|---|------|------|
| C1 | `EvictionPolicy` trait | 统一接口：`select_victims(candidates, count)`、`on_access(block)`、`on_evict(block)` |
| C2 | `LruPolicy` 实现 | 基于 `last_access` 时间戳的 LRU 策略（KCMM 默认） |
| C3 | `LfuPolicy` 实现 | 基于访问频率计数的 LFU 策略 |
| C4 | `FifoPolicy` 实现 | 基于分配时间的 FIFO 策略（最低开销） |
| C5 | 策略运行时切换 | 支持通过配置项或 API 在运行时切换换出策略 |

#### 功能模块 D：CUDA 流管理（优先级 P1 — 第 13 周）

| # | 功能 | 描述 |
|---|------|------|
| D1 | `CudaStream` 封装 | 封装 `CUstream`，使用 `CU_STREAM_NON_BLOCKING` 标志 |
| D2 | 专用换出流 | `evict` 流：GPU→CPU 数据拷贝（D2H） |
| D3 | 专用恢复流 | `restore` 流：CPU→GPU 数据拷贝（H2D） |
| D4 | 专用预取流 | `prefetch` 流：后台异步预取 H2D，不与推理计算竞争 |
| D5 | 流间同步 | CUDA Event 进行 evict/restore/prefetch 三流之间的同步和与推理流的协调 |
| D6 | 异步 memcpy 操作 | `cuda_memcpy_d2h_async()` / `cuda_memcpy_h2d_async()` |

#### 功能模块 E：前缀共享检测（优先级 P2 — 第 16 周，步骤 4 使用）

**步骤 3 预留接口，核心实现在步骤 4。**

| # | 功能 | 描述 |
|---|------|------|
| E1 | `SharingManager` 结构 | 前缀索引 + 引用计数管理 |
| E2 | 前缀注册（`register_prefix`） | 将序列的前缀块注册到共享索引（内容哈希 → 物理块引用） |
| E3 | 前缀查找（`try_share_prefix`） | 检查是否存在可共享前缀，返回匹配的块引用列表 |
| E4 | 引用计数管理 | 写时复制语义：共享前缀被修改时触发块复制 |
| E5 | IPC 通信接口 | 通过 POSIX 共享内存或 Unix Domain Socket 实现跨进程前缀共享 |

#### 功能模块 F：C FFI API（优先级 P0 — 第 16 周）

**暴露给外部推理引擎的稳定接口。**

| # | 函数 | 描述 | 优先级 |
|---|------|------|--------|
| F1 | `kcmm_pool_create` | 创建 KCMM 内存池 | P0 |
| F2 | `kcmm_pool_destroy` | 销毁池，释放所有资源 | P0 |
| F3 | `kcmm_alloc_blocks` | 分配块（自动处理恢复） | P0 |
| F4 | `kcmm_free_blocks` | 释放块 | P0 |
| F5 | `kcmm_touch` | 标记序列活跃 | P0 |
| F6 | `kcmm_cool` | 标记序列可换出 | P0 |
| F7 | `kcmm_get_metrics` | 获取 UFS 指标快照 | P1 |
| F8 | `kcmm_share_prefix` | 跨序列前缀共享 | P2（步骤 4） |
| F9 | `kcmm_set_eviction_policy` | 运行时切换换出策略 | P2 |
| F10 | `kcmm_get_pool_stats` | 获取池统计信息（块使用率、各层分布） | P1 |

#### 功能模块 G：UFS 指标收集（优先级 P1 — 第 15–16 周）

| # | 功能 | 描述 | 来源 |
|---|------|------|------|
| G1 | `FragmentationTracker` 集成 | 接入 KCMM 池，持续采集 IFR/PME/BU/RFI | 复用 `fragmentation_tracker.rs` |
| G2 | `KcmmMetrics` 结构 | 标准化指标快照，与 C API 的 `kcmm_metrics_t` 对应 | 新建 `kcmm/metrics.rs` |
| G3 | 时间序列采样 | 周期性采集指标并保留历史（用于离线分析） | 新增功能 |
| G4 | 跨引擎对比支持 | 通过统一指标格式支持不同引擎间的内存效率对比 | 泛化 `unified_frag.rs` |

#### 功能模块 H：配置与构建（优先级 P0 — 贯穿全阶段）

| # | 功能 | 描述 |
|---|------|------|
| H1 | `KcmmConfig` 结构 | 块大小、最大块数、CPU 缓存路径、分层开关、换出策略、预取窗口等 |
| H2 | Feature flag 控制 | `#![feature(kcmm)]` 可选启用，不影响现有推理引擎行为 |
| H3 | `cdylib` 构建目标 | `Cargo.toml` 添加 `crate-type = ["lib", "cdylib"]` 生成 `libkcmm.so` |
| H4 | vLLM Python 绑定 | `scripts/kcmm_vllm_patch.py` — monkey-patch vLLM 块分配器 |
| H5 | bpftrace 追踪脚本 | `src/trace/kcmm_events.bt` — 追踪换出/恢复事件用于性能诊断 |

---

### 0.4 功能依赖图

```
第 13 周（基础）          第 14–15 周（核心）       第 16 周（集成）
─────────────────────  ───────────────────────  ──────────────────

A1 KcmmPool 创建       ──→ B2 GPU→CPU 换出      ──→ F1-F6 C API
A2 块分配              ──→ B3 CPU→GPU 恢复      ──→ H4 vLLM 绑定
A3 块释放              ──→ B5 批量优化          ──→ E1-E4 前缀共享
A4 序列追踪            ──→ B1 BlockLocation     ──→ G1-G4 UFS 指标
A5 touch/cool          ──→ C1-C4 换出策略       ──→ B6 NVMe 层(可选)
A6 低水位线检测         ──→ B4 CPU 缓冲区        ──→ B7 碎片整理(可选)
A7 超级块管理           ──→ D1-D6 CUDA Stream   ──→ F7-F10 扩展 API

                         └─── 依赖线 ───→
```

### 0.5 向后兼容性约束

- **独立 Rust 推理引擎行为不变：** `cargo run -- --continuous --model-path ...` 仍使用原有的 `PagedKvCache`（不链接 KCMM）
- **KCMM 作为可选 feature flag 启用：** `cargo build --features kcmm` 才编译 KCMM 模块和生成 `libkcmm.so`
- **`src/cache/` 模块保持功能完整：** 不删除或破坏现有代码路径，KCMM 作为新的并行模块存在
- **现有公共 API 不变：** `PagedKvCache`、`SwapManager` 等公开类型的接口保持稳定

### 0.6 关键风险与缓解措施

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| 块粒度换出复杂度：当前 swap.rs 以序列为单位操作 | 高 | 先实现单块换出，验证正确性后再优化为批量操作 |
| CUDA Stream 同步：换出/恢复流需与推理流正确协调 | 高 | 使用 CUDA Event 进行流间同步；编写显式同步测试 |
| `BlockLocation` 状态机正确性：5 状态转换需处理并发 | 中 | 细粒度锁 + 状态转换断言；proptest 模糊测试覆盖所有转换 |
| `cuMemCreate`/`cuMemMap` 在热路径上的延迟（vAttention 实测慢 ~115×） | 中 | 预创建物理句柄（超级块池）；批量 `cuMemMap`/`cuMemUnmap`；后台线程处理 |
| vLLM 内部 API 变化导致 monkey-patch 失效 | 中 | 固定 vLLM 版本；先用自定义 Rust 引擎验证 KCMM 核心功能 |
| 块粒度换出导致的超级块内部碎片 | 低 | `FragmentationTracker` 监控；利用率 < 25% 触发块整理 |
| kvcached 后续添加 CPU/NVMe 分层后直接竞争 | 低 | 差异化竞争：分层存储 + 可插拔策略 + UFS 指标；考虑互操作性 |

---

## 1. 现有代码能力 vs KCMM 差距分析

### 1.1 `src/cache/cuda_vmm.rs` — GPU 物理页管理

| 现有能力 | KCMM 需要的 | 差距 |
|---------|-----------|------|
| `reserve_address` / `free_address` | 保留/释放 GPU VA | **直接可用** |
| `create_physical` / `release_physical` | 创建/释放物理内存句柄 | **直接可用** |
| `map` / `unmap` | 映射/解除映射 | **直接可用** |
| `batch_map_blocks` / `batch_unmap_blocks` | 批量操作 | **直接可用** |
| 无 CUDA Stream 支持 | 专用 CUDA Stream 用于换出/恢复/预取 | **需新增** |
| 硬编码 2MB 对齐 | 配置化粒度（为步骤 4 做准备） | **需改进** |

**需要的修改：**

- 添加 `CudaStream` 封装（换出流、恢复流、预取流），使用 `CU_STREAM_NON_BLOCKING`
- 添加异步 `cudaMemcpy` 操作（D2H, H2D）
- 添加流同步方法

### 1.2 `src/cache/paged_kv.rs` — PagedKvCache（核心文件）

这是最核心的文件，也是 KCMM 的主要演进源。

| 现有概念 | KCMM 对应 | 差距 |
|---------|---------|------|
| `PhysicalBlockAllocator` | KCMM 的块分配器 | 已存在但耦合在 paged_kv.rs 内 |
| `SuperblockInfo` | `Superblock` 结构 | 需提取为独立模块 |
| `BlockHandle` | 同上 | 直接复用 |
| `BlockInfo` + `in_use: bool` | `BlockLocation` 枚举 | **需从 bool 扩展为 5 状态枚举** |
| `LayerKvPool` | 多 layer 物理池 | 结构复用，但换出时需要感知 BlockLocation |
| `SeqMetadata` | `SequenceState` | **需添加 `is_active`、`last_access`、`shared_prefix_len`** |
| `alloc_sequence` / `free_sequence` | `kcmm_alloc_blocks` / `kcmm_free_blocks` | 逻辑类似，需泛化 |
| `alloc_block` | 单块分配（decode 时扩充） | 直接复用 |
| 无 `touch` / `cool` | 活跃/冷却标记 | **需新增** |
| 无引用计数 | 前缀共享的 ref_count | **需新增（步骤 4 使用但步骤 3 预留接口）** |
| `free_sequence` 归还块到 free list | 同，但需考虑 BlockLocation | **需改进** |
| `ensure_capacity` 创建超级块 | 同，但低水位线策略不同 | **需改进** |

**需要的修改（核心工作）：**

- 将 `PhysicalBlockAllocator` 提取为独立的、可配置的块分配器
- 将 `BlockInfo.in_use: bool` 替换为 `BlockLocation` 枚举：

```rust
enum BlockLocation {
    GpuResident(BlockHandle, u64),  // (句柄, GPU VA 偏移)
    CpuResident(usize),              // CPU 交换缓冲区中的偏移
    NvmeResident(u64),               // NVMe 交换文件中的偏移
    Evicting,                         // 传输中
    Restoring,                        // 传输中
}
```

- 将 `SeqMetadata` 扩展为 `SequenceState`，添加：
  - `is_active: bool`（正在解码 vs. 等待中）
  - `last_access: Instant`（用于 LRU）
  - `shared_prefix_len: usize`（与其他序列共享的块数）
- 重构 `ensure_capacity`：当 `free_blocks < low_watermark` 时触发换出（而非总是创建新超级块）
- 添加 `touch` / `cool` 方法

### 1.3 `src/cache/swap.rs` — SwapManager

| 现有能力 | KCMM 需要的 | 差距 |
|---------|-----------|------|
| `evict_sequence`（以序列为单位换出） | 以**块**为单位换出 | **需改为块粒度** |
| `restore_sequence`（以序列为单位恢复） | 以**块**为单位恢复 | **需改为块粒度** |
| D2H / H2D `cudaMemcpy` | 同，但需使用专用 CUDA Stream | **需新增 Stream 参数** |
| 全局 epoch 计数器 | LRU 时间戳 | **改为 Instant** |
| `drop_swapped` 释放 CPU 缓冲区 | 同，但需追踪每个块的位置 | **需改进** |
| 无换出策略选择 | 可插拔策略（LRU、LFU、FIFO） | **需新增** |
| 无 NVMe 层 | GPU ↔ CPU ↔ NVMe 三级分层 | **需新增** |
| 无预取 | 基于启发式的预取 | **需新增** |
| 无 `cuMemUnmap` 调用 | 换出后释放 GPU 物理页 | **需新增** |

**需要的修改（核心工作）：**

- 重构 `SwapManager` 为 `TieringEngine`：

```rust
struct TieringEngine {
    cpu_buffer: *mut u8,          // mmap 的 CPU 交换空间
    cpu_buffer_size: usize,
    nvme_file: Option<File>,      // NVMe 交换文件
    eviction_policy: EvictionPolicy,
    block_states: HashMap<BlockHandle, BlockLocation>,
    evict_queue: BinaryHeap<EvictCandidate>,  // 按 last_access 排序
    prefetch_queue: VecDeque<BlockHandle>,
}
```

- 改为块粒度换出（而非序列粒度）
- 换出流程：选择受害者块 → `cudaMemcpy D2H` → `cuMemUnmap` → 标记 `CpuResident` → 归还块到 free list
- 恢复流程：分配物理块 → `cuMemMap` → `cudaMemcpy H2D` → 标记 `GpuResident`
- 添加 NVMe 层：使用 `cuFileRead`/`cuFileWrite`（GDS）或标准 I/O 回退
- 添加 `EvictionPolicy` trait 和 `LruPolicy`、`LfuPolicy`、`FifoPolicy` 实现
- 添加异步预取后台线程

### 1.4 `src/cache/fragmentation_tracker.rs` — 碎片追踪

| 现有能力 | KCMM 需要的 | 差距 |
|---------|-----------|------|
| `FragmentationSample` | KCMM 指标快照 | 直接复用 |
| `record_unified` | 记录 UFS 指标 | 直接复用 |
| `average_ratio` / `peak_ratio` 等 | 时间序列统计 | 直接复用 |

**需要的修改：** 较小。主要是将指标收集接口暴露给 KCMM C API（`kcmm_get_metrics`）。

### 1.5 `src/cache/unified_frag.rs` — UFS 指标

| 现有能力 | KCMM 需要的 | 差距 |
|---------|-----------|------|
| IFR, BU, PME, RFI 计算 | 相同指标 | **直接复用** |
| `UnifiedFragMetrics` / `UnifiedFragSummary` | 相同结构 | **直接复用** |
| `from_cache` 依赖 `PagedKvCache` | 需要泛化为依赖 `KcmmPool` | **需添加泛化方法** |

**需要的修改：** 添加 `from_kcmm_pool` 方法或泛化现有方法以支持 `KcmmPool`。

---

### 1.6 KCMM 置换策略的进阶优化手段

传统置换策略（LRU/LFU/FIFO）本质上只回答了"**该换出哪个**"这一个问题。但 KCMM 作为一个独立的内存管理服务，在 KV Cache 这个特定领域，拥有比 OS 虚拟内存更丰富的信息优势——KV Cache 的访问模式比通用内存页**可预测得多**。以下从六个维度系统性地分析可用的优化手段。

#### 1.6.1 利用 KV Cache 独有的语义信号

##### 序列生命周期感知

OS 的页表不知道一个页面属于哪个进程的哪个阶段。但 KCMM 知道每个块属于哪个序列，以及该序列处于什么生命周期：

```
prefill 阶段: 大量块一次性分配，之后不再写入（只读）
decode 阶段: 每步分配 1 个新块，所有历史块被 attention 读取
完成阶段: 所有块立即可回收
```

**优化手段：**

1. **分层温控（Tiered Temperature）：** decode 阶段最后几个块是"灼热"的（下一步立刻需要），中间块是"温"的（每步都读），prefill 块在最前面（也每步都读但更可预测）。这比 LRU 的一维"冷-热"模型精细得多。可以为不同温度区间维护独立的换出优先级队列：
   - 灼热区（序列尾部 4 个块）→ 永不换出
   - 温区（序列中间块）→ 正常 LRU
   - 常温区（序列头部 prefill 块）→ 正常 LRU
   - 冷区（已完成序列的块）→ 首选受害者

2. **显式回收（Explicit Reclamation）：** 序列完成后，`kcmm_cool` 不应该把它的块放入普通 LRU 队列，而应该直接标记为"首选受害者"——这些块**永远不会再被访问**。减少 LRU 队列的扫描开销，同时避免"已完成序列的冷块恰好排在队列深处、迟迟不被换出"的问题。

##### 多轮对话的"休眠-唤醒"模式

Chatbot 场景下，同一用户的多次对话之间存在长时间的"休眠期"：

```
Turn 1: [==== 序列 A ====]
                         ... 30秒空闲 ...
Turn 2:                   [==== 序列 A' 继续 ====]
```

LRU 会在 30 秒空闲期内把序列 A 的块全部换出，导致 Turn 2 时全部冷启动——每次换出和恢复都带来数百微秒的延迟和 PCIe 带宽消耗。

**优化手段——"休眠标记"（Hibernation Marker）：**

- 引擎通过 Hint API 告知 KCMM：这个序列可能还会有后续轮次（`HINT_MULTI_TURN`）
- KCMM 对这类序列采用**延迟换出（Delayed Eviction）**策略：比普通冷序列多保留 N 秒（允许内存压力有足够时间消退）
- 配合**对话轮次预取**：检测到同用户新请求到达时，在前一个请求完成 prefill 之前就后台预取上一轮被换出的 KV Cache

```
时间线：
  Turn 1 完成 → 标记 HINT_MULTI_TURN → 延迟换出计时器启动（如 60s）
  30s 后 Turn 2 到达 → 预取 Turn 1 块 → 0 恢复延迟
  （若无 Turn 2，60s 后正常换出）
```

#### 1.6.2 注意力信号指导的差异化置换

这是最具学术前沿性的优化方向——不是所有块都同等重要。近期研究（H2O、StreamingLLM、SnapKV）表明，不同位置的 token 对未来生成的重要性差异巨大。

##### 注意力熵指导的块重要性分级

```
重要性分级（按注意力模式）：
┌────────────────────────────────────────────┐
│ Attention Sink (前 ~4 token)    ⭐⭐⭐⭐⭐  │  ← 几乎所有注意力头高度关注，不可丢弃
│ Heavy Hitters (高注意力 token)   ⭐⭐⭐⭐    │  ← 少数 token 被大多数注意力头关注
│ 普通 token                       ⭐⭐      │  ← 只有局部注意力窗口关注
│ 可丢弃 token                     ⭐       │  ← 注意力分数接近于零
└────────────────────────────────────────────┘
```

**优化手段：**

```c
// 新增 API：标记块的保护级别
void kcmm_protect(pool, seq_id, block_ids, KCMM_PROTECT_NEVER_EVICT);
void kcmm_evictable(pool, seq_id, block_ids, KCMM_EVICT_PREFERRED);
```

引擎在 decode 过程中可以轻量级地追踪注意力分数（取每个 token 被所有注意力头关注的最大值或平均值），把 Attention Sink 和 Heavy Hitters 的物理块标记为受保护。当内存压力来临时，KCMM 优先换出"可丢弃块"——即使它们的 `last_access` 比某些受保护块更近。

**设计要点：**
- 注意力分数的追踪在引擎侧完成（KCMM 只管物理块位置，不解析 attention 矩阵）
- 通过 `kcmm_protect` / `kcmm_evictable` 将重要性信号注入 KCMM 的换出决策
- 保护标记是**软性的**：极端内存压力下（`free_blocks < 5%`），KCMM 仍可降级换出受保护块

##### 逐层差异化策略

不同 Transformer 层的 KV Cache 特性不同：

| 层 | 特性 | 换出策略 |
|---|------|---------|
| 早期层 (L0-L7) | 更位置化，注意力模式更均匀，信息熵高 | 可以更激进地**量化后换出**（损失容忍度高） |
| 中间层 (L8-L23) | 语义丰富，信息密度最高 | **优先保留在 GPU**（对输出质量影响最大） |
| 后期层 (L24-L31) | 任务相关，注意力更集中 | 用注意力信号做**细粒度筛选**，部分块保护、部分换出 |

**优化手段：** 为不同层维护独立的换出优先级队列。内存压力时先从早期层开刀——这与 StreamingLLM 的发现一致：早期层的注意力模式更稳定，换出-恢复带来的信息损失最小。

#### 1.6.3 预测性操作

##### 投机预取的增强

文档中已提到"逻辑块 K 活跃 → 预取 K+1、K+2"，但这可以做得更精细：

1. **批量预取合并（Batch Prefetch Coalescing）：** 多个序列同时在 decode，它们的预取块可以合并为一次批量 H2D 传输，摊销 `cuMemMap` 延迟（`cuMemMap` 是已知瓶颈，vAttention 实测比简单 malloc 慢 ~115×）。

```rust
// 每次预取 tick：收集所有序列的预取候选，合并为一次批量操作
fn prefetch_tick(&mut self) {
    let mut batch: Vec<BlockHandle> = Vec::new();
    for seq in self.active_sequences() {
        batch.extend(seq.prefetch_candidates(self.prefetch_window));
    }
    self.streams.prefetch_batch_async(&batch); // 一次批量 cuMemMap + H2D
}
```

2. **勿预取"即将结束"的序列：** 如果 `max_tokens - current_tokens < prefetch_window`，该序列即将结束，不需要浪费 PCIe 带宽做预取。

3. **预取取消（Prefetch Cancellation）：** 如果预取传输尚未完成但序列已经结束（`free_blocks`），立即取消正在进行的 H2D memcpy 并回收已分配的 GPU 物理页。

##### 自适应水位线（Adaptive Watermark）

固定的低水位线（如"free < 20% 触发换出"）在不同负载下表现差异巨大：

```
负载类型 A — "水滴石穿"：大量短序列 → 自然释放快，水位线可以设低
负载类型 B — "洪峰型"：突发大量长序列 → 需要提前换出，否则来不及
```

**优化手段：** KCMM 追踪最近的分配/释放速率，计算 GPU 池耗尽时间的预测值，让换出操作在 **OOM 发生之前完成**，而不是被动响应：

```rust
fn adaptive_watermark(&self) -> f64 {
    let alloc_rate = self.metrics.alloc_rate_ewma;   // 指数加权移动平均
    let free_rate = self.metrics.free_rate_ewma;
    let net_drain_rate = alloc_rate - free_rate;

    if net_drain_rate <= 0.0 {
        return self.config.low_watermark; // 正常水位线
    }

    let time_to_oom = self.free_blocks as f64 / net_drain_rate;
    let eviction_latency = self.tiering.avg_eviction_latency();

    // 在 OOM 前提前触发，留足换出操作的时间窗口
    if time_to_oom < eviction_latency * SAFETY_FACTOR {
        return self.config.high_watermark; // 提前触发，更激进地换出
    }

    self.config.low_watermark
}
```

#### 1.6.4 传输层面的优化

##### 有损换出（Quantized Eviction）

KV Cache 对精度损失有一定容忍度，尤其是早期层和低重要性块。将块换出到 CPU 前做量化，可以显著降低传输带宽和 CPU 存储占用：

```
GPU HBM (FP16, 128KB/block)
    ↓  量化到 INT8 (64KB/block) 或 INT4 (32KB/block)
CPU DRAM (量化存储)
    ↓  反量化（恢复时）
GPU HBM (FP16, 128KB/block)
```

**收益分析：**

| 量化精度 | 传输带宽 | CPU 存储 | 精度风险 |
|---------|---------|---------|---------|
| 无量化 (FP16) | 1× | 1× | 无 |
| INT8 | 2× | 2× | 极低（广泛验证） |
| INT4 | 4× | 4× | 低（KV Cache 特化量化方案逐步成熟） |
| NF4 (如 QLoRA) | 4× | 4× | 低-中 |

**关键设计选择：**
- 量化/反量化在 **KCMM 的专用 CUDA Stream** 上完成，不阻塞推理计算
- 仅对**低重要性块**（按 1.6.2 的分级）做激进量化；Attention Sink 和 Heavy Hitters 保持无损换出
- 量化内核可以复用 vLLM/SGLang 已有的量化算子，或使用 CUDA 原生 `cublasLtMatmul` 的 INT8 路径

##### 流式恢复（Streaming Restore）

当前设计是"等整个块从 CPU 恢复到 GPU 后才返回给引擎"。可以改为流水线化：

```
标准恢复（当前设计）：
  [cuMemMap] [====== H2D 128KB ======] → 引擎可访问
                                   ↑ 整个传输完成

流式恢复（优化）：
  [cuMemMap] [= H2D 32KB =] → 引擎可访问块的前 25%
             [= H2D 32KB =] → 前 50%
             [= H2D 32KB =] → 前 75%
             [= H2D 32KB =] → 100%
              ↑ 引擎在第一批传输完成后即可开始计算
```

对于长上下文场景（块很大时如 128KB/block），流式恢复的 latency hiding 效果显著——引擎可以在第一批数据到位后立即开始 attention 计算，后续数据以流水线方式进入。

##### 写入缓冲与批量持久化

每次换出一个块立即执行 `cudaMemcpy D2H` + `cuMemUnmap` 会导致大量小粒度的 I/O 操作，无法充分利用 PCIe 带宽。

**优化手段——写入缓冲区（Write Buffer）：**

```rust
struct WriteBuffer {
    pending: Vec<BlockHandle>,    // 待换出的块
    staging: *mut u8,             // 暂存缓冲区（GPU 侧）
    threshold: usize,             // 如 16 个块或 2MB 总量
}

impl WriteBuffer {
    fn flush(&mut self) {
        // 批量 cuMemUnmap → 一次大粒度 cudaMemcpy D2H
        // NVMe 场景：顺序写 vs 随机写，吞吐量差 10× 以上
        self.streams.evict_batch_async(&self.pending);
        self.pending.clear();
    }
}
```

NVMe 层尤其受益——随机小写 vs 顺序大批量写，吞吐量差异可达 **10× 以上**。

#### 1.6.5 跨引擎的全局优化

这是 KCMM 作为独立 OS 服务的**独特优势**——单个引擎内部的置换策略无法获得全局视角。

##### 全局压力平衡

```
引擎 A 池: 90% 利用率，但大部分是已完成序列的冷块
引擎 B 池: 95% 利用率，全部是正在 decode 的热块

→ 传统 LRU（各自决策）: A 和 B 各自换出最冷的块
→ KCMM 全局决策: 从 A 换出更多（即使 B 的利用率更高）
  因为 A 的受害者块更"冷"，换出 A 的块不会影响任何活跃推理
```

**实现关键：** `TieringEngine` 维护所有注册池的全局 victim 优先级队列，而非每个池独立的队列。

##### 跨引擎块借用（Block Lending）

当引擎 A 空间充裕而引擎 B 面临 OOM 时，KCMM 可以在池之间临时借用物理块：

```
引擎 A (空闲，30 blocks free) ──→ 借用 10 blocks ──→ 引擎 B (OOM 危机)
                                                  ↓
                                           B 的 decode 完成后归还
```

"借用"只需要更新 KCMM 内部的映射表（所有权转移），比换出到 CPU 快几个数量级：
- 块借用延迟：~ns（纯内存操作，映射表更新）
- 换出-恢复延迟：~200μs（D2H + H2D + cuMemMap）

这要求 KCMM 的 `KcmmPool` 支持**物理块在池之间动态转移**，这是当前设计中未涵盖的能力。

#### 1.6.6 引擎提示接口（Hint API）

传统置换策略之所以"笨"，是因为它们只能被动观察访问模式——`touch` 和 `cool` 提供了比 OS 页表更多的信号，但仍然不够。推理引擎的调度器**知道得更多**，却无法将这些知识传递给内存管理层。

**当前 API（仅被动观察）：**

```c
kcmm_touch(pool, seq_id);   // "刚才访问了"
kcmm_cool(pool, seq_id);    // "现在空闲了"
```

**扩展 Hint API（主动告知，新增功能模块 I）：**

```c
// 生命周期提示
kcmm_hint(pool, seq_id, KCMM_HINT_MULTI_TURN);     // "这个会话还有后续，延迟换出"
kcmm_hint(pool, seq_id, KCMM_HINT_NEAR_END);        // "快结束了，优先受害者"
kcmm_hint(pool, seq_id, KCMM_HINT_SYSTEM_PROMPT);   // "系统提示词，缓存价值极高"

// 优先级提示
kcmm_hint(pool, seq_id, KCMM_HINT_HIGH_PRIORITY);   // "SLO 关键请求，保护其块"
kcmm_hint(pool, seq_id, KCMM_HINT_LOW_PRIORITY);     // "后台/批处理请求，可激进换出"

// 注意力信号提示
kcmm_hint(pool, block_ids, KCMM_HINT_ATTENTION_SINK); // "注意力沉没，永不换出"
kcmm_hint(pool, block_ids, KCMM_HINT_HEAVY_HITTER);   // "高注意力，受保护"
kcmm_hint(pool, block_ids, KCMM_HINT_EVICTABLE);      // "可丢弃，优先换出"
```

**设计原则：**
- Hint 是**建议性的**（advisory），不是强制命令——KCMM 在极端压力下仍可降级处理
- Hint 不改变 KCMM 的核心机制（块迁移、状态追踪、CUDA Stream 管理），只给策略层提供更丰富的决策信号
- 引擎侧不需要实现任何内存管理逻辑，只需在现有调度循环中插入 1-2 行 hint 调用

#### 1.6.7 优化手段汇总与优先级

```
                     ┌──────────────────────────────────────────────┐
 维度                 │ 优化手段                                      │
                     ├──────────────────────────────────────────────┤
 时间维度（传统）       │ LRU, LFU, FIFO, ARC                         │
 生命周期维度           │ 分层温控、显式回收、多轮对话休眠标记            │
 注意力维度             │ Attention Sink 保护、重要块分级、逐层差异化     │
 预测维度               │ 自适应水位线、投机预取增强、预取取消            │
 传输维度               │ 有损换出（量化）、流式恢复、写入缓冲             │
 跨引擎维度             │ 全局压力平衡、跨引擎块借用                     │
 引擎提示维度           │ Hint API（调度器 → KCMM 主动传递语义信号）     │
                     └──────────────────────────────────────────────┘
```

**建议的实现优先级（按步骤 3 可实现性排序）：**

| 优先级 | 优化手段 | 理由 |
|--------|---------|------|
| P0（步骤 3 即实现） | 分层温控 + 显式回收 | 改动小（仅扩展 `EvictionPolicy`），收益高 |
| P0 | 自适应水位线 | 纯 KCMM 内部逻辑，不依赖引擎侧改动 |
| P1（步骤 3 有额外时间） | Hint API 基础框架 | 定义 C ABI + trait，预留扩展空间 |
| P1 | 批量预取合并 | 与现有 `prefetch_stream` 设计兼容 |
| P2（步骤 4） | 跨引擎块借用 | 需要多池管理成熟后才有意义 |
| P2 | 注意力信号分级 + 有损换出 | 需要引擎侧配合输出注意力分数 |
| P3（未来探索） | 流式恢复、逐层差异化 | 更多工程复杂度，收益需实测验证 |

传统置换策略只用了第一行。KCMM 真正的竞争力来自**后面六行**——这些优化任何一个单独实现都不难，但由于它们散落在不同论文和工程实践中，且都依赖于对推理引擎内部状态的感知，现有系统很少有系统性地整合。KCMM 作为独立的内存管理服务的定位，恰好给了它一个可以同时运用所有这些信号的平台。

---

## 2. 需要新建的文件

### 2.1 `src/kcmm/mod.rs` — KCMM 顶层模块

```rust
pub mod pool;
pub mod superblock;
pub mod tiering;
pub mod sharing;
pub mod metrics;
pub mod ffi;
pub mod streams;

pub use pool::KcmmPool;
pub use ffi::*;
```

核心结构 `KcmmPool`：

```rust
pub struct KcmmPool {
    gpu_va_start: u64,
    gpu_va_size: usize,
    superblocks: Vec<Superblock>,
    free_blocks: VecDeque<BlockHandle>,
    sequences: HashMap<u64, SequenceState>,
    tiering: Option<TieringEngine>,
    sharing: Option<SharingManager>,   // 步骤 4 使用，步骤 3 预留
    metrics: KcmmMetrics,
    fragmentation_tracker: FragmentationTracker,
    streams: KcmmStreams,
}
```

### 2.2 `src/kcmm/pool.rs` — 池生命周期与块分配

从 `paged_kv.rs` 提取并泛化的核心分配逻辑：

- `KcmmPool::new(config)` — 创建池
- `kcmm_alloc_blocks(pool, seq_id, num_blocks)` — 分配块，自动处理恢复
- `kcmm_free_blocks(pool, seq_id, blocks)` — 释放块
- `kcmm_touch(pool, seq_id)` — 标记活跃
- `kcmm_cool(pool, seq_id)` — 标记可换出
- 低水位线检测 → 触发 TieringEngine 换出

### 2.3 `src/kcmm/superblock.rs` — 超级块管理

从 `cuda_vmm.rs` + `paged_kv.rs` 提取：

- `Superblock` 结构（物理句柄、VA 偏移、bitmap、块大小）
- `PhysicalBlockAllocator`（从 paged_kv.rs 移动）
- 超级块生命周期管理

### 2.4 `src/kcmm/tiering.rs` — 分层存储引擎

从 `swap.rs` 演进：

- `TieringEngine` 结构
- `EvictionPolicy` trait + `LruPolicy`、`LfuPolicy`、`FifoPolicy`
- `evict_blocks(count)` — 换出 N 个块
- `restore_blocks(handles)` — 恢复块
- `prefetch_worker` — 后台预取线程
- NVMe 层（可选）

### 2.5 `src/kcmm/sharing.rs` — 前缀共享管理器

**全新代码**（步骤 4 使用，步骤 3 预留接口）：

- `SharingManager` 结构
- `PrefixIndex` — 内容哈希 → 物理块引用
- `try_share_prefix()` — 检查是否存在可共享前缀
- `register_prefix()` — 注册新前缀
- 引用计数管理

### 2.6 `src/kcmm/metrics.rs` — UFS 指标

从 `unified_frag.rs` + `fragmentation_tracker.rs` 泛化：

- `KcmmMetrics` 结构（与 C API 的 `kcmm_metrics_t` 对应）
- 时间序列采样
- 跨引擎指标对比支持

### 2.7 `src/kcmm/ffi.rs` — C FFI API

**全新代码**，暴露给 `libkcmm.so`：

```c
kcmm_pool_t* kcmm_pool_create(size_t block_size, size_t max_blocks, const char* cpu_cache_path);
void kcmm_pool_destroy(kcmm_pool_t* pool);
int kcmm_alloc_blocks(kcmm_pool_t* pool, uint64_t seq_id, size_t num_blocks, uint32_t* out);
void kcmm_free_blocks(kcmm_pool_t* pool, uint64_t seq_id, const uint32_t* blocks, size_t num);
int kcmm_share_prefix(kcmm_pool_t* pool, uint64_t src, uint64_t dst, size_t n, uint32_t* out);
void kcmm_touch(kcmm_pool_t* pool, uint64_t seq_id);
void kcmm_cool(kcmm_pool_t* pool, uint64_t seq_id);
void kcmm_get_metrics(kcmm_pool_t* pool, kcmm_metrics_t* out);
```

### 2.8 `src/kcmm/streams.rs` — CUDA 流管理

**全新代码**：

- `KcmmStreams` 结构（evict / restore / prefetch 三个专用流）
- `CudaStream` 封装

---

## 3. 需要修改的现有文件

### 3.1 `src/config.rs` — 添加 KCMM 配置

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KcmmConfig {
    pub block_size: usize,         // 默认 65536（LLaMA-7B）
    pub max_blocks: usize,         // 默认 16384
    pub cpu_cache_path: String,    // 默认 "/dev/shm/kcmm_swap"
    pub tiering: bool,            // 默认 true
    pub eviction_policy: String,  // 默认 "lru"
    pub prefetch_window: usize,   // 默认 4
}
```

### 3.2 `src/cache/cuda_vmm.rs` — 添加 Stream 支持

- 添加 `CudaStream::new(flags)` 封装
- 添加 `cuda_memcpy_d2h_async()` / `cuda_memcpy_h2d_async()`
- 添加 `stream_synchronize()`
- 添加 `cuMemMap` 延迟批处理支持

### 3.3 `src/cache/paged_kv.rs` — 重构

- 将 `PhysicalBlockAllocator` 和 `SuperblockInfo` 提取到 `kcmm/superblock.rs`
- 保留 `PagedKvCache` 作为兼容层，启用 KCMM 时委托给 `KcmmPool`

### 3.4 `src/cache/swap.rs` — 重构

- `SwapManager` 演进为 `kcmm/tiering.rs` 的 `TieringEngine`
- 保留序列级 API 作为向后兼容的便捷层

### 3.5 `src/cache/mod.rs` — 更新导出

```rust
pub mod kv_cache;
pub mod paged_kv;
pub mod cuda_vmm;
pub mod swap;
pub mod fragmentation_tracker;
pub mod unified_frag;

pub use kv_cache::KvCache;
pub use paged_kv::{PagedKvCache, BLOCK_SIZE};
pub use swap::{EvictedSeqData, SwapManager, advance_epoch, current_epoch};
pub use unified_frag::{UnifiedFragMetrics, UnifiedFragSummary};
```

### 3.6 `src/lib.rs` — 添加 KCMM 模块

```rust
pub mod batch;
pub mod cache;
pub mod config;
pub mod cuda;
pub mod decoder;
pub mod kcmm;     // 新增
pub mod model;
pub mod server;

pub use config::{ModelConfig, ServerConfig, KcmmConfig};
```

### 3.7 `Cargo.toml` — 依赖变更

```toml
[dependencies]
bytes = "1.9"             # 零拷贝缓冲区管理

[lib]
crate-type = ["lib", "cdylib"]  # 新增 cdylib 用于生成 libkcmm.so
```

### 3.8 额外需要补充的脚本

- `scripts/kcmm_vllm_patch.py` — Python 绑定，monkey-patch vLLM 块分配器
- `src/trace/kcmm_events.bt` — bpftrace 脚本，追踪 KCMM 换出/恢复事件

---

## 4. 实现顺序

按照 detailed plan 第 13–18 周的安排：

| 周次 | 任务 | 涉及文件 |
|------|------|---------|
| **13** | 创建 `src/kcmm/` 模块骨架；提取 `PhysicalBlockAllocator` → `kcmm/superblock.rs`；泛化 `PagedKvCache` → `KcmmPool` | `kcmm/mod.rs`, `kcmm/pool.rs`, `kcmm/superblock.rs`, `kcmm/streams.rs` |
| **14** | 实现 `BlockLocation` 追踪；添加 `EvictionPolicy` trait；实现 LRU 换出队列 | `kcmm/pool.rs`, `kcmm/tiering.rs` |
| **15** | 实现 GPU→CPU 换出 + 恢复完整循环；实现 NVMe 层（可选） | `kcmm/tiering.rs`, `cache/swap.rs`（重构） |
| **16** | 构建 `libkcmm.so` + C API | `kcmm/ffi.rs`, `Cargo.toml` |
| **17** | vLLM KCMM 集成（Python 绑定 + monkey-patch） | `scripts/kcmm_vllm_patch.py` |
| **18** | KCMM 评估（内存压力、LRU 对比、CUDA Stream 开销） | 基准测试脚本 |

### 向后兼容性

- 独立 Rust 推理引擎的行为**不变**：`cargo run -- --continuous --model-path ...` 仍使用原有的 `PagedKvCache`（不链接 KCMM）
- KCMM 作为可选 feature flag 启用
- `src/cache/` 模块保持功能完整

### 关键风险

| 风险 | 缓解措施 |
|------|---------|
| 块粒度换出复杂度：当前 swap.rs 以序列为单位操作 | 先实现单块换出，再优化为批量操作 |
| CUDA Stream 同步：换出/恢复流需与推理流正确同步 | 使用 CUDA Event 进行流间同步，显式测试 |
| BlockLocation 状态机正确性：5 状态转换需处理并发 | 使用细粒度锁 + 状态转换断言；proptest 模糊测试 |
| vLLM 内部 API 变化 | 固定 vLLM 版本；先用自定义 Rust 引擎验证功能 |
| cuMemMap 在热路径上的延迟 | 批处理调用；延迟解除映射；使用专用 Stream |

---

## 5. Benchmark 设计方案

### 5.1 Benchmark 分层架构

```
benchmarks/kcmm/
├── micro/                    # 微基准测试（无推理，纯 KCMM API）
│   ├── bench_alloc.rs        # 块分配/释放吞吐量
│   ├── bench_tiering.rs      # 换出/恢复延迟
│   ├── bench_streams.rs      # CUDA Stream 开销
│   └── bench_policies.rs     # 换出策略命中率
├── integration/              # 集成基准测试（Rust 推理引擎 + KCMM）
│   ├── bench_memory_pressure.rs  # 内存压力场景
│   └── bench_ufs_compare.rs      # UFS 指标对比
├── system/                   # 系统级基准测试（vLLM + KCMM）
│   ├── bench_vllm_kcmm.py    # vLLM + KCMM vs vLLM 原生
│   └── bench_concurrency.py  # 并发扩展测试
└── scripts/
    ├── run_micro_bench.sh
    ├── run_integration_bench.sh
    └── run_system_bench.sh
```

---

### 5.2 微基准测试（Micro-benchmarks）

这些是最先需要实现的——**不依赖推理引擎**，直接测试 KCMM API 的性能。

#### Benchmark 1: 块分配/释放吞吐量

**目的：** 衡量 KCMM 在最简单路径（不分层、不换出）上的原始性能，对比 vLLM 内部分配器。

```
场景：预分配 N 个块（无分层存储）
操作：循环 { alloc_blocks(seq_i, 1) → free_blocks(seq_i) }
变量：
  - 块大小: 32KB, 64KB, 128KB（对应不同模型）
  - 池大小: 1024, 4096, 16384 块
指标：
  - alloc_blocks p50/p99 延迟 (ns)
  - free_blocks p50/p99 延迟 (ns)
  - 吞吐量 (alloc+free ops/sec)
基线对比：
  A. vLLM CpuGpuBlockAllocator.allocate() / free()
  B. KCMM kcmm_alloc_blocks() / kcmm_free_blocks()（无分层）
成功标准：KCMM < vLLM × 1.05（回退 < 5%）
```

#### Benchmark 2: 分层存储换出/恢复延迟

**目的：** 测量 GPU↔CPU 数据迁移的单块延迟。这是 KCMM 最关键的性能指标。

```
场景：填满 GPU 池 → 触发换出 → 后续分配触发恢复
变量：
  - 块大小: 32KB, 64KB, 128KB
  - 每次换出块数: 1, 4, 16, 64（批量 vs 单块）
  - 使用专用 CUDA Stream vs 默认流
指标：
  - 换出延迟 p50/p99 (μs)：cudaMemcpy D2H + cuMemUnmap
  - 恢复延迟 p50/p99 (μs)：cuMemMap + cudaMemcpy H2D
  - 批量换出 vs 单块换出的摊销延迟
  - cuMemMap 延迟 p50/p99（单独测量，因其是已知瓶颈）
成功标准：单块恢复 < 200μs (p50)
```

#### Benchmark 3: CUDA Stream 开销

**目的：** 量化专用流对推理计算的干扰。

```
场景：同时运行推理 kernel（模拟）和 KCMM 换出/恢复操作
对比：
  A. 换出/恢复使用默认流（与推理共享）
  B. 换出/恢复使用专用 CUDA Stream（KCMM 设计）
  C. 换出/恢复使用专用流 + CU_STREAM_NON_BLOCKING
指标：
  - 推理 kernel 执行时间（应不受影响）
  - 换出/恢复完成时间
  - GPU 利用率时间线
成功标准：专用流方案推理 kernel 时间增长 < 1%
```

#### Benchmark 4: 换出策略命中率

**目的：** 不依赖真实推理负载，使用合成访问模式评估换出策略。

```
场景：模拟 LRU/LFU/FIFO 访问模式
  - 热-冷交替：80% 的访问集中在 20% 的序列
  - 顺序扫描：每个序列按顺序访问一次
  - 随机访问：均匀随机
  - Zipf 分布：幂律访问模式（最接近真实）
指标：
  - 命中率 = alloc时块已 GPU 常驻的比例
  - 每次未命中的恢复延迟
  - 换出决策时间（选择受害者块的开销）
对比策略：
  A. LRU（KCMM 默认）
  B. LFU
  C. FIFO
  D. Oracle（最优，预知未来访问）
成功标准：LRU 命中率 ≥ Oracle 的 85%
```

---

### 5.3 集成基准测试（Integration Benchmarks）

需要**Rust 推理引擎 + KCMM**，测量端到端推理场景下的 KCMM 表现。

#### Benchmark 5: 内存压力下的分层收益

这是步骤 3 的**核心实验**。

```
环境：A30 (24GB) + LLaMA-7B (~14GB 权重 → ~10GB KV Cache)
模型：LLaMA-7B (或 TinyLLaMA 用于快速迭代)

负载设计（关键）—— 目标：构造总 KV Cache 需求 > GPU 可用容量：

  方法 1：固定 max_tokens，递增并发数
    每个请求 max_tokens = 2048
    并发数 = 1, 2, 4, 8, 16, 32, 64, 128（直到 OOM）

  方法 2：固定并发数，递增 max_tokens
    并发数 = 32
    max_tokens = 256, 512, 1024, 2048, 4096

  方法 3：ShareGPT 真实对话 trace
    使用 ShareGPT 数据集重放，不限并发

对比配置：
  A. vLLM 默认（无 swap）—— 预期在 ~80 并发 OOM
  B. vLLM + vLLM 内置 swap（GPU→CPU）
  C. KCMM 仅 GPU→CPU 分层
  D. KCMM GPU→CPU→NVMe 分层（如 NVMe 可用）

指标（按并发度分组）：

  主要：
  - 最大可接纳并发数（OOM 前的并发上限）
  - TTFT p50/p99 (ms)
  - TPOT p50/p99 (ms/token)
  - 总吞吐量 (tokens/sec)

  内存：
  - GPU 已分配块数 / 总量
  - CPU swap 使用量 (MB)
  - NVMe swap 使用量 (MB，如适用)
  - 换出次数 / 恢复次数
  - 每请求平均换出/恢复次数

  UFS 指标：
  - IFR（内碎片率）
  - PME（物理内存效率）
  - BU（块利用率）
  - RFI（运行时碎片指数）

  CPU：
  - 系统 CPU%
  - 用户 CPU%
  - 上下文切换/秒
  - 换出线程 CPU 使用率

成功标准：
  - KCMM 比 vLLM 无 swap 多接纳 ≥ 30% 并发
  - 无分层时吞吐量回退 < 5%（与 vLLM 内部分配器对比）
  - UFS 指标与 vLLM 内部分配器等效（无分层时）
```

#### Benchmark 6: UFS 指标对比（无分层）

**目的：** 验证 KCMM 在正常负载（不触发换出）下，内存效率指标与 vLLM 内部分配器一致。

```
环境：GPU 内存充足（不触发换出）
负载：
  - 合成均匀负载：32 并发，max_tokens=512
  - 合成变长负载：32 并发，max_tokens 均匀分布在 [128, 1024]
  - ShareGPT 负载

对比：
  A. vLLM 原生 CpuGpuBlockAllocator
  B. KCMM 无分层模式（仅块分配器，不启用 tiering）

指标（时间序列，每秒采样一次）：
  - IFR 时间序列 → avg, peak, stddev
  - PME 时间序列 → avg, min, stddev
  - BU 时间序列 → avg, min, stddev
  - RFI 时间序列 → avg, peak, stddev

成功标准：所有指标在 KCMM 和 vLLM 之间偏差 < 2%（绝对值）
```

---

### 5.4 系统级基准测试（System Benchmarks）

需要 **vLLM + KCMM monkey-patch**。

#### Benchmark 7: vLLM + KCMM 端到端对比

```
环境：A30 + LLaMA-7B，vLLM 固定版本
负载：
  - 合成：128/512/2048 token 定长提示词
  - ShareGPT：真实对话
  - 突发：泊松到达，1→64→1 并发渐变

配置矩阵：
  ┌─────────────────┬──────┬──────────┬──────────┐
  │                 │ 无swap│ vLLM swap│ KCMM     │
  ├─────────────────┼──────┼──────────┼──────────┤
  │ 低内存压力(8并发) │  ✓   │    ✓     │    ✓     │
  │ 中内存压力(32并发)│  ✓   │    ✓     │    ✓     │
  │ 高内存压力(64并发)│  OOM │    ✓     │    ✓     │
  │ 极限压力(128并发) │  OOM │    OOM?  │    ✓     │
  └─────────────────┴──────┴──────────┴──────────┘

指标（同 Benchmark 5）+ 额外：
  - Token 精确匹配率（正确性：KCMM 下输出 token 必须与 vLLM 一致）
  - 首个 token 延迟分解（网络 → 队列 → prefill → 首个 token）
  - 请求超时率（TTFT > 阈值 的比例）
```

#### Benchmark 8: 换出策略对比（真实负载）

```
环境：vLLM + KCMM, LLaMA-7B
负载：高内存压力（64 并发, max_tokens=2048）
对比策略：
  A. LRU
  B. LFU
  C. FIFO
  D. Oracle（事后分析最优策略的理论命中率）

指标：
  - 命中率
  - 平均恢复延迟
  - 吞吐量
  - 换出/恢复操作比例
  - 各策略下 TTFT 分布对比

成功标准：LRU 命中率 ≥ Oracle 的 85%
```

---

### 5.5 Benchmark 实现阶段

#### 阶段 1（第 13–14 周）：微基准测试

此时 KCMM 核心（分配器、BlockLocation）刚完成：

- `bench_alloc.rs` — 不需要分层存储
- `bench_streams.rs` — 测试 CUDA Stream 基础设施

#### 阶段 2（第 14–15 周）：分层存储微基准

换出/恢复实现后：

- `bench_tiering.rs` — 单块换出/恢复延迟
- `bench_policies.rs` — 换出策略命中率

#### 阶段 3（第 16–17 周）：集成基准 + 系统基准

KCMM 完整可用 + vLLM 集成后：

- `bench_memory_pressure.rs` — 内存压力端到端
- `bench_ufs_compare.rs` — UFS 指标对比
- `bench_vllm_kcmm.py` — vLLM 端到端对比

---

### 5.6 关键 Benchmark 伪代码

#### `bench_tiering.rs` — 最重要、最先做的微基准

```rust
// 测量单块换出/恢复延迟
fn bench_single_block_evict_restore() {
    let pool = KcmmPool::new(KcmmConfig {
        block_size: 65536,
        max_blocks: 1024,
        tiering: true,
        ..
    });

    // 填满 GPU 池
    for i in 0..N_SEQS {
        pool.alloc_blocks(i, BLOCKS_PER_SEQ);
    }

    // 测量换出延迟（单块）
    let start = Instant::now();
    pool.evict_blocks(1);
    let evict_latency = start.elapsed();

    // 测量恢复延迟
    let start = Instant::now();
    pool.alloc_blocks(NEW_SEQ, 1);  // 触发恢复
    let restore_latency = start.elapsed();

    // 批量换出
    let start = Instant::now();
    pool.evict_blocks(64);
    let batch_evict_latency = start.elapsed();

    println!("evict_1={:?}, restore_1={:?}, evict_64={:?}, amortized_evict={:?}",
        evict_latency, restore_latency, batch_evict_latency,
        batch_evict_latency / 64);
}
```

#### `bench_memory_pressure.rs` — 核心端到端实验

```rust
// 内存压力下 KCMM vs 无 swap 对比
fn bench_memory_pressure() {
    let model = load_llama_7b();
    let gpu_mem_for_kv = 10 * 1024 * 1024 * 1024; // ~10 GB

    for concurrency in [1, 2, 4, 8, 16, 32, 64, 128] {
        let kv_demand = concurrency * 2048_tokens * bytes_per_token;

        let configs = if kv_demand > gpu_mem_for_kv {
            vec!["kcmm_tiering"]  // vLLM 无 swap 会 OOM
        } else {
            vec!["vllm_no_swap", "vllm_swap", "kcmm_tiering"]
        };

        for config in configs {
            let results = run_benchmark(model, concurrency, config);
            record(concurrency, config, results);
            // results: { ttft_p50, ttft_p99, tpot, throughput,
            //            gpu_blocks_used, cpu_swap_mb,
            //            eviction_count, restore_count,
            //            ifr, pme, bu, rfi }
        }
    }
}
```

---

## 6. Benchmark 清单与优先级

| # | 名称 | 类型 | 依赖 | 优先级 | 成功标准 |
|---|------|------|------|--------|---------|
| 1 | 块分配/释放吞吐量 | 微基准 | KCMM 核心 | **P0** | 回退 < 5% vs vLLM |
| 2 | 分层存储换出/恢复延迟 | 微基准 | TieringEngine | **P0** | 单块恢复 < 200μs p50 |
| 3 | CUDA Stream 开销 | 微基准 | KcmmStreams | P1 | 推理 kernel 影响 < 1% |
| 4 | 换出策略命中率（合成） | 微基准 | TieringEngine | P1 | LRU ≥ Oracle 85% |
| 5 | 内存压力分层收益 | 集成 | KCMM + Rust 引擎 | **P0** | 多接纳 ≥ 30% 并发 |
| 6 | UFS 指标对比（无分层） | 集成 | KCMM + Rust 引擎 | P1 | 偏差 < 2% |
| 7 | vLLM + KCMM 端到端对比 | 系统 | libkcmm.so + vLLM | **P0** | Token 精确匹配 + 回退 < 5% |
| 8 | 换出策略对比（真实负载） | 系统 | libkcmm.so + vLLM | P1 | LRU ≥ Oracle 85% |

**P0** = 步骤 3 必须完成的基准测试，**P1** = 有额外时间时完成。

### 建议执行顺序

建议先从 **Benchmark 1**（分配吞吐量）和 **Benchmark 2**（换出/恢复延迟）开始——它们不需要推理引擎，可以在 KCMM 核心代码写完后立即运行，快速验证设计。这两个 Benchmark 的结果将直接决定后续的优化方向。
