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
| ✅ 所有纯 Rust 代码开发 | ❌ 获得有意义的性能绝对值（GPU-PV 层 `cuMemAlloc_v2` P99=12.96ms vs bare-metal ~10-30µs，差距 400-1000×） |
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

### 阶段一：Rust 引擎集成（第 1-4 周）

**目标：** 让 KCMM 的 tiering 在真实推理路径上端到端跑通，生成 A/B 对比数据。

**总体策略：** 通过引入 `KvCacheBackend` trait 将 `PagedKvCache` 和 `KcmmPool` 统一到同一抽象层，使 `Transformer` trait 和 `ContinuousScheduler` 同时支持两种后端。这避免了代码分叉（两套 scheduler / 两套 forward path），并使用编译时或运行时多态来切换。

#### 1.1 API 补齐：让 `KcmmPool` 与 `PagedKvCache` 接口对齐（~3 天）

**现状：** `KcmmPool` 的公共 API 与 `PagedKvCache` 高度相似（`alloc_block`、`alloc_sequence`、`free_sequence`、`register_sequence`、`unregister_sequence`、`update_seq_len`、`va_k`/`va_v` 等），但缺失以下方法：

- **`get_all_block_offsets_f16()`** — 返回所有 block 的 VA 偏移（以 f16 元素为单位），供 paged attention kernel 使用
- **`append_kv_step(layer_idx, seq_indices, positions, k_src, v_src)`** — 将 KV 投影写入分页缓存，供 `forward_step_paged` 调用
- **`seq_metadata` → 统一为 trait 方法** — `PagedKvCache` 通过 `pub seq_metadata: Mutex<Vec<SeqMetadata>>` 暴露内部字段，`KcmmPool` 使用 `sequences: Mutex<Vec<SequenceState>>`。需要提供 trait 兼容的访问器方法（如 `with_seq_metadata()`/`seq_block_table()`/`seq_len()`）

**具体改动：**
- `pool.rs`: 移植 `append_kv_step` 逻辑（从 `paged_kv.rs` 复制，内部结构一致）
- `pool.rs`: 添加 `get_all_block_offsets_f16()` 方法
- `pool.rs`: 添加 `with_seq_metadata<R>(&self, f: impl FnOnce(&[SeqMetadata]) -> R) -> R` 风格的闭包访问器，或直接暴露 `seq_metadata()` 返回迭代器
- 单元测试覆盖新增方法

#### 1.2 定义 `KvCacheBackend` trait（~2 天）

**现状：** `Transformer` trait 中 `forward_step_paged` 的签名硬编码了 `&PagedKvCache`，无法直接传入 `KcmmPool`。

**方案：** 在 `src/cache/` 下新建 `backend.rs`，定义：

```rust
pub trait KvCacheBackend: Send + Sync {
    // Block allocation
    fn alloc_block(&self) -> Result<u32>;
    fn alloc_sequence(&self, num_blocks: usize) -> Result<Vec<u32>>;
    fn free_sequence(&self, block_table: &[u32]);
    fn append_block_to_sequence(&self, seq_idx: usize, block_idx: u32);

    // Sequence management
    fn register_sequence(&self, block_table: Vec<u32>) -> usize;
    fn unregister_sequence(&self, seq_idx: usize);
    fn update_seq_len(&self, seq_idx: usize, len: usize);
    fn get_seq_len(&self, seq_idx: usize) -> usize;
    fn get_block_table(&self, seq_idx: usize) -> Option<Vec<u32>>;

    // VA layout
    fn va_k(&self, layer: usize) -> u64;
    fn va_v(&self, layer: usize) -> u64;
    fn get_block_va_offset(&self, block_idx: u32) -> Option<usize>;
    fn get_all_block_offsets_f16(&self) -> Vec<u64>;

    // KV write (used by forward_step_paged)
    fn append_kv_step(&self, layer_idx: usize, seq_indices: &[usize],
        positions: &[usize], k_src: &CudaSlice<f16>, v_src: &CudaSlice<f16>) -> Result<()>;

    // Config accessors
    fn block_size(&self) -> usize;
    fn max_blocks_per_seq(&self) -> usize;

    // Pool stats
    fn blocks_in_use(&self) -> usize;
    fn has_free_blocks(&self) -> bool;

    // Sequence metadata access (for paged attention kernel)
    fn with_seq_metadata<R>(&self, f: impl FnOnce(&[SeqMetadata]) -> R) -> R;
}
```

- `PagedKvCache` 和 `KcmmPool` 均 `impl KvCacheBackend`
- 对于 `PagedKvCache`，大部分方法已有，只需添加 `with_seq_metadata`
- 对于 `KcmmPool`，`SequenceState` 需转换为 `SeqMetadata` 或直接在 trait 中返回 `block_table` + `seq_len` 的引用

**关键决策点：** `with_seq_metadata` 的闭包方式持有锁期间暴露引用，可能导致死锁。备选方案是提供 `seq_block_table(seq_idx) -> Option<&[u32]>` 的单独访问器，避免暴露内部 Mutex。具体选择在实现时根据锁竞争模式决定。

#### 1.3 修改 `Transformer` trait 支持 trait object（~1 天）

- 将 `forward_step_paged` 签名从 `cache: &PagedKvCache` 改为 `cache: &dyn KvCacheBackend`
- `NaiveTransformer` 和 `LlamaTransformer` 的实现同步修改
- 将所有 `cache.seq_metadata.lock()` 直接访问替换为 `cache.with_seq_metadata()`
- 将所有 `cache.block_size` / `cache.max_blocks_per_seq` 字段访问替换为 `cache.block_size()` / `cache.max_blocks_per_seq()`

#### 1.4 重构 `ContinuousScheduler` 支持 KCMM（~3 天）

**现状：** `ContinuousScheduler` 内部有两套 eviction 机制与 KCMM 冲突：
- `SwapManager` — 自己的 swap-out/restore 逻辑 → 替换为 KCMM `TieringEngine`
- `seq_last_epoch: HashMap<usize, u64>` + `select_victim()` → 替换为 KCMM `touch()`/`cool()` + `EvictionPolicy`

**具体改动：**

1. **`main.rs` KCMM 路径选择：**
   - 在 `--continuous` 模式下，通过 `--kcmm` CLI flag 选择后端
   - `kcmm` feature flag（`Cargo.toml` 已定义 `kcmm = []`）控制编译时包含 KCMM 依赖
   - 实现 `fn create_cache_backend(…) -> Arc<dyn KvCacheBackend>` 工厂函数

2. **`ContinuousScheduler` 重构：**
   - 将 `cache: Arc<PagedKvCache>` 改为 `cache: Arc<dyn KvCacheBackend>`
   - 将 `swap_manager: SwapManager` 替换为 `kcmm_enabled: bool` 标记
   - 替换 `select_victim()` → 当 `kcmm_enabled` 时调用 `TieringEngine::evict_blocks()`
     - `admit_waiting()` 中 OOM 路径：KCMM 模式走 `tiering.evict_blocks()` → `TieringEngine` 自动换出
     - `try_restore_swapped()` 中恢复逻辑：KCMM 模式调用 `pool.restore_evicted_blocks()`
   - 插入 `pool.touch(seq_idx)` / `pool.cool(seq_idx)` 调用
     - `touch()`: 每次 `run_step()` 对所有 running sequence 调用
     - `cool()`: 序列完成（`remove_completed`）时调用，标记 block 为首选受害者
   - 处理 `CpuResident` 块自动恢复：在 `run_step()` 之前检查 block 位置，若为 `CpuResident` 则触发 restore

3. **保持向后兼容：**
   - 非 KCMM 模式使用 `PagedKvCache` + 原有 `SwapManager`（行为不变）
   - KCMM 模式使用 `KcmmPool` + `TieringEngine`（新代码路径）

#### 1.5 正确性验证（~2 天）

- **GPU KV 写入正确性：** 使用 `NaiveTransformer`（zero weights）+ `KcmmPool` 后端，验证 `append_kv_step` 写入后 block 数据与 `PagedKvCache` 后端一致
- **Token 序列等价性：** 固定 seed + 相同 prompt，对比 `PagedKvCache` vs `KcmmPool`（tiering 关闭）的生成 token 序列。当前推理引擎使用 `greedy_sample`（无随机性），zero-weight 模型下 token 序列完全确定（始终输出 token 0）
- **Tiering 数据完整性：** 在内存压力场景下触发 eviction → restoration，验证 GPU↔CPU roundtrip 后 block 内容无损
  - 通过写入已知 pattern → evict → restore → 读出对比
- **Scheduler 状态机正确性：** 验证 KCMM 路径下序列生命周期（admit → touch → decode → cool → evict → restore）不丢失 block、不 double-free

#### 1.6 集成 Benchmark（~2 天）

- **新 Benchmark：`kcmm_engine_integration`**
  - 在 WSL2 上使用 `NaiveTransformer` + `KcmmPool`（tiering on）跑多请求并发
  - 对比指标：`PagedKvCache` vs `KcmmPool`（tiering off）的 throughput、延迟分布
  - 记录 KCMM tiering 启用时的 eviction count、restore count、per-step latency overhead
- **复用已有 Benchmark 基础设施：**
  - Benchmark 1-5 已独立验证 KCMM 各组件的正确性；集成阶段重点是端到端数据
  - 使用 `tests/kcmm_bench_memory_pressure.rs` 的内存压力生成模式，嵌入到集成测试中

### 阶段二：C FFI API 实现（第 4-6 周）

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

## 4. Bare-Metal 计划

Bare-metal 环境的完整验证与优化计划已独立为专项文档，详见：

→ **[`docs/dev/kcmm-baremetal-plan.md`](kcmm-baremetal-plan.md)**

该文档涵盖：
- **基准测试复现**：`run_kcmm_benches.sh` 和 `run_kcmm_integration_bench.sh` 在 A30 上的完整运行计划
- **换出策略对比**：LRU / LFU / FIFO 在真实负载下的命中率与延迟对比
- **高级换出策略**：分层温控、自适应水位线、Hint API、有损换出、流式恢复等 P0/P1 优化的实现与测试方案
- **文档与可观测性**：bpftrace 追踪脚本、API 使用指南、时间序列指标采样、开发文档
- **Benchmark 完善**：UFS 指标对比、策略命中率增强、per-sequence latency、真实 trace 回放

## 5. 里程碑检查点

```
Week 1:   KcmmPool API 补齐（append_kv_step, get_all_block_offsets_f16, with_seq_metadata）
Week 2:   KvCacheBackend trait 定义 + Transformer trait 重构完成
Week 3:   ContinuousScheduler KCMM 适配完成，touch/cool 路径跑通
Week 4:   正确性验证通过（token 等价性 + tiering roundtrip），集成 benchmark 数据产出
Week 6:   (阶段二) libkcmm.so 符号导出，C 测试程序通过
```

> **注意：** 阶段三（策略进阶优化）、阶段四（文档与可观测性）、阶段五（Benchmark 完善）已迁移至 Bare-Metal 计划 → [`docs/dev/kcmm-baremetal-plan.md`](kcmm-baremetal-plan.md)。

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
