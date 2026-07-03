# KCMM 下一步实现计划

**日期:** 2026-06-05
**状态:** 第 13 周骨架完成，进入第 14-15 周核心实现
**参考:** `docs/task/kcmm-implementation-analysis.md`、`docs/task/kcmm-related-research.md`、`docs/task/detailed-plan.md`

---

## 一、当前实现状态总览

### 1.1 已完成（第 13 周，全部 37 个测试通过）

| 功能编号 | 功能 | 文件 | 行数 | 状态 |
|---------|------|------|------|------|
| A1 | `KcmmPool` 创建与销毁 | `pool.rs` | 1053 | ✅ 完成 |
| A2 | 块分配（`alloc_block`/`alloc_sequence`） | `pool.rs` | — | ✅ 完成 |
| A3 | 块释放（`free_sequence`） | `pool.rs` | — | ✅ 完成 |
| A4 | 序列注册与追踪（`SequenceState`） | `pool.rs` | — | ✅ 完成 |
| A5 | `touch` / `cool` 操作 | `pool.rs` | — | ✅ 完成 |
| A6 | 低水位线检测 | `pool.rs` | — | ✅ 完成 |
| A7 | `PhysicalBlockAllocator` 提取 | `superblock.rs` | 368 | ✅ 完成 |
| D1-D5 | CUDA Stream 封装 + 三流管理 | `streams.rs` | 90 | ✅ 完成 |
| B4 骨架 | CPU 缓冲区 mmap | `tiering.rs` | 343 | ⚠️ 仅骨架 |
| H1-H3 | 配置 + feature flag + cdylib | `config.rs`/`Cargo.toml` | — | ✅ 完成 |
| G1-G2 | 指标结构与采集 | `metrics.rs` + `pool.rs::collect_metrics()` | 83 | ✅ 完成 |

### 1.2 仅骨架（类型/接口已定义，逻辑为空）

| 模块 | 已有 | 缺失 |
|------|------|------|
| `tiering.rs` — `EvictionPolicy` trait | trait 定义完整（`select_victims`、`on_access`、`on_evict`） | 三个策略实现全部返回空 `Vec` |
| `tiering.rs` — `TieringEngine` | mmap CPU 缓冲区、文件持久化、`Send+Sync` | 无 `evict_blocks()`、无 `restore_blocks()` |
| `ffi.rs` | 类型定义（`kcmm_pool_t`、`kcmm_metrics_t`、`kcmm_hint_t`） | **所有 10 个 `extern "C"` 函数被注释掉** |
| `sharing.rs` | 结构体 + 接口签名 | 所有方法为 `// Placeholder`（Step 4） |

### 1.3 完全未开始

- B2 块粒度 GPU→CPU 换出逻辑
- B3 块粒度 CPU→GPU 恢复逻辑
- B5 批量换出/恢复优化
- B6 NVMe 层
- B7 超级块碎片整理
- C2-C5 三种换出策略完整实现
- D6 异步 memcpy 封装（`cuda_memcpy_d2h_async` / `cuda_memcpy_h2d_async`）
- F1-F10 C FFI 函数实现
- H4 vLLM Python 绑定
- H5 bpftrace 追踪脚本
- 全部 8 个 Benchmark

---

## 二、核心差距详解

### 差距 1：分层存储引擎 — 换出/恢复逻辑完全空白

这是整个 KCMM 最核心的差异化能力，也是实现复杂度最高的部分。

```
当前 TieringEngine 的能力矩阵：
  ✅ mmap CPU 缓冲区（/dev/shm/kcmm_swap, MAP_SHARED）
  ✅ 文件持久化验证（写入 → 重新打开 → 读回比对）
  ✅ Drop 时自动 munmap
  ❌ evict_blocks(count) -> Vec<BlockHandle>       — 不存在
  ❌ restore_block(block_idx) -> ()                 — 不存在
  ❌ 块粒度的 cudaMemcpy D2H（使用专用 evict 流）     — 不存在
  ❌ 块粒度的 cudaMemcpy H2D（使用专用 restore 流）   — 不存在
  ❌ cuMemUnmap 释放 GPU 物理页                      — 不存在
  ❌ cuMemMap 重新映射 GPU 物理页                     — 不存在
  ❌ BlockLocation 状态转换验证                       — 枚举已定义，无转换逻辑
  ❌ CPU 缓冲区槽位分配/释放                          — 无 alloc_cpu_slot() / free_cpu_slot()
  ❌ 批量换出/恢复（批量 cudaMemcpy + 批量 cuMemMap）  — 不存在
  ❌ NVMe 层（GDS 或标准 I/O）                       — 不存在
```

**设计要点：**
- `TieringEngine` 当前不持有对 `KcmmPool` 的引用，这意味着它无法访问 GPU VA 区域和 CUDA VMM 句柄
- 换出/恢复操作需要 `TieringEngine` 与 `KcmmPool` 之间的双向调用

### 差距 2：置换策略 — 接口已有，逻辑全空

```rust
// 当前 tiering.rs 中三个策略的 select_victims() 全部是：
fn select_victims(&self, _candidates: &[BlockHandle], _count: usize) -> Vec<BlockHandle> {
    Vec::new()  // ← 占位符
}
```

**需要实现的核心逻辑：**

| 策略 | 需要维护的数据 | `select_victims` 逻辑 |
|------|-------------|---------------------|
| `LruPolicy` | 每块的 `last_access: Instant` | 按 `last_access` 升序（最旧优先），取前 `count` 个 |
| `LfuPolicy` | 每块的 `access_count: u64` | 按 `access_count` 升序（最少访问优先），取前 `count` 个 |
| `FifoPolicy` | 每块的 `alloc_time: Instant` | 按 `alloc_time` 升序（最早分配优先），取前 `count` 个 |

### 差距 3：C FFI 函数 — 类型定义完成，无实现

```rust
// ffi.rs 中所有关键函数都只是注释：
// extern "C" {
//     pub fn kcmm_pool_create(...)
//     pub fn kcmm_alloc_blocks(...)
//     pub fn kcmm_free_blocks(...)
//     pub fn kcmm_touch(...)
//     pub fn kcmm_cool(...)
//     pub fn kcmm_get_metrics(...)
//     pub fn kcmm_share_prefix(...)
// }
```

---

## 三、下一步实现计划（按优先级排序）

### 阶段 A：置换策略实现（预计 2 天）

#### A.1 让 `TieringEngine` 持有对 `KcmmPool` 关键组件的引用

当前 `TieringEngine` 不持有任何对 `KcmmPool` 的引用。为实现换出/恢复，需要建立以下数据通路：

```rust
// TieringEngine 需要新增的字段（在 evict/restore 方法中作为参数传入也可）
struct TieringEngine {
    // 现有字段保留...
    
    // 新增：GPU VA 引用（从 KcmmPool 传入）
    // 换出/恢复时通过方法参数传入，避免循环引用
}
```

**设计决策：** 采用**方法参数传递**而非持有引用，避免 `Arc` 循环引用问题。`TieringEngine` 的方法通过参数接收 `&KcmmPool` 或必要的组件引用。

#### A.2 实现 `LruPolicy`

```rust
use std::collections::HashMap;
use std::time::Instant;

pub struct LruPolicy {
    /// BlockHandle → last access timestamp.
    access_times: Mutex<HashMap<BlockHandle, Instant>>,
}

impl EvictionPolicy for LruPolicy {
    fn select_victims(&self, candidates: &[BlockHandle], count: usize) -> Vec<BlockHandle> {
        let times = self.access_times.lock();
        let mut sorted: Vec<_> = candidates.iter()
            .filter_map(|h| times.get(h).map(|t| (*h, *t)))
            .collect();
        sorted.sort_by_key(|(_, t)| *t);  // 最旧优先
        sorted.truncate(count);
        sorted.into_iter().map(|(h, _)| h).collect()
    }

    fn on_access(&mut self, block: BlockHandle) {
        self.access_times.lock().insert(block, Instant::now());
    }

    fn on_evict(&mut self, block: BlockHandle) {
        self.access_times.lock().remove(&block);
    }
}
```

#### A.3 实现 `LfuPolicy` 和 `FifoPolicy`

参照 `LruPolicy` 的模式，使用 `access_count: HashMap<BlockHandle, u64>` 和 `alloc_time: HashMap<BlockHandle, Instant>` 分别实现。

#### A.4 策略运行时切换

在 `TieringEngine::new()` 中根据 `KcmmConfig::eviction_policy` 字段选择策略：

```rust
let eviction_policy: Box<dyn EvictionPolicy> = match config.eviction_policy.as_str() {
    "lru" => Box::new(LruPolicy::new()),
    "lfu" => Box::new(LfuPolicy::new()),
    "fifo" => Box::new(FifoPolicy::new()),
    _ => Box::new(LruPolicy::new()), // 默认 LRU
};
```

---

### 阶段 B：单块换出实现（预计 3 天）

#### B.1 CPU 缓冲区槽位管理

```rust
impl TieringEngine {
    /// 分配 CPU swap 缓冲区槽位，返回字节偏移量
    fn alloc_cpu_slot(&mut self, block_bytes: usize) -> Result<usize> {
        // 简单方案：顺序分配 + 空闲列表回收
        // 复杂方案：buddy allocator（如果碎片严重）
        let offset = self.next_free_offset;
        if offset + block_bytes > self.cpu_buffer_size {
            return Err(anyhow!("CPU swap buffer exhausted"));
        }
        self.next_free_offset += block_bytes;
        Ok(offset)
    }

    /// 释放 CPU swap 缓冲区槽位
    fn free_cpu_slot(&mut self, _offset: usize, _block_bytes: usize) {
        // Phase 1: 加入 free list 供后续复用
        // Phase 2: 可选紧凑整理（当碎片率超过阈值时）
    }
}
```

#### B.2 单块换出流程

```rust
impl TieringEngine {
    /// 换出 count 个块从 GPU 到 CPU。
    /// 返回成功换出的块句柄列表。
    pub fn evict_blocks(
        &mut self,
        pool: &KcmmPool,           // 用于访问 GPU VA 和 VMM
        candidates: &[BlockHandle],
        count: usize,
    ) -> Result<Vec<BlockHandle>> {
        // 1. 选择受害者
        let victims = self.eviction_policy.select_victims(candidates, count);
        
        // 2. 对每个受害者执行换出
        for &victim in &victims {
            self.evict_single_block(pool, victim)?;
        }
        
        Ok(victims)
    }

    fn evict_single_block(&mut self, pool: &KcmmPool, block: BlockHandle) -> Result<()> {
        let block_bytes = pool.block_bytes;
        
        // 2.1 在 CPU swap buffer 中分配槽位
        let cpu_offset = self.alloc_cpu_slot(block_bytes)?;
        
        // 2.2 更新 BlockLocation → Evicting（标记传输中）
        pool.set_block_location(block, BlockLocation::Evicting)?;
        
        // 2.3 执行 D2H memcpy（使用专用 evict CUDA Stream）
        unsafe {
            self.streams.evict.memcpy_d2h_async(
                self.cpu_buffer.add(cpu_offset),
                pool.gpu_va_for_block(block)?,
                block_bytes,
            )?;
        }
        
        // 2.4 等待 evict 流完成
        self.streams.evict.synchronize()?;
        
        // 2.5 释放 GPU 物理页
        pool.vmm_unmap_block(block)?;
        
        // 2.6 更新 BlockLocation → CpuResident
        pool.set_block_location(block, BlockLocation::CpuResident(cpu_offset))?;
        
        // 2.7 通知策略
        self.eviction_policy.on_evict(block);
        
        Ok(())
    }
}
```

**关键注意事项：**
- `BlockLocation::Evicting` 中间状态防止并发访问在传输期间读到不一致数据
- `synchronize()` 确保 memcpy 完成后才 unmap（否则 GPU 可能还在读）
- 每层 K+V 都需要拷贝（`pool.num_layers * 2` 次 memcpy）

#### B.3 按层展开的完整拷贝

换出一个块需要拷贝所有层的 K 和 V 数据：

```rust
fn evict_single_block_all_layers(
    &mut self,
    pool: &KcmmPool,
    block: BlockHandle,
    cpu_offset: usize,
) -> Result<()> {
    let block_bytes = pool.block_bytes;
    let num_layers = pool.num_layers;
    let mut byte_offset = cpu_offset;

    for l in 0..num_layers {
        // 拷贝 K 层
        let gpu_va_k = pool.va_k(l) + pool.block_va_offset(block)? as u64;
        unsafe {
            self.streams.evict.memcpy_d2h_async(
                self.cpu_buffer.add(byte_offset),
                gpu_va_k,
                block_bytes,
            )?;
        }
        byte_offset += block_bytes;

        // 拷贝 V 层
        let gpu_va_v = pool.va_v(l) + pool.block_va_offset(block)? as u64;
        unsafe {
            self.streams.evict.memcpy_d2h_async(
                self.cpu_buffer.add(byte_offset),
                gpu_va_v,
                block_bytes,
            )?;
        }
        byte_offset += block_bytes;
    }

    self.streams.evict.synchronize()?;
    Ok(())
}
```

---

### 阶段 C：单块恢复实现（预计 2 天）

#### C.1 恢复流程

```rust
impl TieringEngine {
    /// 从 CPU swap 恢复一个块到 GPU。
    pub fn restore_block(
        &mut self,
        pool: &KcmmPool,
        block: BlockHandle,
        cpu_offset: usize,
    ) -> Result<()> {
        let block_bytes = pool.block_bytes;
        
        // 1. 标记状态 → Restoring
        pool.set_block_location(block, BlockLocation::Restoring)?;
        
        // 2. 分配 GPU 物理块并映射
        let (va_offset, sb_idx, blk_in_sb) = pool.alloc_one_block_internal()?;
        
        // 3. H2D memcpy（使用专用 restore CUDA Stream）
        self.restore_block_all_layers(pool, block, cpu_offset, va_offset)?;
        
        // 4. 等待 restore 流完成
        self.streams.restore.synchronize()?;
        
        // 5. 更新 BlockLocation → GpuResident
        pool.set_block_location(
            block,
            BlockLocation::GpuResident(block, va_offset as u64),
        )?;
        
        // 6. 释放 CPU 槽位
        self.free_cpu_slot(cpu_offset, block_bytes * pool.num_layers * 2);
        
        // 7. 通知策略
        self.eviction_policy.on_access(block);
        
        Ok(())
    }
}
```

#### C.2 与 `alloc_blocks` 路径集成

在 `KcmmPool::install_block()` 中检测 `BlockLocation` 并自动触发恢复：

```rust
// 在 alloc_blocks 路径中：
match block.location {
    BlockLocation::CpuResident(offset) => {
        // 需要先从 CPU 恢复
        self.tiering.as_ref()
            .ok_or_else(|| anyhow!("block is CPU-resident but tiering is disabled"))?
            .restore_block(self, handle, offset)?;
    }
    BlockLocation::NvmeResident(offset) => {
        // NVMe 恢复路径（后续实现）
    }
    BlockLocation::Evicting | BlockLocation::Restoring => {
        return Err(anyhow!("block is in transit"));
    }
    BlockLocation::GpuResident(_, _) => {
        // 已 GPU 常驻，直接返回
    }
}
```

---

### 阶段 D：异步 memcpy 封装（预计 1 天）

在 `CudaStream` 上添加：

```rust
impl CudaStream {
    /// 异步 Device-to-Host memcpy（GPU → CPU）。
    pub unsafe fn memcpy_d2h_async(
        &self,
        dst: *mut u8,
        src: CUdeviceptr,
        nbytes: usize,
    ) -> Result<()> {
        let r = sys::lib().cuMemcpyDtoHAsync_v2(
            dst as *mut std::ffi::c_void,
            src,
            nbytes,
            self.inner,  // ← 使用专用流
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyDtoHAsync failed: {:?}", r));
        }
        Ok(())
    }

    /// 异步 Host-to-Device memcpy（CPU → GPU）。
    pub unsafe fn memcpy_h2d_async(
        &self,
        dst: CUdeviceptr,
        src: *const u8,
        nbytes: usize,
    ) -> Result<()> {
        let r = sys::lib().cuMemcpyHtoDAsync_v2(
            dst,
            src as *const std::ffi::c_void,
            nbytes,
            self.inner,  // ← 使用专用流
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyHtoDAsync failed: {:?}", r));
        }
        Ok(())
    }
}
```

---

### 阶段 E：微基准测试 Benchmark 1-2（预计 2 天）

#### E.1 Benchmark 1: 块分配/释放吞吐量

```rust
// tests/kcmm_bench_alloc.rs
#[test]
fn bench_alloc_throughput() {
    // 场景：预分配 N 个块（无分层存储）
    // 操作：循环 { alloc_blocks(seq_i, 1) → free_blocks(seq_i) }
    // 变量：块大小 32KB/64KB/128KB；池大小 1024/4096/16384
    // 指标：alloc_blocks p50/p99 (ns), free_blocks p50/p99 (ns)
    // 成功标准：KCMM < vLLM × 1.05（回退 < 5%）
}
```

#### E.2 Benchmark 2: 分层存储换出/恢复延迟

```rust
// tests/kcmm_bench_tiering.rs
#[test]
fn bench_single_block_evict_restore() {
    // 场景：填满 GPU 池 → 触发换出 → 后续分配触发恢复
    // 变量：块大小 32KB/64KB/128KB；每次换出块数 1/4/16/64
    // 指标：
    //   - 换出延迟 p50/p99 (μs)：cudaMemcpy D2H + cuMemUnmap
    //   - 恢复延迟 p50/p99 (μs)：cuMemMap + cudaMemcpy H2D
    //   - 批量换出 vs 单块换出的摊销延迟
    //   - cuMemMap 延迟 p50/p99（单独测量，已知瓶颈）
    // 成功标准：单块恢复 < 200μs (p50)
}
```

---

### 阶段 F：后续工作（第 15-16 周）

#### F.1 批量换出/恢复优化

单块换出的 `cuMemMap`/`cuMemUnmap` 延迟是已知瓶颈（vAttention 实测 ~115× vs malloc）。批量操作可显著摊销：

```rust
pub fn evict_batch(&mut self, pool: &KcmmPool, count: usize) -> Result<Vec<BlockHandle>> {
    // 1. 一次性选择 count 个受害者
    // 2. 一次大粒度 cudaMemcpy D2H（合并所有受害者数据）
    // 3. 一次批量 cuMemUnmap
    // 4. 批量更新 BlockLocation
}
```

#### F.2 C FFI 完整实现

将 `ffi.rs` 中注释的 10 个函数变为实际实现。需要：
- 全局 `HashMap<usize, Arc<KcmmPool>>` 管理池句柄
- `kcmm_pool_create()` 返回不透明指针
- 所有函数的参数校验和错误码返回

#### F.3 NVMe 层（可选）

- 优先使用 GDS（`cuFileRead`/`cuFileWrite`）
- 标准 I/O 回退路径
- CPU 缓冲区作为 NVMe 的 staging area

#### F.4 Benchmark 3-4

- Benchmark 3: CUDA Stream 开销（推理 kernel 干扰 < 1%）
- Benchmark 4: 换出策略命中率（合成负载，LRU ≥ Oracle 85%）

---

## 四、文件变更清单

### 需要修改的现有文件

| 文件 | 变更内容 |
|------|---------|
| `src/kcmm/tiering.rs` | **主要变更** — 添加 `LruPolicy`/`LfuPolicy`/`FifoPolicy` 完整实现；添加 `evict_blocks()`/`restore_block()`；添加 CPU 槽位管理；添加按层展开的批量 memcpy |
| `src/kcmm/pool.rs` | 添加 `set_block_location()` / `gpu_va_for_block()` / `block_va_offset()` 供 TieringEngine 调用；`alloc_blocks` 路径中集成自动恢复；`free_sequence` 中处理 `CpuResident` 块的 CPU 槽位释放 |
| `src/kcmm/streams.rs` | 添加 `memcpy_d2h_async()` / `memcpy_h2d_async()` 方法 |
| `src/kcmm/mod.rs` | 无重大变更（模块结构已定） |
| `src/kcmm/ffi.rs` | 实现被注释的 `extern "C"` 函数 |

### 需要新建的文件

| 文件 | 内容 |
|------|------|
| `tests/kcmm_bench_alloc.rs` | Benchmark 1：块分配/释放吞吐量 |
| `tests/kcmm_bench_tiering.rs` | Benchmark 2：换出/恢复延迟微基准 |

---

## 五、实现顺序与时间估算

```
第 14 周（6 天）：
  Day 1-2: 阶段 A — LruPolicy/LfuPolicy/FifoPolicy 完整实现 + 策略切换
  Day 3-5: 阶段 B — 单块 GPU→CPU 换出（含 CPU 槽位管理 + 按层 memcpy）
  Day 6:   阶段 C — 单块 CPU→GPU 恢复 + 与 alloc_blocks 路径集成

第 15 周（6 天）：
  Day 1:   阶段 D — 异步 memcpy 封装
  Day 2-3: 阶段 E — Benchmark 1-2 编写与运行
  Day 4-6: 阶段 F.1 — 批量换出/恢复优化

第 16 周（6 天）：
  Day 1-3: 阶段 F.2 — C FFI 完整实现
  Day 4-6: 阶段 F.3 — NVMe 层（可选）+ Benchmark 3-4
```

---

## 六、风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| `cuMemMap` 延迟成为瓶颈（~115× vs malloc） | 高 | Benchmark 2 中优先单独测量；批量操作摊销；后台线程预分配 |
| `BlockLocation` 状态机并发 bug | 中 | 使用 `Mutex<BlockInfo>` 保护状态转换；添加转换合法性断言；`Evicting`/`Restoring` 中间状态阻止并发访问 |
| `TieringEngine` 与 `KcmmPool` 之间的数据通路设计不当导致循环引用 | 中 | 采用方法参数传递模式（非持有引用）；`TieringEngine` 不 `Arc` 指向 `KcmmPool` |
| 按层展开的 memcpy 次数过多（每块 `num_layers * 2` 次） | 低 | 使用专用 CUDA Stream 异步执行；批量合并；后续可考虑跨层合并的内存布局 |
| CPU swap 缓冲区碎片化 | 低 | 初始采用顺序分配 + 空闲列表；后续引入 buddy allocator |

---

## 七、成功标准（阶段 A-E 完成时）

- [ ] `LruPolicy::select_victims()` 正确选择 `last_access` 最旧的受害者块
- [ ] 单块 GPU→CPU 换出完成（cudaMemcpy D2H → cuMemUnmap → CpuResident）
- [ ] 单块 CPU→GPU 恢复完成（cuMemMap → cudaMemcpy H2D → GpuResident）
- [ ] `BlockLocation` 状态转换正确（GpuResident ↔ Evicting ↔ CpuResident ↔ Restoring ↔ GpuResident）
- [ ] 换出触发条件正确（低水位线 → `evict_blocks`）
- [ ] 恢复触发条件正确（`alloc_blocks` 发现 `CpuResident` → `restore_block`）
- [ ] Benchmark 1: 无分层时 KCMM 块分配吞吐量回退 < 5% vs vLLM
- [ ] Benchmark 2: 单块恢复延迟 p50 < 200μs
- [ ] `cargo test --features kcmm` 全部测试通过（含新增 Benchmark）
- [ ] `cargo check --features kcmm` 编译无警告

---

## 八、与后续 Step 的接口预留

以下接口在 Step 3 各阶段仅做预留（定义类型签名和空实现），具体逻辑属于 Step 4：

| 接口 | 说明 | 实现阶段 |
|------|------|---------|
| `SharingManager::register_prefix()` | 前缀注册 | Step 4 |
| `SharingManager::try_share_prefix()` | 前缀查找 | Step 4 |
| `KcmmPool::sharing` 字段 | `Option<SharingManager>`，Step 3 始终 `None` | Step 4 |
| `kcmm_hint()` C API | Hint API 函数声明 | Step 4 |
| `kcmm_protect()` / `kcmm_evictable()` | 注意力信号保护级别 | Step 4 |
