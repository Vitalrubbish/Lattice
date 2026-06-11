# KCMM 压力测试设计分析与性能瓶颈诊断

**Date:** 2026-06-09
**Status:** Analysis Report
**Related:** `docs/dev/kcmm-pressure-tiering-test.md`, `tests/kcmm_bench_memory_pressure.rs`

---

## 1. 压力测试设计评估

### 1.1 测试设计概要

`kcmm_bench_memory_pressure_single` 测试对比了两种配置在 GPU 内存受限时的吞吐能力：

| 维度 | Baseline (PagedKvCache) | KCMM (KcmmPool + Tiering) |
|------|------------------------|---------------------------|
| 策略 | 无 tiering，OOM 即 cap/reject | 触达容量上限时触发 eviction 释放 GPU 空间 |
| 关键指标 | completed sequences | completed sequences |
| 目标 ratio | — | KCMM / Baseline ≥ 1.3× |

### 1.2 设计优点

1. **端到端验证核心价值主张。** 测试直接回答了 KCMM 最根本的问题：eviction 能否在 GPU 内存受限时真正提升并发容量。

2. **动态负载模拟真实场景。** 采用 continuous batching 模式——序列在 decode 过程中完成、新序列不断到达——比简单的填充-排空测试更有代表性。

3. **与 Baseline 共用 workload config。** 两种配置使用完全相同的 `WorkloadConfig`，消除了参数差异对结论的影响。

4. **多层统计输出。** 记录了 completed/capped/rejected/peak_concurrent/eviction_count 等多个维度，覆盖了"容量""健康度""机制活跃度"三类信号。

5. **Sweep 覆盖多个压力剖面。** `kcmm_bench_memory_pressure_sweep` 测试了 4 种不同配置（不同的 block_size、max_batch、max_seq_len、arrival_interval），增加了结论的鲁棒性。

### 1.3 设计缺陷

#### 1.3.1 缺少预热 (Warmup)

两个 `run_*_workload` 函数都没有预热阶段。CUDA 的首次 kernel launch 和首次 memcpy 通常比后续调用慢得多（驱动初始化、JIT 编译缓存等），导致前几个 iteration 的耗时具有系统性偏差。

**建议：** 在正式 workload 前插入一个 mini warmup phase（例如 2 个序列，各 4 步 decode），使 GPU 驱动达到稳态。

#### 1.3.2 单次测量缺乏统计显著性

`kcmm_bench_memory_pressure_single` 只运行一次完整的 workload。在 WSL2 GPU-PV 环境中，GPU 操作的时间方差可能达到 ±20%——这意味着单次运行的 `throughput_ratio` 可能不可复现。`completed` 这个指标虽然是离散的（序列计数）而非连续的（时延），但 capacity 受 timing 影响：本次运行中 baseline 恰好 cap 了 8 个序列，换一个 timing 窗口可能不同。

**建议：** 至少运行 5 次独立 trial，报告 mean ± stddev of `throughput_ratio`。

#### 1.3.3 缺少时延维度的评估

测试只关注"完成多少序列"（容量），没有关注"每个序列完成需要多长时间"（时延）。一个 tiering 系统可以实现 2× 容量提升，但如果每个序列的 end-to-end latency 增加了 10×，在实际部署中将不可接受。

当前结果中 KCMM 的 `elapsed_ms=356ms` vs Baseline 的 `elapsed_ms=48ms` 已经是 7.4× 的减速。这不仅仅是"慢了一点"的问题——它意味着 KCMM 在当前 WSL2 环境下虽然能容纳更多序列，但每个序列的响应时间显著增加。

**建议：** 增加 per-sequence latency 指标（P50/P99 decode step time），并在综合评分中同时考虑 capacity_ratio 和 latency_ratio。

#### 1.3.4 Baseline 的 "Pool is full" 判定不精确

```rust
// run_baseline_workload, line 126:
match cache.alloc_sequence(initial_blocks) {
    Ok(bt) => bt,
    Err(_) => break, // Pool is full — pre-fill complete.
}
```

`PagedKvCache::alloc_sequence` 返回 `Err` 时并不一定意味着"池已满"——可能只是当前恰好没有足够的连续空闲块。这会导致 baseline 提前停止 pre-fill，使 baseline 的 `completed` 被低估，KCMM 的 `throughput_ratio` 被高估。

**建议：** 检查 `Err` 的具体原因，或者在 OOM 后尝试释放一些已完成序列后再试。

#### 1.3.5 KCMM workload 比 Baseline 多了额外的管理开销

KCMM 路径在每次 decode step 中多了以下操作，这些不在 tiering 的性能模型中：
- `pool.touch()` / `pool.cool()` 调用（line 424, 455, 472, 485）
- `pool.update_seq_len()` 调用
- 每 8 步的 cool/touch 周期（line 411-427）
- 每次 block 分配失败的 eviction 重试逻辑（line 460-481）

这些操作本身很快（µs 级），但增加了代码路径的差异。

---

## 2. 测试结果分析

### 2.1 实测数据

```
Config: bs16_mb16_msl640_pl[128,256]_mnt384_arr32
block_bytes=180224 (176 KiB), VA blocks=640 (~110 MiB), total_arrivals=32

  Baseline (PagedKvCache, no tiering):
    completed=24, capped=8, rejected=0, peak_concurrent=32
    elapsed=48ms

  KCMM (KcmmPool, tiering ON):
    completed=32, capped=0, rejected=0, peak_concurrent=32
    evictions=30, cpu_swap_peak=5406720 B, peak_blocks=795
    elapsed=356ms

  throughput_ratio = 32 / 24 = 1.33×  ✅ PASS
```

### 2.2 容量分析

- KCMM 确实提升了吞吐：32 vs 24 个完成的序列，**1.33× 达到目标**。
- 30 次 eviction 说明 tiering 机制被充分激活——内存压力确实存在。
- Baseline cap 了 8 个序列（在 decode 过程中 OOM），KCMM cap 了 0 个——**eviction 有效地将 OOM 转化为 CPU swap，保持了序列的完整性**。

### 2.3 时延分析——核心问题

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Baseline total | 48ms | — |
| KCMM total | **356ms** | 100% |
| KCMM eviction | **309ms** | 87% |
| KCMM alloc | 14ms | 4% |
| KCMM cool/touch | 0ms | 0% |

**结论：KCMM 的 87% 时间花在 eviction 上。每个 eviction batch (8 blocks) 耗时 ~10ms。**

### 2.4 关键观测

每次 eviction 的详细日志：
```
[evict detail] batch=8  scan=0µs  evict=10062µs  total=10062µs
[evict detail] batch=8  scan=0µs  evict=9744µs   total=9744µs
[evict detail] batch=8  scan=0µs  evict=11219µs  total=11219µs
...
```

- `scan=0µs`：候选块选择（扫描序列+策略排序）几乎不耗时，瓶颈不在这里。
- `evict=~10ms`：**eviction 本身是瓶颈**。对于 22 层、8 个块的 batch，每个块 180KB 数据，总共 ~1.4MB 的 GPU→CPU 传输量，这应该是远小于 1ms 的操作。

---

## 3. 根因分析：为什么 KCMM Eviction 这么慢？

### 3.1 瓶颈定位

Batched eviction 路径 (`evict_blocks_batched`) 对每个 layer×KV pair 执行以下操作：

```
for layer in 0..22 {          // 22 layers
    for is_v in [K, V] {      // 2 (K + V) = 44 iterations total
        1. 构建 CPU 端指针数组                   (CPU, ~µs)
        2. device.alloc_zeros::<u64>(8)  ← 关键!  (GPU alloc, ~200µs each in WSL2)
        3. cuMemcpyHtoDAsync_v2 (上传 64 bytes)  (GPU API call)
        4. launch_kv_gather (kernel launch)      (GPU kernel launch)
        5. cuMemcpyDtoHAsync_v2 (下载 64KB)      (GPU API call)
    }
}
6. device.synchronize()  ← 等待所有 44 轮 GPU 操作
7. CPU scatter: staging → per-block CPU slots
8. Finalize (释放物理块, 标记 CpuResident)
```

### 3.2 根因：每轮迭代的 GPU 内存分配

**每个迭代中，`device.alloc_zeros::<u64>(8)` 分配了一个 64 字节的 GPU 内存块用于存储指针数组。** 对于 44 个迭代，这就是 **44 次 GPU 内存分配**。

> **API 说明：** `cudarc::alloc_zeros` 底层调用 CUDA Driver API 的 `cuMemAlloc_v2`（传统线性分配）。这与 KCMM 中 KV cache **超块（superblock）**使用的 `cuMemCreate`（CUDA VMM API，`src/cache/cuda_vmm.rs:101`）是不同的分配路径。本文中讨论的 44 次分配特指 eviction 过程中的小粒度指针数组分配，而非超块分配。

在 WSL2 的 GPU-PV (Paravirtualization) 环境下：
- 每次 `cuMemAlloc_v2` 调用的开销约为 **200–300µs**（需要通过 hypercall 进入宿主机内核）
- 44 × 250µs ≈ **11ms** → 与观测到的 ~10ms eviction 时间高度吻合

此外，每个迭代还有：
- 1 次 `cuMemcpyHtoDAsync_v2`（64 bytes 指针上传）：~30–50µs GPU-PV 开销
- 1 次 kernel launch (`gather_kv_layer`)：~20–40µs 开销
- 1 次 `cuMemcpyDtoHAsync_v2`（64KB staging 下载）：~30–50µs GPU-PV 开销

**44 × (250 + 40 + 30 + 40) ≈ 15.8ms** 的理论上限（部分操作可能被流水线重叠），与实测 ~10ms 一致。

### 3.3 问题量化

| 操作 | 每轮次数 | 单次开销 (WSL2) | 总开销 (44 轮) |
|------|---------|----------------|---------------|
| `cuMemAlloc_v2` (64 bytes) | 1 | ~200–300µs | ~8.8–13.2ms |
| `cuMemcpyHtoDAsync` (64B) | 1 | ~30–50µs | ~1.3–2.2ms |
| Kernel launch (gather) | 1 | ~20–40µs | ~0.9–1.8ms |
| `cuMemcpyDtoHAsync` (64KB) | 1 | ~30–50µs | ~1.3–2.2ms |
| **合计** | | | **~12–19ms** |

观测值 ~10ms 接近预测范围的下界，说明部分操作在硬件层面被流水线重叠了，但 GPU 内存分配仍然是绝对主导因素。

### 3.4 在 Bare-Metal 上的预期表现

在 bare-metal Linux 上，`cuMemAlloc_v2` 的开销通常是 **5–20µs** 而非 200–300µs。其他 CUDA API 调用的开销也会相应降低。因此：

- Bare-metal 预估每轮：**44 × (10 + 5 + 5 + 10) ≈ 1.3ms**
- Bare-metal batch evict（8 blocks）：**~1.5ms** vs WSL2 的 ~10ms
- KCMM total 预估：baseline 48ms + 30 evictions × 1.5ms ≈ **93ms**（约为 WSL2 的 1/4）

但即使如此，44 次 GPU 分配仍然是浪费的——**在任何平台上，44 次 64 字节分配都不如 1 次 2.8KB 分配高效**。

---

## 4. 修复方案

### 4.1 立即修复（高优先级）：预分配指针设备数组

**问题：** `evict_blocks_batched` 和 `restore_blocks_batched` 中每次 layer×KV 迭代都分配一个新的 `ptrs_dev_layer` (64 bytes)。

**方案：** 在 `TieringEngine::new()` 中一次性分配一个足够大的指针数组：
```rust
// 在 TieringEngine 结构体中新增字段:
ptr_pool: CudaSlice<u64>,  // 大小: max_batch_blocks * num_layers * 2

// 使用时按 offset 取:
let ptrs_dev_offset = layer_idx * max_batch_blocks;
// 上传指针到 ptr_pool[ptrs_dev_offset..ptrs_dev_offset + actual_n]
```

**预期收益：** 
- 将 44 次 `cuMemAlloc_v2` 调用减少为 0 次（首次初始化时已分配）
- Eviction per-batch 从 ~10ms 降至 **~1–2ms**（WSL2）或 **<0.5ms**（bare-metal）
- KCMM total 从 356ms 降至 **~80–110ms**（WSL2，约为 2.3× Baseline 而非 7.4×）

### 4.2 中期优化：减少 CUDA API 调用次数

当前每个 layer×KV 对发出 3 个独立的 CUDA API 调用（H2D ptrs + kernel + D2H staging）。可以考虑：

1. **使用 CUDA Graph** 将重复的 44 步操作录制为单个 graph，然后一次性 launch。CUDA Graph 消除了每次 kernel launch 和 memcpy 的 API 调用开销。

2. **合并所有层的指针数组为一次大 H2D**：将所有 44 个指针数组预计算在 CPU 端，一次性上传到 GPU，然后依次 launch 44 个 gather kernel（此时只需要 kernel launch，不需要 memcpy API 调用）。

### 4.3 长期优化：消除 Gather/Scatter Kernel

当前 gather kernel 的作用是将 N 个分散的 GPU 源指针收集到连续的 staging buffer，以便进行一次 batched D2H。这是必要的，因为 `cuMemcpyDtoH` 只能从连续的 GPU 地址拷贝。

但是如果使用 GPU Direct RDMA 或 CUDA VMM 的特定模式，可以消除 gather kernel 的需求。这需要在 Phase 2 或更后期探索。

---

## 5. 修复实施优先级

| 优先级 | 修复 | 预期效果 | 工作量 |
|--------|------|---------|--------|
| **P0** | 预分配 ptrs_dev 数组，消除 44×alloc | Eviction 10ms→~1.5ms (WSL2) | ~2h |
| **P1** | 同样优化 restore_blocks_batched 中的对称分配 | Restore batch 同步加速 | ~1h |
| **P2** | 增加 warmup + 多次 trial + latency 指标 | 测试结论更可靠 | ~3h |
| **P3** | CUDA Graph 消除 API 调用开销 | 进一步降低 ~30% 时延 | ~1d |
| **未来** | GPU Direct / 替代 gather-scatter 策略 | 大幅降低 tiering 开销 | 研究级 |

---

## 6. 总结

1. **压力测试设计总体合理**，正确验证了 KCMM 的核心价值主张（容量提升 1.33×），但缺少统计显著性（单次测量）和时延维度的评估。

2. **KCMM 运行慢的根因是每轮 batched eviction 的 44 次 GPU 内存分配**。每次分配只有 64 bytes，但在 WSL2 GPU-PV 环境下每次 `cuMemAlloc_v2` 调用开销约 200–300µs，44 次合计 ~10ms，占 eviction 总时延的绝大部分。

3. **修复方案明确且低风险**：预分配指针设备数组（一次性分配 2.8KB 代替 44 次 64B 分配），预计可将 eviction 时延降低 6–8×，KCMM 总时延从 7.4× Baseline 降至 ~2× Baseline。

4. **在 bare-metal 上该问题同样存在但程度较轻**。44 次 GPU alloc 在 bare-metal 上约 0.5ms（44 × 10µs），仍然是不必要的浪费。预分配修正在两个平台都有效。
