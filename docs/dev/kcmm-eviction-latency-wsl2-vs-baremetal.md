# KCMM Eviction 时延：WSL2 vs Bare-Metal 定量分析

**Date:** 2026-06-09
**Related:** `docs/dev/kcmm-pressure-test-analysis.md`

---

## 1. 实测微基准数据 (WSL2)

| 操作 | P50 | Mean | P99 | Max |
|------|-----|------|-----|-----|
| `cuMemAlloc(64B)` | **9.7µs** | 63.7µs | 168.3µs | 9.97ms |
| `cuMemAlloc(64B) + cuMemset` (模拟 `alloc_zeros`) | **19.4µs** | **162.5µs** | **12.96ms** | 12.96ms |
| `cuCtxSynchronize()` (空闲) | 2.4µs | 7.5µs | 410.6µs | 410.6µs |
| 44 × (alloc + memset + free) | — | **1.7ms** | — | 2.1ms |

**关键发现：WSL2 的 `cuMemAlloc` 不是均匀地慢，而是有严重的厚尾分布。**

- P50 只有 19.4µs（很快）
- Mean 被拉到 162.5µs（被尾部的 12.96ms spike 拉高了 8.4×）
- 单次调用 P99 = **12.96ms**，几乎等于整个 eviction batch 的耗时

每批 eviction 有 44 次分配，30 批 eviction 共 1320 次分配——哪怕 P99 只有 1% 概率触发长尾，1320 次中也有很高的概率命中多次。

---

## 2. Eviction Batch 时延分解

当前观测的 10ms/batch 拆解如下：

| 组件 | WSL2 (当前) | 说明 |
|------|-----------|------|
| 44 × `alloc_zeros` | **~7.2ms** | mean 162.5µs × 44 = 7.15ms (厚尾主导) |
| 44 × kernel launch + memcpy API | ~1.3ms | P50 ~30µs × 44 |
| 44 × 实际 GPU 工作 (gather + D2H) | ~0.5ms | 64KB × 44 的 DMA + kernel |
| CPU scatter (Phase 4) | ~0.1ms | `std::ptr::copy_nonoverlapping` |
| `device.synchronize()` | ~0.5ms | 等待 GPU pipe 排空 |
| 其他 (policy, finalize) | ~0.4ms | |
| **合计** | **~10ms** | ✅ 与观测一致 |

---

## 3. Bare-Metal 预期

在 bare-metal Linux (如 d7525 的 AMD EPYC + NVIDIA A30) 上，没有 GPU-PV 层：

| 操作 | WSL2 P50 | Bare-Metal P50 | 改善倍数 |
|------|---------|---------------|---------|
| `cuMemAlloc(64B)` | 9.7µs | ~3–5µs | 2–3× |
| `cuMemAlloc` tails | **12.96ms P99** | **~10–30µs P99** | **400–1000×** |
| `cuMemAlloc` mean | 162.5µs | ~5–8µs | **20–30×** |
| `cuCtxSynchronize` | 2.4µs | ~1–2µs | 1.5–2× |
| kernel launch | 20–40µs | ~5–10µs | 3–5× |
| cuMemcpy API | 10–30µs | ~3–5µs | 3–6× |

### 场景一：Bare-Metal 不做任何代码修改

```
44 × alloc_zeros:    44 × 8µs = 0.35ms  (vs 7.2ms,  20× faster)
44 × kernel+memcpy:  44 × 15µs = 0.66ms (vs 1.3ms,  2× faster)
实际 GPU 工作:       ~0.5ms            (相同)
其他:                ~1.0ms            (相似)

Per-batch eviction:  ~2.5ms  (vs 10ms, 4× faster)
30 batches:          ~75ms   (vs 309ms, 4.1× faster)
KCMM total:          ~48ms + 75ms = ~123ms
vs Baseline:         123/48 = 2.6×
```

### 场景二：Bare-Metal + 预分配修复

```
44 × alloc_zeros:    0ms              (已预分配)
44 × kernel+memcpy:  44 × 15µs = 0.66ms
实际 GPU 工作:       ~0.5ms
其他:                ~1.0ms

Per-batch eviction:  ~2.2ms
30 batches:          ~66ms
KCMM total:          ~48ms + 66ms = ~114ms
vs Baseline:         114/48 = 2.4×
```

### 场景三：Bare-Metal + 预分配 + CUDA Graph

```
单次 Graph launch:   ~0.05ms          (消除 44 次 API 调用)
实际 GPU 工作:       ~0.5ms
其他:                ~1.0ms

Per-batch eviction:  ~1.6ms
30 batches:          ~48ms
KCMM total:          ~48ms + 48ms = ~96ms
vs Baseline:         96/48 = 2.0×
```

---

## 4. 各平台对比汇总

| 配置 | Per-Batch Evict | KCMM Total | vs Baseline |
|------|----------------|-----------|-------------|
| WSL2 (当前) | **10ms** | **356ms** | **7.4×** |
| WSL2 + 预分配修复 | ~3.5ms | ~150ms | 3.1× |
| Bare-Metal (无修改) | ~2.5ms | ~123ms | 2.6× |
| Bare-Metal + 预分配修复 | ~2.2ms | ~114ms | 2.4× |
| Bare-Metal + 预分配 + CUDA Graph | ~1.6ms | ~96ms | 2.0× |

---

## 5. 结论

### 真机能否解决分配时延问题？

**能解决大部分，但不能完全解决。**

1. **Bare-metal 最大收益来自消除厚尾。** WSL2 的 `cuMemAlloc` P99 是 **12.96ms**，这是 GPU-PV 调度的 artifact。Bare-metal 的 P99 预期是 10–30µs — **400–1000× 改善**。这个改善直接使 eviction 从 10ms/batch → ~2.5ms/batch。

2. **但结构性浪费仍然存在。** 即使在 bare-metal 上，44 次 × 8µs = 0.35ms 的分配开销也完全可以避免。预分配修复在 bare-metal 上仍能节省 ~12% 的 eviction 时间。

3. **WSL2 + 预分配修复 ≈ Bare-Metal 无修改的效果。** WSL2 的 3.5ms/batch (修复后) 和 bare-metal 的 2.5ms/batch (未修改) 差别不大——说明当前最大的 WSL2 性能损耗确实来自分配厚尾，修复后两者差距缩小。

4. **要达到 <2.0× Baseline 需要 Bare-Metal + 预分配 + CUDA Graph。** 单独靠任何一个都不够。CUDA Graph 消除 kernel launch API 开销是关键的最后一步。

### 推荐路径

- **立即（WSL2）：** 实施预分配修复 → eviction 10ms→3.5ms，KCMM 从 7.4× → 3.1× baseline
- **短期（Bare-Metal可用时）：** 跑完整 benchmark suite → 获得 ~2.6× baseline 基线
- **中期：** CUDA Graph → 目标 2.0× baseline
