# GPU 显存碎片化：UFS 指标的深入分析

**日期:** 2026-06-01
**上下文:** Step 3 实验结果分析
**相关结果:** `results/baremetal/step3_compare_20260601_053321/`

---

## 目录

1. [背景：TinyLlama 参数与派生常量](#1-背景tinyllama-参数与派生常量)
2. [四个 UFS 指标的公式与含义](#2-四个-ufs-指标的公式与含义)
3. [为什么 PME 恒等于 BU？](#3-为什么-pme-恒等于-bu)
4. [为什么 RFI = IFR（vLLM）但 RFI > IFR（Baseline）？](#4-为什么-rfi--ifrvllm但-rfi--ifrbaseline)
5. [Superblock 锯齿效应](#5-superblock-锯齿效应)
6. [碎片化的三层嵌套结构](#6-碎片化的三层嵌套结构)
7. [哪些指标可以跨系统比较？](#7-哪些指标可以跨系统比较)
8. [vLLM 测量中的陷阱：nvidia-smi Diff Trap](#8-vllm-测量中的陷阱nvidia-smi-diff-trap)
9. [负载-效率曲线：Grow-on-Demand 的完整画像](#9-负载-效率曲线grow-on-demand-的完整画像)
10. [架构权衡总结](#10-架构权衡总结)
11. [已知问题与改进建议](#11-已知问题与改进建议)

---

## 1. 背景：TinyLlama 参数与派生常量

本实验使用 TinyLlama 1.1B 模型，关键配置：

```
kv_heads   = 4
head_dim   = 64
num_layers = 22
block_size = 16
```

由此推导的所有常量：

| 常量 | 公式 | 值 |
|------|------|-----|
| K 层每 token 字节数 | kv_heads × head_dim × sizeof(f16) | 4 × 64 × 2 = **512 bytes** |
| block_bytes（单层 K） | block_size × K_bytes_per_token | 16 × 512 = **8,192 bytes** |
| blocks_per_superblock | 2 MiB / block_bytes | 2,097,152 / 8,192 = **256 blocks** |
| BPT（全层 K+V 每 token 字节） | K_bytes × num_layers × 2 | 512 × 22 × 2 = **22,528 bytes** ≈ 0.0215 MiB |
| 一个 SB 位置物理内存（全层 K+V） | 2 MiB × 22 × 2 | **88 MiB** |
| 一个 K 层 SB 可容纳 token 数 | 2 MiB / 512 | **4,096 tokens** |

**最后一个常数最为关键：** 一个 K 层的 superblock 能装 4,096 个 token 的 KV 数据。这意味着：

- seq_len=64 时，需要 **64** 个并发请求才能填满一个 K superblock
- seq_len=128 时，需要 **32** 个并发
- seq_len=256 时，需要 **16** 个并发

在并发数低于此阈值时，永远至少存在一个半空的 superblock。

---

## 2. 四个 UFS 指标的公式与含义

UFS（Unified Fragmentation Standard）定义在 `src/cache/unified_frag.rs`。四个指标及其在两个系统中的实际公式如下。

### 2.1 IFR — Internal Fragmentation Rate（内部碎片率）

```
IFR = (total_slots - total_tokens) / total_slots
     其中 total_slots = total_blocks_used_by_seqs × block_size
```

**测量什么：** 每个序列的最后一个 block 中，有多少 slot 是空的。这是 block_size 粒度的浪费。

**跨系统可比性：** ✅ **可直接比较。** 公式在所有系统中完全相同，仅取决于 block_size 和 workload 的序列长度分布。Range: [0, 1)。

### 2.2 BU — Block Utilization（Block 利用率）

```
BU = blocks_in_use / total_blocks_allocated
```

**测量什么：** 所有已分配物理内存的 block 中，有多少正在被序列使用。

**跨系统可比性：** ❌ **不可直接比较绝对值。** 分母含义在不同系统中截然不同：

| 系统 | total_blocks_allocated 的含义 |
|------|------------------------------|
| **Baseline (CUDA VMM)** | `sb_count × 256`，随负载增长。低负载时只有 1 个 SB = 256 blocks；高负载时增至 22 个 SB = 5,632 blocks |
| **vLLM (PyTorch)** | 启动时一次性预分配的全池大小。本实验中估算为 56,301 blocks，固定不变 |

**CONTEXT.md 明确规定：「Compare the trend (rising under load = grow-on-demand working), not the absolute value。」**

### 2.3 PME — Physical Memory Efficiency（物理内存效率）

```
PME = ideal_physical_bytes / actual_physical_bytes
```

**两个系统的具体公式：**

| | Baseline (CUDA VMM) | vLLM (PyTorch) |
|---|---|---|
| ideal_physical_bytes | blocks_in_use × 8192 × 44 | blocks_in_use × 8192 × 44 |
| actual_physical_bytes | sb_count × 2MiB × 44 | total_blocks_allocated × 8192 × 44 |
| 化简后 | = blocks_in_use / (sb_count × 256) | = blocks_in_use / total_blocks_allocated |
| **结果** | **= BU** | **= BU** |

**跨系统可比性：** ❌ **不可直接比较。** 且 PME 在两个系统中都恒等于 BU，没有提供额外信息。详见 [第 3 节](#3-为什么-pme-恒等于-bu)。

### 2.4 RFI — Runtime Fragmentation Index（运行时碎片指数）

```
RFI = 1 - (total_tokens × BPT) / actual_active_bytes
```

**两个系统的具体公式：**

| | Baseline (CUDA VMM) | vLLM (PyTorch) |
|---|---|---|
| actual_active_bytes | ⌈blocks_in_use / 256⌉ × 2MiB × 44 | blocks_in_use × 8192 × 44 |
| 化简后 | 1 − tokens / (⌈blocks_in_use/256⌉ × 256 × 16) | 1 − tokens / (blocks_in_use × 16) |

**关键差异：Baseline 的 `actual_active_bytes` 将 `blocks_in_use` 向上取整到 superblock 边界（⌈·/256⌉），而 vLLM 没有这个步骤。** 详见 [第 4 节](#4-为什么-rfi--ifrvllm但-rfi--ifrbaseline)。

**跨系统可比性：** ⚠️ **差值 RFI − IFR 可比。** RFI − IFR = 纯 superblock 粒度浪费。vLLM 的 RFI − IFR = 0（无 superblock），Baseline 的 RFI − IFR > 0。

---

## 3. 为什么 PME 恒等于 BU？

### 3.1 代数推导

以 Baseline 为例。查看 `src/cache/unified_frag.rs:135-144`：

```rust
let actual_physical_bytes =
    (superblock_count * superblock_size * num_layers * 2) as u64;
    // = sb_count × 2MiB × 44

let ideal_physical_bytes =
    (blocks_in_use * block_bytes * num_layers * 2) as u64;
    // = blocks_in_use × 8192 × 44

let physical_memory_efficiency = ideal_physical_bytes / actual_physical_bytes;
```

分子分母的 `num_layers × 2`（即 ×44）直接抵消：

```
PME = (blocks_in_use × 8192 × 44) / (sb_count × 2MiB × 44)
    = (blocks_in_use × 8192) / (sb_count × 2MiB)
    = blocks_in_use / (sb_count × 2MiB / 8192)
    = blocks_in_use / (sb_count × 256)
    = blocks_in_use / total_physical_blocks
    = BU
```

对于 vLLM，过程完全相同：

```
PME = (blocks_in_use × 8192 × 44) / (total_blocks_allocated × 8192 × 44)
    = blocks_in_use / total_blocks_allocated
    = BU
```

**这是一个代数的必然结果，与 block 满不满完全无关。** 无论 CUDA VMM 还是 PyTorch 分配器，PME 都退化为 BU。

### 3.2 问题所在

`ideal_physical_bytes` 的定义是 `blocks_in_use × block_bytes`——它隐含假设**每个 in-use block 都是 100% 满的**。Block 内部有多少 slot 空着，有多少物理内存其实没存任何 KV 数据，PME 完全看不到。

举个例子。4 并发，28 blocks in use，256 total_tokens：

```
当前 PME 计算:
  ideal = 28 × 8192 × 44 = 10.1 MiB     ← 假设 28 个 block 全满
  actual = 1 × 2MiB × 44 = 88 MiB       ← 1 个 superblock
  PME = 10.1 / 88 = 0.115 = BU ✓

如果要捕捉"真正存了有用 KV 数据的物理内存":
  ideal_correct = 256 × 22528 = 5.5 MiB  ← 只有 256 个 token 的真实数据
  actual = 88 MiB
  PME_correct = 5.5 / 88 = 0.0625        ← 远小于 BU=0.115

差值 0.115 − 0.0625 = 0.0525 = 内部碎片（block 没满）在物理内存维度的体现
```

**由于 PME ≡ BU，UFS 实际上只有三个有效指标（IFR、BU、RFI），而不是四个。** PME 没有独立的信息量。

### 3.3 修复方案

若将 `ideal_physical_bytes` 改为使用真实 token 数：

```rust
let ideal_physical_bytes = (total_tokens * BPT) as u64;
// 而非 (blocks_in_use * block_bytes * num_layers * 2)
```

则四个指标各自正交：

```
PME  = total_tokens × BPT / (sb_count × 2MiB × 44)     ← 物理内存维度
IFR  = 1 − total_tokens / total_slots                   ← block 内部维度
BU   = blocks_in_use / total_blocks_allocated            ← block 池维度
RFI  = 1 − total_tokens × BPT / actual_active_bytes      ← 活跃超块维度

且满足: PME ≈ BU × (1 − IFR)                             ← 当 blocks_in_use ≈ blocks_used_by_seqs 时精确
```

---

## 4. 为什么 RFI = IFR（vLLM）但 RFI > IFR（Baseline）？

### 4.1 根源：actual_active_bytes 的向上取整

两个系统的 RFI 公式相同：

```
RFI = 1 - (total_tokens × BPT) / actual_active_bytes
```

区别全在 `actual_active_bytes` 怎么算。

**vLLM（无 superblock）：**

```
actual_active_bytes = blocks_in_use × 8192 × 44
```

代入 RFI，消去公因子 `8192 × 44`：

```
RFI_vllm = 1 − total_tokens × (8192/16) × 44
           ─────────────────────────────────
              blocks_in_use × 8192 × 44

         = 1 − (total_tokens / 16) / blocks_in_use
         = 1 − total_tokens / (blocks_in_use × 16)
         = IFR                        ← blocks_in_use ≈ blocks_used_by_seqs 时恒等
```

**Baseline（CUDA VMM，2 MiB superblock）：**

```
actual_active_bytes = ⌈blocks_in_use / 256⌉ × 2MiB × 44
```

注意 `⌈·/256⌉`——向上取整。CUDA VMM 的最小物理操作粒度是 2 MiB，哪怕只有 1 个 block 在被使用，整个 superblock 也必须被计数为「活跃」。

将 `2MiB = 256 × 8192` 代入：

```
actual_active_bytes = ⌈blocks_in_use/256⌉ × 256 × 8192 × 44
                    = rounded_up_blocks × 8192 × 44

其中 rounded_up_blocks = ⌈blocks_in_use/256⌉ × 256
```

代入 RFI：

```
RFI_baseline = 1 − total_tokens × (8192/16) × 44
               ─────────────────────────────────
               ⌈blocks_in_use/256⌉ × 256 × 8192 × 44

             = 1 − total_tokens / (⌈blocks_in_use/256⌉ × 256 × 16)
```

### 4.2 数值对比：同一场景

假设 `blocks_in_use=28, total_tokens=256`：

| | vLLM | Baseline | 差异来源 |
|---|------|---------|---------|
| IFR | 1 − 256/(28×16) = 1 − 256/448 = **0.43** | 同左 = **0.43** | 相同 |
| RFI 分母 | 28 × 16 = **448 slots** | ⌈28/256⌉ × 256 × 16 = 1 × 256 × 16 = **4,096 slots** | **9.14 倍** |
| RFI | 1 − 256/448 = **0.43** | 1 − 256/4096 = **0.94** | ⌈·/256⌉ 取整 |
| **RFI − IFR** | **0** | **0.51** | **纯 superblock 粒度浪费** |

### 4.3 直观解释

```
vLLM 的 actual_active_bytes:
  ┌──────────────────────────────────────────────────┐
  │████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░│
  │← 28 blocks in use = 3584 bytes (K层) →           │
  │  每个 block 可以独立分配/释放，没有更大的粒度约束    │
  └──────────────────────────────────────────────────┘
  RFI 分母 = 28 blocks 的物理内存，空闲 block 不计入

Baseline 的 actual_active_bytes:
  ┌────────────────── 1 superblock = 256 blocks ──────────────────────┐
  │████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
  │← 28 blocks used →← 这 228 个空闲 block 也被迫算在 RFI 分母里 →  │
  │   因为 cuMemMap 以 2MiB 为单位，superblock 不能部分物理分配        │
  └────────────────────────────────────────────────────────────────────┘
  RFI 分母 = 256 blocks 的物理内存，包括所有同 superblock 的空闲 block
```

**本质：vLLM 的最小分配粒度是 1 个 block（8 KiB per K layer），CUDA VMM 是 256 个 block（2 MiB）。粒度差 256 倍，RFI 分母差 256 倍。**

---

## 5. Superblock 锯齿效应

### 5.1 碎片率不是平滑变化的

由于物理内存以 2 MiB × 44 = 88 MiB 为单位跳跃增长，碎片率不是平滑函数，而是在 superblock 边界处剧烈振荡。

GPU 模拟测试（avg 28.6 并发，200 请求，bimodal prompt）中的 6 个采样快照：

| step | active seqs | mem_not_free | mem_tokens | ratio | 解释 |
|------|------------|-------------|-----------|-------|------|
| 0 | 32 | 2 MiB | 1.54 MiB | **0.23** | 填了 77%，在一个 SB 内，不错 |
| 64 | 32 | **4 MiB** | 2.15 MiB | **0.46** | 刚跨过 2 MiB，第二个 SB 几乎全空 |
| 128 | 32 | 4 MiB | 2.14 MiB | **0.47** | 仍在第二个 SB 边缘 |
| 192 | 32 | 2 MiB | 1.89 MiB | **0.06** | 回落到一个 SB 内，效率极好！ |
| 256 | 32 | **4 MiB** | 1.92 MiB | **0.52** | 1.92 MiB 略超 2 MiB 边界，却需付 4 MiB |
| 320 | 8 | 2 MiB | 1.49 MiB | **0.26** | 再次回落（并发降低，token 总量减少） |

**step 192 的 ratio=0.06 说明：当 token 总量恰好在一个 superblock 容量以内时，效率可以非常好（94% 利用率）。但 step 256 的 ratio=0.52 说明：仅仅多出 ~0.03 MiB 的 K 层数据（1.92 vs 1.89 MiB），就要分配第二个 superblock，效率瞬间减半。**

### 5.2 锯齿图案

```
  RFI
  1.0 ┤
      │
  0.8 ┤
      │
  0.6 ┤           ┌─┐         ┌─┐
      │           │ │         │ │
  0.4 ┤     ┌─┐   │ │   ┌─┐   │ │   ┌─┐
      │     │ │   │ │   │ │   │ │   │ │
  0.2 ┤ ┌─┐ │ │ ┌─┘ └─┐ │ │ ┌─┘ └─┐ │ │ ┌─
      │ │ │ │ │ │     │ │ │ │     │ │ │
  0.0 ┼─┴─┴─┴─┴─┴─────┴─┴─┴─┴─────┴─┴─┴──→ total_tokens
      0       4096       8192      12288    (per K layer)

  每次跨过 4096 的倍数，RFI 就跳升一次，然后随 token 增多逐渐回落。
```

**这不是 bug，是 2 MiB 最小物理粒度的必然数学结果。** 只要 CUDA VMM 要求以 2 MiB 为单位分配，锯齿就不可避免。唯一的缓解手段是提高并发数，使 total_tokens 远大于 4,096 的阈值。

---

## 6. 碎片化的三层嵌套结构

### 6.1 Baseline（CUDA VMM）的三层浪费

```
┌──────────────────────────── 88 MiB physical (1 superblock × 44 layers) ────────────┐
│                                                                                    │
│  ┌─────────────────────── 256 blocks allocated ────────────────────────┐           │
│  │                                                                     │           │
│  │  ┌── 28 blocks in use ──┐                                          │           │
│  │  │ ██████████████████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│           │
│  │  │   BU = 28/256 = 0.11 │   空闲 block 浪费                         │           │
│  │  └──────────────────────┘   (1 − BU = 0.89)                        │           │
│  │                                                                     │           │
│  │  这 228 个空闲 block 的物理内存随 superblock 已被分配，无法单独释放     │           │
│  └─────────────────────────────────────────────────────────────────────┘           │
│                                                                                    │
│  第 3 层：即使 28 个 in-use block 里面，也不是全满的：                               │
│    只有 256 个 token 的数据 (256 × 22528 = 5.5 MiB)                                │
│    IFR = 1 − 256/(28×16) = 0.43                                                    │
│                                                                                    │
│  综合浪费 (RFI) = 1 − (256 × 22528) / (1 × 2MiB × 44)                              │
│                  = 1 − 5.5 MiB / 88 MiB                                             │
│                  = 0.94                                                             │
│                                                                                    │
│  其中: IFR 贡献 ≈ 0.10 (占 10.6%)                                                     │
│       Superblock 粒度贡献 ≈ 0.84 (占 89.4%)                                           │
└────────────────────────────────────────────────────────────────────────────────────┘
```

**三层浪费的数值分解（4 并发吞吐 benchmark）：**

| 层级 | 指标 | 数值 | 浪费的物理内存 | 占总浪费比 |
|------|------|------|---------------|-----------|
| 第一层 | 内部碎片 (IFR) | 0.43 | ~4.6 MiB | 5.2% |
| 第二层 | Block 池未使用 (1−BU) | 0.89 | ~77.9 MiB | 88.6% |
| 第三层 | SB 粒度（隐含在 RFI 分母） | — | — | 6.2% |
| **合计** | **RFI** | **0.94** | **~82.5 MiB** | **100%** |

> 注：第二层和第三层在 CUDA VMM 下紧密耦合——因为 block 池随 superblock 整体分配，BU 低本身就是因为 SB 粒度过大。实际可独立归因的只有 IFR。

### 6.2 vLLM（PyTorch 预分配）只有一层

```
┌───────── 20+ GB pre-allocated pool ───────────────────────┐
│                                                            │
│  ┌── ~200 blocks in use ──┐                                │
│  │█████████████████████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  └────────────────────────┘                                │
│                                                            │
│ IFR ≈ 0.002（几乎可以忽略）                                  │
│ 无 superblock 粒度浪费（RFI = IFR）                          │
│ 空闲 block 物理内存在预分配时已被占用，但 vLLM 没有             │
│ "不活跃 SB 也算活跃"的问题                                   │
└────────────────────────────────────────────────────────────┘
```

vLLM 没有 superblock 层级，所以它的碎片化只有一个来源：block 内部的 slot 浪费（IFR）。只要 block 填得足够满，碎片化就极低。

---

## 7. 哪些指标可以跨系统比较？

| 指标 | 可跨系统比较？ | 说明 |
|------|-------------|------|
| **IFR** | ✅ **可直接比较** | 公式完全相同，在所有系统中只取决于 block_size 和 workload |
| **BU** | ❌ **不可比较绝对值** | 分母含义不同：Baseline = 当前 SB 中的总 block 数（随负载增长）；vLLM = 启动时全预分配池（固定且极大）。只能比较**趋势**（是否随负载增长） |
| **PME** | ❌ **不可比较** | 恒等于 BU，没独立信息。且当前定义未捕捉内部碎片 |
| **RFI** | ⚠️ **差值可比** | RFI − IFR = 纯 superblock/allocator 粒度浪费。vLLM 该差值为 0，Baseline > 0 |
| **RFI − IFR** | ✅ **可直接比较** | 衡量「分配器最小粒度造成的额外浪费」，是跨系统比较的**最佳单一指标** |

### 7.1 正确的对比方式

**不要对比绝对 BU/PME 值。** 而应对比：

1. **IFR** — 内部碎片率，相同 workload 下应该接近
2. **RFI − IFR** — 分配器粒度浪费。Baseline > 0, vLLM = 0
3. **绝对物理内存用量** — 同一 workload 下，哪个系统实际使用了更多 GPU VRAM
4. **BU 随负载的变化趋势** — Baseline 的 BU 应从低负载的低值上升到高负载的高值（grow-on-demand 生效）

---

## 8. vLLM 测量中的陷阱：nvidia-smi Diff Trap

### 8.1 问题

vLLM benchmark 输出中的关键警告：

```
WARNING: could not query vLLM block pool, estimated 56301 blocks
```

56,301 这个数字是从 GPU VRAM 反推的（CONTEXT.md 定义的「nvidia-smi Diff Trap」）：

```
(24 GB VRAM − 模型权重 − 其他开销) / (block_bytes × num_layers × 2)
≈ 20.3 GB / (8192 × 44)
≈ 20.3 × 1024³ / 360,448
≈ 56,301 blocks
```

### 8.2 后果

用 56,301 作为 `total_blocks_allocated`，vLLM 的 BU 和 PME 完全失去意义：

| 时刻 | blocks_in_use | total_blocks_allocated | BU | 真实含义 |
|------|---------------|----------------------|-----|---------|
| 启动 | 0 | 56,301 | 0.000 | — |
| 4 并发稳态 | ~200 | 56,301 | 0.004 | 全预分配池中只有 0.4% 被使用 |
| 满载 | ~682 | 56,301 | 0.012 | 全预分配池中只有 1.2% 被使用 |

**这些 BU 数字只是在说「vLLM 的预分配池很大」，不反映 vLLM 的内存管理效率。** 要获得有意义的 BU，必须从 vLLM 日志或 `/metrics` 端点直接查询 `num_gpu_blocks`。

### 8.3 正确做法

vLLM 的 `num_gpu_blocks` 可以从以下渠道获取：
- 服务器启动日志：`"Num GPU blocks: XXXX"`
- `/metrics` 端点（如果启用了 Prometheus）
- vLLM API 的内部接口

**永远不要用 `nvidia-smi` 差值来估算 KV cache 的 block 数。** 预分配池在基线测量中已被计入，差值法会隐藏整个预分配池。

---

## 9. 负载-效率曲线：Grow-on-Demand 的完整画像

### 9.1 实验数据

将三次 Baseline 测试的结果排列，揭示负载与碎片化的关系：

| 场景 | 并发数 | 物理内存 | IFR | BU | RFI | 浪费主导因素 |
|------|--------|---------|-----|-----|-----|------------|
| 吞吐 benchmark | 4 | **88 MiB** | 0.51 | 0.12 | **0.94** | SB 粒度（89%） |
| GPU 模拟测试 | ~29 | **264 MiB** | 0.04 | 0.51 | **0.33** | SB 粒度（~80%） |
| 最大并发测试 | 1,024 | **1,936 MiB** | — | **0.97** | — | 极低 |

### 9.2 趋势解读

```
BU ↑
1.0 ┤                                           ● (1024 concurrent, BU=0.97)
    │                                      ......
    │                              ......
0.8 ┤                      ......
    │              ......
0.6 ┤      ......
    │  ...             ● (29 concurrent, BU=0.51)
0.4 ┤
    │
0.2 ┤  ● (4 concurrent, BU=0.12)
    │
0.0 ┼──────────────────────────────────────────────────────────→ 并发数
    0     200     400     600     800    1000

RFI ↑
1.0 ┤  ● (4 concurrent, RFI=0.94)
    │
0.8 ┤
    │
0.6 ┤      ......
    │          ......
0.4 ┤              ● (29 concurrent, RFI=0.33)
    │                  ......
0.2 ┤                      ......
    │                          ......   ● (estimated <0.05 at 1024)
0.0 ┼──────────────────────────────────────────────────────────→ 并发数
    0     200     400     600     800    1000
```

**Grow-on-demand 的核心特征：**
- 低负载时 BU 差（12%），但绝对物理内存用量极低（88 MiB = 0.35% VRAM）
- 随负载增长，BU 持续改善至 97%
- RFI 的改善在跨过 4,096 tokens/K-SB 阈值后尤为显著

### 9.3 生产负载估计

对于 TinyLlama，设典型生产场景为 seq_len≈128, 32 并发：
- total_tokens/K-layer ≈ 32 × 128 = 4,096 → 恰好填满 1 个 K-SB
- 超过此阈值后，每增加一个 SB 都是接近满载的
- 预测 RFI < 0.1 at 64+ 并发

---

## 10. 架构权衡总结

| 维度 | Baseline (CUDA VMM) | vLLM (PyTorch) |
|------|---------------------|----------------|
| **分配策略** | Grow-on-Demand | Pre-allocation |
| **最小物理粒度** | 2 MiB superblock (256 blocks × 44 layers = 88 MiB) | 1 block (~360 KiB per all layers) |
| **低负载 VRAM 占用** | **88 MiB**（极低） | ~20 GB（占满 85% VRAM） |
| **低负载碎片率** | RFI=0.94（差，但绝对值极小） | IFR≈0.002（极低） |
| **高负载碎片率** | RFI≈0.05-0.33（好） | IFR≈0.002（极低） |
| **运行时分配开销** | ~254 µs/map + ~11 ms/superblock | ~0（全预分配） |
| **可释放空闲内存** | ✅ 可以（cuMemUnmap + cuMemRelease） | ❌ 不可以（PyTorch 持有） |
| **虚拟内存控制** | ✅ 显式（cuMemMap/cuMemUnmap） | ❌ 无 |
| **最佳场景** | 负载波动大、VRAM 受限、多租户 | 稳态高负载、追求极致吞吐 |
| **RFI − IFR** | > 0（superblock 粒度浪费） | = 0（无此浪费） |

**两种策略没有绝对的优劣，取决于场景。** Baseline 用「低负载时碎片率数字难看」换取了「低负载时几乎不占 VRAM」和「高负载时效率接近 vLLM」的弹性。

---

## 11. 已知问题与改进建议

### 11.1 PME 冗余

**问题：** PME 在两个系统中恒等于 BU，没有提供独立信息。

**建议：** 将 `ideal_physical_bytes` 改为 `total_tokens × BPT`，使四个 UFS 指标各自正交：
- IFR：block 内部浪费
- BU：block 池利用率
- PME（修复后）：物理内存中存了多少真实 KV 数据
- RFI：活跃超块中的综合浪费

### 11.2 vLLM total_blocks_allocated 估算错误

**问题：** 无法查询 vLLM block pool 时使用 `nvidia-smi` 差值估算 56,301 blocks，导致 BU/PME 无意义。

**建议：** 
1. 从 vLLM 启动日志解析 `num_gpu_blocks`
2. 或通过 vLLM `/metrics` 端点获取
3. 在无法获取时，明确标注 BU/PME 为「不可用」，而非展示无意义的小数

### 11.3 缺少多并发级别的碎片化报告

**问题：** 目前只有一个并发级别（4）的吞吐 benchmark。单一数据点不足以展示 grow-on-demand 的效率曲线。

**建议：** 使用 `--stress-concurrency` 模式，在多个并发级别（1, 2, 4, 8, 16, 32, 64, 128）下报告 UFS 指标，绘制负载-效率曲线。

### 11.4 EOS 控制的对齐

**问题：** Baseline benchmark 固定生成 64 token（EOS 被抑制），vLLM 允许 EOS 提前结束。这导致两个系统在 benchmark 中生成了不同数量的 token，影响了 IFR 和性能对比的公平性。

**建议：** vLLM benchmark 也应配置 EOS 抑制（通过设置不可达的 `eos_token_id`），使两个系统生成相同数量的 token。CONTEXT.md 中已将其列为「EOS-Controlled Benchmark」。

---

## 附录 A：关键源码引用

| 概念 | 文件 | 行号 |
|------|------|------|
| UFS 指标计算（Baseline） | `src/cache/unified_frag.rs` | 99-175 (`from_cache`) |
| UFS 指标计算（vLLM, from_raw） | `src/cache/unified_frag.rs` | 185-243 (`from_raw`) |
| 运行时碎片追踪（含 legacy 指标） | `src/cache/fragmentation_tracker.rs` | 55-121 (`record`) |
| UFS 汇总统计 | `src/cache/unified_frag.rs` | 275-337 (`from_samples`) |
| 术语定义 | `CONTEXT.md` | 全文 |

## 附录 B：实验数据文件

| 文件 | 说明 |
|------|------|
| `results/baremetal/step3_compare_20260601_053321/baseline_llama_output.txt` | Baseline 吞吐 benchmark 输出 + UFS 汇总 |
| `results/baremetal/step3_compare_20260601_053321/baseline_gpu_tests.txt` | GPU 单元测试（含模拟碎片化测试） |
| `results/baremetal/step3_compare_20260601_053321/baseline_llama_results.frag.csv` | 308 个 UFS 采样点 |
| `results/baremetal/step3_compare_20260601_053321/vllm_output.txt` | vLLM 综合 benchmark（并发、碎片、吞吐） |
| `results/baremetal/step3_compare_20260601_053321/vllm_fragmentation.csv` | 57 个 UFS 采样点 |
| `results/baremetal/step3_compare_20260601_053321/fragmentation.json` | vLLM 专用碎片测试数据 |
| `results/baremetal/step3_compare_20260601_053321/max_concurrency.json` | vLLM 最大并发 ramp 数据 |
| `results/baremetal/step3_compare_20260601_053321/throughput.json` | vLLM 吞吐 + UFS 详细数据 |
