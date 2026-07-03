# KCMM 测试修复方案

## Context

在 `results/` 下两份测试报告中发现了 3 个异常：

1. **Batch Eviction 摊还因子 <1.0** — 批量化本该降低单块开销，但 batch=16 时反而慢 9×
2. **Memory Pressure Sweep 表格缺列** — 输出缺少 CappedB/CappedK，导致 `completed+capped+rejected` 不可验证
3. **Engine Integration Config 4 回归 (CapRat=0.97×)** — tiering ON 完成数反而更少，且无 thrashing 告警

---

## Fix 1: Batch Eviction Amortisation — 池容量不足导致测量中混入 superblock 创建

### 根因

测试用 `make_tiering_pool` 创建 `max_blocks=512` 的池（每个 layer 约 256 blocks）。每轮 alloc→evict→restore→free 周期中，`release_block_physical` 将 block 归还到**其原始 superblock**，但下一轮的 `alloc_one_block_internal` 可能从**不同 superblock** 分配。随着 superblock 间负载不均衡累积，某些 superblock 先耗尽 → 触发 `ensure_capacity()` → cuMemCreate + cuMemMap (~600µs per 2MB)。

- batch=1：30 轮 × 1 block/layer = 30 blocks/layer < 192 空闲 → 永不触发
- batch=16：30 轮 × 16 blocks/layer = 480 blocks/layer > 192 空闲 → **必触发**
- batch=64：触发但摊还到 64 blocks，per-block 影响较小

证据：batch=16 的分布呈双峰 — min=59µs（与 batch=1 相同），P50=602µs — 说明快速轮次和慢速轮次交替出现。

### 修复方案（仅修改测试）

**文件**: `tests/kcmm_bench_tiering.rs`

1. **将 `max_blocks` 从 512 提高到 4096**，保证 30 轮 × 64 blocks × 2 layers = 3840 blocks/layer 不触发 superblock 创建

2. **改进预热**：从单周期改为 5 个全周期（alloc 64 → evict 64 → restore 64 → free 64），充分稳定所有 lazy 分配

3. **增加 per-batch-size 预热**：每个 batch_size 测量前先跑 5 个静默轮次

4. **使用 median 替代 mean** 计算摊还因子，消除离群抖动

### 具体改动点

**Change 1** — `make_tiering_pool` 调用 (line 163)：`max_blocks: 512` → `max_blocks: 4096`

**Change 2** — 预热循环 (lines 175-185)：单周期 → 5 周期
```rust
for _ in 0..5 {
    let pairs = alloc_blocks(&pool, 64);
    // ... evict + restore + free
}
```

**Change 3** — 测量循环前增加 per-batch 预热 (after line 193)：
```rust
for _ in 0..5 {
    let pairs = alloc_blocks(&pool, batch_size);
    // ... evict + restore + free (不计时)
}
```

**Change 4** — 摊还因子计算 (line 216)：`mean(&per_block_latencies)` → `median(&mut per_block_latencies)`

### 验证

```bash
cargo test kcmm_bench_batch_eviction_amortization -- --nocapture
```

预期：摊还因子 batch=16 > 1.0×, batch=64 > batch=16

---

## Fix 2: Sweep Table 添加 Capped 列

### 根因

Sweep 表只输出 `RejB/RejK`（admission 阶段拒绝），不输出 `CappedB/CappedK`（decode 阶段 OOM）。导致无法验证 `completed + capped + rejected` 守恒。Single-config 测试本身有完整的三个字段。

### 修复方案

**文件**: `tests/kcmm_bench_memory_pressure.rs`，`kcmm_bench_memory_pressure_sweep` 函数

1. 表头增加 `CappedB` / `CappedK` 两列
2. 数据行输出 `baseline.capped` / `kcmm.capped`
3. 加宽分隔线

### 具体改动点

**Change 5** — 表头 (lines 787-789)：
```
Config  Base  KCMM  Ratio  RejB  RejK  CappedB  CappedK  Evict
```

**Change 6** — 数据行 (lines 813-823)：在 `kcmm.rejected` 之后插入 `baseline.capped` / `kcmm.capped`

### 验证

```bash
cargo test kcmm_bench_memory_pressure_sweep -- --nocapture
```

验证每行 `completed + capped + rejected` 与预期一致。

---

## Fix 3: Engine Integration Thrashing 检测

### 根因

Config 4 (`ari=4, max_batch=10, reqs=40`) 是极高 churn 场景。KCMM 执行 321 次 eviction + 41 次 restore 仅完成 35 个请求（~9.2 evictions/completion），tiering 开销超过了容量收益。这是合法场景但测试应显式告警。

### 修复方案

**文件**: `tests/kcmm_bench_engine_integration.rs`

1. Sweep 循环中，每行输出后检查 `evictions/completion > 3`
2. Single-config 测试的 box-table 分析区域也加入同样检测

### 具体改动点

**Change 7** — Sweep 循环 (after line 1323)：添加 thrashing 检测
```rust
if on.completed > 0 {
    let epc = on.eviction_count as f64 / on.completed as f64;
    if epc > 3.0 {
        println!("  ⚠️  Thrashing: {:.1} evictions/completion ({} evictions, {} completed)",
            epc, on.eviction_count, on.completed);
    }
}
```

**Change 8** — Single-config 分析 (after line 1205)：同样逻辑，在 "No evictions" warning 的 else 分支之后

### 验证

```bash
cargo test kcmm_engine_integration_sweep -- --nocapture
```

预期 Config 4 输出 `⚠️  Thrashing: ~9 evictions/completion`，Config 1-3 不输出。

---

## 修改文件清单

| 文件 | 改动数 | 风险 |
|------|--------|------|
| `tests/kcmm_bench_tiering.rs` | 4 处 | 低 — 仅测试参数调整 |
| `tests/kcmm_bench_memory_pressure.rs` | 2 处 | 无 — 仅输出格式 |
| `tests/kcmm_bench_engine_integration.rs` | 2 处 | 无 — 仅输出格式 |

总计：**3 个文件，8 处改动，无生产代码修改**。

## 完整验证

```bash
# 全部 KCMM 测试
cargo test --features kcmm --release -- \
  kcmm_bench_batch_eviction_amortization \
  kcmm_bench_memory_pressure_sweep \
  kcmm_engine_integration_sweep \
  kcmm_engine_integration_single \
  --nocapture
```
