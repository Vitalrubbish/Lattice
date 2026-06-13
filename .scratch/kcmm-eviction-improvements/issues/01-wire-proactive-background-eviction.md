# Wire proactive background eviction

Status: done
Type: AFK

## What to build

`KcmmPool::below_low_watermark(threshold)` 已经实现——它比较空闲物理 block 比例与阈值，返回 `true` 表示自由 block 不足。但当前没有任何调用方真正使用它来**提前触发** eviction。

当前 eviction 只在 admission path 上被动发生：admission 失败 → `select_victim` → evict → retry。这意味着 eviction 延迟直接打在请求关键路径上。

加一个 background driver（thread 或 tokio task），周期性地检查 low watermark，当空闲 block 比例低于可配置阈值时，**不等 admission 失败就提前**将冷 block evict 到 CPU swap buffer，保证 admission path 始终有充足的 free block。

background driver 应使用与 admission path 相同的 `EvictionPolicy` 来选择 victim，并且每次 evict 的 block 数量应可配置（batch size）。

## Acceptance criteria

- [ ] `KcmmConfig` 增加 `low_watermark_threshold: f32`（默认 0.2）和 `background_evict_batch: usize`（默认 16）。
- [ ] 一个 background driver（thread 或 async task）周期性地调用 `below_low_watermark`，低于阈值时触发 `evict_blocks`。
- [ ] background eviction 不与 admission-path eviction 竞争（使用同一 policy mutex 时的锁争用可接受，但不能破坏 policy 状态）。
- [ ] `KcmmMetrics` 新增 `background_eviction_count` 字段并在此路径上递增。
- [ ] 集成 benchmark 中 admission 延迟的 tail latency（p99）在启用 background eviction 时有所改善，或至少不退化。
- [ ] 集成 benchmark compile check 通过。
- [ ] 新增单元测试：`test_background_eviction_triggered_below_watermark`。

## Blocked by

None - can start immediately.
