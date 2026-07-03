# Eviction observability and metrics

Status: done
Type: AFK

## What to build

`KcmmMetrics` 结构体已定义了 `eviction_count: u64` 和 `restoration_count: u64` 字段，但它们在 tiering engine 中从未被 populate（注释标记为 future work）。

为 eviction 子系统加上完整的 observability：

**计数器**：
- `eviction_count`：GPU→CPU eviction 总次数
- `restoration_count`：CPU→GPU restoration 总次数
- `evicted_blocks_total`：累计被 evict 的 block 数
- `restored_blocks_total`：累计被 restore 的 block 数
- `eviction_failures`：eviction 失败次数（如 CPU slot 不足）
- `restoration_failures`：restoration 失败次数

**per-policy 统计**（按当前 active policy 标签分组）：
- eviction count per policy
- average blocks per eviction batch

**延迟 histogram**：
- eviction 延迟（从 `evict_blocks` 调用到 `cuStreamSynchronize` 完成）
- restoration 延迟

**Benchmark 输出**：
- 集成 benchmark sweep 表格增加 eviction-related 列（evictions/completion, avg evict latency）
- 使不同 config（policy、block_size、max_blocks）下的 eviction 行为可直接对比

## Acceptance criteria

- [ ] `KcmmMetrics` 中的 `eviction_count` 和 `restoration_count` 在每次 evict/restore 操作时递增。
- [ ] 新增 `evicted_blocks_total`、`restored_blocks_total`、`eviction_failures`、`restoration_failures` 计数器。
- [ ] 延迟 histogram 记录每次 eviction 和 restoration 的 wall-clock 时间。
- [ ] per-policy 统计在运行时 policy 切换时正确分桶。
- [ ] 集成 benchmark sweep 输出包含 eviction metrics 列。
- [ ] Benchmark 结果 CSV/JSON 包含所有 eviction 指标。
- [ ] 新增单元测试：`test_eviction_metrics_increment`、`test_restoration_metrics_increment`。
- [ ] 集成 benchmark compile check 通过。

## Blocked by

None - can start immediately. 建议在其他 eviction 改进之前先做，因为它们都需要 metrics 来验证效果。
