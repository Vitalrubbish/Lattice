# Implement block-granularity victim selection in the scheduler

Status: done
Type: AFK

## What to build

当前 `ContinuousScheduler::select_victim()` 选中一个 victim sequence 后，`admit_waiting()` 把它**全部 block** 传给 `tiering.evict_blocks(&pool, &candidates, candidates.len())`。这导致 evict 的 block 数 = sequence 的全部 block 数，可能远超实际需要的 free block 数量，造成不必要的 D2H 带宽消耗和后续 restore 开销。

`EvictionPolicy::select_victims(candidates, count)` trait 方法已经支持从候选 block 中只选出 `count` 个——但这个能力从未被真正使用（`count` 总是等于 `candidates.len()`）。

改为：
- admission path 计算实际需要释放多少个 block（`needed = required - free`）
- `select_victim` 仍选 sequence，但只取该 sequence 中由 `EvictionPolicy` 排出的前 `needed` 个 block
- 如果单个 sequence 的 block 不够，继续选下一个 victim sequence

这样保留了整 sequence 作为 victim 选择单位的简单性，同时实现了 partial eviction 减少浪费。

## Acceptance criteria

- [ ] `admit_waiting()` 或等效路径计算 `needed_blocks = required_blocks.saturating_sub(free_blocks)`。
- [ ] Victim selection loop 累积候选 block 直到满足 `needed_blocks`，可能在多个 sequence 间分摊。
- [ ] `evict_blocks` 只 evict 实际需要的 block 数而非全 sequence。
- [ ] `EvictionPolicy::select_victims` 的 `count` 参数在生产路径上小于 `candidates.len()` 的情况被真实覆盖。
- [ ] 集成 benchmark 验证：被 evict 的 sequence 在 restore 后仍可继续 decode（partial eviction 不影响正确性）。
- [ ] 新增单元测试：`test_partial_sequence_eviction`、`test_multi_sequence_eviction_in_one_batch`。
- [ ] 集成 benchmark compile check 通过。

## Blocked by

None - can start immediately. 建议先做 #01（background eviction），因为 proactive eviction + partial eviction 之间有协同效应——background path 可以更激进地做小批量 partial evict。
