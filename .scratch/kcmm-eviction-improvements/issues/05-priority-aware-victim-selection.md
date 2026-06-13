# Priority-aware victim selection

Status: done
Type: AFK

## What to build

Hint API（`kcmm_hint_t`）已经定义了 `KCMM_HINT_HIGH_PRIORITY`、`KCMM_HINT_LOW_PRIORITY`、`KCMM_HINT_EVICTABLE` 等标记，但当前实现只是调 `touch()` / `cool()`——在 LRU policy 下等于改了访问时间戳，影响微弱且不保证语义。

让 scheduler 的 `select_victim()` 真正理解 sequence 优先级：
- `LOW_PRIORITY` / `EVICTABLE` sequence 优先选为 victim（即使 epoch 更近）
- `HIGH_PRIORITY` sequence 只在没有其他可选时才考虑
- 同一 priority class 内部仍按 epoch-based LRU 排序

需要在 `SequenceState` 中增加一个 `priority: SequencePriority` 字段，并在 hint API（`kcmm_hint`）中写入。优先级应可动态变更（一个 batch job 中途可能被降级）。

## Acceptance criteria

- [ ] `SequenceState` 增加 `priority: SequencePriority` 枚举（`High`, `Normal`, `Low`, `Evictable`）。
- [ ] `kcmm_hint` / `kcmm_hint_sequence` 真正写入 priority 字段而非仅调 `touch`/`cool`。
- [ ] `select_victim` 按 priority tier 分层选择：先扫 `Evictable`，再扫 `Low`，最后 `Normal`/`High`。
- [ ] `HIGH_PRIORITY` sequence 的 eviction 需要额外的保护（如更高的 water mark 阈值）。
- [ ] Benchmark 验证：在 mixed-priority workload 下，`HIGH_PRIORITY` sequence 的 eviction 次数显著低于 `LOW_PRIORITY`。
- [ ] `EVICTABLE` hint 标记的 sequence 在内存压力下最先被整 sequence 丢弃（不 restore）。
- [ ] 集成 benchmark compile check 通过。

## Blocked by

None - can start immediately. 但 #02（block-granularity eviction）会使 priority 在 partial eviction 场景下更加灵活，建议在其后或并行做。
