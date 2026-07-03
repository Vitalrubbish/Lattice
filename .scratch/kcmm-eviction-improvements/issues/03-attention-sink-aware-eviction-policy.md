# Attention-sink-aware eviction policy

Status: done
Type: AFK

## What to build

实现一个新的 `EvictionPolicy` 实现——`SinkWindowPolicy`，它知道 attention sink 的存在：一个 sequence 中前几个 token（sink tokens）被几乎所有后续 token attend，evict 它们的代价极高；最近几个 token（local window）是下一个 decode step 的必需输入。

这个 policy 在 `select_victims` 时：
- **永不 evict** 前 `S` 个 token 对应的 block（sink blocks）
- **永不 evict** 最后 `W` 个 token 对应的 block（window blocks）
- 只从中间区域选 victim（按 LRU 排序）

效果上等价于 StreamingLLM 的 eviction 策略，使得长 sequence 可以在恒定 GPU 内存下持续 decode。

`S` 和 `W` 应为 `KcmmConfig` 的可配置参数（`attention_sink_blocks: usize`，默认 1；`recent_window_blocks: usize`，默认 4）。

Block 需要一个方式知道自己在 sequence 中的 token 位置——当前 `BlockInfo` 没有这个字段。需要在不破坏现有抽象的前提下增加此信息，或通过 `EvictionPolicy` 的 `on_allocate` 扩展来接收位置信息。

## Acceptance criteria

- [ ] 实现 `SinkWindowPolicy` 并注册为 `"sink_window"` policy。
- [ ] `KcmmConfig` 增加 `attention_sink_blocks` 和 `recent_window_blocks`。
- [ ] Block 的 position-in-sequence 信息以合理方式提供给 policy（扩展 `on_allocate` 签名，或通过 per-block metadata）。
- [ ] `SinkWindowPolicy::select_victims` 跳过 sink 和 window 区域的 block。
- [ ] 新增长 sequence benchmark：`test_sink_window_policy_long_sequence`，验证 sink+window 外的 block 被 evict 后 decode 仍正确。
- [ ] 对比 benchmark：在同一 workload 下，`sink_window` vs `lru` 在 block 不足时的 decode 成功率和 eviction count。
- [ ] 集成 benchmark compile check 通过。

## Blocked by

- `#02-block-granularity-victim-selection-in-scheduler` — sink-window policy 只有在 partial eviction 可用时才有意义（整 sequence eviction 下无法保留 sink+window）。
