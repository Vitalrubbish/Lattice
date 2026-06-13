# Adaptive Replacement Cache (ARC) policy

Status: done
Type: AFK

## What to build

实现 ARC（Adaptive Replacement Cache）作为新的 `EvictionPolicy`，配置名 `"arc"`。

ARC 维护两个 LRU 链表（T1：recent，T2：frequent）和对应的 ghost list（B1，B2）。核心思想：
- 首次访问的 block 进入 T1（只被访问过一次）
- 在 T1 中再次被访问的 block 晋升到 T2（被访问过多次）
- Ghost list B1/B2 记录最近被 evict 的 block 的 metadata（不存数据），用于学习 workload pattern
- 当 B1 命中时增大 T1 容量（说明 recent 更重要），当 B2 命中时增大 T2 容量（说明 frequent 更重要）
- 目标容量 p 在 [0, c] 之间动态调整，自动适应 access pattern

ARC 相比纯 LRU 的优势：
- **抗扫描污染**：一次 sequential scan 不会把整个 T2 冲掉
- **自动平衡 recency 和 frequency**：无需手动调参
- **在 mixed workload 下命中率更好**

Ghost list 不存储实际 block 数据，只存储 `BlockHandle` + 少量 metadata（如 access count），内存开销可忽略。

## Acceptance criteria

- [ ] 实现 `ArcPolicy`，包含 T1、T2、B1、B2 四个链表和自适应 p 值。
- [ ] Ghost list 在 `on_evict` 时记录被 evict 的 block handle。
- [ ] Ghost hit（B1/B2 命中）正确调整 p 值。
- [ ] `select_victims` 从 T1 尾部选取 victim（T1 优先于 T2）。
- [ ] 注册为 `"arc"` policy，可通过 config 和运行时 `set_policy` 切换。
- [ ] Benchmark sweep 对比 arc vs lru vs lfu 在以下场景的 eviction count 和命中率：
  - Sequential scan（扫描污染）
  - Zipfian access（有 hot/cold 区分）
  - Mixed（扫描 + hot set）
- [ ] ARC 在扫描场景的 eviction count 显著低于 LRU。
- [ ] 集成 benchmark compile check 通过。

## Blocked by

None - 纯新 policy，不依赖其他 issue。可以与现有 whole-sequence eviction 或 #02 的 partial eviction 配合使用。
