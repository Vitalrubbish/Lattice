# KCMM 跨进程 GPU 显存管理设计

**Date:** 2026-06-13
**Status:** Design document
**Related:** [`kcmm-baremetal-plan.md`](kcmm-baremetal-plan.md), [`kcmm-ffi-roadmap.md`](kcmm-ffi-roadmap.md), [`kcmm-implementation-analysis.md`](../task/kcmm-implementation-analysis.md)

---

## 1. 动机

### 1.1 现状问题

当前 KCMM 采用 per-process 模型：每个推理进程独立创建 `KcmmPool`，各自管理 GPU 显存。这导致：

```
进程A (vLLM)              进程B (SGLang)
┌──────────────┐          ┌──────────────┐
│  KcmmPool    │          │  KcmmPool    │
│  128 blocks  │          │  128 blocks  │
│              │          │              │
│  独立 LRU    │          │  独立 LRU    │
│  独立水位线  │          │  独立水位线  │
│  独立 Tiering│          │  独立 Tiering │
└──────────────┘          └──────────────┘
───────────────────────────────────────────
      GPU 显存（被割裂为孤岛）
```

| 问题 | 表现 |
|------|------|
| **全局信息缺失** | 每个池只看自己的碎片率和压力，无法做出全局最优决策 |
| **资源割裂** | 池 A 空闲块不能被池 B 利用，即使 B 正在 OOM |
| **Victim 选择局部最优** | 每个池独立选 victim，可能是全局最不该换出的热块 |
| **协调成本高** | 跨池协调需要额外的分布式协议（如跨池块借用），复杂且易出错 |

### 1.2 核心洞察

**GPU 显存管理的本质决定了它应该是全局的。** 这与操作系统中内核作为全局内存管理者、而非每个进程自己管理物理页的原因相同：

- 物理显存是系统级稀缺资源，不是进程私有的
- 换出决策需要全局视角——从最冷的块开始，而不是从最满的进程开始
- 碎片化和容量利用只能从全局层面评估

KCMM 作为"GPU 显存的 OS 层"，**跨进程是架构的第一性原理，不是后装功能**。

---

## 2. 架构设计：KCMM Daemon

### 2.1 整体架构

```
进程A (vLLM)           进程B (SGLang)          进程C (TRT-LLM)
    │                       │                       │
    │ libkcmm.so            │ libkcmm.so            │ libkcmm.so
    │ (轻量 client)          │ (轻量 client)          │ (轻量 client)
    │                       │                       │
    │ 职责：                 │                       │
    │ - 注册本进程 CUDA ctx  │                       │
    │ - 发送 RPC 请求        │                       │
    │ - 管理本地 VA 映射     │                       │
    │                       │                       │
    └───────────┬───────────┴───────────┬───────────┘
                │                       │
                │  Unix Domain Socket   │
                │  + Shared Memory      │
                │                       │
                ▼                       ▼
┌──────────────────────────────────────────────────────────┐
│                     kcmm-daemon                           │
│                                                          │
│  全局唯一实例。管理所有 GPU 物理显存，作为跨进程的        │
│  内存管理中枢。                                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 全局 Block 池                                       │  │
│  │                                                    │  │
│  │ - 所有进程的分配来自同一个池                         │  │
│  │ - 天然看到全局内存压力                               │  │
│  │ - victim 选择天然跨进程（LRU 上的所有 block）        │  │
│  │ - 不存在"借块"——所有块本来就是全局的                 │  │
│  ├────────────────────────────────────────────────────┤  │
│  │ 全局 Tiering Engine                                 │  │
│  │                                                    │  │
│  │ - GPU↔CPU↔NVMe 传输统一调度                          │  │
│  │ - 所有进程共享同一 CPU buffer / NVMe 存储            │  │
│  │ - daemon 拥有 CUDA primary context，负责 GPU 操作    │  │
│  ├────────────────────────────────────────────────────┤  │
│  │ 全局 Eviction Policy                                │  │
│  │                                                    │  │
│  │ - 所有进程的 block 在同一 LRU/LFU/FIFO 上           │  │
│  │ - 天然从全局最冷的块换出，而非最满进程的块           │  │
│  │ - 策略切换对所有进程立即可见                        │  │
│  ├────────────────────────────────────────────────────┤  │
│  │ 全局水位线（Adaptive Watermark）                     │  │
│  │                                                    │  │
│  │ - 汇总所有进程的 alloc/free 速率                     │  │
│  │ - 信息完整 → 预测更准确                             │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  跨进程 GPU 物理页共享（CUDA IPC）：                       │
│    cuIpcGetMemHandle  →  daemon 内部交换 handle          │
│    cuIpcOpenMemHandle →  client 映射到自己的 VA 空间     │
└──────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────
                 GPU 显存（统一管理，无割裂）
```

### 2.2 Client 端（libkcmm.so）

每个推理进程链接 `libkcmm.so`，它是 daemon 的轻量 client stub：

```
libkcmm.so 在一个进程内的职责：

1. 连接管理
   - 启动时通过 Unix socket 连接到 kcmm-daemon
   - 注册本进程的 CUDA context 给 daemon

2. RPC 转发
   - 将 API 调用（alloc/free/touch/cool/...）封装为消息发给 daemon
   - 接收 daemon 返回的结果并返回给调用者

3. 本地 VA 管理
   - daemon 分配物理块后，返回 handle 给 client
   - client 将 handle 映射到本进程的 GPU VA 空间
   - 映射后的 VA 指针直接传给推理引擎的 attention kernel

4. 不做的事
   - 不做物理页管理（由 daemon 做）
   - 不做 tiering 决策（由 daemon 做）
   - 不维护 block 状态（由 daemon 做）
```

### 2.3 Daemon 端（kcmm-daemon）

Daemon 是一个独立的 Rust 二进制，在系统启动时以守护进程方式运行：

```
kcmm-daemon 的职责：

1. GPU 物理显存管理
   - 通过 CUDA VMM API（cuMemCreate/cuMemMap/cuMemUnmap）管理物理页
   - 持有 CUDA primary context
   - 管理所有超级块（superblock）的分配和回收

2. 全局 Block 池
   - 维护全局的 free list 和 in-use 列表
   - 维护 per-sequence metadata（哪个进程的哪个序列用了哪些块）
   - 维护全局引用计数

3. Tiering Engine
   - 管理 CPU buffer（mmap 的大块 pinned memory）
   - 管理 NVMe 存储（可选）
   - CUDA stream 管理和异步传输
   - Gather/scatter kernel 执行

4. Eviction Policy
   - 全局 LRU/LFU/FIFO 状态
   - victim 选择（天然跨进程）
   - 温控（HOT/WARM/COLD 标记）

5. 水位线监控
   - 汇总所有进程的分配/释放速率
   - 预判 OOM 并提前触发换出

6. 进进程管理
   - 跟踪连接的 client 进程
   - 进程异常退出时清理其持有的 block
   - 心跳检测
```

---

## 3. IPC 机制设计

### 3.1 控制路径：Unix Domain Socket

```
消息格式（简化）：

Request:
  ┌────────┬──────────┬──────────────────────┐
  │ opcode │  seq_id  │  payload             │
  │ 1 byte │  8 bytes │  variable            │
  └────────┴──────────┴──────────────────────┘

Response:
  ┌────────┬─────────────────────────────────┐
  │ status │  payload                        │
  │ 1 byte │  variable                       │
  └────────┴─────────────────────────────────┘

操作码：
  - ALLOC_SEQUENCE    → 返回 block_indices
  - FREE_SEQUENCE     → ()
  - ALLOC_BLOCK       → 返回 block_index
  - TOUCH             → ()
  - COOL              → ()
  - GET_BLOCK_HANDLE  → 返回 IPC mem handle
  - GET_STATS         → 返回 KcmmMetrics
  - SET_POLICY        → ()
  - SHUTDOWN          → ()
```

### 3.2 数据路径：CUDA IPC + 零拷贝 VA 映射

物理块的 GPU VA 映射流程：

```
1. Daemon 分配物理块
   daemon: cuMemCreate(&handle, size, &prop)
   daemon: cuMemMap(daemon_va, size, 0, handle, 0)

2. Daemon 导出 IPC handle
   daemon: cuIpcGetMemHandle(&ipc_handle, daemon_va_ptr)

3. Daemon 通过 Unix socket 发送 ipc_handle 给 client

4. Client 导入 IPC handle
   client: cuIpcOpenMemHandle(&client_va, ipc_handle, CU_IPC_MEM_LAZY_ENABLE)

5. Client 获得可直接用于 attention kernel 的 GPU VA 指针
   → 零拷贝。不经过 CPU。不经过 daemon。
```

```
                     Daemon                         Client
                    ┌──────────┐                 ┌──────────┐
 cuMemCreate ──→    │ 物理页   │                 │          │
 cuMemMap    ──→    │ VA:0x100 │                 │          │
                    │          │                 │          │
 cuIpcGetMemHandle  │          │                 │          │
 ──────────────────→│ ipc_hdl  │─── socket ──→   │ ipc_hdl  │
                    │          │                 │          │
                    │          │   cuIpcOpenMemHandle          │
                    │          │ ←──────────────────────────── │
                    │          │                 │ VA:0x200  │
                    │          │                 │           │
                    │    GPU 显存（同一块物理页）   │           │
                    │   ┌───────────────────┐    │           │
                    │   │   KV Cache Data   │    │           │
                    │   └───────────────────┘    │           │
                    │    ▲                  ▲     │           │
                    │    │ daemon CUDA stream│    │ client    │
                    │    │ (tiering 传输)    │    │ attention │
                    │    │                   │    │ kernel    │
                    └────┼───────────────────┼────┘           │
                         └───────────────────┘
                          两者直接读写同一物理页
```

### 3.3 为什么不需要共享内存做数据传输

CUDA IPC 映射后，client 的 attention kernel **直接读写 daemon 创建的物理页**，不需要中间拷贝。Tiering（D2H/H2D）由 daemon 在自己的 CUDA stream 上执行，对同一物理页的读写与 client 的 attention kernel 通过 CUDA stream 顺序保证正确性。

唯一的例外是控制面信息（如 block table 内容、sequence 元数据），这些通过 Unix socket 传递，数据量极小（每条消息几十字节），不需要共享内存。

---

## 4. 生命周期管理

### 4.1 Daemon 启动

```
1. 系统启动时，systemd 启动 kcmm-daemon（或手动启动）
2. Daemon 初始化 CUDA primary context
3. Daemon 通过 cuMemCreate 预分配物理显存池（或按需分配）
4. Daemon 创建监听 socket：/var/run/kcmm.sock
5. Daemon 初始化 CPU buffer（mmap）和可选的 NVMe 存储
6. Daemon 初始化全局 LRU policy
7. Daemon 进入事件循环，等待 client 连接
```

### 4.2 Client 连接

```
1. 推理进程启动
2. 调用 kcmm_pool_connect("/var/run/kcmm.sock") 或 kcmm_pool_create()
   → libkcmm.so 内部连接 daemon
3. 发送 REGISTER 消息：
   { pid, cuda_context_handle, requested_capacity, config }
4. Daemon 注册 client：
   - 记录 pid 和 CUDA context
   - 从全局池中为 client 预留容量（可超额分配）
   - 返回 client_id
5. 后续所有操作使用 client_id + seq_id 寻址
```

### 4.3 Client 断开

```
正常断开：
  1. Client 发送 SHUTDOWN
  2. Daemon 释放该进程所有 block 的引用
  3. ref_count = 0 的 block → 回到 free list
  4. ref_count > 0 的 block（被其他进程共享）→ 保留

异常断开（crash）：
  1. Daemon 检测到 socket 断开（EPOLLHUP / heartbeat 超时）
  2. Daemon 立即释放该进程持有的所有 block
  3. 其他进程不受影响
```

### 4.4 Daemon 重启

```
1. 所有 client 失去连接 → 进入降级模式
2. Client 可选：
   a. 继续运行（KV cache 全部丢失，推理引擎退化为无缓存模式）
   b. 等待 daemon 恢复后重连
3. Daemon 恢复后，client 自动重连、重新注册
4. 已有的 CPU/NVMe 缓存数据可以保留（tiering 恢复）
```

---

## 5. 全局 Eviction 的语义

### 5.1 Victim 选择

```
场景：全局池 256 块已满，进程 C 请求分配

    进程A                  进程B                  进程C
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │ 80 blocks│          │ 90 blocks│          │ 86 blocks│
  │          │          │          │          │          │
  │ 访问模式：│          │ 访问模式：│          │ 访问模式：│
  │ 热: 活跃  │          │ 热: 活跃  │          │ 冷: 刚启动│
  │ 大部分热块│          │ 全部热块  │          │ 全部冷块  │
  └──────────┘          └──────────┘          └──────────┘

传统 per-process：
  进程C 满了 → 从 C 的 86 个块中选 victim
  → 可能选中相对较热的块（C 的块虽然全局看冷，但内部看热）

KCMM daemon（全局 LRU）：
  全 256 块在同一 LRU 上排序
  → victim 自然是全局最久未访问的块
  → 如果 C 的块确实最冷 → 从 C 换出（公平）
  → 如果 A 有更冷的块 → 从 A 换出（全局最优）
```

### 5.2 公平性

daemon 不偏向任何进程。LRU 天然按"全局最近一次 touch 的时间"排序，哪个进程的块最久没被访问，就从哪里换出。

如果需要优先级（例如 VIP 租户），可以通过 Hint API 标记 `KCMM_HINT_HIGH_PRIORITY`，让某些进程的块在 LRU 上获得时间偏移，更难被选中。

---

## 6. 状态分布

### 6.1 哪些状态在 Daemon，哪些在 Client

```
┌─────────────────────────────────────────────────────────┐
│                       Daemon 端                          │
│                                                         │
│  BlockInfo (per block):                                  │
│    - superblock_idx, block_index_in_sb                   │
│    - va_offset                                          │
│    - in_use: bool                                       │
│    - location: BlockLocation (Gpu/Cpu/Nvme)             │
│    - ref_count: u32                                     │
│    - owning_seq: (client_id, seq_id)                     │
│                                                         │
│  SequenceState (per seq):                               │
│    - client_id, seq_id                                  │
│    - block_indices: Vec<u32>                             │
│    - seq_len: usize                                     │
│    - is_active: bool (hot/cold)                         │
│    - last_access: Instant                               │
│                                                         │
│  Global:                                                 │
│    - free_block_indices                                 │
│    - LRU access_times                                   │
│    - IPC handles cache                                  │
│    - connected_clients                                  │
│    - alloc_rate_ewma, free_rate_ewma                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                       Client 端                          │
│                                                         │
│  Per-sequence:                                          │
│    - seq_id (local handle, maps to daemon's seq_id)     │
│    - GPU VA pointers for each block's K/V region        │
│      → 这些 VA 是 cuIpcOpenMemHandle 返回的              │
│      → attention kernel 直接通过 VA 读写                 │
│                                                         │
│  Block table (per seq):                                 │
│    - Vec<*mut f16> 或 Vec<usize> (GPU VA per block)     │
│      → client 本地维护一份轻量的 block table 镜像         │
│      → 每次 alloc/free 操作后与 daemon 同步               │
│      → 推理时不需要跨进程调用                             │
└─────────────────────────────────────────────────────────┘
```

### 6.2 推理热路径：无 IPC

关键设计目标：**推理过程中的 attention kernel 调用不经过 daemon，不产生 IPC 开销。**

```
Client 推理热路径（每个 decode step）：

  1. token_ids = [next_tokens]                               ← 本地
  2. positions = [p+1 for p in positions]                    ← 本地
  3. block_table = [VA_ptr0, VA_ptr1, ...]                  ← 本地（已映射的 VA）
  4. attention_kernel(hidden, block_table, positions, ...)   ← 直接 GPU 操作
                                                             ← 不经过 daemon
                                                             ← 不经过 socket
                                                             ← 不需要锁
```

IPC 只发生在：
- `alloc_block()`：需要 daemon 分配新物理块 + 返回 IPC handle
- `free_sequence()`：通知 daemon 释放
- `touch()`/`cool()`：更新 LRU 状态（可批量异步）

---

## 7. 迁移路径

### 7.1 当前 → Daemon 架构

```
Phase A（当前状态）:
  每个进程独立 KcmmPool，API 为直接函数调用

Phase B（重构）:
  1. 抽取 trait KcmmBackend { alloc, free, touch, cool, ... }
  2. 实现 LocalPool: KcmmBackend（当前代码，进程内）
  3. 实现 RemotePool: KcmmBackend（Unix socket client）

Phase C（daemon）:
  1. 提取现有 KcmmPool 核心逻辑 → kcmm-daemon binary
  2. libkcmm.so 变为 RemotePool + 本地 VA 管理
  3. LocalPool 作为开发/测试回退保留（单进程无 daemon 模式）
```

### 7.2 兼容性

```
// 单进程模式（开发/测试/无 daemon 环境）
let pool = KcmmBackend::local(config);  // 当前 KcmmPool

// 多进程模式（生产环境）
let pool = KcmmBackend::remote("/var/run/kcmm.sock", config);

// 上层 API 完全一致
pool.alloc_sequence(num_blocks)?;
pool.touch(seq_idx);
pool.free_sequence(&block_table);
```

---

## 8. 错误处理

### 8.1 Daemon 不可用

```
场景：daemon 未启动或崩溃

Client 行为（可配置）：
  Option A: 返回错误 → 引擎自行处理（退化到纯 GPU 缓存，无 tiering）
  Option B: 阻塞等待 daemon 恢复（超时可设）
  Option C: 自动切换到 LocalPool 降级模式

推荐：Option A + 自动重连。推理引擎通常有自己的错误处理。
```

### 8.2 Client 崩溃

```
Daemon 检测到 socket 断开：
  1. 标记该 client 为 "disconnected"
  2. 遍历该 client 的所有 seq → 释放所有 block
  3. 共享块（ref_count > 1）：ref_count -= 1，不释放物理页
  4. 清理该 client 的 CUDA IPC 导出
```

---

## 9. 性能考量

| 操作 | 当前（进程内） | Daemon 模型 | 开销 |
|------|-------------|-----------|------|
| 推理热路径 | 直接 GPU kernel | 直接 GPU kernel | **0** |
| alloc_block | 直接函数调用 | Unix socket RPC | ~10-50 µs |
| free_sequence | 直接函数调用 | Unix socket RPC | ~10-50 µs |
| touch | 直接函数调用 | 可批量异步，或本地缓存 | ~0 (amortized) |
| evict/restore | 本进程执行 | daemon 执行（同一 GPU） | **相同** |

关键：alloc/free 不在推理热路径上（每秒几次 vs 每秒数千 step），~10-50µs 的 socket 开销完全可以接受。

---

## 10. 安全性

| 关注点 | 措施 |
|--------|------|
| 未授权进程连接 | Unix socket 权限（0600）+ SO_PEERCRED 检查 PID/UID |
| 进程 A 访问进程 B 的 block | daemon 验证 client_id → 不返回非本进程的 IPC handle |
| Daemon 内存泄漏 | Rust 所有权模型 + per-client 资源追踪，client 断开时全部清理 |
| 恶意 client 耗尽资源 | daemon 支持 per-client quota 限制 |

---

## 11. 与现有系统的对比

| 系统 | 跨进程管理 | 分层存储 | 架构模型 |
|------|----------|---------|---------|
| **KCMM (daemon)** | ✅ 原生支持 | ✅ GPU↔CPU↔NVMe | OS 风格守护进程 |
| vAttention | ❌ 嵌入引擎内 | ❌ | 引擎内库 |
| kvcached | ✅ 支持（IPC） | ❌ 仅 GPU | 守护进程 |
| KVBM (Dynamo) | ✅ 分布式 | ✅ 四层 | 重量级框架 |
| LMCache | ✅ 分布式 | ✅ 多层 | 中间件 |

KCMM 的独特定位：**轻量级 OS 风格守护进程 + 完整多级分层存储 + 引擎无关 API**。

---

## 12. 未包含在本设计中的内容

以下能力暂不在此设计中，根据需要后续扩展：

- ❌ **跨进程 prefix sharing 全局索引**：`kcmm_share_prefix()` API 保留，但全局 prefix index（跨进程发现共享机会）推迟
- ❌ **跨节点显存管理**（多 GPU 服务器 / RDMA）：单 daemon 管理单 GPU，多 GPU 可运行多个 daemon 实例
- ❌ **NVMe GDS 路径**：评估后按需加入
- ❌ **CUDA Graph 优化**：daemon 的 tiering 操作可用 CUDA Graph 减少 API launch 开销
