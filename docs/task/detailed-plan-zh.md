# 面向大语言模型推理的 GPU 内存操作系统 —— 详细实施方案

## 目录

1. [愿景与动机](#1-愿景与动机)
2. [系统架构](#2-系统架构)
3. [步骤 1：I/O 路径分析与延迟特征刻画（15%）](#3-步骤-1io-路径分析与延迟特征刻画15)
4. [步骤 2：基于 eBPF 的推理请求网络旁路（30%）](#4-步骤-2基于-ebpf-的推理请求网络旁路30)
5. [步骤 3：KV Cache 内存管理器 —— KCMM（30%）](#5-步骤-3kv-cache-内存管理器--kcmm30)
6. [步骤 4：跨引擎前缀共享与细粒度 GPU 页表（25%）](#6-步骤-4跨引擎前缀共享与细粒度-gpu-页表25)
7. [评估策略](#7-评估策略)
8. [时间线与里程碑](#8-时间线与里程碑)
9. [风险登记与缓解](#9-风险登记与缓解)
10. [发表策略](#10-发表策略)
11. [代码库迁移方案](#11-代码库迁移方案)

---

## 1. 愿景与动机

### 1.1 核心论点

> **做一个更强的推理引擎是错误的目标。做一个让*每一个*推理引擎都受益的 OS 层——这才是正确的目标。**

根本洞见：成熟的推理引擎（vLLM、SGLang）在 PagedAttention 优化、FlashInfer 集成和生产级打磨方面拥有多年的领先优势。对于一个小的研究团队来说，在内存碎片率或 decode 吞吐量上与它们正面竞争是一场注定失败的战斗。

但这些引擎共享一个关键的盲区：它们是**单进程、用户空间系统**，无法做到操作系统能做的事：

| OS 能力 | 推理引擎的局限 | 我们的机会 |
|--------|-------------|----------|
| 跨进程内存共享 | 无法跨引擎实例共享 KV Cache | KCMM 管理的共享前缀缓存 |
| 透明内存分层 | 每个引擎临时实现自己的 swap | OS 级 GPU↔CPU↔NVMe 分层存储 |
| 内核级网络旁路 | 必须穿越完整 TCP/IP 协议栈 | eBPF/XDP 零拷贝请求路径 |
| 全局资源可见性 | 每个引擎仅看到自己的内存 | 系统级 GPU 内存压力管理 |

### 1.2 设计原则

1. **透明性优先于集成度**：OS 层应加速推理引擎而不需要侵入式修改。引擎使用简单的分配器 API；所有分层、共享和预取操作在幕后发生。

2. **提供机制，而非策略**：KCMM 提供*机制*（按需分页、分层存储、引用计数）。推理引擎提供*策略*（哪些序列应当换出、何时预取）。

3. **可组合性优先于单体架构**：支柱 A（网络）和支柱 B（内存）各自独立可用、独立可评估。它们相互组合但不耦合。

4. **Rust 作为 OS 语言**：所有新的 OS 层组件使用 Rust 编写。C FFI 边界保持最小且定义清晰。

---

## 2. 系统架构

### 2.1 组件图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        客户端应用                                      │
│  (OpenAI SDK, curl, 基准测试工具, 其他推理客户端)                        │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ TCP/IP（以太网或本地回环）
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Rust OS 支持层                                      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  支柱 A：eBPF 网络旁路                                         │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │   │
│  │  │ XDP eBPF     │  │ AF_XDP UMEM   │  │ GDRCopy 引擎     │  │   │
│  │  │ 程序         │──│ (Rust xdpilone│──│ (NIC→GPU DMA)    │  │   │
│  │  │ (数据包分类)  │  │  或 libbpf-rs)│  │                  │  │   │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘  │   │
│  │                         │                                     │   │
│  │  ┌──────────────────────▼──────────────────────────────────┐ │   │
│  │  │  Rust 代理核心 (src/proxy/)                              │ │   │
│  │  │  - TCP 重组（XDP 模式）                                   │ │   │
│  │  │  - HTTP/JSON 解析                                        │ │   │
│  │  │  - OpenAI API 转换                                       │ │   │
│  │  │  - 请求路由 → 推理后端                                    │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  支柱 B：KV Cache 内存管理器（KCMM）                            │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │   │
│  │  │ 块分配器      │  │ 分层引擎       │  │ 前缀共享管理器    │  │   │
│  │  │              │  │               │  │                  │  │   │
│  │  │ cuMemCreate  │  │ LRU 换出      │  │ 引用计数          │  │   │
│  │  │ cuMemMap     │  │ 热/冷追踪     │  │ 块级去重          │  │   │
│  │  │ 空闲列表     │  │ GPU↔CPU↔NVMe  │  │ 跨引擎共享        │  │   │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘  │   │
│  │                                                               │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │  KCMM 客户端 API（C FFI / Rust crate）                    │ │   │
│  │  │  kcmm_pool_create() / kcmm_alloc_blocks() / ...          │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  观测层 (src/trace/, scripts/)                                 │   │
│  │  - bpftrace 脚本: trace_vfs, trace_tcp, trace_nvme, ...      │   │
│  │  - UFS 指标: IFR, BU, PME, RFI                                │   │
│  │  - 延迟火焰图（请求路径分解）                                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     ┌─────────┐     ┌─────────┐     ┌──────────┐
     │  vLLM   │     │ SGLang  │     │  自定义   │
     │ (Python)│     │ (Python)│     │  引擎     │
     │         │     │         │     │  (Rust)  │
     └─────────┘     └─────────┘     └──────────┘
     推理后端——无需修改，透明加速
```

### 2.2 数据流——请求生命周期

```
1.  客户端发送 HTTP POST /v1/completions → NIC
2.  XDP eBPF 程序匹配数据包（端口 8000）→ XDP_REDIRECT 到 AF_XDP 套接字
3.  AF_XDP UMEM 环形缓冲区 → Rust 代理消费原始数据包
4.  Rust 代理：TCP 重组 → HTTP 解析 → 提取 token_ids JSON
5.  可选：GDRCopy 将 token_ids 直接写入 GPU 缓冲区（NIC→GPU DMA）
6.  Rust 代理：POST /v1/completions → vLLM@localhost:8001
7.  vLLM：通过 KCMM API 分配 KV Cache 块
8.  KCMM：若 GPU 内存已满 → 将冷区块换出到 CPU RAM/NVMe → 映射新区块
9.  vLLM：运行推理，返回 token
10. vLLM：通过 KCMM 释放已完成请求的 KV Cache 块
11. vLLM：流式返回响应 → Rust 代理 → 客户端
```

### 2.3 关键接口

#### KCMM C API（用于 vLLM / Python 集成，通过 ctypes/cffi 调用）

```c
// 不透明句柄
typedef struct kcmm_pool kcmm_pool_t;

// 池生命周期
kcmm_pool_t* kcmm_pool_create(
    size_t block_size,         // 每块字节数（例如 LLaMA-7B 为 65536）
    size_t max_blocks,         // 所有序列的最大块数
    const char* cpu_cache_path  // CPU/NVMe 交换文件路径
);

void kcmm_pool_destroy(kcmm_pool_t* pool);

// 块分配——返回 GPU 常驻的块索引
// 返回时保证块已在 GPU 虚拟地址空间中物理映射
int kcmm_alloc_blocks(
    kcmm_pool_t* pool,
    uint64_t seq_id,
    size_t num_blocks,
    uint32_t* out_block_indices   // 调用者分配的数组
);

// 释放序列的块
void kcmm_free_blocks(
    kcmm_pool_t* pool,
    uint64_t seq_id,
    const uint32_t* block_indices,
    size_t num_blocks
);

// 在序列之间共享前缀块
// dst_seq 的前缀部分获得与 src_seq 相同的物理块
// 共享块的引用计数递增
int kcmm_share_prefix(
    kcmm_pool_t* pool,
    uint64_t src_seq_id,
    uint64_t dst_seq_id,
    size_t num_prefix_blocks,
    uint32_t* out_block_indices   // 填充共享块索引
);

// 提示：此序列正在解码中（防止被换出）
void kcmm_touch(kcmm_pool_t* pool, uint64_t seq_id);

// 提示：此序列处于空闲状态（可被换出）
void kcmm_cool(kcmm_pool_t* pool, uint64_t seq_id);

// 指标
typedef struct {
    size_t total_blocks;        // 池中的总块数
    size_t allocated_blocks;    // 当前已分配块数
    size_t shared_blocks;       // 跨 ≥2 个序列共享的块数
    size_t evicted_blocks;      // 换出到 CPU/NVMe 的块数
    size_t gpu_resident_blocks; // 当前在 GPU 中映射的块数
    double internal_frag;       // IFR 指标
    double phys_mem_eff;        // PME 指标
} kcmm_metrics_t;

void kcmm_get_metrics(kcmm_pool_t* pool, kcmm_metrics_t* out);
```

#### Rust 代理配置（TOML）

```toml
[proxy]
listen_addr = "0.0.0.0:8000"
backend_addr = "127.0.0.1:8001"
backend_type = "vllm"  # 或 "sglang"、"custom"

[proxy.xdp]
enabled = true
iface = "eth0"
xdp_mode = "native"  # 或 "skb" 用于测试
umem_frames = 4096
umem_frame_size = 4096

[proxy.gdrcopy]
enabled = false  # 第二阶段
gpu_id = 0

[kcmm]
block_size = 65536
max_blocks = 16384
cpu_cache_path = "/dev/shm/kcmm_swap"
tiering = true
eviction_policy = "lru"
prefetch_window = 4  # 解码前方预取块数
```

---

## 3. 步骤 1：I/O 路径分析与延迟特征刻画（15%）

### 3.1 目标

产出推理请求路径从 NIC 中断到首个 token 生成的**完整延迟分解**。识别哪些内核子系统贡献最多的开销——这些数据直接激发和指导步骤 2 中的 eBPF 旁路设计。

### 3.2 研究问题

1. 端到端请求延迟中有多大比例花在 OS 内核中（网络栈、调度器、中断）vs. 推理引擎中（排队、prefill、decode）？
2. OS 开销如何随请求并发度变化？内核 TCP 栈在负载下是否成为瓶颈？
3. `read` → `cudaMemcpy` 模型加载路径的代价是多少？`mmap` 或 GDS 在什么情况下有帮助？

### 3.3 任务

#### 任务 1.1：请求路径延迟追踪（第 1-2 周）

扩展现有的 bpftrace 脚本以覆盖**完整请求路径**：

| 层次 | 追踪点 | 测量内容 |
|-----|-------|---------|
| NIC 驱动 | `mlx5e_poll_rx_cq`（Mellanox） | 数据包到达、中断延迟 |
| XDP | `xdp_do_redirect` | XDP 处理时间 |
| IP 栈 | `ip_rcv`、`ip_local_deliver` | IP 处理开销 |
| TCP 栈 | `tcp_v4_rcv`、`tcp_rcv_established` | TCP 处理、重组 |
| Socket | `sock_recvmsg`、`tcp_recvmsg` | Socket 缓冲区拷贝 |
| 用户空间 | `schedule`（上下文切换到用户态） | 唤醒延迟 |
| 推理 | 应用层时间戳 | 队列等待、prefill、首个 token |

**交付物**：`scripts/trace_request_path.bt`——单个 bpftrace 脚本，产出带逐层分解的延迟直方图。

```
# 示例输出：
# 层次                | p50 (μs) | p99 (μs) | 占总比例
# -------------------+----------+----------+------------
# NIC DMA + IRQ      |     2.3  |    15.7  |   0.5%
# XDP 处理           |     0.8  |     3.2  |   0.2%
# IP 栈              |     1.5  |     8.1  |   0.3%
# TCP 栈             |    12.4  |   124.3  |   3.1%
# Socket → 用户态    |     8.2  |    45.6  |   2.0%
# 代理 HTTP 解析     |    15.3  |    89.2  |   3.8%
# vLLM 队列等待      |    45.1  |  1200.5  |  11.2%
# vLLM prefill       |   280.3  |  3500.1  |  69.7%
# vLLM 首个 token    |    35.2  |   210.3  |   8.8%
# -------------------+----------+----------+------------
# 总计（首个 token） |   402.1  |  5197.0  | 100.0%
```

#### 任务 1.2：模型加载 I/O 路径对比（第 2-3 周）

基于现有加载器代码（`src/model/loader.rs`）进行严格对比：

| 方法 | 数据路径 | 内核子系统 | CPU 拷贝次数 |
|-----|---------|----------|------------|
| `read(2)` | 磁盘 → 页缓存 → `cudaMemcpy` | VFS、页缓存、块层 | 2（磁盘→RAM，RAM→GPU） |
| `mmap` | 磁盘 → 页缓存（按需）→ `cudaMemcpy` | VFS、页缓存（缺页驱动）、块层 | 1（RAM→GPU） |
| `O_DIRECT` | 磁盘 → 用户缓冲区 → `cudaMemcpy` | VFS、块层、bio | 1（缓冲区→GPU） |
| GDS（`cuFileRead`） | 磁盘 → GPU（PCIe P2P DMA） | NVMe 驱动、PCIe | 0 |

**交付物**：四路对比表，包含延迟、CPU 利用率和页缓存效率指标，在 d7525 裸金属服务器上使用其 NVMe SSD 和 A30 GPU 进行测量。

#### 任务 1.3：并发扩展分析（第 3-4 周）

在递增并发度（1→2→4→8→...→64 并发请求）下运行追踪，以识别：
- 何种并发度下内核 TCP 栈成为 CPU 瓶颈
- `ksoftirqd` 在高数据包速率下是否消耗不成比例的 CPU
- `schedule` 延迟尖峰是否表明上下文切换压力

**交付物**：带逐层分解的并发-延迟曲线，识别 OS 开销开始占主导的"拐点"。

### 3.4 成功标准

- [ ] 覆盖 NIC→GPU 的完整推理请求路径延迟火焰图
- [ ] 量化负载下 TCP 栈开销占端到端延迟的百分比
- [ ] 带瓶颈识别的并发扩展曲线
- [ ] 裸金属上的四种模型加载方式对比
- [ ] 对步骤 2 的 eBPF 旁路必须越过哪些内核层给出明确理由

---

## 4. 步骤 2：基于 eBPF 的推理请求网络旁路（30%）

### 4.1 目标

构建一个 eBPF 加速代理，在 NIC 层（XDP）拦截推理请求，绕过内核 TCP/IP 协议栈，以最小拷贝次数将请求数据直接交付到推理后端。测量相对于标准 TCP 的延迟改善。

### 4.2 架构决策：分阶段推进

不做一次性大爆炸实现，而是分三个阶段构建。每个阶段产出独立可测量、独立可发表的成果。

```
Phase 1：纯 Rust 代理（第 5-6 周）
  └─→ 测量："Rust 代理相比直连 TCP 增加了多少开销？"

Phase 2：AF_XDP 旁路（第 7-12 周）
  └─→ 测量："XDP 旁路相比 Phase 1 节省了多少？"

Phase 3：GDRCopy 直写 GPU（第 13-16 周，延展目标）
  └─→ 测量："能否将 token 从 NIC 直接写入 GPU 内存？"
```

### 4.3 Phase 1：纯 Rust 代理基线

#### 任务 2.1a：代理核心（第 5 周）

构建一个最小 Rust TCP 代理：
1. 监听 `0.0.0.0:8000`
2. 接受推理客户端的 TCP 连接
3. 解析线协议（兼容 OpenAI 的 JSON 或自定义二进制格式）
4. 转发到推理后端（`localhost:8001`）
5. 将响应 token 流式返回给客户端

```rust
// src/proxy/mod.rs
pub struct ProxyConfig {
    pub listen_addr: SocketAddr,
    pub backend_addr: SocketAddr,
    pub backend_type: BackendType,  // Vllm, Sglang, Custom
    pub max_concurrent: usize,
}

pub struct Proxy {
    config: ProxyConfig,
    backend: Box<dyn InferenceBackend>,
    metrics: ProxyMetrics,
}

#[async_trait]
pub trait InferenceBackend {
    async fn generate(&self, req: InferenceRequest) -> Result<InferenceResponse>;
    async fn health(&self) -> Result<HealthStatus>;
}
```

#### 任务 2.1b：基线基准测试（第 6 周）

对三种配置进行基准测试：
1. **直连**：客户端直接连接 vLLM（无代理）
2. **Rust 代理**：客户端 → Rust 代理 → vLLM
3. **Python 代理**：客户端 → Python 代理 → vLLM（公平对比）

在不同并发度（1、4、16、64）和请求大小（128、512、2048 token 提示词）下测量。

**交付物**：`docs/report/step2-phase1-proxy-baseline.md`——延迟分布、吞吐量曲线、CPU 开销对比。

**可发表吗？** 是的——这成为 eBPF 实验的对照组，数据是测量论文（贡献 1）的一部分。

### 4.4 Phase 2：AF_XDP 旁路

这是步骤 2 的核心技术贡献。

#### 任务 2.2a：XDP eBPF 程序（第 7-8 周）

编写 XDP eBPF 程序：
1. 分类数据包：匹配目标端口（8000）和协议（TCP）
2. 匹配的数据包 → `XDP_REDIRECT` 到 AF_XDP 套接字
3. 不匹配的数据包 → `XDP_PASS`（对其他流量透明）

```c
// src/proxy/xdp_filter.bpf.c
SEC("xdp")
int xdp_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;

    // 仅处理 IP
    if (eth->h_proto != __constant_htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;

    // 仅处理 TCP
    if (ip->protocol != IPPROTO_TCP) return XDP_PASS;

    struct tcphdr *tcp = (void *)(ip + 1);
    if ((void *)(tcp + 1) > data_end) return XDP_PASS;

    // 匹配目标端口
    if (tcp->dest == __constant_htons(INFERENCE_PORT)) {
        return XDP_REDIRECT; // 重定向到 AF_XDP 套接字
    }

    return XDP_PASS; // 其余流量：走内核协议栈
}

char _license[] SEC("license") = "GPL";
```

**关键设计决策：**
- 使用 `xdpilone` crate（纯 Rust）或 `libbpf-rs`（C 库包装器）处理 AF_XDP
- 预分配 UMEM：4096 个帧，每帧 4096 字节（共 16 MB）
- 绑定到专用 NIC RX 队列以避免争用

#### 任务 2.2b：Rust 用户态 TCP 重组（第 8-10 周）

这是步骤 2 中最难的技术挑战。当数据包通过 AF_XDP 到达时，它们是包含 IP 数据报、IP 数据报又包含 TCP 段的原始以太网帧。Rust 用户态代码必须：

1. 解析以太网 → IP → TCP 头部
2. 维护每个连接的 TCP 状态机：
   ```
   ConnectionState:
     SYN_RCVD → ESTABLISHED → FIN_WAIT / CLOSE_WAIT → CLOSED
   ```
3. 追踪 TCP 序列号并重组乱序段
4. 检测完整 HTTP 请求（在重组流中扫描 `\r\n\r\n`）
5. 处理重传、重复 ACK 和窗口缩放

```rust
// src/proxy/tcp_reasm.rs
pub struct TcpReassembler {
    connections: HashMap<ConnectionKey, TcpConnection>,
    config: ReasmConfig,
}

struct TcpConnection {
    state: TcpState,
    recv_buf: Vec<u8>,        // 重组后的流
    next_expected_seq: u32,
    out_of_order: BTreeMap<u32, Vec<u8>>,  // seq → 段数据
    send_buf: Vec<u8>,        // 待发送的响应
    send_next: u32,
    last_ack_sent: u32,
}

impl TcpReassembler {
    /// 处理原始 TCP 段，当完整 HTTP 请求在 recv_buf 中组装完成时
    /// 返回 Some(request)
    pub fn ingest_segment(&mut self, key: ConnectionKey, tcp_header: &Tcphdr,
                          payload: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// 当代理想要发送响应数据时调用
    pub fn enqueue_response(&mut self, key: ConnectionKey, data: &[u8]);
    
    /// 生成下一个待发送的 TCP 段（如有）
    pub fn next_tx_segment(&mut self, key: ConnectionKey) -> Option<Vec<u8>>;
}
```

**v1 设计简化**：由于主要使用场景是**本地回环或单跳 LAN**（客户端和服务器在同一台机器或同一机架），我们可以假设：
- 无丢包（初始阶段跳过重传逻辑）
- 无乱序（单个 NIC 队列，同一 NUMA 节点）
- 小连接数（< 1000 并发）

这大幅降低了复杂度，允许我们在 3 周而非 3 个月内产出可工作的原型。

#### 任务 2.2c：AF_XDP 集成（第 10-12 周）

将 XDP 程序、AF_XDP 套接字和 TCP 重组器集成到一个统一的事件循环中：

```rust
// src/proxy/af_xdp_loop.rs
pub struct AfXdpProxy {
    umem: Umem,
    rx_q: RxQueue,
    tx_q: TxQueue,
    fill_q: FillQueue,
    completion_q: CompletionQueue,
    reasm: TcpReassembler,
    backend: Box<dyn InferenceBackend>,
}

impl AfXdpProxy {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            // 1. 补充 RX 描述符
            self.fill_q.fill_free_frames()?;

            // 2. 轮询接收数据包
            let n = self.rx_q.poll_and_consume(|frame| {
                let pkt = parse_eth_ip_tcp(frame.data)?;
                if let Some(request) = self.reasm.ingest_segment(
                    pkt.conn_key, &pkt.tcp, pkt.payload)? {
                    // 完整的 HTTP 请求已组装
                    let backend = self.backend.clone();
                    tokio::spawn(async move {
                        let response = backend.generate(parse_request(&request)?).await?;
                        // 响应 token 排队到重组器的发送缓冲区
                        enqueue_response(pkt.conn_key, &serialize_response(&response)?);
                    });
                }
            })?;

            // 3. 刷新待发送数据
            for (conn_key, segment) in self.reasm.drain_tx_segments() {
                self.tx_q.send(conn_key.addr, segment)?;
            }

            // 4. 唤醒 TX 队列
            self.tx_q.wake()?;

            // 让出给 tokio 以处理异步后端调用
            tokio::task::yield_now().await;
        }
    }
}
```

**性能关键路径优化：**
- UMEM 帧预分配且绝不释放（环形缓冲区）
- TCP 重组使用 `bytes` crate 的 `BytesMut` 实现零拷贝缓冲区管理
- HTTP 解析使用 SIMD 加速的 `memchr` 检测 `\r\n\r\n` 边界
- 后端调用非阻塞（Tokio async）

#### 任务 2.2d：Phase 2 基准测试（第 12 周）

对比 Phase 2（AF_XDP 旁路）与 Phase 1（纯代理）和直连：

| 指标 | 直连 TCP | Rust 代理 | AF_XDP 旁路 | 改进幅度 |
|-----|---------|----------|-----------|---------|
| 中位请求延迟（128 token 提示词） | Tp50_direct | Tp50_proxy | Tp50_xdp | Δ |
| 尾部延迟 p99 | Tp99_direct | Tp99_proxy | Tp99_xdp | Δ |
| 吞吐量 @ 64 并发 | Q_direct | Q_proxy | Q_xdp | Δ |
| CPU 利用率（sys%） | C_direct | C_proxy | C_xdp | Δ |
| 上下文切换/秒 | S_direct | S_proxy | S_xdp | Δ |

### 4.5 Phase 3：GDRCopy——NIC→GPU 直写（延展目标）

**目标**：在 AF_XDP 路径将 token 交付到 Rust 代理之后，使用 GDRCopy 或 GPU Direct RDMA 直接将它们写入 vLLM 的 GPU 输入缓冲区。

**为何是延展目标**：这需要：
- 修改 vLLM 以暴露其输入缓冲区 GPU 地址（侵入式），或
- 构建一个独立的 CUDA kernel 演示来展示概念（影响较低）

**建议**：推迟到步骤 4 之后或以独立短论文形式发表。仅 AF_XDP 旁路已是足够强的贡献。

### 4.6 成功标准

- [ ] Phase 1：Rust 代理相比直连 TCP 增加 < 100μs 中位开销
- [ ] Phase 2：AF_XDP 旁路在负载下相比 Phase 1 减少 ≥ 30% 中位延迟
- [ ] Phase 2：系统 CPU 时间减少 ≥ 40%（TCP 栈旁路的证据）
- [ ] Phase 2：无正确性回归（与直连 TCP 基线 100% token 匹配）
- [ ] 完整延迟分解：对比内核 TCP 路径 vs. AF_XDP 旁路路径

---

## 5. 步骤 3：KV Cache 内存管理器 —— KCMM（30%）

### 5.1 目标

构建 KCMM——一个**用户空间 OS 服务**，提供带透明分层存储（GPU↔CPU↔NVMe）的 GPU KV Cache 内存管理。KCMM 替换推理引擎内置的 KV Cache 分配器，提供跨引擎内存压力管理、基于 LRU 的换出策略以及可选的分层存储。

### 5.2 为何不只使用 vLLM 的 Swap？

| 特性 | vLLM 内置 Swap | KCMM |
|-----|--------------|------|
| 作用范围 | 单个 vLLM 进程 | 使用 KCMM API 的任何进程 |
| 跨引擎共享 | 否 | 是 |
| 换出策略 | vLLM 内部逻辑 | 可插拔（LRU、LFU、FIFO） |
| 分层存储 | 仅 GPU ↔ CPU | GPU ↔ CPU ↔ NVMe |
| 内存压力视图 | 仅 vLLM 自己的池 | 系统级（所有注册的池） |
| 预取 | 无 | 基于启发式的预取 |
| 指标 | vLLM 内部 | UFS 兼容的跨引擎指标 |

### 5.3 核心设计

#### 5.3.1 内存模型

```
┌─────────────────────────────────────────────────────────────┐
│  GPU 虚拟地址空间（每个引擎进程独立）                           │
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ 序列 A  │ │ 序列 B  │ │ 序列 C  │ │  空闲    │           │
│  │ Block 0 │ │ Block 0 │ │ Block 0 │ │  VA      │           │
│  │ Block 1 │ │ Block 1 │ │ Block 1 │ │  空间    │           │
│  │ Block 2 │ │   ...   │ │   ...   │ │         │           │
│  │   ...   │ │         │ │         │ │         │           │
│  └────┬────┘ └───┬─────┘ └───┬─────┘ └─────────┘           │
│       │          │           │                               │
│       │   cuMemMap（按需映射）                                 │
│       ▼          ▼           ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GPU 物理内存（2MB 超级块，固定块数）                    │   │
│  │                                                      │   │
│  │  [Block 0] [Block 1] [Block 2] ... [Block N-1]      │   │
│  │     ↑                     ↑                          │   │
│  │     │  换出                │  恢复                    │   │
│  │     ▼                     │                          │   │
│  └───────────────────────────┼──────────────────────────┘   │
│                              │                               │
│  ┌───────────────────────────▼──────────────────────────┐   │
│  │  CPU RAM（mmap 文件或共享内存）                        │   │
│  │  [/dev/shm/kcmm_swap]                                │   │
│  │  Block N → Block N+1 → ...                           │   │
│  │                           │                           │   │
│  │                           │  溢出（可选）              │   │
│  │                           ▼                           │   │
│  │  NVMe SSD（cuFileRead/Write 或标准 I/O）              │   │
│  │  [/mnt/nvme/kcmm_swap]                               │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

关键设计选择：

1. **GPU VA 空间以引擎进程为单位。** 每个引擎通过 `cuMemAddressReserve` 预留自己的连续 VA 区域。KCMM 管理此区域内的物理页映射。

2. **物理页使用 2 MB 超级块**（CUDA VMM 默认）。每个超级块被划分为固定大小的块（例如，LLaMA-7B 使用 32 KB 块时，每超级块 64 个块）。这是当前 `src/cache/cuda_vmm.rs` 的设计，予以保留。

3. **分层存储以块为粒度，而非超级块粒度。** 换出时，KCMM 可以换出单个冷块，而非整个 2MB 超级块。这需要先将块数据拷贝出来（通过 `cudaMemcpy D2H`），然后再释放物理页。

4. **KCMM 以库的形式链接进每个引擎进程**，而非作为独立守护进程。这避免了关键路径上的 IPC 开销。跨引擎协调（用于共享前缀检测）通过共享内存或轻量级 Unix 套接字连接到协调守护进程来实现。

#### 5.3.2 KCMM 内部架构

```rust
// src/kcmm/mod.rs

/// 顶层 KCMM 池。每个引擎进程一个（或每个 GPU 一个）。
pub struct KcmmPool {
    // GPU 虚拟地址空间
    gpu_va_start: u64,
    gpu_va_size: usize,

    // 物理内存管理
    superblocks: Vec<Superblock>,
    free_blocks: VecDeque<BlockHandle>,
    
    // 序列追踪
    sequences: HashMap<u64, SequenceState>,
    
    // 分层存储
    tiering: Option<TieringEngine>,
    
    // 前缀共享
    sharing: Option<SharingManager>,
    
    // 指标
    metrics: KcmmMetrics,
    fragmentation_tracker: FragmentationTracker,
}

struct Superblock {
    handle: CudaMemHandle,    // 来自 cuMemCreate
    va_offset: usize,         // GPU VA 中的偏移
    block_size: usize,
    blocks_per_sb: usize,
    block_bitmap: Bitmap,     // 哪些块已被分配
}

struct SequenceState {
    seq_id: u64,
    blocks: Vec<BlockRef>,    // 逻辑块 → 物理块映射
    is_active: bool,          // 正在解码 vs. 等待中
    last_access: Instant,     // 用于 LRU
    shared_prefix_len: usize, // 与其他序列共享的块数
}

enum BlockLocation {
    GpuResident(BlockHandle, u64),  // (句柄, GPU VA 偏移)
    CpuResident(usize),             // CPU 交换缓冲区中的偏移
    NvmeResident(u64),              // NVMe 交换文件中的偏移
    Evicting,                       // 传输中
    Restoring,                      // 传输中
}

struct TieringEngine {
    cpu_buffer: *mut u8,       // mmap 的 CPU 交换空间
    cpu_buffer_size: usize,
    nvme_file: Option<File>,   // NVMe 交换文件
    eviction_policy: EvictionPolicy,
    block_states: HashMap<BlockHandle, BlockLocation>,
    evict_queue: BinaryHeap<EvictCandidate>,  // 按 last_access 排序
    prefetch_queue: VecDeque<BlockHandle>,
}

struct SharingManager {
    // 映射：块内容哈希 → 引用列表 (engine_id, seq_id, block_idx)
    prefix_index: HashMap<u64, Vec<BlockOwnership>>,
    ref_counts: HashMap<BlockHandle, u32>,
}
```

#### 5.3.3 分层存储算法

**换出（GPU → CPU）：**
```
触发条件: free_blocks.len() < low_watermark（例如 < 总量的 10%）

1. 选择牺牲者：从 evict_queue 弹出（LRU——非活跃序列中最冷的块）
2. 分配 CPU 缓冲区槽位
3. cudaMemcpy D2H：GPU 块 → CPU 缓冲区
4. cuMemUnmap：移除 GPU 物理页映射
5. 更新 BlockLocation → CpuResident
6. 将 GPU 块归还 free_blocks
7. 重复直到 free_blocks > target
```

**恢复（CPU → GPU）：**
```
触发条件: kcmm_alloc_blocks() 被调用，请求的块当前为 CpuResident

1. 从 free_blocks 分配 GPU 物理块（若不足则先换出）
2. cuMemMap：在目标 VA 处映射 GPU 物理页
3. cudaMemcpy H2D：CPU 缓冲区 → GPU 块
4. 更新 BlockLocation → GpuResident
5. 将块返回给调用者
```

**异步预取（可选优化）：**
```
后台线程：
1. 对每个活跃序列，预测接下来需要的块（例如序列当前在 logical_block K，预取 K+1、K+2）
2. 若预取候选为 CpuResident，启动异步 cudaMemcpy H2D
3. 当分配请求到达时，块已 GPU 常驻
```

#### 5.3.4 CUDA 流管理

所有 KCMM 的 GPU 操作使用专用 CUDA 流以避免干扰推理计算：

```rust
pub struct KcmmStreams {
    pub evict: CudaStream,     // D2H 拷贝
    pub restore: CudaStream,   // H2D 拷贝  
    pub prefetch: CudaStream,  // 异步预取 H2D
}

impl KcmmPool {
    pub fn new(config: KcmmConfig) -> Result<Self> {
        let streams = KcmmStreams {
            evict: CudaStream::new(CU_STREAM_NON_BLOCKING)?,
            restore: CudaStream::new(CU_STREAM_NON_BLOCKING)?,
            prefetch: CudaStream::new(CU_STREAM_NON_BLOCKING)?,
        };
        // ...
    }
}
```

### 5.4 与现有代码的集成

KCMM 直接从当前代码库演进而来：

| 当前文件 | KCMM 中的新角色 |
|--------|-------------|
| `src/cache/cuda_vmm.rs` | KCMM 的 GPU 物理页管理（超级块、cuMemMap） |
| `src/cache/paged_kv.rs` | KCMM 的块分配 + 序列追踪 |
| `src/cache/swap.rs` | KCMM 的 TieringEngine（GPU↔CPU 迁移） |
| `src/cache/fragmentation_tracker.rs` | KCMM 的指标收集（IFR、PME、RFI） |
| `src/cache/unified_frag.rs` | 用于跨引擎对比的 UFS 指标收集 |

### 5.5 任务

#### 任务 3.1：KCMM 核心——带分层存储的块分配器（第 13-16 周）

1. **提取并泛化**现有的 `PagedKvCache` 为 `KcmmPool`
2. **添加 `BlockLocation` 追踪**——每个块知道自己是 GPU 常驻、CPU 常驻还是 NVMe 常驻
3. **实现换出**：
   - LRU 换出队列
   - 换出时 `cudaMemcpy D2H`
   - 拷贝完成后 `cuMemUnmap`
4. **实现恢复**：
   - `cuMemMap` 物理页
   - 恢复时 `cudaMemcpy H2D`
5. **实现 NVMe 层**（第 15 周）：
   - 使用 `cuFileRead`/`cuFileWrite` 进行 GPU↔NVMe 直接传输（GDS）
   - 若 GDS 不可用则回退到标准 I/O
   - NVMe 层是可选的——仅 CPU 层已足以完成步骤 3

#### 任务 3.2：通过 KCMM C API 集成 vLLM（第 16-17 周）

1. **构建 `libkcmm.so`**——暴露 KCMM API 的 C 共享库
2. **编写 Python 绑定**，使用 `ctypes` 或 `cffi`
3. **Monkey-patch vLLM 的块分配器**以使用 KCMM：

```python
# kcmm_vllm_patch.py
import ctypes
import vllm.core.block_manager as bm

libkcmm = ctypes.CDLL("./libkcmm.so")

class KcmmBlockAllocator:
    """vLLM CpuGpuBlockAllocator 的即插即用替代"""

    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.pool = libkcmm.kcmm_pool_create(block_size, num_gpu_blocks, ...)

    def allocate(self, block_tables):
        # 将 vLLM 的分配请求翻译为 KCMM API 调用
        for seq_id, num_blocks in block_tables.items():
            out = (ctypes.c_uint32 * num_blocks)()
            libkcmm.kcmm_alloc_blocks(self.pool, seq_id, num_blocks, out)
            # ...

    def free(self, seq_id):
        libkcmm.kcmm_free_blocks(self.pool, seq_id, ...)
```

目标是**最小化对 vLLM 的修改**——理想情况下仅需一个 `--block-allocator-backend kcmm` 标志。

#### 任务 3.3：KCMM 评估（第 17-18 周）

**实验 1：内存压力下的分层存储收益**

```
环境：A30（24 GB 显存）+ vLLM + KCMM，LLaMA-7B（14 GB 权重 → 约 10 GB 用于 KV Cache）
负载：128 并发请求，每个 max_tokens=2048
     （总 KV Cache 需求：约 16 GB → 超出可用的约 10 GB）
对比：
  A. vLLM 默认（约 80 并发后 OOM，拒绝剩余请求）
  B. vLLM + vLLM swap（GPU→CPU swap，同进程）
  C. vLLM + KCMM（GPU→CPU 分层存储，外部服务）

指标：最大可接纳并发数、TTFT p50/p99、吞吐量（tok/s）、CPU RAM 使用量
```

**实验 2：换出策略对比**

```
对比：KCMM 中的 LRU vs. LFU vs. FIFO vs. Oracle（最优）换出策略
测量：命中率（分配请求中块已 GPU 常驻的比例）、平均恢复延迟、吞吐量
```

**实验 3：CUDA 流开销**

```
测量：不同批量大小下 cuMemMap 延迟（p50、p99）
测量：专用流上 cudaMemcpy D2H/H2D 开销 vs. 推理流
对比：单块换出 vs. 批量换出（一次流操作换出 N 个块）
```

### 5.6 成功标准

- [ ] KCMM 在非分层模式下成功替换 vLLM 的块分配器，吞吐量回退 < 5%
- [ ] 在内存压力下，KCMM 比无 swap 的 vLLM 多接纳 ≥ 30% 的并发请求
- [ ] KCMM GPU→CPU 分层存储为块分配增加 < 200μs 延迟（p50）
- [ ] LRU 换出相比最优（Oracle）策略达到 ≥ 85% 的命中率
- [ ] 在无分层存储时，UFS 指标（IFR、PME、RFI）与 vLLM 内部分配器等效

---

## 6. 步骤 4：跨引擎前缀共享与细粒度 GPU 页表（25%）

### 6.1 目标

为 KCMM 扩展两项单进程推理引擎无法实现的能力：
1. **跨引擎前缀共享**：多个引擎实例（或同一引擎内的多个请求）自动为相同的前缀共享 KV Cache 块
2. **细粒度 GPU 页**：修改 NVIDIA 开源内核模块，将 CUDA VMM 最小分配粒度从 2 MB 降低到 64 KB

### 6.2 前缀共享设计

#### 6.2.1 工作原理

```
场景：两个请求带有相同的 500-token system prompt

请求 A 到达：
  - KCMM 为前缀分配块 0..31（500 token / 16 token-per-block）
  - KCMM 对前缀内容哈希：SHA256(token_ids[0..500]) → hash
  - KCMM 存入 prefix_index：{hash → [(pool_id, seq_A, blocks 0..31)]}

请求 B 到达（相同前缀）：
  - 分配前，KCMM 哈希 B 的前缀 token
  - 哈希命中！KCMM 调用 kcmm_share_prefix(seq_A, seq_B, 32)
  - seq_B 的 block table[0..31] 指向与 seq_A 相同的物理块
  - 块 0..31 的引用计数变为 2

请求 A 完成：
  - kcmm_free_blocks 被调用，释放 seq_A
  - 块 0..31：ref_count 降至 1（仍被 seq_B 引用）→ 不释放
  - 块 32..N：ref_count = 1 → 释放

请求 B 完成：
  - 块 0..31：ref_count 降至 0 → 释放
```

#### 6.2.2 内容可寻址前缀索引

```rust
// src/kcmm/sharing.rs

pub struct PrefixIndex {
    // 映射：内容哈希 → 物理块引用列表
    entries: HashMap<Hash, Vec<SharedPrefix>>,
    // 映射：(pool_id, seq_id) → 此序列前缀匹配的哈希集合
    seq_prefixes: HashMap<(u64, u64), Vec<Hash>>,
}

struct SharedPrefix {
    superblock_idx: u32,
    block_offset: u32,
    num_blocks: u32,
    ref_count: AtomicU32,
    content_hash: Hash,
}

impl SharingManager {
    /// 检查此前缀是否已存在于任何池中，若找到则共享
    pub fn try_share_prefix(
        &mut self,
        pool: &KcmmPool,
        seq_id: u64,
        prefix_token_ids: &[u32],
        block_size_tokens: usize,
    ) -> Option<Vec<u32>> {
        // 1. 计算前缀的内容哈希
        let hash = hash_prefix(prefix_token_ids, block_size_tokens);
        
        // 2. 在索引中查找
        if let Some(existing) = self.entries.get(&hash) {
            // 3. 递增现存块的引用计数
            for shared in existing {
                shared.ref_count.fetch_add(1, Ordering::SeqCst);
            }
            // 4. 返回现存的块索引
            Some(existing.iter().flat_map(|s| s.block_indices()).collect())
        } else {
            // 5. 新前缀——调用者必须分配新块，然后注册
            None
        }
    }
    
    /// 注册新分配的前缀以供将来共享
    pub fn register_prefix(
        &mut self,
        hash: Hash,
        pool_id: u64,
        seq_id: u64,
        blocks: &[u32],
    ) {
        let shared = SharedPrefix {
            num_blocks: blocks.len() as u32,
            ref_count: AtomicU32::new(1),
            content_hash: hash,
            // ...
        };
        self.entries.entry(hash).or_default().push(shared);
    }
}
```

#### 6.2.3 跨引擎场景

同一台机器上的多个 vLLM 实例：

```
实例 A（GPU 0，端口 8001）：服务模型 X
实例 B（GPU 1，端口 8002）：服务模型 X（相同模型，不同 GPU）

两者收到带有相同 500-token system prompt 的请求。

无 KCMM：
  - 实例 A：在 GPU 0 上为前缀分配 32 个块（500 token）
  - 实例 B：在 GPU 1 上为前缀分配 32 个块（500 token）
  - 总 GPU 内存：64 个块（2× 浪费）

有 KCMM（同一 GPU）：
  - 实例 A 和 B 共享同一 GPU 上的 KCMM 池
  - 总 GPU 内存：32 个块（零浪费）

有 KCMM（不同 GPU）：
  - 每个 GPU 管理自己的池
  - KCMM 协调守护进程检测到跨池的重复前缀哈希
  - 无法跨 GPU 共享物理页（A30 无 NVLink）
  - 但：可以共享 CPU 侧缓存的前缀块
    （实例 B 从 CPU 缓存恢复，而非重新计算前缀 KV）
```

对于项目范围，同 GPU 共享是优先目标。跨 GPU 通过 CPU 缓存共享是延展目标。

### 6.3 细粒度 GPU 页

#### 6.3.1 动机

CUDA VMM 的 `cuMemCreate` 默认最小分配粒度为 2 MB。对于前缀共享：
- 2 MB 超级块 = 64 块 × 每块 32 KB（LLaMA-7B）
- 一个前缀可能只有 100 token ≈ 6 块 ≈ 192 KB
- 使用 2 MB 粒度时，若不需要完整超级块，剩余约 1.8 MB 被浪费

使用 64 KB 粒度：
- 一个 64 KB "微块" = 2 块 × 32 KB
- 6 块的前缀：分配 3 个微块 = 192 KB（无浪费）
- 对小前缀实现更精确的物理内存分配

#### 6.3.2 技术方案

NVIDIA 的 `open-gpu-kernel-modules` 仓库在 `kernel-open/nvidia-uvm/` 下包含 UVM（统一虚拟内存）驱动代码。2 MB 最小粒度在 UVM 页表构造中强制执行。

**修改（概念性）：**

```c
// kernel-open/nvidia-uvm/uvm_va_range.c（概念性位置）

// 当前：最小分配大小为 UVM_CHUNK_SIZE_MAX（2MB）
// 目标：支持 UVM_PAGE_SIZE_64K 作为最小值

// 1. 定位映射 2MB 页的 GPU 页表层
// 2. 启用下一级（A30/Ampere 上为 64KB 页）
// 3. 修改 cuMemCreate 路径以接受 64KB 对齐的大小
// 4. 确保 cuMemMap/cuMemUnmap 在 64KB 粒度下正确工作
// 5. 更新 TLB 无效化逻辑以处理 64KB 页表项

// 需要修改的关键结构：
// - uvm_va_range_create() — 接受 64KB 对齐
// - uvm_page_tree — 添加 64KB 页表项
// - TLB shootdown 逻辑 — 处理 64KB 粒度
```

**分步进行：**

1. **环境搭建**（第 19 周）：
   - 在 d7525 上使用 `--kernel-module-type=open` 安装 NVIDIA 驱动 580.x
   - 克隆匹配标签的 `open-gpu-kernel-modules`
   - 验证：构建并加载未修改的模块 → 运行 CUDA VMM 测试

2. **代码探索**（第 19-20 周）：
   - 映射 UVM 页表遍历路径：`uvm_va_range.c` → `uvm_page_tree.c` → 硬件 PTE
   - 识别控制最小分配大小的常量/枚举
   - 追踪 `cuMemCreate` → `uvm_va_range_create` 调用路径
   - 记录 A30（Ampere）的多级 GPU 页表结构

3. **实现**（第 20-22 周）：
   - 添加模块参数：`uvm_min_allocation_size=65536`（默认 2097152）
   - 修改 `uvm_va_range_create` 以接受 64KB 对齐的大小
   - 更新 64KB 页的页表项格式
   - 处理边界情况：同一 VA 范围内混合使用 2MB/64KB 分配

4. **测试与验证**（第 22-23 周）：
   - 通过 `cuMemCreate` 分配 64KB GPU 内存 → 验证成功
   - 通过 `cuMemMap` 映射 64KB 页 → 验证 GPU 访问正常
   - 运行带宽微基准测试：64KB 访问 vs. 2MB 访问
   - 运行 TLB 缺失率微基准测试（随机访问模式）

5. **与 KCMM 集成**（第 23-24 周）：
   - 当检测到修改后的驱动时，KCMM 使用 64KB 超级块
   - 与 2MB 超级块对比：浪费减少、TLB 缺失影响

#### 6.3.3 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|-----|---------|
| 在 UVM 驱动中找不到正确的代码修改位置 | 中 | 高 | 从 vAttention 团队的笔记开始（他们在 A100 上做过此工作）。搜索 `UVM_CHUNK_SIZE` 或 `PAGE_SIZE` 常量。 |
| 修改后的驱动导致 GPU 崩溃 | 高 | 中 | 保留原始驱动备份。先用最小 CUDA kernel 测试。 |
| 64KB 页降低 TLB 性能 | 中 | 中 | 显式测量修改前后的 TLB 缺失率。若退化 >10%，在论文中明确讨论此权衡。 |
| CUDA 13.0 不兼容 | 低 | 高 | 项目开始时锁定 CUDA 13.0 和驱动 580.x。 |
| WSL2 无法加载自定义驱动 | 确定 | — | 所有步骤 4 的内核工作必须在裸金属（d7525）上进行。 |

### 6.4 任务

#### 任务 4.1：KCMM 中的前缀共享（第 19-21 周）

1. 按 §6.2 所述实现 `SharingManager`
2. 向 C API 添加 `kcmm_share_prefix()`
3. 与 vLLM 集成：在请求队列中检测公共前缀，对匹配请求调用 `kcmm_share_prefix()` 而非 `kcmm_alloc_blocks()`
4. 基准测试：测量内存节省和吞吐量改善 vs. 无共享

#### 任务 4.2：前缀共享评估（第 21-22 周）

**实验：共享前缀场景**

```
环境：A30 + vLLM + KCMM，LLaMA-7B
负载：
  - 50% 的请求共享一个 2048-token system prompt
  - 50% 的请求具有唯一提示词
对比：
  A. vLLM 无前缀共享（APC 禁用）
  B. vLLM 启用自动前缀缓存（APC）
  C. KCMM 前缀共享

指标：
  - 内存：GPU 已分配总块数、共享块数、相比无共享节省的内存
  - 性能：共享前缀请求的 TTFT、吞吐量
  - 正确性：所有配置之间 token 精确匹配
```

**实验：跨引擎共享**

```
环境：同一 GPU 上两个 vLLM 实例，相同模型
负载：两个实例收到带有相同 system prompt 的请求
对比：
  A. 各实例独立运行（无法共享）
  B. KCMM 跨实例共享

指标：GPU 总内存使用量、共享块数、每实例吞吐量
```

#### 任务 4.3：NVIDIA UVM 驱动修改（第 22-25 周）

详见 §6.3.2 的技术方案。

#### 任务 4.4：64KB 页评估（第 25-26 周）

**实验：粒度影响**

```
环境：修改后的驱动（64KB）vs. 原厂驱动（2MB），相同 KCMM 池
负载：混合前缀共享 + 唯一请求
对比：
  A. 2MB 超级块（原厂驱动）
  B. 64KB 超级块（修改后的驱动）

指标：
  - 物理内存效率（PME）——使用 64KB 应有所改善
  - 内碎片率（IFR）——不变（块级指标）
  - 平均块分配延迟（cuMemCreate + cuMemMap 时间）
  - TLB 缺失率（通过 nvprof 或自定义微基准测试）
  - 端到端吞吐量（tok/s）
```

### 6.5 成功标准

- [ ] 前缀共享：共享前缀负载相比无共享节省 ≥ 80% 内存
- [ ] 前缀共享：token 精确输出与无共享基线匹配
- [ ] 跨引擎共享：两个 vLLM 实例共享块，总内存 ≤ 1.1× 单实例内存
- [ ] 64KB 驱动：`cuMemCreate` 以 64KB 大小成功执行
- [ ] 64KB 驱动：前缀共享负载的 PME 相比 2MB 粒度改善 ≥ 30%
- [ ] 64KB 驱动：性能退化（吞吐量）相比 2MB 粒度 ≤ 5%

---

## 7. 评估策略

### 7.1 测试环境

| 资源 | 规格 |
|-----|-----|
| 服务器 | d7525：2× AMD EPYC 7302（16 核），128 GB RAM |
| GPU | NVIDIA A30（24 GB HBM2e，Ampere，PCIe Gen4） |
| NVMe | 1.6 TB PCIe Gen4（型号待定） |
| NIC | Mellanox ConnectX-6 DX 100Gb（单端口） |
| 操作系统 | Ubuntu 22.04 或 24.04，Linux 6.6+ |
| CUDA | 13.0 |
| 驱动 | 580.x（开源内核模块） |

### 7.2 负载类型

| 负载 | 描述 | 用途 |
|-----|-----|-----|
| **合成负载** | 固定长度提示词，均匀 token 分布 | 微基准测试，可复现性 |
| **ShareGPT** | 真实 ChatGPT 对话（可变长度） | 真实推理服务负载 |
| **前缀密集型** | 80% 的请求共享 2048-token 前缀 | 前缀共享评估 |
| **突发负载** | 泊松到达，1→64→1 并发渐变 | 压力测试，分层评估 |

### 7.3 基线

| 基线 | 描述 | 参与对比的步骤 |
|-----|-----|-------------|
| vLLM（原厂） | 标准 vLLM 安装，无修改 | 所有步骤 |
| vLLM + 内置 swap | vLLM 的 GPU→CPU swap 已启用 | 步骤 3-4 |
| vLLM + APC | vLLM 的自动前缀缓存已启用 | 步骤 4 |
| SGLang | 替代推理后端 | 步骤 3-4（可选） |
| 直连 TCP（无代理） | 原始 TCP 到 vLLM，无代理层 | 步骤 2 |
| 自定义 Rust 引擎 | 我们现有的推理引擎 | 步骤 3-4（对照点） |

### 7.4 关键指标

| 指标 | 定义 | 工具 |
|-----|-----|-----|
| **TTFT** | 首个 Token 时间（端到端）| 应用层时间戳 |
| **TPOT** | 每输出 Token 时间（decode 速度）| 应用层时间戳 |
| **吞吐量** | 所有请求的总 token/秒 | 应用层时间戳 |
| **IFR** | 内碎片率 | UFS（unified_frag.rs） |
| **PME** | 物理内存效率 | UFS |
| **BU** | 块利用率 | UFS |
| **RFI** | 运行时碎片指数 | UFS |
| **逐层延迟** | 每内核层延迟分解 | bpftrace 脚本 |
| **CPU 利用率** | 系统/用户 CPU % | `perf stat`、`mpstat` |
| **GPU 利用率** | GPU SM/内存利用率 | `nvidia-smi`、CUDA profiler |
| **上下文切换** | 自愿 + 非自愿/秒 | `perf stat` |
| **cuMemMap 延迟** | 映射时间 p50/p99 | 自定义插桩 |
| **换出延迟** | 块换出时间 p50/p99 | 自定义插桩 |
| **TLB 缺失率** | 每次访问的 GPU TLB 缺失 | `nvprof` 或自定义基准测试 |

---

## 8. 时间线与里程碑

```
周次   阶段   步骤   里程碑                                        交付物
────  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
 1    环境    —     d7525 环境搭建                                 裸金属工作环境
 2    步骤1  1.1    NIC→GPU 延迟追踪完成                            trace_request_path.bt
 3    步骤1  1.2    模型加载 I/O 对比                               四种加载器对比报告
 4    步骤1  1.3    并发扩展分析                                    延迟-并发曲线
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
 5    步骤2  2.1a   纯 Rust 代理实现                                src/proxy/（Phase 1）
 6    步骤2  2.1b   代理基线基准测试                                Phase 1 基准测试报告
 7    步骤2  2.2a   XDP eBPF 程序 + AF_XDP 搭建                    xdp_filter.bpf.c、UMEM 搭建
 8    步骤2  2.2a   XDP 重定向验证（丢弃测试）                       XDP 功能测试
 9    步骤2  2.2b   TCP 重组（SYN/ACK 握手）                        TcpReassembler（基础版）
10    步骤2  2.2b   TCP 重组（数据传输、FIN）                        TcpReassembler（完整版）
11    步骤2  2.2c   AF_XDP 与代理循环集成                            AfXdpProxy 可工作
12    步骤2  2.2d   Phase 2 基准测试                                AF_XDP vs. 直连 TCP 报告
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
13    步骤3  3.1    KCMM 核心：泛化 PagedKvCache                     src/kcmm/（核心）
14    步骤3  3.1    KCMM 核心：BlockLocation + 换出队列              TieringEngine（仅换出）
15    步骤3  3.1    KCMM：GPU→CPU 换出 + 恢复可工作                  完整分层存储循环
16    步骤3  3.2    libkcmm.so + C API                              libkcmm.so
17    步骤3  3.2    vLLM KCMM 集成                                   Monkey-patch vLLM 块分配器
18    步骤3  3.3    KCMM 评估（内存压力、LRU）                        步骤 3 评估报告
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
19    步骤4  4.1    KCMM 中的 PrefixIndex                            SharingManager（同引擎）
20    步骤4  4.1    KCMM 跨引擎协调守护进程                            SharingManager（跨引擎）
21    步骤4  4.2    前缀共享评估                                     前缀共享报告
22    步骤4  4.3    阅读/理解 UVM 页表代码                            UVM 驱动分析文档
23    步骤4  4.3    在 UVM 驱动中实现 64KB 支持                       修改后的 nvidia-uvm.ko
24    步骤4  4.3    测试 64KB 驱动（稳定性 + 正确性）                  64KB 驱动测试报告
25    步骤4  4.4    64KB 与 KCMM 集成 + 基准测试                      64KB vs 2MB 对比
26    步骤4  4.4    TLB 缺失率分析                                    64KB 最终报告
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
27    写作    —     论文草稿（测量论文）                              初稿
28    写作    —     论文草稿（系统论文）                              初稿
29    打磨    —     修改、 rebuttal 准备、artifact 评估               终稿
30    缓冲    —     溢出缓冲（2 周）                                  —
```

**关键决策点：**
- **第 4 周检查点**：TCP 栈开销是否显著到足以证明 AF_XDP 旁路的合理性？若 < 端到端延迟的 5%，则将步骤 2 转向"网络可观测性"论文。
- **第 12 周检查点**：AF_XDP 旁路是否显示 ≥ 15% 的延迟改善？若否，将步骤 2 限制在 Phase 1 + 测量，将资源转移到步骤 3-4。
- **第 18 周检查点**：KCMM 分层存储是否比 vLLM swap 多接纳 ≥ 20% 的并发请求？若否，将步骤 3 聚焦于跨引擎共享方面。
- **第 22 周检查点**：UVM 驱动修改是否可行？若代码过于不透明或崩溃频繁，将步骤 4 转为仅做 2MB vs. 理想页大小的测量研究。

---

## 9. 风险登记与缓解

### 9.1 关键风险

| ID | 风险 | 概率 | 影响 | 缓解措施 |
|----|-----|-----|-----|---------|
| R1 | AF_XDP 增加延迟而非减少延迟 | 中 | 高 | 分阶段构建。Phase 1（纯代理）即使 Phase 2 失败也是可发表的测量成果。转向"分析为何 AF_XDP 对推理无帮助"论文。 |
| R2 | 用户态 TCP 重组存在 bug（数据损坏、连接泄漏） | 高 | 高 | 初始仅支持本地回环（v1 完全跳过 TCP）。使用 `tokio::net::TcpStream` 作为回退。仅在本地回环稳定后再为远程流量添加 XDP 旁路。 |
| R3 | 无法在不做侵入式修改的情况下将 KCMM 与 vLLM 集成 | 中 | 高 | 先用自定义 Rust 引擎构建 KCMM（代码可控）。与 vLLM 集成作为延展目标。即使是独立的 KCMM + Rust 引擎对比也是可发表的。 |
| R4 | 修改后的 NVIDIA 驱动不稳定/崩溃 | 高 | 中 | 将 64KB 驱动工作限制在 4 周内。若不稳，发表 2MB 结果，将 64KB 作为"未来工作与初步分析"。 |
| R5 | WSL2 是主要开发环境，但无法运行 XDP 或自定义内核模块 | 确定 | 高 | 所有 WSL2 工作：代理逻辑、KCMM 核心、基准测试。所有裸金属工作：XDP、UVM 驱动。保持明确分离。 |
| R6 | vLLM 版本更迭破坏集成 | 中 | 中 | 项目开始时固定 vLLM 版本。记录所固定的版本。小 API 变化在最终论文修订时是可接受的。 |

### 9.2 技术风险

| ID | 风险 | 概率 | 影响 | 缓解措施 |
|----|-----|-----|-----|---------|
| R7 | AF_XDP 需要特定的 NIC 驱动（仅 mlx5、i40e、ice 得到良好支持） | 低 | 高 | d7525 配备 Mellanox ConnectX-6 DX，使用 mlx5——完全支持。 |
| R8 | 高换手率负载中 cuMemMap/unmap 开销占主导 | 中 | 中 | 批量调用 cuMemMap。使用延迟解除映射（vAttention 策略）。测量并对比。 |
| R9 | 64KB 页导致 GPU TLB 抖动 | 中 | 中 | 显式基准测试 TLB 缺失率。若 >2× 增加，在论文中讨论粒度权衡。 |
| R10 | Rust CUDA 绑定（cudarc crate）缺少 VMM 相关 API | 低 | 中 | 已有 `src/cache/cuda_vmm.rs` 通过 FFI 封装原始 CUDA 驱动 API。按需扩展。 |

### 9.3 进度风险

| ID | 风险 | 概率 | 影响 | 缓解措施 |
|----|-----|-----|-----|---------|
| R11 | d7525 服务器不可用（硬件故障、调度冲突） | 低 | 高 | 预留备份：使用任何带 NVIDIA GPU + Linux 的机器。GDS 和 100Gb NIC 测试可推迟或模拟。 |
| R12 | 步骤 2 用时超过预估（TCP 重组难度大） | 高 | 中 | 预置回退方案：用 TC BPF 代替 XDP（避免 TCP 重组）。优雅降级为"无内核旁路的智能代理"。 |
| R13 | 步骤 3-4 被步骤 2 超期挤压 | 中 | 高 | 步骤间是解耦的。可在完成步骤 3 实现的同时写步骤 2 的论文。在第 24-30 周并行化写作和编码。 |

---

## 10. 发表策略

### 10.1 论文计划

| 论文 | 目标会议 | 核心贡献 | 步骤依赖 | 最早投稿时间 |
|-----|---------|---------|---------|------------|
| **论文 1：测量** | EuroSys '27、ATC '27 | LLM 推理请求路径的 OS 延迟特征刻画（步骤 1 + 步骤 2 Phase 1） | 步骤 1 + 步骤 2 Phase 1 | 2026 年 5 月（EuroSys）或 2027 年 1 月（ATC） |
| **论文 2：系统** | SOSP '27、OSDI '28 | eBPF 网络旁路 + KCMM（步骤 2-4） | 所有步骤 | 2027 年 4 月（SOSP）或 2027 年 12 月（OSDI） |
| **论文 3：短论文** | HotOS '27、APSys '27 | 单独组件：GDS 模型加载或 64KB GPU 页 | 步骤 1 GDS 或步骤 4 64KB | 视情况而定 |

### 10.2 叙事线索

**论文 1（测量）：** *"OS 是否是大语言模型推理的瓶颈？"*
- 钩子：所有人都在优化 GPU kernel，但 OS 呢？
- 贡献：LLM 推理服务的首个全面 OS 延迟分解
- 数据：步骤 1 追踪（带并发扩展的 NIC→GPU 延迟分解）
- 价值：识别出具体的内核子系统瓶颈，为论文 2 提供动机

**论文 2（系统）：** *"面向大语言模型推理的 OS 支持层"*
- 钩子：推理引擎在用户空间重新实现 OS 功能（内存管理、调度、I/O）
- 贡献：KCMM + eBPF 旁路——一个加速*任何*推理引擎的 OS 层
- 数据：步骤 2-4 评估（AF_XDP 延迟降低、KCMM 分层收益、前缀共享内存节省）
- 价值：OS 抽象是 LLM 推理优化的正确层次

### 10.3 Artifact 评估计划

- 所有代码：开源（MIT 或 Apache 2.0），GitHub
- 基准测试：可复现脚本（`scripts/bench_*.sh`），记录负载说明
- 追踪数据：匿名化，包含在仓库中
- 硬件需求：文档化（A30 24GB 或以上，Linux 6.6+）
- Docker 镜像：用于 vLLM + KCMM 的可复现环境

---

## 11. 代码库迁移方案

### 11.1 当前 → 目标映射

```
src/
├── main.rs                 →  保留，现在启动代理或独立引擎
├── lib.rs                  →  保留
├── config.rs               →  扩展：添加 [proxy] 和 [kcmm] 配置节
├── server/
│   ├── mod.rs              →  保留
│   ├── http.rs             →  演进：成为代理请求解析器 (src/proxy/http_parse.rs)
│   └── pipeline.rs         →  保留用于独立模式
├── proxy/                  →  新增：eBPF 代理
│   ├── mod.rs              →  代理主模块
│   ├── config.rs           →  代理配置
│   ├── af_xdp_loop.rs      →  AF_XDP 事件循环
│   ├── tcp_reasm.rs        →  TCP 重组状态机
│   ├── http_parse.rs       →  HTTP/JSON 解析（从 server/http.rs 演进）
│   ├── backend.rs          →  InferenceBackend trait + vLLM/SGLang 实现
│   └── xdp_filter.bpf.c    →  XDP eBPF 程序
├── kcmm/                   →  新增：KV Cache 内存管理器
│   ├── mod.rs              →  KcmmPool 顶层
│   ├── pool.rs             →  池生命周期、块分配
│   ├── superblock.rs       →  超级块管理（来自 cuda_vmm.rs）
│   ├── tiering.rs          →  TieringEngine（来自 swap.rs）
│   ├── sharing.rs          →  SharingManager（前缀缓存）
│   ├── metrics.rs          →  UFS 指标（来自 unified_frag.rs）
│   ├── ffi.rs              →  C API（用于 libkcmm.so）
│   └── streams.rs          →  CUDA 流管理
├── cache/                  →  保留：但现在专用于独立引擎模式
│   ├── mod.rs
│   ├── kv_cache.rs         →  简单连续缓存（仅作基线）
│   ├── paged_kv.rs         →  重构：启用 KCMM 时委托给 KCMM
│   ├── cuda_vmm.rs         →  重构：在 cache/ 和 kcmm/ 之间共享
│   ├── swap.rs             →  重构：在 cache/ 和 kcmm/ 之间共享
│   └── ...
├── model/                  →  保留：不变（模型加载、权重、Transformer）
├── batch/                  →  保留：独立调度器
├── cuda/                   →  保留：自定义 CUDA kernel
│   ├── kernels/
│   └── runtime.rs
├── decoder/                →  保留：独立解码器
├── trace/                  →  新增：eBPF 追踪程序
│   ├── request_path.bt     →  请求路径延迟追踪
│   └── kcmm_events.bt      →  KCMM 换出/恢复事件
└── bin/
    ├── latttice            →  独立推理引擎（向后兼容）
    ├── latttice-proxy      →  eBPF 代理二进制文件
    └── kcmm-bench          →  KCMM 微基准测试
```

### 11.2 向后兼容性

独立 Rust 推理引擎全程保持可用：
- `cargo run -- --standalone --model llama-7b` → 运行原始引擎
- `cargo run -- --proxy --backend vllm` → 运行 eBPF 代理
- `cargo run -- --proxy --kcmm` → 运行代理 + KCMM

### 11.3 依赖变更

```toml
# 新增依赖
[dependencies]
xdpilone = "0.6"          # 或 libbpf-rs = "0.25"
etherparse = "0.16"       # 以太网/IP/TCP 头部解析
memchr = "2.7"            # SIMD 加速字节搜索
bytes = "1.9"             # 零拷贝缓冲区管理
reqwest = { version = "0.12", features = ["json"] }

[build-dependencies]
# 用于编译 XDP eBPF 程序
cargo-bpf = "..."         # 或手动 clang 调用
```

### 11.4 测试策略

| 组件 | 测试类型 | 工具 |
|-----|---------|-----|
| TCP 重组 | 使用构造数据包的单元测试 | `cargo test` |
| KCMM 块分配器 | 基于属性的测试（proptest） | `cargo test` |
| KCMM 分层存储 | 集成测试：填满池、验证换出 | `cargo test` + 真实 GPU |
| AF_XDP 代理 | 集成测试：发送请求、验证 token | `scripts/test_proxy.sh` |
| vLLM + KCMM 集成 | 集成测试：带 KCMM 补丁的 vLLM 服务器 | `scripts/test_kcmm_vllm.sh` |
| 64KB 驱动 | 冒烟测试：cuMemCreate(64KB) → GPU 访问 → cuMemFree | `scripts/test_64kb.sh` |
| 完整系统 | 端到端基准测试（同评估） | `scripts/run_bench.sh` |

---

## 12. 总结：为何此方案胜出

| 维度 | 原始 tasks.md | 本方案 |
|-----|-------------|-------|
| **核心叙事** | "我们做了一个更强的推理引擎" | "我们做了一个让所有推理引擎都受益的 OS 层" |
| **与 vLLM 的差异化** | 竞争（注定失败的战斗） | 互补（vLLM 因我们的工作而受益） |
| **步骤 3-4 贡献** | block-table PagedAttention（重新发明 vLLM） | 跨引擎 GPU 内存分层存储 + 前缀共享（新颖） |
| **eBPF 贡献** | NCCL 旁路（小众） | 推理请求路径旁路（广泛影响） |
| **学术新颖性** | 低（很多推理引擎项目） | 高（OS + ML 交叉领域未被充分探索） |
| **风险画像** | 高（必须在核心指标上击败 vLLM） | 中等（每步独立可发表） |
| **代码复用** | 步骤 3-4 丢弃现有工作 | 现有 cache/cuda_vmm/swap 代码演进为 KCMM |
| **发表路径** | 1 篇论文（必须全部成功） | 2-3 篇论文（每步都是可行的发表单元） |
