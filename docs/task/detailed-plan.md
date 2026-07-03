# GPU Memory OS for LLM Inference — Detailed Implementation Plan

## Table of Contents

1. [Vision & Motivation](#1-vision--motivation)
2. [System Architecture](#2-system-architecture)
3. [Step 1: I/O Path Analysis & Latency Characterization (15%)](#3-step-1-io-path-analysis--latency-characterization-15)
4. [Step 2: eBPF Network Bypass for Inference Requests (30%)](#4-step-2-ebpf-network-bypass-for-inference-requests-30)
5. [Step 3: KV Cache Memory Manager — KCMM (30%)](#5-step-3-kv-cache-memory-manager--kcmm-30)
6. [Step 4: Cross-Engine Prefix Sharing & Fine-Grained GPU Pages (25%)](#6-step-4-cross-engine-prefix-sharing--fine-grained-gpu-pages-25)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Timeline & Milestones](#8-timeline--milestones)
9. [Risk Register & Mitigation](#9-risk-register--mitigation)
10. [Publication Strategy](#10-publication-strategy)
11. [Codebase Transition Plan](#11-codebase-transition-plan)

---

## 1. Vision & Motivation

### 1.1 The Core Thesis

> **Building a better inference engine is the wrong goal. Building an OS layer that makes *every* inference engine better — that's the right goal.**

The fundamental insight: mature inference engines (vLLM, SGLang) have multi-year leads in PagedAttention optimization, FlashInfer integration, and production hardening. Competing head-to-head on memory fragmentation rates or decode throughput is a losing battle for a small research team.

But these engines share a critical blind spot: they are **single-process, user-space systems** that cannot do what an OS can:

| OS Capability | Inference Engine Limitation | Our Opportunity |
|---------------|---------------------------|-----------------|
| Cross-process memory sharing | Cannot share KV cache across engine instances | KCMM-managed shared prefix cache |
| Transparent memory tiering | Each engine implements its own swap ad-hoc | OS-level GPU↔CPU↔NVMe tiering |
| Kernel-level network bypass | Must traverse full TCP/IP stack | eBPF/XDP zero-copy request path |
| Global resource visibility | Each engine sees only its own memory | System-wide GPU memory pressure management |

### 1.2 Design Principles

1. **Transparency over Integration**: The OS layer should accelerate inference engines without requiring invasive modifications. Engines use a simple allocator API; all tiering, sharing, and prefetching happens behind the scenes.

2. **Mechanism, Not Policy**: KCMM provides the *mechanisms* (demand paging, tiered storage, reference counting). The inference engine provides the *policy* (which sequences to evict, when to prefetch).

3. **Composability over Monolith**: Pillar A (network) and Pillar B (memory) are independently useful and independently evaluable. They compose but do not couple.

4. **Rust as the OS Language**: All new OS-layer components are written in Rust. C FFI boundaries are kept minimal and well-defined.

---

## 2. System Architecture

### 2.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Client Applications                            │
│  (OpenAI SDK, curl, benchmark harness, other inference clients)       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ TCP/IP (Ethernet or localhost)
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Rust OS Support Layer                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Pillar A: eBPF Network Bypass                                │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │   │
│  │  │ XDP eBPF     │  │ AF_XDP UMEM   │  │ GDRCopy Engine   │  │   │
│  │  │ Program      │──│ (Rust xdpilone│──│ (NIC→GPU DMA)    │  │   │
│  │  │ (packet      │  │  or libbpf-rs)│  │                  │  │   │
│  │  │  classifier) │  │               │  │                  │  │   │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘  │   │
│  │                         │                                     │   │
│  │  ┌──────────────────────▼──────────────────────────────────┐ │   │
│  │  │  Rust Proxy Core (src/proxy/)                            │ │   │
│  │  │  - TCP reassembly (if XDP)                               │ │   │
│  │  │  - HTTP/JSON parsing                                     │ │   │
│  │  │  - OpenAI API translation                                │ │   │
│  │  │  - Request routing → inference backend                   │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Pillar B: KV Cache Memory Manager (KCMM)                     │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │   │
│  │  │ Block         │  │ Tiering       │  │ Prefix Sharing   │  │   │
│  │  │ Allocator     │  │ Engine        │  │ Manager          │  │   │
│  │  │              │  │               │  │                  │  │   │
│  │  │ cuMemCreate  │  │ LRU eviction  │  │ Reference count  │  │   │
│  │  │ cuMemMap     │  │ Hot/cold      │  │ Block-level      │  │   │
│  │  │ Free list    │  │ tracking      │  │ deduplication    │  │   │
│  │  │              │  │ GPU↔CPU↔NVMe  │  │ Cross-engine     │  │   │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘  │   │
│  │                                                               │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │  KCMM Client API (C FFI / Rust crate)                     │ │   │
│  │  │  kcmm_pool_create() / kcmm_alloc_blocks() / ...          │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Observability Layer (src/trace/, scripts/)                   │   │
│  │  - bpftrace scripts: trace_vfs, trace_tcp, trace_nvme, ...   │   │
│  │  - UFS metrics: IFR, BU, PME, RFI                            │   │
│  │  - Latency flame graphs (request path breakdown)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     ┌─────────┐     ┌─────────┐     ┌──────────┐
     │  vLLM   │     │ SGLang  │     │  Custom  │
     │ (Python)│     │ (Python)│     │  Engine  │
     │         │     │         │     │  (Rust)  │
     └─────────┘     └─────────┘     └──────────┘
     Inference backends — unchanged, accelerated transparently
```

### 2.2 Data Flow — Request Lifecycle

```
1. Client sends HTTP POST /v1/completions → NIC
2. XDP eBPF program matches packet (port 8000) → XDP_REDIRECT to AF_XDP socket
3. AF_XDP UMEM ring buffer → Rust proxy consumes raw packet
4. Rust proxy: TCP reassembly → HTTP parse → extract token_ids JSON
5. Optional: GDRCopy writes token_ids directly to GPU buffer (NIC→GPU DMA)
6. Rust proxy: POST /v1/completions → vLLM@localhost:8001
7. vLLM: allocates KV cache blocks via KCMM API
8. KCMM: if GPU memory full → evicts cold blocks to CPU RAM/NVMe → maps new blocks
9. vLLM: runs inference, returns tokens
10. vLLM: frees completed request's KV cache blocks via KCMM
11. vLLM: streams response → Rust proxy → Client
```

### 2.3 Key Interfaces

#### KCMM C API (for vLLM / Python integration via ctypes/cffi)

```c
// Opaque handle
typedef struct kcmm_pool kcmm_pool_t;

// Pool lifecycle
kcmm_pool_t* kcmm_pool_create(
    size_t block_size,        // bytes per block (e.g., 65536 for LLaMA-7B)
    size_t max_blocks,        // maximum blocks across all sequences
    const char* cpu_cache_path // path for CPU/NVMe swap file
);

void kcmm_pool_destroy(kcmm_pool_t* pool);

// Block allocation — returns GPU-resident block indices
// Blocks are guaranteed to be physically mapped in GPU VA space on return
int kcmm_alloc_blocks(
    kcmm_pool_t* pool,
    uint64_t seq_id,
    size_t num_blocks,
    uint32_t* out_block_indices  // caller-allocated array
);

// Free blocks for a sequence
void kcmm_free_blocks(
    kcmm_pool_t* pool,
    uint64_t seq_id,
    const uint32_t* block_indices,
    size_t num_blocks
);

// Share prefix blocks between sequences
// dst_seq gets the same physical blocks as src_seq for the prefix portion
// Reference count on shared blocks is incremented
int kcmm_share_prefix(
    kcmm_pool_t* pool,
    uint64_t src_seq_id,
    uint64_t dst_seq_id,
    size_t num_prefix_blocks,
    uint32_t* out_block_indices  // filled with shared block indices
);

// Hint: this sequence is actively being decoded (protect from eviction)
void kcmm_touch(kcmm_pool_t* pool, uint64_t seq_id);

// Hint: this sequence is idle (eligible for eviction)
void kcmm_cool(kcmm_pool_t* pool, uint64_t seq_id);

// Metrics
typedef struct {
    size_t total_blocks;       // total blocks in pool
    size_t allocated_blocks;   // currently allocated
    size_t shared_blocks;      // shared across >=2 sequences
    size_t evicted_blocks;     // evicted to CPU/NVMe
    size_t gpu_resident_blocks;// currently mapped in GPU
    double internal_frag;      // IFR metric
    double phys_mem_eff;       // PME metric
} kcmm_metrics_t;

void kcmm_get_metrics(kcmm_pool_t* pool, kcmm_metrics_t* out);
```

#### Rust Proxy Configuration (TOML)

```toml
[proxy]
listen_addr = "0.0.0.0:8000"
backend_addr = "127.0.0.1:8001"
backend_type = "vllm"  # or "sglang", "custom"

[proxy.xdp]
enabled = true
iface = "eth0"
xdp_mode = "native"  # or "skb" for testing
umem_frames = 4096
umem_frame_size = 4096

[proxy.gdrcopy]
enabled = false  # phase 2
gpu_id = 0

[kcmm]
block_size = 65536
max_blocks = 16384
cpu_cache_path = "/dev/shm/kcmm_swap"
tiering = true
eviction_policy = "lru"
prefetch_window = 4  # blocks to prefetch ahead of decode
```

---

## 3. Step 1: I/O Path Analysis & Latency Characterization (15%)

### 3.1 Objective

Produce a **complete latency decomposition** of the inference request path, from NIC interrupt to the first token generated. Identify which kernel subsystems contribute the most overhead — this data directly motivates and guides the eBPF bypass design in Step 2.

### 3.2 Research Questions

1. What fraction of end-to-end request latency is spent in the OS kernel (network stack, scheduler, interrupts) vs. the inference engine (queueing, prefill, decode)?
2. How does OS overhead scale with request concurrency? Does the kernel TCP stack become a bottleneck under load?
3. What is the cost of the `read` → `cudaMemcpy` model loading path? When does `mmap` or GDS help?

### 3.3 Tasks

#### Task 1.1: Request-Path Latency Tracing (Week 1-2)

Extend existing bpftrace scripts to cover the **full request path**:

| Layer | Trace Point | What We Measure |
|-------|------------|-----------------|
| NIC Driver | `mlx5e_poll_rx_cq` (Mellanox) | Packet arrival, interrupt latency |
| XDP | `xdp_do_redirect` | XDP processing time |
| IP Stack | `ip_rcv`, `ip_local_deliver` | IP processing overhead |
| TCP Stack | `tcp_v4_rcv`, `tcp_rcv_established` | TCP processing, reassembly |
| Socket | `sock_recvmsg`, `tcp_recvmsg` | Socket buffer copy |
| User-Space | `schedule` (context switch to userspace) | Wake-up latency |
| Inference | Application-level timestamps | Queue wait, prefill, first token |

**Deliverable**: `scripts/trace_request_path.bt` — a single bpftrace script that produces a latency histogram with per-layer breakdown.

```
# Sample output:
# Layer              | p50 (μs) | p99 (μs) | % of total
# -------------------+----------+----------+------------
# NIC DMA + IRQ      |     2.3  |    15.7  |   0.5%
# XDP processing     |     0.8  |     3.2  |   0.2%
# IP stack           |     1.5  |     8.1  |   0.3%
# TCP stack          |    12.4  |   124.3  |   3.1%
# Socket → userspace |     8.2  |    45.6  |   2.0%
# Proxy HTTP parse   |    15.3  |    89.2  |   3.8%
# vLLM queue wait    |    45.1  |  1200.5  |  11.2%
# vLLM prefill       |   280.3  |  3500.1  |  69.7%
# vLLM first token   |    35.2  |   210.3  |   8.8%
# -------------------+----------+----------+------------
# Total (first token)|   402.1  |  5197.0  | 100.0%
```

#### Task 1.2: Model Loading I/O Path Comparison (Week 2-3)

Build on existing loader code (`src/model/loader.rs`) to produce a rigorous comparison:

| Method | Data Path | Kernel Subsystems | CPU Copies |
|--------|-----------|-------------------|------------|
| `read(2)` | Disk → Page Cache → `cudaMemcpy` | VFS, page cache, block layer | 2 (disk→RAM, RAM→GPU) |
| `mmap` | Disk → Page Cache (on-demand) → `cudaMemcpy` | VFS, page cache (fault-driven), block layer | 1 (RAM→GPU) |
| `O_DIRECT` | Disk → User buffer → `cudaMemcpy` | VFS, block layer, bio | 1 (buffer→GPU) |
| GDS (`cuFileRead`) | Disk → GPU (PCIe P2P DMA) | NVMe driver, PCIe | 0 |

**Deliverable**: A 4-way comparison table with latency, CPU utilization, and page cache efficiency metrics, measured on the d7525 bare metal server with its NVMe SSD and A30 GPU.

#### Task 1.3: Concurrency Scaling Analysis (Week 3-4)

Run the trace under increasing concurrency (1→2→4→8→...→64 concurrent requests) to identify:
- At what concurrency the kernel TCP stack becomes CPU-bound
- Whether `ksoftirqd` consumes disproportionate CPU under high packet rates
- Whether `schedule` latency spikes indicate context-switch pressure

**Deliverable**: Concurrency-vs-latency curves with per-layer breakdown, identifying the "knee point" where OS overhead becomes dominant.

### 3.4 Success Criteria

- [ ] Complete latency flame graph covering NIC→GPU for the inference request path
- [ ] Quantified TCP stack overhead as % of end-to-end latency under load
- [ ] Concurrency scaling curve with identified bottlenecks
- [ ] 4-way model loading comparison on bare metal
- [ ] Clear justification for which kernel layers Step 2's eBPF bypass must eliminate

---

## 4. Step 2: eBPF Network Bypass for Inference Requests (30%)

### 4.1 Objective

Build an eBPF-accelerated proxy that intercepts inference requests at the NIC level (XDP), bypasses the kernel TCP/IP stack, and delivers request data directly to the inference backend with minimal copies. Measure the latency improvement over standard TCP.

### 4.2 Architecture Decision: Phased Rollout

Rather than one big bang implementation, we build in three phases. Each phase produces independently measurable and publishable results.

```
Phase 1: Vanilla Rust Proxy (Week 5-6)
  └─→ Measure: "How much overhead does a Rust proxy add vs. direct TCP?"

Phase 2: AF_XDP Bypass (Week 7-12)
  └─→ Measure: "How much does XDP bypass save vs. Phase 1?"

Phase 3: GDRCopy Direct-to-GPU (Week 13-16, stretch goal)
  └─→ Measure: "Can we write tokens directly to GPU memory from the NIC?"
```

### 4.3 Phase 1: Vanilla Rust Proxy Baseline

#### Task 2.1a: Proxy Core (Week 5)

Build a minimal Rust TCP proxy that:
1. Listens on `0.0.0.0:8000`
2. Accepts TCP connections from inference clients
3. Parses the wire protocol (OpenAI-compatible JSON or custom binary)
4. Forwards to the inference backend (`localhost:8001`)
5. Streams response tokens back to the client

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

#### Task 2.1b: Baseline Benchmark (Week 6)

Benchmark three configurations:
1. **Direct**: Client → vLLM directly (no proxy)
2. **Rust Proxy**: Client → Rust proxy → vLLM
3. **Python Proxy**: Client → Python proxy → vLLM (for fairness)

Measure at varying concurrency (1, 4, 16, 64) and request sizes (128, 512, 2048 tokens prompt).

**Deliverable**: `docs/report/step2-phase1-proxy-baseline.md` — latency distribution, throughput curves, CPU overhead comparison.

**Publishable?** Yes — this becomes the control group for the eBPF experiment, and the data is part of the measurement paper (Contribution 1).

### 4.4 Phase 2: AF_XDP Bypass

This is the core technical contribution of Step 2.

#### Task 2.2a: XDP eBPF Program (Week 7-8)

Write an XDP eBPF program that:
1. Classifies packets: match on destination port (8000) and protocol (TCP)
2. Matching packets → `XDP_REDIRECT` to AF_XDP socket
3. Non-matching packets → `XDP_PASS` (transparent to other traffic)

```c
// src/proxy/xdp_filter.bpf.c
SEC("xdp")
int xdp_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;

    // Only IP
    if (eth->h_proto != __constant_htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;

    // Only TCP
    if (ip->protocol != IPPROTO_TCP) return XDP_PASS;

    struct tcphdr *tcp = (void *)(ip + 1);
    if ((void *)(tcp + 1) > data_end) return XDP_PASS;

    // Match on destination port
    if (tcp->dest == __constant_htons(INFERENCE_PORT)) {
        return XDP_REDIRECT; // Redirect to AF_XDP socket
    }

    return XDP_PASS; // All other traffic: kernel stack
}

char _license[] SEC("license") = "GPL";
```

**Key design decisions:**
- Use `xdpilone` crate (pure Rust) or `libbpf-rs` (C library wrapper) for AF_XDP
- Pre-allocate UMEM with 4096 frames of 4096 bytes each (16 MB total)
- Bind to a dedicated NIC RX queue to avoid contention

#### Task 2.2b: TCP Reassembly in Rust (Week 8-10)

This is the hardest technical challenge in Step 2. When packets arrive via AF_XDP, they are raw Ethernet frames containing IP datagrams containing TCP segments. The Rust userspace code must:

1. Parse Ethernet → IP → TCP headers
2. Maintain per-connection TCP state machine:
   ```
   ConnectionState:
     SYN_RCVD → ESTABLISHED → FIN_WAIT / CLOSE_WAIT → CLOSED
   ```
3. Track TCP sequence numbers and reassemble out-of-order segments
4. Detect complete HTTP requests (scan for `\r\n\r\n` in the reassembled stream)
5. Handle retransmissions, duplicate ACKs, and window scaling

```rust
// src/proxy/tcp_reasm.rs
pub struct TcpReassembler {
    connections: HashMap<ConnectionKey, TcpConnection>,
    config: ReasmConfig,
}

struct TcpConnection {
    state: TcpState,
    recv_buf: Vec<u8>,        // reassembled stream
    next_expected_seq: u32,
    out_of_order: BTreeMap<u32, Vec<u8>>,  // seq → segment data
    send_buf: Vec<u8>,        // response to send
    send_next: u32,
    last_ack_sent: u32,
}

impl TcpReassembler {
    /// Process a raw TCP segment, return Some(request) when a complete HTTP
    /// request is assembled in recv_buf
    pub fn ingest_segment(&mut self, key: ConnectionKey, tcp_header: &Tcphdr,
                          payload: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// Called when the proxy wants to send response data
    pub fn enqueue_response(&mut self, key: ConnectionKey, data: &[u8]);
    
    /// Generate the next TCP segment to send (if any)
    pub fn next_tx_segment(&mut self, key: ConnectionKey) -> Option<Vec<u8>>;
}
```

**Design simplification for v1**: Since the primary use case is **localhost or single-hop LAN** (client and server on same machine or same rack), we can assume:
- No packet loss (skip retransmission logic initially)
- No reordering (single NIC queue, same NUMA node)
- Small connection count (< 1000 concurrent)

This dramatically reduces complexity, allowing us to produce a working prototype in 3 weeks instead of 3 months.

#### Task 2.2c: AF_XDP Integration (Week 10-12)

Integrate the XDP program, AF_XDP socket, and TCP reassembler into a unified event loop:

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
            // 1. Refill RX descriptors
            self.fill_q.fill_free_frames()?;

            // 2. Poll for received packets
            let n = self.rx_q.poll_and_consume(|frame| {
                let pkt = parse_eth_ip_tcp(frame.data)?;
                if let Some(request) = self.reasm.ingest_segment(
                    pkt.conn_key, &pkt.tcp, pkt.payload)? {
                    // Complete HTTP request assembled
                    let backend = self.backend.clone();
                    tokio::spawn(async move {
                        let response = backend.generate(parse_request(&request)?).await?;
                        // Response tokens are enqueued in the reassembler's send buffer
                        enqueue_response(pkt.conn_key, &serialize_response(&response)?);
                    });
                }
            })?;

            // 3. Flush pending TX data
            for (conn_key, segment) in self.reasm.drain_tx_segments() {
                self.tx_q.send(conn_key.addr, segment)?;
            }

            // 4. Wake TX queue
            self.tx_q.wake()?;

            // Yield to tokio for async backend calls
            tokio::task::yield_now().await;
        }
    }
}
```

**Performance-Critical Path Optimization:**
- UMEM frames are pre-allocated and never freed (ring buffer)
- TCP reassembly uses `BytesMut` from the `bytes` crate for zero-copy buffer management
- HTTP parsing uses SIMD-accelerated `memchr` for `\r\n\r\n` boundary detection
- Backend calls are non-blocking (Tokio async)

#### Task 2.2d: Phase 2 Benchmark (Week 12)

Compare Phase 2 (AF_XDP bypass) against Phase 1 (vanilla proxy) and Direct:

| Metric | Direct TCP | Rust Proxy | AF_XDP Bypass | Improvement |
|--------|-----------|------------|---------------|-------------|
| Median request latency (128 tokens prompt) | Tp50_direct | Tp50_proxy | Tp50_xdp | Δ |
| Tail latency p99 | Tp99_direct | Tp99_proxy | Tp99_xdp | Δ |
| Throughput @ 64 concurrent | Q_direct | Q_proxy | Q_xdp | Δ |
| CPU utilization (sys%) | C_direct | C_proxy | C_xdp | Δ |
| Context switches/sec | S_direct | S_proxy | S_xdp | Δ |

### 4.5 Phase 3: GDRCopy — NIC → GPU Direct (Stretch)

**Goal**: After the AF_XDP path delivers tokens to the Rust proxy, use GDRCopy or GPU Direct RDMA to write them directly into the vLLM GPU input buffer.

**Why stretch**: This requires either:
- Modifying vLLM to expose its input buffer GPU address (invasive), or
- Building a standalone CUDA kernel demo showing the concept (lower impact)

**Recommendation**: Defer to post-Step-4 or a separate short paper. The AF_XDP bypass alone is a strong enough contribution.

### 4.6 Success Criteria

- [ ] Phase 1: Rust proxy adds < 100μs median overhead over direct TCP
- [ ] Phase 2: AF_XDP bypass reduces median latency by ≥ 30% vs. Phase 1 under load
- [ ] Phase 2: System CPU time reduced by ≥ 40% (TCP stack bypass evidence)
- [ ] Phase 2: No correctness regressions (100% token-match against direct TCP baseline)
- [ ] Complete latency breakdown comparing kernel TCP path vs. AF_XDP bypass path

---

## 5. Step 3: KV Cache Memory Manager — KCMM (30%)

### 5.1 Objective

Build KCMM — a **user-space OS service** that provides GPU KV cache memory management with transparent tiering (GPU↔CPU↔NVMe). KCMM replaces the inference engine's built-in KV cache allocator with one that offers cross-engine memory pressure management, LRU-based eviction, and optional tiered storage.

### 5.2 Why Not Just Use vLLM's Swap?

| Feature | vLLM's Built-in Swap | KCMM |
|---------|---------------------|------|
| Scope | Single vLLM process | Any process using the KCMM API |
| Cross-engine sharing | No | Yes |
| Eviction policy | vLLM's internal logic | Pluggable (LRU, LFU, FIFO) |
| Tiering | GPU ↔ CPU only | GPU ↔ CPU ↔ NVMe |
| Memory pressure view | vLLM's own pool only | System-wide (all registered pools) |
| Prefetching | None | Heuristic-based prefetch |
| Metrics | vLLM internal | UFS-compatible cross-engine |

### 5.3 Core Design

#### 5.3.1 Memory Model

```
┌─────────────────────────────────────────────────────────────┐
│  GPU Virtual Address Space (per engine process)              │
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Seq A   │ │ Seq B   │ │ Seq C   │ │  Free   │           │
│  │ Block 0 │ │ Block 0 │ │ Block 0 │ │  VA     │           │
│  │ Block 1 │ │ Block 1 │ │ Block 1 │ │  Space  │           │
│  │ Block 2 │ │   ...   │ │   ...   │ │         │           │
│  │   ...   │ │         │ │         │ │         │           │
│  └────┬────┘ └───┬─────┘ └───┬─────┘ └─────────┘           │
│       │          │           │                               │
│       │   cuMemMap (on-demand)                               │
│       ▼          ▼           ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GPU Physical Memory (2MB superblocks, fixed #blocks) │   │
│  │                                                      │   │
│  │  [Block 0] [Block 1] [Block 2] ... [Block N-1]      │   │
│  │     ↑                     ↑                          │   │
│  │     │  evict              │  restore                 │   │
│  │     ▼                     │                          │   │
│  └───────────────────────────┼──────────────────────────┘   │
│                              │                               │
│  ┌───────────────────────────▼──────────────────────────┐   │
│  │  CPU RAM (mmap'd file or shm)                        │   │
│  │  [/dev/shm/kcmm_swap]                                │   │
│  │  Block N → Block N+1 → ...                           │   │
│  │                           │                           │   │
│  │                           │  spill (optional)         │   │
│  │                           ▼                           │   │
│  │  NVMe SSD (cuFileRead/Write or standard I/O)         │   │
│  │  [/mnt/nvme/kcmm_swap]                               │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

Key design choices:

1. **GPU VA space is per-engine-process.** Each engine reserves its own contiguous VA region via `cuMemAddressReserve`. KCMM manages physical page mappings within this region.

2. **Physical pages are in 2 MB superblocks** (CUDA VMM default). Each superblock is carved into fixed-size blocks (e.g., 64 blocks per superblock for LLaMA-7B with 32 KB blocks). This is the current `src/cache/cuda_vmm.rs` design, preserved.

3. **Tiering is block-granular, not superblock-granular.** When evicting, KCMM can evict individual cold blocks, not entire 2MB superblocks. This requires the block data to be copied out (via `cudaMemcpy D2H`) before the physical page is freed.

4. **KCMM runs as a library linked into each engine process**, not as a separate daemon. This avoids IPC overhead on the critical path. Cross-engine coordination (for shared prefix detection) happens via shared memory or a lightweight Unix socket to a coordinator daemon.

#### 5.3.2 KCMM Internal Architecture

```rust
// src/kcmm/mod.rs

/// Top-level KCMM pool. One per engine process (or one per GPU).
pub struct KcmmPool {
    // GPU virtual address space
    gpu_va_start: u64,
    gpu_va_size: usize,

    // Physical memory management
    superblocks: Vec<Superblock>,
    free_blocks: VecDeque<BlockHandle>,
    
    // Sequence tracking
    sequences: HashMap<u64, SequenceState>,
    
    // Tiering
    tiering: Option<TieringEngine>,
    
    // Prefix sharing
    sharing: Option<SharingManager>,
    
    // Metrics
    metrics: KcmmMetrics,
    fragmentation_tracker: FragmentationTracker,
}

struct Superblock {
    handle: CudaMemHandle,    // from cuMemCreate
    va_offset: usize,         // offset in GPU VA
    block_size: usize,
    blocks_per_sb: usize,
    block_bitmap: Bitmap,     // which blocks are allocated
}

struct SequenceState {
    seq_id: u64,
    blocks: Vec<BlockRef>,    // logical block → physical block mapping
    is_active: bool,          // decode in progress vs. waiting
    last_access: Instant,     // for LRU
    shared_prefix_len: usize, // number of blocks shared with another sequence
}

enum BlockLocation {
    GpuResident(BlockHandle, u64),  // (handle, GPU VA offset)
    CpuResident(usize),             // offset in CPU swap buffer
    NvmeResident(u64),              // offset in NVMe swap file
    Evicting,                       // in transit
    Restoring,                      // in transit
}

struct TieringEngine {
    cpu_buffer: *mut u8,       // mmap'd CPU swap space
    cpu_buffer_size: usize,
    nvme_file: Option<File>,   // NVMe swap file
    eviction_policy: EvictionPolicy,
    block_states: HashMap<BlockHandle, BlockLocation>,
    evict_queue: BinaryHeap<EvictCandidate>,  // sorted by last_access
    prefetch_queue: VecDeque<BlockHandle>,
}

struct SharingManager {
    // Hash: block content hash → list of (engine_id, seq_id, block_idx) references
    prefix_index: HashMap<u64, Vec<BlockOwnership>>,
    ref_counts: HashMap<BlockHandle, u32>,
}
```

#### 5.3.3 Tiering Algorithm

**Eviction (GPU → CPU):**
```
Trigger: free_blocks.len() < low_watermark (e.g., < 10% of total)

1. Select victim: pop from evict_queue (LRU — coldest block from inactive sequence)
2. Allocate CPU buffer slot
3. cudaMemcpy D2H: GPU block → CPU buffer
4. cuMemUnmap: remove GPU physical page mapping
5. Update BlockLocation → CpuResident
6. Return GPU block to free_blocks
7. Repeat until free_blocks > target
```

**Restore (CPU → GPU):**
```
Trigger: kcmm_alloc_blocks() called for a block that is CpuResident

1. Allocate GPU physical block from free_blocks (or evict first)
2. cuMemMap: map GPU physical page at target VA
3. cudaMemcpy H2D: CPU buffer → GPU block
4. Update BlockLocation → GpuResident
5. Return block to caller
```

**Async Prefetch (optional optimization):**
```
Background thread:
1. For each active sequence, predict next needed blocks
   (e.g., sequence is at logical_block K, prefetch K+1, K+2)
2. If prefetch candidates are CpuResident, initiate async cudaMemcpy H2D
3. When alloc request arrives, block is already GPU-resident
```

#### 5.3.4 CUDA Stream Management

All KCMM GPU operations use dedicated CUDA streams to avoid interfering with inference compute:

```rust
pub struct KcmmStreams {
    pub evict: CudaStream,     // D2H copies
    pub restore: CudaStream,   // H2D copies  
    pub prefetch: CudaStream,  // async prefetch H2D
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

### 5.4 Integration with Existing Code

KCMM directly evolves from the current codebase:

| Current File | New Role in KCMM |
|-------------|-----------------|
| `src/cache/cuda_vmm.rs` | KCMM's GPU physical page management (superblocks, cuMemMap) |
| `src/cache/paged_kv.rs` | KCMM's block allocation + sequence tracking |
| `src/cache/swap.rs` | KCMM's TieringEngine (GPU↔CPU migration) |
| `src/cache/fragmentation_tracker.rs` | KCMM's metrics (IFR, PME, RFI) |
| `src/cache/unified_frag.rs` | UFS metrics collection for cross-engine comparison |

### 5.5 Tasks

#### Task 3.1: KCMM Core — Block Allocator with Tiering (Week 13-16)

1. **Extract and generalize** the existing `PagedKvCache` into `KcmmPool`
2. **Add `BlockLocation` tracking** — every block knows whether it's GPU-resident, CPU-resident, or NVMe-resident
3. **Implement eviction**:
   - LRU eviction queue
   - `cudaMemcpy D2H` on eviction
   - `cuMemUnmap` after copy completes
4. **Implement restore**:
   - `cuMemMap` physical page
   - `cudaMemcpy H2D` on restore
5. **Implement NVMe tier** (Week 15):
   - Use `cuFileRead`/`cuFileWrite` for GPU↔NVMe direct transfers (GDS)
   - Fall back to standard I/O if GDS unavailable
   - NVMe tier is optional — CPU tier alone is sufficient for Step 3

#### Task 3.2: vLLM Integration via KCMM C API (Week 16-17)

1. **Build `libkcmm.so`** — C shared library exposing the KCMM API
2. **Write Python bindings** using `ctypes` or `cffi`
3. **Monkey-patch vLLM's block allocator** to use KCMM:

```python
# kcmm_vllm_patch.py
import ctypes
import vllm.core.block_manager as bm

libkcmm = ctypes.CDLL("./libkcmm.so")

class KcmmBlockAllocator:
    """Drop-in replacement for vLLM's CpuGpuBlockAllocator"""
    
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.pool = libkcmm.kcmm_pool_create(block_size, num_gpu_blocks, ...)
    
    def allocate(self, block_tables):
        # Translate vLLM allocation requests to KCMM API
        for seq_id, num_blocks in block_tables.items():
            out = (ctypes.c_uint32 * num_blocks)()
            libkcmm.kcmm_alloc_blocks(self.pool, seq_id, num_blocks, out)
            # ...
    
    def free(self, seq_id):
        libkcmm.kcmm_free_blocks(self.pool, seq_id, ...)
```

The goal is **minimal vLLM modification** — ideally just a `--block-allocator-backend kcmm` flag.

#### Task 3.3: KCMM Evaluation (Week 17-18)

**Experiment 1: Tiering Benefit Under Memory Pressure**

```
Setup: vLLM + KCMM on A30 (24 GB VRAM), LLaMA-7B (14 GB weights → ~10 GB for KV cache)
Workload: 128 concurrent requests, 2048 max_tokens each
         (total KV cache need: ~16 GB → exceeds available ~10 GB)
Compare:
  A. vLLM default (OOM after ~80 concurrent, rejects remaining)
  B. vLLM + vLLM swap (GPU→CPU swap, same process)
  C. vLLM + KCMM (GPU→CPU tiering, external service)

Metrics: max concurrent admitted, TTFT p50/p99, throughput (tok/s), CPU RAM usage
```

**Experiment 2: Eviction Policy Comparison**

```
Compare: LRU vs. LFU vs. FIFO vs. Oracle (optimal) eviction policies in KCMM
Measure: hit rate (fraction of alloc requests where block was already GPU-resident),
         average restore latency, throughput
```

**Experiment 3: CUDA Stream Overhead**

```
Measure: cuMemMap latency (p50, p99) for different batch sizes
Measure: cudaMemcpy D2H/H2D overhead on dedicated streams vs. inference stream
Compare: single-block eviction vs. batched eviction (evict N blocks in one stream operation)
```

### 5.6 Success Criteria

- [ ] KCMM successfully replaces vLLM's block allocator with < 5% throughput regression in non-tiering mode
- [ ] Under memory pressure, KCMM admits ≥ 30% more concurrent requests than vLLM without swap
- [ ] KCMM GPU→CPU tiering adds < 200μs latency to block allocation (p50)
- [ ] LRU eviction achieves ≥ 85% hit rate compared to optimal (oracle) policy
- [ ] UFS metrics (IFR, PME, RFI) are equivalent to vLLM's internal allocator in absence of tiering

---

## 6. Step 4: Cross-Engine Prefix Sharing & Fine-Grained GPU Pages (25%)

### 6.1 Objective

Extend KCMM with two capabilities that are impossible for single-process inference engines:
1. **Cross-engine prefix sharing**: Multiple engine instances (or multiple requests within one engine) automatically share KV cache blocks for identical prefixes
2. **Fine-grained GPU pages**: Modify NVIDIA open-source kernel modules to reduce CUDA VMM minimum allocation granularity from 2 MB to 64 KB

### 6.2 Prefix Sharing Design

#### 6.2.1 How It Works

```
Scenario: Two requests with the same 500-token system prompt

Request A arrives:
  - KCMM allocates blocks 0..31 for the prefix (500 tokens / 16 tokens-per-block)
  - KCMM hashes the prefix content: SHA256(token_ids[0..500]) → hash
  - KCMM stores in prefix_index: {hash → [(pool_id, seq_A, blocks 0..31)]}

Request B arrives (same prefix):
  - Before allocating, KCMM hashes B's prefix tokens
  - Hash match found! KCMM calls kcmm_share_prefix(seq_A, seq_B, 32)
  - seq_B's block table[0..31] points to the SAME physical blocks as seq_A
  - Reference count on blocks 0..31 becomes 2

Request A finishes:
  - kcmm_free_blocks called for seq_A
  - Block 0..31: ref_count decreases to 1 (still referenced by seq_B) → NOT freed
  - Blocks 32..N: ref_count = 1 → freed

Request B finishes:
  - Block 0..31: ref_count decreases to 0 → freed
```

#### 6.2.2 Content-Addressable Prefix Index

```rust
// src/kcmm/sharing.rs

pub struct PrefixIndex {
    // Map: content_hash → list of physical block references
    entries: HashMap<Hash, Vec<SharedPrefix>>,
    // Map: (pool_id, seq_id) → set of hashes this sequence's prefix matches
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
    /// Check if this prefix already exists in any pool, and share if found
    pub fn try_share_prefix(
        &mut self,
        pool: &KcmmPool,
        seq_id: u64,
        prefix_token_ids: &[u32],
        block_size_tokens: usize,
    ) -> Option<Vec<u32>> {
        // 1. Compute content hash of prefix
        let hash = hash_prefix(prefix_token_ids, block_size_tokens);
        
        // 2. Look up in index
        if let Some(existing) = self.entries.get(&hash) {
            // 3. Increment refcounts on existing blocks
            for shared in existing {
                shared.ref_count.fetch_add(1, Ordering::SeqCst);
            }
            // 4. Return the existing block indices
            Some(existing.iter().flat_map(|s| s.block_indices()).collect())
        } else {
            // 5. New prefix — caller must allocate fresh, then register
            None
        }
    }
    
    /// Register a newly allocated prefix for future sharing
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

#### 6.2.3 Cross-Engine Scenario

For multiple vLLM instances on the same machine:

```
Instance A (GPU 0, port 8001): serves Model X
Instance B (GPU 1, port 8002): serves Model X (same model, different GPU)

Both receive requests with the same 500-token system prompt.

Without KCMM:
  - Instance A: allocates 32 blocks for prefix (500 tokens) on GPU 0
  - Instance B: allocates 32 blocks for prefix (500 tokens) on GPU 1
  - Total GPU memory: 64 blocks (2× waste)

With KCMM (same GPU):
  - Instance A and B share a KCMM pool on the same GPU
  - Total GPU memory: 32 blocks (0 waste)

With KCMM (different GPUs):
  - Each GPU manages its own pool
  - KCMM coordinator daemon detects duplicate prefix hash across pools
  - Cannot share physical pages across GPUs (no NVLink on A30)
  - BUT: can share the CPU-side cached prefix blocks
    (Instance B restores from CPU cache instead of recomputing the prefix KV)
```

For the project scope, same-GPU sharing is the priority. Cross-GPU sharing via CPU cache is a stretch goal.

### 6.3 Fine-Grained GPU Pages

#### 6.3.1 Motivation

CUDA VMM's `cuMemCreate` default minimum allocation granularity is 2 MB. For prefix sharing:
- A 2 MB superblock = 64 blocks × 32 KB per block (LLaMA-7B)
- A prefix might be only 100 tokens ≈ 6 blocks ≈ 192 KB
- With 2 MB granularity, the remaining ~1.8 MB of the superblock is wasted if not needed

With 64 KB granularity:
- A 64 KB "miniblock" = 2 blocks × 32 KB
- Prefix of 6 blocks: allocate 3 miniblocks = 192 KB (no waste)
- Much more precise physical memory allocation for small prefixes

#### 6.3.2 Technical Approach

NVIDIA's `open-gpu-kernel-modules` repository contains the UVM (Unified Virtual Memory) driver code under `kernel-open/nvidia-uvm/`. The 2 MB minimum granularity is enforced in the UVM page table construction.

**The modification (conceptual):**

```c
// kernel-open/nvidia-uvm/uvm_va_range.c (conceptual locations)

// Current: minimum allocation size is UVM_CHUNK_SIZE_MAX (2MB)
// Goal: support UVM_PAGE_SIZE_64K as minimum

// 1. Locate the GPU page table level that maps 2MB pages
// 2. Enable the next level down (64KB pages on A30/Ampere)
// 3. Modify cuMemCreate path to accept 64KB-aligned sizes
// 4. Ensure cuMemMap/cuMemUnmap work correctly at 64KB granularity
// 5. Update TLB invalidation to handle 64KB page entries

// Key structures to modify:
// - uvm_va_range_create() — accept 64KB alignment
// - uvm_page_tree — add 64KB page table entries
// - TLB shootdown logic — handle 64KB granularity
```

**Step-by-step:**

1. **Environment Setup** (Week 19):
   - Install NVIDIA driver 580.x with `--kernel-module-type=open` on d7525
   - Clone `open-gpu-kernel-modules` at matching tag
   - Verify: build and load unmodified modules → run CUDA VMM test

2. **Code Exploration** (Week 19-20):
   - Map the UVM page table walk: `uvm_va_range.c` → `uvm_page_tree.c` → hardware PTEs
   - Identify the constant/enum controlling minimum allocation size
   - Trace `cuMemCreate` → `uvm_va_range_create` call path
   - Document the multi-level GPU page table structure for A30 (Ampere)

3. **Implementation** (Week 20-22):
   - Add a module parameter: `uvm_min_allocation_size=65536` (default 2097152)
   - Modify `uvm_va_range_create` to accept 64KB-aligned sizes
   - Update page table entry format for 64KB pages
   - Handle edge cases: mixed 2MB/64KB allocations in same VA range

4. **Testing & Validation** (Week 22-23):
   - Allocate 64KB GPU memory via `cuMemCreate` → verify success
   - Map 64KB pages via `cuMemMap` → verify GPU access works
   - Run bandwidth microbenchmark: 64KB access vs. 2MB access
   - Run TLB miss rate microbenchmark (random access pattern)

5. **Integration with KCMM** (Week 23-24):
   - KCMM uses 64KB superblocks when the modified driver is detected
   - Compare with 2MB superblocks: waste reduction, TLB miss impact

#### 6.3.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cannot find the right code to modify in UVM driver | Medium | High | Start with the vAttention team's notes (they did this for A100). Search for `UVM_CHUNK_SIZE` or `PAGE_SIZE` constants. |
| Modified driver crashes GPU | High | Medium | Keep a backup of the original driver. Test with minimal CUDA kernels first. |
| 64KB pages degrade TLB performance | Medium | Medium | Measure TLB miss rate before/after. If degradation >10%, discuss trade-off explicitly. |
| CUDA 13.0 incompatibility | Low | High | Pin to CUDA 13.0 and driver 580.x at the start. |
| WSL2 cannot load custom drivers | Certain | — | All Step 4 kernel work MUST be on bare metal (d7525). |

### 6.4 Tasks

#### Task 4.1: Prefix Sharing in KCMM (Week 19-21)

1. Implement `SharingManager` as described in §6.2
2. Add `kcmm_share_prefix()` to the C API
3. Integrate with vLLM: detect common prefixes in the request queue, call `kcmm_share_prefix()` instead of `kcmm_alloc_blocks()` for matching requests
4. Benchmark: measure memory savings and throughput improvement vs. no sharing

#### Task 4.2: Prefix Sharing Evaluation (Week 21-22)

**Experiment: Shared Prefix Scenario**

```
Setup: vLLM + KCMM on A30, LLaMA-7B
Workload:
  - 50% of requests share a 2048-token system prompt
  - 50% of requests have unique prompts
Compare:
  A. vLLM without prefix sharing (APC disabled)
  B. vLLM with Automatic Prefix Caching (APC) enabled
  C. KCMM with prefix sharing

Metrics:
  - Memory: total GPU blocks allocated, shared blocks, memory saved vs. no sharing
  - Performance: TTFT for shared-prefix requests, throughput
  - Correctness: token-exact match between all configurations
```

**Experiment: Cross-Engine Sharing**

```
Setup: Two vLLM instances on same GPU, same model
Workload: Both instances receive requests with the same system prompt
Compare:
  A. Each instance independently (no sharing possible)
  B. KCMM sharing across instances

Metrics: total GPU memory usage, number of shared blocks, throughput per instance
```

#### Task 4.3: NVIDIA UVM Driver Modification (Week 22-25)

See §6.3.2 for the detailed technical approach.

#### Task 4.4: 64KB Page Evaluation (Week 25-26)

**Experiment: Granularity Impact**

```
Setup: Modified driver (64KB) vs. stock driver (2MB), same KCMM pool
Workload: Mixed prefix sharing + unique requests
Compare:
  A. 2MB superblocks (stock driver)
  B. 64KB superblocks (modified driver)

Metrics:
  - Physical Memory Efficiency (PME) — should improve with 64KB
  - Internal Fragmentation Rate (IFR) — unchanged (block-level metric)
  - Average block allocation latency (cuMemCreate + cuMemMap time)
  - TLB miss rate (via nvprof or custom microbenchmark)
  - End-to-end throughput (tok/s)
```

### 6.5 Success Criteria

- [ ] Prefix sharing: ≥ 80% memory savings for shared-prefix workload compared to no sharing
- [ ] Prefix sharing: token-exact output match against no-sharing baseline
- [ ] Cross-engine sharing: two vLLM instances share blocks, total memory ≤ 1.1× single-instance memory
- [ ] 64KB driver: `cuMemCreate` succeeds with 64KB size
- [ ] 64KB driver: PME improves by ≥ 30% for prefix-sharing workload vs. 2MB granularity
- [ ] 64KB driver: Performance regression (throughput) ≤ 5% vs. 2MB granularity

---

## 7. Evaluation Strategy

### 7.1 Test Environment

| Resource | Specification |
|----------|--------------|
| Server | d7525: 2× AMD EPYC 7302 (16-core), 128 GB RAM |
| GPU | NVIDIA A30 (24 GB HBM2e, Ampere, PCIe Gen4) |
| NVMe | 1.6 TB PCIe Gen4 (model TBD) |
| NIC | Mellanox ConnectX-6 DX 100Gb (single port) |
| OS | Ubuntu 22.04 or 24.04, Linux 6.6+ |
| CUDA | 13.0 |
| Driver | 580.x (open kernel modules) |

### 7.2 Workloads

| Workload | Description | Use Case |
|----------|-------------|----------|
| **Synthetic** | Fixed-length prompts, uniform token distribution | Microbenchmarks, reproducibility |
| **ShareGPT** | Real ChatGPT conversations (variable lengths) | Realistic serving workload |
| **Prefix-Heavy** | 80% of requests share a 2048-token prefix | Prefix sharing evaluation |
| **Burst** | Poisson arrival, 1→64→1 concurrent ramp | Stress test, tiering evaluation |

### 7.3 Baselines

| Baseline | Description | Compared In |
|----------|-------------|------------|
| vLLM (stock) | Standard vLLM installation, no modifications | All steps |
| vLLM + built-in swap | vLLM's GPU→CPU swap enabled | Steps 3-4 |
| vLLM + APC | vLLM's Automatic Prefix Caching enabled | Step 4 |
| SGLang | Alternative inference backend | Steps 3-4 (optional) |
| Direct TCP (no proxy) | Raw TCP to vLLM, no proxy layer | Step 2 |
| Custom Rust engine | Our existing inference engine | Steps 3-4 (comparison point) |

### 7.4 Key Metrics

| Metric | Definition | Tool |
|--------|-----------|------|
| **TTFT** | Time to First Token (end-to-end) | Application timestamps |
| **TPOT** | Time per Output Token (decode speed) | Application timestamps |
| **Throughput** | Total tokens/sec across all requests | Application timestamps |
| **IFR** | Internal Fragmentation Rate | UFS (unified_frag.rs) |
| **PME** | Physical Memory Efficiency | UFS |
| **BU** | Block Utilization | UFS |
| **RFI** | Runtime Fragmentation Index | UFS |
| **Layer latency** | Per-kernel-layer latency breakdown | bpftrace scripts |
| **CPU util** | System/user CPU % | `perf stat`, `mpstat` |
| **GPU util** | GPU SM/memory utilization | `nvidia-smi`, CUDA profiler |
| **Context switches** | Voluntary + involuntary/sec | `perf stat` |
| **cuMemMap latency** | p50/p99 map time | Custom instrumentation |
| **eviction latency** | p50/p99 block eviction time | Custom instrumentation |
| **TLB miss rate** | GPU TLB misses per access | `nvprof` or custom benchmark |

---

## 8. Timeline & Milestones

```
Week  Stage  Step  Milestone                                      Deliverable
────  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
 1    Setup  —     Environment setup on d7525                     Working bare-metal environment
 2    Step1  1.1   NIC→GPU latency tracing complete               trace_request_path.bt
 3    Step1  1.2   Model loading I/O comparison                   4-way loader comparison report
 4    Step1  1.3   Concurrency scaling analysis                   Latency-vs-concurrency curves
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
 5    Step2  2.1a  Vanilla Rust proxy implemented                 src/proxy/ (Phase 1)
 6    Step2  2.1b  Proxy baseline benchmark                       Phase 1 benchmark report
 7    Step2  2.2a  XDP eBPF program + AF_XDP setup               xdp_filter.bpf.c, UMEM setup
 8    Step2  2.2a  XDP redirection verified (drop test)           XDP functional test
 9    Step2  2.2b  TCP reassembly (SYN/ACK handshake)             TcpReassembler (basic)
10    Step2  2.2b  TCP reassembly (data transfer, FIN)            TcpReassembler (complete)
11    Step2  2.2c  AF_XDP integrated with proxy loop              AfXdpProxy working
12    Step2  2.2d  Phase 2 benchmark                              AF_XDP vs. direct TCP report
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
13    Step3  3.1   KCMM core: generalize PagedKvCache             src/kcmm/ (core)
14    Step3  3.1   KCMM core: BlockLocation + evict queue         TieringEngine (evict only)
15    Step3  3.1   KCMM: GPU→CPU eviction + restore working       Full tiering loop
16    Step3  3.2   libkcmm.so + C API                             libkcmm.so
17    Step3  3.2   vLLM KCMM integration                          Monkey-patched vLLM block allocator
18    Step3  3.3   KCMM evaluation (memory pressure, LRU)         Step 3 evaluation report
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
19    Step4  4.1   PrefixIndex in KCMM                            SharingManager (same-engine)
20    Step4  4.1   KCMM cross-engine coordinator daemon           SharingManager (cross-engine)
21    Step4  4.2   Prefix sharing evaluation                      Prefix sharing report
22    Step4  4.3   Read/understand UVM page table code            UVM driver analysis document
23    Step4  4.3   Implement 64KB support in UVM driver           Modified nvidia-uvm.ko
24    Step4  4.3   Test 64KB driver (stability + correctness)     64KB driver test report
25    Step4  4.4   Integrate 64KB with KCMM + benchmark           64KB vs 2MB comparison
26    Step4  4.4   TLB miss rate analysis                         64KB final report
───  ─────  ────  ──────────────────────────────────────────     ──────────────────────────────
27    Write  —     Paper draft (measurement paper)                First draft
28    Write  —     Paper draft (systems paper)                    First draft
29    Polish —     Revision, rebuttal prep, artifact evaluation   Final paper
30    Buffer —     Overflow buffer (2 weeks)                      —
```

**Key Decision Points:**
- **Week 4 checkpoint**: Is TCP stack overhead significant enough to justify AF_XDP bypass? If < 5% of end-to-end latency, pivot Step 2 to "network observability" paper.
- **Week 12 checkpoint**: Does AF_XDP bypass show ≥ 15% latency improvement? If not, cap Step 2 at Phase 1 + measurement and shift resources to Steps 3-4.
- **Week 18 checkpoint**: Does KCMM tiering admit ≥ 20% more concurrent requests than vLLM swap? If not, focus Step 3 on the cross-engine sharing aspect.
- **Week 22 checkpoint**: Is the UVM driver modification feasible? If the code is too opaque or crashes are frequent, pivot Step 4 to a measurement-only study of 2MB vs. ideal page sizes.

---

## 9. Risk Register & Mitigation

### 9.1 Critical Risks

| ID | Risk | P | I | Mitigation |
|----|------|---|---|------------|
| R1 | AF_XDP adds latency instead of reducing it | M | H | Build in phases. Phase 1 (vanilla proxy) is a publishable measurement even if Phase 2 fails. Pivot to "characterizing why AF_XDP doesn't help for inference" paper. |
| R2 | TCP reassembly in user-space is buggy (data corruption, connection leaks) | H | H | Start with localhost-only (skip TCP entirely for v1). Use `tokio::net::TcpStream` as the fallback. Add XDP bypass for remote traffic only after localhost is solid. |
| R3 | Cannot integrate KCMM with vLLM without invasive changes | M | H | Build KCMM with the custom Rust engine first (we control the code). Integrate with vLLM as a stretch. Even standalone KCMM + Rust engine comparison is publishable. |
| R4 | Modified NVIDIA driver is unstable / crashes | H | M | Timebox the 64KB driver work to 4 weeks. If unstable, publish the 2MB results and frame 64KB as "future work with preliminary analysis." |
| R5 | WSL2 is the primary dev environment but can't run XDP or custom kernel modules | C | H | All WSL2 work: proxy logic, KCMM core, benchmarks. All bare-metal work: XDP, UVM driver. Keep a clear separation. |
| R6 | vLLM version churn breaks integration | M | M | Pin vLLM version at project start. Document the pinned version. Minor API changes are acceptable for final paper revision. |

### 9.2 Technical Risks

| ID | Risk | P | I | Mitigation |
|----|------|---|---|------------|
| R7 | AF_XDP requires specific NIC drivers (only mlx5, i40e, ice are well-supported) | L | H | d7525 has Mellanox ConnectX-6 DX which uses mlx5 — fully supported. |
| R8 | cuMemMap/unmap overhead dominates in high-churn workloads | M | M | Batch cuMemMap calls. Use deferred unmapping (vAttention strategy). Measure and compare. |
| R9 | GPU TLB thrashing with 64KB pages | M | M | Benchmark TLB miss rate explicitly. If >2× increase, discuss granularity trade-off in paper. |
| R10 | Rust CUDA bindings (cudarc crate) have missing APIs for VMM | L | M | We already have `src/cache/cuda_vmm.rs` wrapping the raw CUDA driver API via FFI. Extend as needed. |

### 9.3 Schedule Risks

| ID | Risk | P | I | Mitigation |
|----|------|---|---|------------|
| R11 | d7525 server unavailable (hardware failure, scheduling conflict) | L | H | Reserve backup: use any machine with NVIDIA GPU + Linux. GDS and 100Gb NIC tests can be deferred or simulated. |
| R12 | Step 2 takes longer than estimated (TCP reassembly is hard) | H | M | Pre-built fallback: TC BPF instead of XDP (avoids TCP reassembly). Graceful degradation to "smart proxy without kernel bypass." |
| R13 | Steps 3-4 compressed by Step 2 overrun | M | H | Steps are decoupled. Can write Step 2 paper while finishing Step 3 implementation. Parallelize writing and coding in Weeks 24-30. |

---

## 10. Publication Strategy

### 10.1 Paper Plan

| Paper | Target Venue | Core Contribution | Step Dependency | Earliest Submission |
|-------|-------------|-------------------|----------------|--------------------|
| **Paper 1: Measurement** | EuroSys '27, ATC '27 | OS latency characterization of LLM inference request path (Step 1 + Step 2 Phase 1) | Step 1 + Step 2 Phase 1 | May 2026 (EuroSys) or Jan 2027 (ATC) |
| **Paper 2: Systems** | SOSP '27, OSDI '28 | eBPF Network Bypass + KCMM (Steps 2-4) | All steps | Apr 2027 (SOSP) or Dec 2027 (OSDI) |
| **Paper 3: Short** | HotOS '27, APSys '27 | Individual component: GDS model loading or 64KB GPU pages | Step 1 GDS or Step 4 64KB | Varies |

### 10.2 Narrative Arcs

**Paper 1 (Measurement):** *"Is the OS the bottleneck for LLM inference?"*
- Hook: Everyone optimizes GPU kernels, but what about the OS?
- Contribution: First comprehensive OS latency breakdown for LLM inference serving
- Data: Step 1 traces (NIC→GPU latency decomposition with concurrency scaling)
- Takeaway: Identifies specific kernel subsystems as bottlenecks, motivating Paper 2

**Paper 2 (Systems):** *"An OS Support Layer for LLM Inference"*
- Hook: Inference engines reimplement OS functionality in user space (memory management, scheduling, I/O)
- Contribution: KCMM + eBPF bypass — an OS layer that accelerates ANY inference engine
- Data: Steps 2-4 evaluation (AF_XDP latency reduction, KCMM tiering benefit, prefix sharing memory savings)
- Takeaway: OS abstractions are the right level for LLM inference optimization

### 10.3 Artifact Evaluation Plan

- All code: open-source (MIT or Apache 2.0), GitHub
- Benchmarks: reproducible scripts (`scripts/bench_*.sh`), documented workloads
- Trace data: anonymized, included in repository
- Hardware requirements: documented (A30 24GB or larger, Linux 6.6+)
- Docker image: for vLLM + KCMM reproducible setup

---

## 11. Codebase Transition Plan

### 11.1 Current → Target Mapping

```
src/
├── main.rs                 →  kept, now launches proxy or standalone engine
├── lib.rs                  →  kept
├── config.rs               →  extended: add [proxy] and [kcmm] sections
├── server/
│   ├── mod.rs              →  kept
│   ├── http.rs             →  EVOLVED: becomes proxy request parser (src/proxy/http.rs)
│   └── pipeline.rs         →  kept for standalone mode
├── proxy/                  →  NEW: eBPF proxy
│   ├── mod.rs              →  proxy main
│   ├── config.rs           →  proxy configuration
│   ├── af_xdp_loop.rs      →  AF_XDP event loop
│   ├── tcp_reasm.rs        →  TCP reassembly state machine
│   ├── http_parse.rs       →  HTTP/JSON parsing (evolved from server/http.rs)
│   ├── backend.rs          →  InferenceBackend trait + vLLM/SGLang impls
│   └── xdp_filter.bpf.c    →  XDP eBPF program
├── kcmm/                   →  NEW: KV Cache Memory Manager
│   ├── mod.rs              →  KcmmPool top-level
│   ├── pool.rs             →  pool lifecycle, block allocation
│   ├── superblock.rs       →  superblock management (from cuda_vmm.rs)
│   ├── tiering.rs          →  TieringEngine (from swap.rs)
│   ├── sharing.rs          →  SharingManager (prefix cache)
│   ├── metrics.rs          →  UFS metrics (from unified_frag.rs)
│   ├── ffi.rs              →  C API (for libkcmm.so)
│   └── streams.rs          →  CUDA stream management
├── cache/                  →  KEPT: but now specific to standalone engine mode
│   ├── mod.rs
│   ├── kv_cache.rs         →  simple contiguous cache (baseline only)
│   ├── paged_kv.rs         →  refactored: delegates to KCMM when KCMM is enabled
│   ├── cuda_vmm.rs         →  refactored: shared between cache/ and kcmm/
│   ├── swap.rs             →  refactored: shared between cache/ and kcmm/
│   └── ...
├── model/                  →  KEPT: unchanged (model loading, weights, transformer)
├── batch/                  →  KEPT: standalone scheduler
├── cuda/                   →  KEPT: custom CUDA kernels
│   ├── kernels/
│   └── runtime.rs
├── decoder/                →  KEPT: standalone decoder
├── trace/                  →  NEW: eBPF tracing programs
│   ├── request_path.bt     →  request-path latency trace
│   └── kcmm_events.bt      →  KCMM eviction/restore events
└── bin/
    ├── latttice            →  standalone inference engine (backward compat)
    ├── latttice-proxy      →  eBPF proxy binary
    └── kcmm-bench          →  KCMM microbenchmarks
```

### 11.2 Backward Compatibility

The standalone Rust inference engine remains functional throughout:
- `cargo run -- --standalone --model llama-7b` → runs the original engine
- `cargo run -- --proxy --backend vllm` → runs the eBPF proxy
- `cargo run -- --proxy --kcmm` → runs proxy + KCMM

### 11.3 Dependency Changes

```toml
# New dependencies
[dependencies]
xdpilone = "0.6"          # or libbpf-rs = "0.25"
etherparse = "0.16"       # Ethernet/IP/TCP header parsing
memchr = "2.7"            # SIMD-accelerated byte search
bytes = "1.9"             # zero-copy buffer management
reqwest = { version = "0.12", features = ["json"] }

[build-dependencies]
# For compiling XDP eBPF programs
cargo-bpf = "..."         # or manual clang invocation
```

### 11.4 Testing Strategy

| Component | Test Type | Tool |
|-----------|-----------|------|
| TCP reassembly | Unit tests with crafted packets | `cargo test` |
| KCMM block allocator | Property-based tests (proptest) | `cargo test` |
| KCMM tiering | Integration test: fill pool, verify eviction | `cargo test` + real GPU |
| AF_XDP proxy | Integration test: send requests, verify tokens | `scripts/test_proxy.sh` |
| vLLM + KCMM integration | Integration test: vLLM server with KCMM patch | `scripts/test_kcmm_vllm.sh` |
| 64KB driver | Smoke test: cuMemCreate(64KB) → GPU access → cuMemFree | `scripts/test_64kb.sh` |
| Full system | End-to-end benchmark (same as evaluation) | `scripts/run_bench.sh` |

---

## 12. Summary: Why This Plan Wins

| Dimension | Original tasks.md | This Plan |
|-----------|------------------|-----------|
| **Core narrative** | "We built a better inference engine" | "We built an OS layer that makes all inference engines better" |
| **Differentiation from vLLM** | Competes (losing battle) | Complements (vLLM benefits from our work) |
| **Step 3-4 contribution** | Block-table PagedAttention (rediscovering vLLM) | Cross-engine GPU memory tiering + prefix sharing (novel) |
| **eBPF contribution** | NCCL bypass (niche) | Inference request path bypass (broad impact) |
| **Academic novelty** | Low (many inference engine projects) | High (OS + ML intersection is underexplored) |
| **Risk profile** | High (must beat vLLM on core metrics) | Moderate (each step is independently publishable) |
| **Code reuse** | Steps 3-4 throw away existing work | Existing cache/cuda_vmm/swap code evolves into KCMM |
| **Publication path** | 1 paper (must succeed entirely) | 2-3 papers (each step is a viable publication) |
