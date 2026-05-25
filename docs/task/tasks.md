# An Efficient OS Support Layer for Large Language Model (LLM) Inference

## Background and Motivation

The inference process of large language models (e.g., GPT, LLaMA series) is divided into two stages: Prefill and Decode. These two stages differ significantly in computational intensity and memory access patterns. Long-context inference generates enormous Key-Value Caches (KV Cache). Under traditional OS memory allocation mechanisms, the dynamic growth of KV Cache leads to severe external and internal memory fragmentation, greatly limiting GPU memory utilization efficiency and the overall throughput of inference systems. Top systems conferences in 2024 focused heavily on this pain point, producing heavyweight research such as DistServe (decoupling prefill and decode computation) and LoongServe (elastic sequence parallelism for long contexts).

## Expected Goals

In this task, you need to design and implement an OS support layer tailored specifically for LLM inference workloads. The core challenges lie in the kernel-level PagedAttention mechanism and virtual GPU memory management, as well as coroutine-based heterogeneous scheduling and eBPF network offloading.

### Kernel-Level PagedAttention Mechanism and Offloading Management

Provide automatic GPU memory offloading and on-demand loading mechanisms at the kernel level. When inference requests dynamically generate new tokens and need to store corresponding KV Caches, the kernel's page fault handling mechanism should allocate fine-grained physical memory on demand and dynamically update the corresponding process's page table structures.

### Decoding Algorithms

Support advanced decoding algorithms such as Beam Search that require generating multiple candidate sequences. Implement a highly optimized Copy-on-Write (COW) mechanism in the kernel, allowing multiple candidate sequences to share the same underlying physical KV Cache data pages before forking. The kernel must maintain rigorous physical page reference counts and trigger actual memory allocation and data copying only when a sequence attempts to modify the data.

### eBPF-Based Network Offloading

Embed eBPF hooks in the kernel's network data path. When external network packets containing inference requests arrive at the NIC, your eBPF program should parse the data in kernel space and trigger GPU computation, achieving zero-copy data flow that bypasses the traditional socket buffer layer.

## Baseline Code

Use Linux + Rust as the system's baseline code, with Rust for execution. Compare against baselines: vLLM, SGLang.

## Suggested Steps

### Step 1: Model Weight Loading and I/O Stack Analysis (15%)

The baseline system provided by the TA uses naive `read` + `cudaMemcpy` to load model weights. You need to use eBPF (bpftrace or custom BPF programs) to trace key kernel functions such as `vfs_read`, `filemap_get_pages`, `submit_bio`, and NVMe completion, analyzing the data flow and latency distribution along the complete I/O path: VFS → page cache → block layer → device driver → DMA.

Based on this analysis, you should implement and compare the following loading methods in sequence: `mmap` (page-fault-driven on-demand loading), `O_DIRECT` (bypassing page cache), and GDS `cuFileRead` (NVMe direct DMA to GPU memory). Through trace data and performance comparisons, you should analyze: whether the page cache helps or hurts in model loading scenarios, which kernel subsystems each method's data path traverses, and why GDS can reduce CPU involvement and memory copies.

### Step 2: eBPF-Based Distributed Inference Network Acceleration (25%)

Extend the system to multi-GPU pipeline parallel inference. Under TCP-based Ethernet interconnects, activation transfers between pipeline stages must traverse the full kernel TCP/IP protocol stack, introducing multiple memory copies and context switches.

You need to write eBPF programs at the XDP or TC layer to identify packets belonging to NCCL communication and bypass them from the kernel protocol stack, delivering data directly to user-space communication buffers via AF_XDP sockets. Combined with GDRCopy or CUDA IPC, explore further zero-copy writes of received data into GPU memory. Non-NCCL traffic should continue through the kernel protocol stack unaffected. You should measure single-transfer latency before and after bypass, pipeline bubble ratio, and end-to-end inference throughput.

### Step 3: Continuous Batching and KV Cache Memory Management (35%)

Implement a continuous batching scheduler for the system, supporting dynamic addition and removal of requests. KV cache allocation should use the CUDA VMM API (`cuMemCreate` / `cuMemAddressReserve` / `cuMemMap`) to implement paged GPU memory management: reserve a contiguous GPU virtual address space as a KV cache address pool, allocate physical GPU memory in fixed-size blocks and maintain a free list, map physical blocks to virtual addresses on demand when requests arrive, and unmap/reclaim when requests finish. The attention kernel indexes virtual addresses via a block table, while underlying physical blocks can be discretely distributed.

You should compare this implementation against vLLM in terms of memory fragmentation rate, maximum concurrent requests, and throughput, and analyze the invocation overhead of `cuMemMap`/`cuMemUnmap` along with optimization strategies (e.g., batch mapping).

### Step 4: Prefix Sharing and Fine-Grained Page Tables (25%)

Build on Step 3 to implement prefix caching: for requests sharing the same system prompt or few-shot prefix, their KV Cache for the prefix portion should map to the same physical blocks, with lifecycle managed by reference counting. Physical blocks corresponding to a prefix are reclaimed when the reference count reaches zero.

The CUDA VMM API default minimum allocation granularity is 2MB, which is too coarse for fine-grained prefix sharing. You need to read the page table construction and allocation granularity related code in the NVIDIA open-source kernel modules (`open-gpu-kernel-modules`) under `kernel-open/nvidia-uvm/`, understand the mechanisms of large pages and small pages in GPU multi-level page tables, modify the relevant code to enable the VMM API to support smaller allocation granularities (64KB recommended), and analyze the impact of granularity changes on page table size, TLB miss rate, and inference performance.
