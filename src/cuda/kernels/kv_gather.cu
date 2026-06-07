#include "cuda_fp16.h"

// Gather same-layer KV data from N scattered source pointers into a
// contiguous destination buffer.  Each source block contributes
// (block_bytes / sizeof(__half)) elements.
//
// Launched with one thread per half-precision element.  Grid covers
// num_blocks * (block_bytes / sizeof(__half)) elements.
extern "C" __global__ void gather_kv_layer(
    const unsigned long long *src_ptrs,  // array of N device pointers (CUdeviceptr)
    __half *dst,                          // contiguous staging destination
    int half_count,                       // elements per block (= block_bytes / sizeof(__half))
    int num_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_blocks * half_count;
    if (tid >= total) return;

    int blk = tid / half_count;
    int off = tid % half_count;
    const __half *src = (const __half *)src_ptrs[blk];
    dst[tid] = src[off];
}

// Scatter same-layer KV data from a contiguous source buffer to N
// scattered destination pointers.  Reverse of gather_kv_layer.
//
// Launched with one thread per half-precision element.
extern "C" __global__ void scatter_kv_layer(
    const __half *src,                    // contiguous staging source
    const unsigned long long *dst_ptrs,   // array of N device pointers (CUdeviceptr)
    int half_count,                       // elements per block (= block_bytes / sizeof(__half))
    int num_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_blocks * half_count;
    if (tid >= total) return;

    int blk = tid / half_count;
    int off = tid % half_count;
    __half *dst = (__half *)dst_ptrs[blk];
    dst[off] = src[tid];
}
