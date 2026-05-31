#include "cuda_fp16.h"

// Per-row online softmax for scores matrix [rows, cols]
// Uses max-subtraction for numerical stability, output in f16.
extern "C" __global__ void softmax_f16(
    const __half *inp,      // [rows, cols]
    __half *out,            // [rows, cols]
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    // Find max per row
    extern __shared__ float smem[];
    float my_max = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = __half2float(inp[row * cols + i]);
        if (v > my_max) my_max = v;
    }
    smem[tid] = my_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Compute exp sum
    float my_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        my_sum += expf(__half2float(inp[row * cols + i]) - row_max);
    }
    smem[tid] = my_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0] + 1e-10f;

    // Compute softmax
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = expf(__half2float(inp[row * cols + i]) - row_max) / row_sum;
        out[row * cols + i] = __float2half(v);
    }
}
