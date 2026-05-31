#include "cuda_fp16.h"
extern "C" __global__ void rms_norm_f16(
    const __half *x,       // [rows, cols]
    const __half *weight,  // [cols]
    __half *out,           // [rows, cols]
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    // Compute mean of squares via shared memory
    extern __shared__ float shared[];
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = __half2float(x[row * cols + i]);
        sum_sq += v * v;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)cols + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < cols; i += blockDim.x) {
        float v = __half2float(x[row * cols + i]) * inv_rms;
        float w = __half2float(weight[i]);
        out[row * cols + i] = __float2half(v * w);
    }
}
