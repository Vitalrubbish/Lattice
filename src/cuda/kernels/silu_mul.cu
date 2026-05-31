#include "cuda_fp16.h"

extern "C" __global__ void silu_mul_f16(
    const __half *gate,
    const __half *up,
    __half *out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float silu_g = g / (1.0f + expf(-g));
    out[i] = __float2half(silu_g * u);
}
