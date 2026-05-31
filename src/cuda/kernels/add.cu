#include "cuda_fp16.h"

extern "C" __global__ void add_f16(
    __half *a,
    const __half *b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    a[i] = __float2half(va + vb);
}
