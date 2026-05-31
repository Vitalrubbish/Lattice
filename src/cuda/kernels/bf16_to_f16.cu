#include "cuda_fp16.h"
#include "cuda_bf16.h"

// In-place BF16→F16 conversion.  Both formats are 2 bytes per element,
// so the conversion can be done in-place on the same buffer.
extern "C" __global__ void bf16_to_f16(
    __nv_bfloat16 *buf,       // input: BF16, output: F16 (same buffer)
    int n                      // number of elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(buf[i]);
    __half *out = (__half *)buf;
    out[i] = __float2half(v);
}
