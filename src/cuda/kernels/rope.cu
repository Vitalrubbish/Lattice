#include "cuda_fp16.h"
extern "C" __global__ void rope_f16(
    __half *q,              // [batch, num_q_heads, 2, half_dim] — last dim is even/odd pairs
    __half *k,              // [batch, num_kv_heads, 2, half_dim]
    int batch, int num_q_heads, int num_kv_heads, int half_dim,
    int pos                  // current position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = batch * num_q_heads * half_dim;
    int total_k = batch * num_kv_heads * half_dim;

    // Q rotation
    if (idx < total_q) {
        int b = idx / (num_q_heads * half_dim);
        int h = (idx / half_dim) % num_q_heads;
        int d = idx % half_dim;

        float theta = 1.0f / powf(10000.0f, (float)(2 * (d / 2)) / (float)(half_dim * 2));

        int base = b * num_q_heads * half_dim * 2 + h * half_dim * 2;
        __half x0 = q[base + 2 * d];
        __half x1 = q[base + 2 * d + 1];

        float cos_theta = cosf((float)pos * theta);
        float sin_theta = sinf((float)pos * theta);

        float v0 = __half2float(x0) * cos_theta - __half2float(x1) * sin_theta;
        float v1 = __half2float(x0) * sin_theta + __half2float(x1) * cos_theta;
        q[base + 2 * d] = __float2half(v0);
        q[base + 2 * d + 1] = __float2half(v1);
    }

    // K rotation
    if (idx < total_k) {
        int b = idx / (num_kv_heads * half_dim);
        int h = (idx / half_dim) % num_kv_heads;
        int d = idx % half_dim;

        float theta = 1.0f / powf(10000.0f, (float)(2 * (d / 2)) / (float)(half_dim * 2));

        int base = b * num_kv_heads * half_dim * 2 + h * half_dim * 2;
        __half x0 = k[base + 2 * d];
        __half x1 = k[base + 2 * d + 1];

        float cos_theta = cosf((float)pos * theta);
        float sin_theta = sinf((float)pos * theta);

        float v0 = __half2float(x0) * cos_theta - __half2float(x1) * sin_theta;
        float v1 = __half2float(x0) * sin_theta + __half2float(x1) * cos_theta;
        k[base + 2 * d] = __float2half(v0);
        k[base + 2 * d + 1] = __float2half(v1);
    }
}
