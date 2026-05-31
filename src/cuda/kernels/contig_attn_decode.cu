#include "cuda_fp16.h"

// Decode attention for contiguous KV cache with GQA support.
// Q: [batch, num_q_heads, head_dim] — packed as [batch, num_q_heads * head_dim]
// K: [batch, kv_heads, seq_len, head_dim] — flattened with stride
// V: same layout as K
// Output: [batch, num_q_heads * head_dim]
extern "C" __global__ void contig_attn_decode_f16(
    const __half *q,           // [batch * num_q_heads * head_dim]
    const __half *k,           // [batch * kv_heads * seq_len * head_dim]
    const __half *v,           // [batch * kv_heads * seq_len * head_dim]
    __half *out,               // [batch * num_q_heads * head_dim]
    int batch, int num_q_heads, int kv_heads, int head_dim, int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_q_heads;
    if (idx >= total) return;

    int b = idx / num_q_heads;
    int qh = idx % num_q_heads;
    int kvh = qh * kv_heads / num_q_heads; // GQA: map Q head to KV head
    float scale = rsqrtf((float)head_dim);

    // Online softmax: m (max), s (sum exp), accum output
    float m = -1e30f;
    float s = 0.0f;

    float acc[64]; // for head_dim up to 256
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    // K stride: kv_heads * seq_len * head_dim per batch item
    int k_batch_offset = b * kv_heads * seq_len * head_dim;
    int k_head_offset = k_batch_offset + kvh * seq_len * head_dim;
    int v_head_offset = kvh * seq_len * head_dim;  // same batch offset base

    const __half *q_ptr = q + b * num_q_heads * head_dim + qh * head_dim;
    float q_val[64];
    for (int d = 0; d < head_dim; d++) {
        q_val[d] = __half2float(q_ptr[d]);
    }

    for (int pos = 0; pos < seq_len; pos++) {
        const __half *k_ptr = k + k_head_offset + pos * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_val[d] * __half2float(k_ptr[d]);
        }
        dot *= scale;

        float old_m = m;
        m = fmaxf(m, dot);
        float e = expf(dot - m);
        float correction = expf(old_m - m);
        s = s * correction + e;

        const __half *v_ptr = v + k_batch_offset + v_head_offset + pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction + e * __half2float(v_ptr[d]);
        }
    }

    __half *out_ptr = out + b * num_q_heads * head_dim + qh * head_dim;
    float inv_s = 1.0f / s;
    for (int d = 0; d < head_dim; d++) {
        out_ptr[d] = __float2half(acc[d] * inv_s);
    }
}
