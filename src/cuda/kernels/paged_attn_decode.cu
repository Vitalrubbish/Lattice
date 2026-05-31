#include "cuda_fp16.h"

// Paged attention for decode (1 query token per sequence).
// K/V are stored in paged blocks within a VA region.
//
// block_tables: [batch * max_blocks_per_seq] — seq i starts at i*max_blocks_per_seq
// seq_lens: [batch] — tokens processed so far for each sequence
// va_k, va_v: base addresses of the K and V VA regions (per layer)
// block_offsets_f16: [num_blocks] — byte offset within VA region, divided by sizeof(f16)
extern "C" __global__ void paged_attn_decode_f16(
    const __half *q,
    const __half *va_k,
    const __half *va_v,
    const int *block_tables,
    const int *seq_lens,
    const unsigned long long *block_offsets_f16,
    __half *out,
    int total_heads,
    int num_q_heads,
    int kv_heads,
    int head_dim,
    int packed_bs          // lower 16 bits = block_size, upper 16 bits = max_blocks_per_seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_heads) return;

    int b = idx / num_q_heads;
    int qh = idx % num_q_heads;
    int kvh = qh * kv_heads / num_q_heads;
    float scale = rsqrtf((float)head_dim);

    int block_size = packed_bs & 0xFFFF;
    int max_blocks_per_seq = (packed_bs >> 16) & 0xFFFF;

    int seq_len = seq_lens[b];
    if (seq_len == 0) return;

    float m = -1e30f;
    float s = 0.0f;
    float acc[64]; // head_dim up to 256
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    // Q values for this head
    const __half *q_ptr = q + (unsigned long long)b * num_q_heads * head_dim + (unsigned long long)qh * head_dim;
    float q_val[64];
    for (int d = 0; d < head_dim; d++) q_val[d] = __half2float(q_ptr[d]);

    #pragma unroll 1
    for (int pos = 0; pos < seq_len; pos++) {
        int lb = pos / block_size;
        int off = pos % block_size;
        int block_idx = block_tables[(unsigned long long)b * max_blocks_per_seq + lb];
        unsigned long long block_off = block_offsets_f16[block_idx];

        const __half *k_token = va_k + block_off + (unsigned long long)(off * kv_heads * head_dim);
        const __half *k_head = k_token + (unsigned long long)(kvh * head_dim);
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) { dot += q_val[d] * __half2float(k_head[d]); }
        dot *= scale;

        float old_m = m;
        m = fmaxf(m, dot);
        float e = expf(dot - m);
        float correction = expf(old_m - m);
        s = s * correction + e;

        const __half *v_token = va_v + block_off + (unsigned long long)(off * kv_heads * head_dim);
        const __half *v_head = v_token + (unsigned long long)(kvh * head_dim);
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction + e * __half2float(v_head[d]);
        }
    }

    __half *out_ptr = out + (unsigned long long)b * num_q_heads * head_dim + (unsigned long long)qh * head_dim;
    float inv_s = 1.0f / s;
    for (int d = 0; d < head_dim; d++) { out_ptr[d] = __float2half(acc[d] * inv_s); }
}
