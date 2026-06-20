#include "cuda_fp16.h"

// vLLM-facing KCMM paged attention decode kernel.
//
// All pointer arguments are passed as integer CUDA virtual addresses so this
// kernel can be launched from the C ABI without constructing Rust CudaSlice
// wrappers around PyTorch-owned tensors.
extern "C" __global__ void kcmm_vllm_paged_attn_decode_f16(
    unsigned long long q_ptr,
    unsigned long long va_k_ptr,
    unsigned long long va_v_ptr,
    unsigned long long block_tables_ptr,
    unsigned long long seq_lens_ptr,
    unsigned long long block_offsets_f16_ptr,
    unsigned long long out_ptr,
    int total_heads,
    int packed_heads,
    int head_dim,
    int packed_blocks,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_heads) return;

    const __half *q = reinterpret_cast<const __half *>(q_ptr);
    const __half *va_k = reinterpret_cast<const __half *>(va_k_ptr);
    const __half *va_v = reinterpret_cast<const __half *>(va_v_ptr);
    const int *block_tables = reinterpret_cast<const int *>(block_tables_ptr);
    const int *seq_lens = reinterpret_cast<const int *>(seq_lens_ptr);
    const unsigned long long *block_offsets_f16 =
        reinterpret_cast<const unsigned long long *>(block_offsets_f16_ptr);
    __half *out = reinterpret_cast<__half *>(out_ptr);
    int num_q_heads = packed_heads & 0xFFFF;
    int kv_heads = (packed_heads >> 16) & 0xFFFF;
    int block_size = packed_blocks & 0xFFFF;
    int max_blocks_per_seq = (packed_blocks >> 16) & 0xFFFF;

    int b = idx / num_q_heads;
    int qh = idx % num_q_heads;
    int kvh = qh * kv_heads / num_q_heads;
    int seq_len = seq_lens[b];
    if (seq_len <= 0) return;

    float m = -1e30f;
    float s = 0.0f;
    float acc[64];
    float q_val[64];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    const __half *q_head =
        q + (unsigned long long)b * num_q_heads * head_dim
          + (unsigned long long)qh * head_dim;
    for (int d = 0; d < head_dim; d++) {
        q_val[d] = __half2float(q_head[d]);
    }

    #pragma unroll 1
    for (int pos = 0; pos < seq_len; pos++) {
        int logical_block = pos / block_size;
        int offset_in_block = pos % block_size;
        int block_id = block_tables[
            (unsigned long long)b * max_blocks_per_seq + logical_block
        ];
        unsigned long long block_offset = block_offsets_f16[block_id];
        unsigned long long token_offset =
            (unsigned long long)offset_in_block * kv_heads * head_dim;

        const __half *k_head =
            va_k + block_offset + token_offset
                 + (unsigned long long)kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_val[d] * __half2float(k_head[d]);
        }
        dot *= scale;

        float old_m = m;
        m = fmaxf(m, dot);
        float e = expf(dot - m);
        float correction = expf(old_m - m);
        s = s * correction + e;

        const __half *v_head =
            va_v + block_offset + token_offset
                 + (unsigned long long)kvh * head_dim;
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction + e * __half2float(v_head[d]);
        }
    }

    __half *out_head =
        out + (unsigned long long)b * num_q_heads * head_dim
            + (unsigned long long)qh * head_dim;
    float inv_s = 1.0f / s;
    for (int d = 0; d < head_dim; d++) {
        out_head[d] = __float2half(acc[d] * inv_s);
    }
}
