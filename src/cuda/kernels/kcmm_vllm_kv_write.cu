#include "cuda_fp16.h"

// vLLM-facing KCMM KV write kernel.
//
// The host-side slot writer takes a CPU int64 slot array and issues one D2D
// copy per row. This kernel consumes the original device-resident vLLM
// slot_mapping tensor directly and copies contiguous K/V rows into KCMM VA.
extern "C" __global__ void kcmm_vllm_kv_write_slots_f16(
    unsigned long long va_k_ptr,
    unsigned long long va_v_ptr,
    unsigned long long slot_mapping_ptr,
    unsigned long long block_offsets_f16_ptr,
    unsigned long long k_src_ptr,
    unsigned long long v_src_ptr,
    unsigned long long status_ptr,
    int total_elements,
    int step_elements,
    int block_size,
    int block_offsets_f16_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const long long *slot_mapping =
        reinterpret_cast<const long long *>(slot_mapping_ptr);
    const unsigned long long *block_offsets_f16 =
        reinterpret_cast<const unsigned long long *>(block_offsets_f16_ptr);
    const __half *k_src = reinterpret_cast<const __half *>(k_src_ptr);
    const __half *v_src = reinterpret_cast<const __half *>(v_src_ptr);
    __half *va_k = reinterpret_cast<__half *>(va_k_ptr);
    __half *va_v = reinterpret_cast<__half *>(va_v_ptr);
    int *status = reinterpret_cast<int *>(status_ptr);

    int row = idx / step_elements;
    int col = idx - row * step_elements;
    long long slot = slot_mapping[row];
    if (slot < 0) return;

    long long block_id = slot / block_size;
    if (block_id < 0 || block_id >= block_offsets_f16_len) {
        if (status != 0) {
            atomicCAS(status, 0, 1);
        }
        return;
    }

    int offset_in_block = static_cast<int>(slot % block_size);
    unsigned long long block_offset = block_offsets_f16[block_id];
    unsigned long long token_offset =
        static_cast<unsigned long long>(offset_in_block) * step_elements;
    unsigned long long dst_idx = block_offset + token_offset + col;
    unsigned long long src_idx =
        static_cast<unsigned long long>(row) * step_elements + col;

    va_k[dst_idx] = k_src[src_idx];
    va_v[dst_idx] = v_src[src_idx];
}
