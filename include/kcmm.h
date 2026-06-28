/**
 * kcmm.h — C API for the KCMM (KV Cache Memory Manager) library.
 *
 * KCMM provides OS-style GPU KV Cache memory management with multi-tier
 * storage (GPU HBM → CPU DRAM → NVMe SSD) and pluggable eviction policies
 * (LRU, LFU, FIFO).
 *
 * All functions are thread-safe.  Pool handles (kcmm_pool_t) are opaque
 * and safe to share across threads.
 *
 * Error handling:
 *   - Functions that return `int` use 0 for success and -1 for failure.
 *   - Retrieve descriptive error messages via kcmm_get_last_error().
 *   - Functions that return pointers return NULL on error.
 *
 * Lifecycle:
 *   kcmm_pool_create()  →  use pool  →  kcmm_pool_destroy()
 */

#ifndef KCMM_H
#define KCMM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Opaque handle
 * --------------------------------------------------------------------------- */

/** Opaque handle to a KCMM pool.  Created by kcmm_pool_create(). */
typedef struct kcmm_pool_t kcmm_pool_t;

/* ---------------------------------------------------------------------------
 * Enumerations
 * --------------------------------------------------------------------------- */

/** Eviction hint types.  Influence how the eviction policy treats a sequence. */
typedef enum {
    KCMM_HINT_MULTI_TURN     = 1,  /**< Multi-turn conversation; protect from eviction. */
    KCMM_HINT_NEAR_END       = 2,  /**< Near end-of-life; preferred victim. */
    KCMM_HINT_SYSTEM_PROMPT  = 3,  /**< System prompt tokens; high cache value. */
    KCMM_HINT_HIGH_PRIORITY  = 4,  /**< SLO-critical request; strong protection. */
    KCMM_HINT_LOW_PRIORITY   = 5,  /**< Background batch; prefer eviction. */
    KCMM_HINT_ATTENTION_SINK = 6,  /**< Attention sink tokens; high sharing value. */
    KCMM_HINT_HEAVY_HITTER   = 7,  /**< Heavy-hitter attention tokens; protect. */
    KCMM_HINT_EVICTABLE      = 8,  /**< Can be discarded without restore. */
} kcmm_hint_t;

/** Block-granularity protection levels. */
typedef enum {
    KCMM_PROTECT_NEVER_EVICT = 0,  /**< Never evict this block. */
    KCMM_PROTECT_PREFERRED   = 1,  /**< Protect from eviction if possible. */
    KCMM_EVICT_PREFERRED     = 2,  /**< Prefer this block for eviction. */
} kcmm_protection_t;

/** Where a block's data currently resides in the storage hierarchy. */
typedef enum {
    KCMM_LOC_GPU_RESIDENT  = 0,  /**< Block is resident in GPU HBM. */
    KCMM_LOC_CPU_RESIDENT  = 1,  /**< Block data is in CPU DRAM swap buffer. */
    KCMM_LOC_NVME_RESIDENT = 2,  /**< Block data is on NVMe SSD. */
    KCMM_LOC_EVICTING      = 3,  /**< Block is being evicted (transfer in flight). */
    KCMM_LOC_RESTORING     = 4,  /**< Block is being restored (transfer in flight). */
} kcmm_block_location_t;

/* ---------------------------------------------------------------------------
 * Configuration
 * --------------------------------------------------------------------------- */

/**
 * Pool creation configuration.
 *
 * All fields have sensible defaults (zero-initialize to use defaults for
 * most fields).  The model-specific fields (num_layers, kv_heads, head_dim,
 * max_batch, max_seq_len) must be set to match the inference model.
 */
typedef struct {
    /** Tokens per block.  Default: 16. */
    size_t block_size;

    /** Maximum number of blocks in the pool.  Default: 16384. */
    size_t max_blocks;

    /**
     * Path to the CPU swap buffer file (typically in /dev/shm).
     * Use empty string "" for the default "/dev/shm/kcmm_swap".
     * Buffer limited to 256 bytes including null terminator.
     */
    char cpu_cache_path[256];

    /** Enable multi-tier storage (GPU→CPU).  Default: 1 (true). */
    int32_t tiering;

    /**
     * Eviction policy.
     *   0 = LRU (Least Recently Used)
     *   1 = LFU (Least Frequently Used)
     *   2 = FIFO (First In, First Out)
     * Default: 0 (LRU).
     */
    int32_t eviction_policy;

    /** Look-ahead blocks to prefetch per active sequence.  Default: 4. */
    size_t prefetch_window;

    /** Maximum blocks per eviction/restore batch.  Default: 64. */
    size_t max_batch_blocks;

    /** GPU device ordinal.  Default: 0. */
    size_t device_ordinal;

    /* --- Model-specific (REQUIRED) --- */

    /** Number of transformer layers. */
    size_t num_layers;

    /** Number of KV attention heads. */
    size_t kv_heads;

    /** Dimension of each attention head. */
    size_t head_dim;

    /** Maximum batch size. */
    size_t max_batch;

    /** Maximum sequence length in tokens. */
    size_t max_seq_len;

    /** Low watermark threshold for proactive background eviction. Default: 0.2. */
    float low_watermark_threshold;

    /** Background eviction check interval in milliseconds. Default: 100. */
    uint64_t background_evict_interval_ms;

    /** Number of attention sink blocks protected by sink-window policy. Default: 1. */
    size_t attention_sink_blocks;

    /** Number of recent blocks protected by sink-window policy. Default: 4. */
    size_t recent_window_blocks;
} kcmm_config_t;

/* ---------------------------------------------------------------------------
 * Statistics structures
 * --------------------------------------------------------------------------- */

/** UFS-compatible fragmentation metrics snapshot. */
typedef struct {
    double ifr;               /**< Internal fragmentation ratio. */
    double pme;               /**< Physical memory efficiency (1.0 = optimal). */
    double bu;                /**< Block utilization ratio. */
    double rfi;               /**< Runtime fragmentation index. */
    uint64_t gpu_blocks;      /**< Blocks currently in GPU. */
    uint64_t cpu_blocks;      /**< Blocks currently in CPU swap. */
    uint64_t nvme_blocks;     /**< Blocks currently on NVMe. */
    uint64_t eviction_count;  /**< Total eviction operations. */
    uint64_t restoration_count; /**< Total restoration operations. */
} kcmm_metrics_t;

/** Runtime pool statistics. */
typedef struct {
    uint32_t blocks_in_use;         /**< Logical blocks currently in use. */
    uint32_t total_blocks;          /**< Total block indices (incl. recycled). */
    uint32_t total_physical_blocks; /**< Total physical blocks across all layers. */
    uint32_t free_physical_blocks;  /**< Free physical blocks available. */
    uint32_t active_sequences;      /**< Number of registered sequences. */
    uint32_t num_layers;            /**< Number of transformer layers. */
    uint32_t blocks_per_superblock; /**< Blocks per superblock. */
    uint32_t superblock_count;      /**< Number of superblocks allocated. */
    uint32_t block_size;            /**< Block size in tokens. */
    uint32_t max_blocks_per_seq;    /**< Maximum blocks per sequence. */
    uint32_t block_bytes;           /**< Byte size of each block. */
    int32_t  tiering_enabled;       /**< 1 if tiering is active, 0 otherwise. */
    int32_t  sharing_enabled;       /**< 1 if prefix sharing is active, 0 otherwise. */
    float    physical_idle_ratio;   /**< Fraction of superblock capacity idle. */
} kcmm_pool_stats_t;

/* ===========================================================================
 * Pool Lifecycle
 * =========================================================================== */

/**
 * Create a new KCMM pool.
 *
 * @param config  Pool configuration; must not be NULL.  Model-specific fields
 *                (num_layers, kv_heads, head_dim, max_batch, max_seq_len) are
 *                required.  Other fields may be zero for defaults.
 * @return Opaque pool handle on success, NULL on error.
 */
kcmm_pool_t *kcmm_pool_create(const kcmm_config_t *config);

/**
 * Destroy a KCMM pool and release all GPU/CPU resources.
 *
 * Blocks until all in-flight CUDA operations complete, unmaps VA regions,
 * and releases physical memory.  The handle is invalid after this call.
 *
 * @param pool  Pool handle from kcmm_pool_create().  NULL is a no-op.
 */
void kcmm_pool_destroy(kcmm_pool_t *pool);

/* ===========================================================================
 * Error Handling
 * =========================================================================== */

/**
 * Retrieve the last error message for a pool.
 *
 * @param pool     Pool handle.
 * @param buf      Output buffer for the null-terminated message.
 * @param max_len  Size of `buf` in bytes.
 * @return Number of bytes written (excluding null terminator), or 0 if no
 *         error has occurred or the buffer is too small.
 */
size_t kcmm_get_last_error(kcmm_pool_t *pool, char *buf, size_t max_len);

/**
 * Clear the last error for a pool.
 *
 * @param pool  Pool handle.
 */
void kcmm_clear_error(kcmm_pool_t *pool);

/* ===========================================================================
 * Block Allocation
 * =========================================================================== */

/**
 * Allocate `num_blocks` blocks.
 *
 * @param pool        Pool handle.
 * @param num_blocks  Number of blocks to allocate.
 * @param out_blocks  Pre-allocated output buffer (≥ num_blocks × sizeof(u32)).
 * @return 0 on success, -1 on error.
 */
int kcmm_alloc_blocks(kcmm_pool_t *pool, uint32_t num_blocks, uint32_t *out_blocks);

/**
 * Free blocks, returning them to the per-layer physical allocators.
 *
 * @param pool    Pool handle.
 * @param blocks  Array of block indices to free.
 * @param num     Number of entries in `blocks`.
 * @return 0 on success.  Freeing zero blocks is a successful no-op.
 */
int kcmm_free_blocks(kcmm_pool_t *pool, const uint32_t *blocks, uint32_t num);

/* ===========================================================================
 * Sequence Management
 * =========================================================================== */

/**
 * Register a new sequence with its block table.
 *
 * @param pool         Pool handle.
 * @param block_table  Array of block indices allocated for this sequence.
 * @param num_blocks   Number of entries in `block_table`.
 * @param out_seq_idx  Output: assigned sequence index.
 * @return 0 on success, -1 on error.
 */
int kcmm_register_sequence(kcmm_pool_t *pool,
                           const uint32_t *block_table, uint32_t num_blocks,
                           uint32_t *out_seq_idx);

/**
 * Unregister a sequence and free its blocks.
 *
 * Safe to call with an out-of-bounds index (no-op with return 0).
 *
 * @param pool     Pool handle.
 * @param seq_idx  Sequence index to unregister.
 * @return 0 on success.
 */
int kcmm_unregister_sequence(kcmm_pool_t *pool, uint32_t seq_idx);

/**
 * Mark a sequence as recently accessed (hot).
 *
 * Updates the last-access timestamp and sets `is_active = true`.
 * Call this each time a sequence is scheduled for decoding.
 *
 * @param pool     Pool handle.
 * @param seq_idx  Sequence index.
 */
void kcmm_touch(kcmm_pool_t *pool, uint32_t seq_idx);

/**
 * Mark a sequence as cool (eligible for eviction).
 *
 * Sets `is_active = false`.  The sequence's blocks become eviction candidates
 * when memory pressure triggers the tiering engine.
 *
 * @param pool     Pool handle.
 * @param seq_idx  Sequence index.
 */
void kcmm_cool(kcmm_pool_t *pool, uint32_t seq_idx);

/**
 * Update the sequence length in tokens.
 *
 * @param pool     Pool handle.
 * @param seq_idx  Sequence index.
 * @param len      New sequence length in tokens.
 */
void kcmm_update_seq_len(kcmm_pool_t *pool, uint32_t seq_idx, uint32_t len);

/**
 * Get the current sequence length in tokens.
 *
 * @param pool     Pool handle.
 * @param seq_idx  Sequence index.
 * @return Sequence length, or 0 if the index is invalid.
 */
uint32_t kcmm_get_seq_len(kcmm_pool_t *pool, uint32_t seq_idx);

/**
 * Append a block to an existing sequence's block table.
 *
 * @param pool      Pool handle.
 * @param seq_idx   Sequence index.
 * @param block_idx Block index to append.
 * @return 0 on success, -1 if seq_idx is out of bounds.
 */
int kcmm_append_block_to_sequence(kcmm_pool_t *pool,
                                  uint32_t seq_idx, uint32_t block_idx);

/**
 * Get the block table for a sequence.
 *
 * @param pool       Pool handle.
 * @param seq_idx    Sequence index.
 * @param out_table  Pre-allocated output buffer (≥ max_blocks × sizeof(u32)).
 * @param max_blocks Capacity of `out_table`.
 * @param out_count  Output: actual number of blocks written.
 * @return 0 on success, -1 if seq_idx is invalid.
 */
int kcmm_get_block_table(kcmm_pool_t *pool, uint32_t seq_idx,
                         uint32_t *out_table, uint32_t max_blocks,
                         uint32_t *out_count);

/* ===========================================================================
 * Block Queries
 * =========================================================================== */

/**
 * Get the virtual address (byte) offset for a given block index.
 *
 * @param pool      Pool handle.
 * @param block_idx Block index.
 * @return VA byte offset, or 0 if the block index is invalid.
 */
uint64_t kcmm_get_block_va_offset(kcmm_pool_t *pool, uint32_t block_idx);

/**
 * Get VA offsets for all blocks in f16-element units.
 *
 * Each offset is the byte offset divided by sizeof(f16) — the format
 * expected by paged-attention CUDA kernels.  Inactive blocks yield 0.
 *
 * @param pool        Pool handle.
 * @param out_offsets Pre-allocated output buffer (≥ max_blocks × sizeof(u64)).
 * @param max_blocks  Capacity of `out_offsets`.
 * @param out_count   Output: actual number of offsets written.
 * @return 0 on success, -1 on error.
 */
int kcmm_get_all_block_offsets_f16(kcmm_pool_t *pool,
                                   uint64_t *out_offsets, uint32_t max_blocks,
                                   uint32_t *out_count);

/**
 * Get the storage location of a block.
 *
 * @param pool         Pool handle.
 * @param block_idx    Block index.
 * @param out_location Output: block location.
 * @return 0 on success, -1 if block_idx is invalid.
 */
int kcmm_get_block_location(kcmm_pool_t *pool, uint32_t block_idx,
                            kcmm_block_location_t *out_location);

/**
 * Get VA byte offsets for all blocks belonging to a sequence.
 *
 * @param pool        Pool handle.
 * @param seq_idx     Sequence index.
 * @param out_offsets Pre-allocated output buffer (≥ max_blocks × sizeof(u64)).
 * @param max_blocks  Capacity of `out_offsets`.
 * @param out_count   Output: actual number of offsets written.
 * @return 0 on success, -1 if seq_idx is invalid.
 */
int kcmm_get_block_table_va_offsets(kcmm_pool_t *pool, uint32_t seq_idx,
                                    uint64_t *out_offsets, uint32_t max_blocks,
                                    uint32_t *out_count);

/* ===========================================================================
 * Virtual Address Accessors
 * =========================================================================== */

/**
 * Get the K-cache virtual address base for a given layer.
 *
 * @param pool   Pool handle.
 * @param layer  Layer index (0 .. num_layers-1).
 * @return VA base in bytes, or 0 if the layer index is out of bounds.
 */
uint64_t kcmm_get_va_k(kcmm_pool_t *pool, uint32_t layer);

/**
 * Get the V-cache virtual address base for a given layer.
 *
 * @param pool   Pool handle.
 * @param layer  Layer index (0 .. num_layers-1).
 * @return VA base in bytes, or 0 if the layer index is out of bounds.
 */
uint64_t kcmm_get_va_v(kcmm_pool_t *pool, uint32_t layer);

/* ===========================================================================
 * KV Cache Write
 * =========================================================================== */

/**
 * Write one step of KV data for a batch of sequences.
 *
 * @param pool         Pool handle.
 * @param layer_idx    Transformer layer index (0 .. num_layers-1).
 * @param seq_indices  Array of sequence indices (length = batch).
 * @param positions    Array of per-sequence token positions (length = batch).
 * @param batch        Number of sequences in the batch.
 * @param k_src_ptr    GPU virtual address of K source data
 *                     (layout: [batch, kv_heads * head_dim] F16).
 * @param v_src_ptr    GPU virtual address of V source data
 *                     (layout: [batch, kv_heads * head_dim] F16).
 * @return 0 on success, -1 on error.
 */
int kcmm_append_kv_step(kcmm_pool_t *pool, uint32_t layer_idx,
                        const uint32_t *seq_indices,
                        const uint32_t *positions, uint32_t batch,
                        uint64_t k_src_ptr, uint64_t v_src_ptr);

/**
 * Write one step of KV data using vLLM-style physical slot ids.
 *
 * Non-negative slots are interpreted as
 * `slot = block_idx * block_size + offset_in_block`; negative slots are
 * padding and are skipped. K and V source tensors are FP16 rows laid out as
 * [batch, kv_heads * head_dim].
 *
 * @param pool         Pool handle.
 * @param layer_idx    Transformer layer index.
 * @param slot_mapping CPU array of vLLM physical slots, length = batch.
 * @param batch        Number of source rows.
 * @param k_src_ptr    GPU virtual address of K source data.
 * @param v_src_ptr    GPU virtual address of V source data.
 * @return 0 on successful enqueue, -1 on error.
 */
int kcmm_append_kv_slots(kcmm_pool_t *pool, uint32_t layer_idx,
                         const int64_t *slot_mapping, uint32_t batch,
                         uint64_t k_src_ptr, uint64_t v_src_ptr);

/**
 * Write one step of KV data using vLLM-style physical slot ids on a caller
 * CUDA stream.
 *
 * This has the same data contract as `kcmm_append_kv_slots`, but enqueues D2D
 * copies on `stream_ptr` and returns without synchronizing. The caller is
 * responsible for passing the current framework stream and preserving source
 * tensor lifetimes until the stream reaches this work.
 *
 * @param stream_ptr Raw CUDA stream handle, or 0 for the legacy default stream.
 * @return 0 on successful enqueue, -1 on error.
 */
int kcmm_append_kv_slots_on_stream(kcmm_pool_t *pool, uint32_t layer_idx,
                                   const int64_t *slot_mapping, uint32_t batch,
                                   uint64_t k_src_ptr, uint64_t v_src_ptr,
                                   uint64_t stream_ptr);

/**
 * Launch KCMM paged-attention decode for vLLM-owned query/output tensors.
 *
 * All pointer arguments are CUDA virtual addresses. `query_ptr` and `out_ptr`
 * are FP16 tensors shaped [batch, num_q_heads, head_dim]. `block_tables_ptr`
 * and `seq_lens_ptr` are int32 tensors. `block_offsets_f16_ptr` is an
 * int64/u64 table indexed by block_id, where each value is the KCMM K/V VA
 * byte offset divided by sizeof(f16).
 *
 * @param pool                  Pool handle.
 * @param layer_idx             Transformer layer index.
 * @param query_ptr             CUDA VA of query tensor.
 * @param out_ptr               CUDA VA of output tensor to fill.
 * @param block_tables_ptr      CUDA VA of int32 block table tensor.
 * @param seq_lens_ptr          CUDA VA of int32 sequence lengths tensor.
 * @param block_offsets_f16_ptr CUDA VA of int64/u64 block offset table.
 * @param batch                 Number of sequences.
 * @param num_q_heads           Number of query heads.
 * @param kv_heads              Number of KV heads.
 * @param head_dim              Attention head dimension. Current kernel max: 256.
 * @param block_size            Tokens per KV block.
 * @param max_blocks_per_seq    Columns in `block_tables`.
 * @param scale                 Attention scale.
 * @return 0 on success, -1 on error.
 */
int kcmm_paged_attn_decode_f16(kcmm_pool_t *pool, uint32_t layer_idx,
                               uint64_t query_ptr, uint64_t out_ptr,
                               uint64_t block_tables_ptr, uint64_t seq_lens_ptr,
                               uint64_t block_offsets_f16_ptr,
                               uint32_t batch, uint32_t num_q_heads,
                               uint32_t kv_heads, uint32_t head_dim,
                               uint32_t block_size, uint32_t max_blocks_per_seq,
                               float scale);

/**
 * Launch KCMM paged-attention decode on a caller-owned CUDA stream.
 *
 * This has the same tensor contract as `kcmm_paged_attn_decode_f16`, but it
 * enqueues the kernel on `stream_ptr` and returns without synchronizing. The
 * caller is responsible for passing the current framework stream and preserving
 * tensor lifetimes until the stream reaches this work.
 *
 * @param stream_ptr Raw CUDA stream handle, or 0 for the legacy default stream.
 * @return 0 on successful enqueue, -1 on error.
 */
int kcmm_paged_attn_decode_f16_on_stream(
    kcmm_pool_t *pool, uint32_t layer_idx, uint64_t query_ptr, uint64_t out_ptr,
    uint64_t block_tables_ptr, uint64_t seq_lens_ptr,
    uint64_t block_offsets_f16_ptr, uint32_t batch, uint32_t num_q_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t block_size,
    uint32_t max_blocks_per_seq, float scale, uint64_t stream_ptr);

/* ===========================================================================
 * Tiering Operations
 * =========================================================================== */

/**
 * Trigger eviction of up to `count` blocks from GPU to CPU memory.
 *
 * The tiering engine selects victim blocks according to the current
 * eviction policy.  Data is copied asynchronously on the eviction CUDA stream.
 * Call kcmm_synchronize() to wait for completion.
 *
 * @param pool   Pool handle.
 * @param count  Maximum number of blocks to evict.
 * @return Actual number of blocks evicted, or 0 if tiering is disabled or
 *         no eligible blocks exist.
 */
uint32_t kcmm_evict_blocks(kcmm_pool_t *pool, uint32_t count);

/**
 * Restore a single evicted block from CPU memory back to GPU.
 *
 * If the block is already GpuResident, returns the current VA offset immediately.
 * If the block is CpuResident, allocates a new GPU physical block and copies
 * data from CPU to GPU.
 *
 * @param pool      Pool handle.
 * @param block_idx Block index to restore.
 * @return GPU VA byte offset on success, 0 on error.
 */
uint64_t kcmm_restore_evicted_block(kcmm_pool_t *pool, uint32_t block_idx);

/**
 * Restore multiple evicted blocks from CPU memory back to GPU.
 *
 * Blocks already in GPU are silently skipped.  For batches of ≥4 blocks,
 * the scatter-kernel path is used for better throughput.
 *
 * @param pool         Pool handle.
 * @param block_indices Array of block indices to restore.
 * @param count        Number of entries in `block_indices`.
 * @return 0 on success, -1 on error.
 */
int kcmm_restore_evicted_blocks(kcmm_pool_t *pool,
                                const uint32_t *block_indices, uint32_t count);

/**
 * Check whether a block is currently resident in GPU memory.
 *
 * @param pool      Pool handle.
 * @param block_idx Block index.
 * @return 1 if GpuResident, 0 otherwise.
 */
int kcmm_is_gpu_resident(kcmm_pool_t *pool, uint32_t block_idx);

/* ===========================================================================
 * Metrics and Statistics
 * =========================================================================== */

/**
 * Collect a UFS-compatible metrics snapshot.
 *
 * @param pool Pool handle.
 * @param out  Output: metrics snapshot.
 * @return 0 on success, -1 if arguments are invalid.
 */
int kcmm_get_metrics(kcmm_pool_t *pool, kcmm_metrics_t *out);

/**
 * Get runtime pool statistics.
 *
 * @param pool Pool handle.
 * @param out  Output: pool statistics.
 * @return 0 on success, -1 if arguments are invalid.
 */
int kcmm_get_pool_stats(kcmm_pool_t *pool, kcmm_pool_stats_t *out);

/** @return Number of logical blocks currently in use, or 0 if pool is NULL. */
uint32_t kcmm_blocks_in_use(kcmm_pool_t *pool);

/** @return Total number of block indices (including recycled slots). */
uint32_t kcmm_total_blocks(kcmm_pool_t *pool);

/** @return Number of free physical blocks across all layers. */
uint32_t kcmm_free_physical_blocks(kcmm_pool_t *pool);

/** @return 1 if free blocks are available, 0 otherwise. */
int kcmm_has_free_blocks(kcmm_pool_t *pool);

/** @return Number of active (registered) sequences. */
uint32_t kcmm_active_sequences(kcmm_pool_t *pool);

/* ===========================================================================
 * Policy Configuration
 * =========================================================================== */

/**
 * Set the eviction policy at runtime.
 *
 * @param pool    Pool handle.
 * @param policy  Policy name: "lru", "lfu", or "fifo" (null-terminated).
 * @return 0 on success, -1 if tiering is disabled or the policy name is
 *         invalid.
 */
int kcmm_set_eviction_policy(kcmm_pool_t *pool, const char *policy);

/**
 * Get the current eviction policy name.
 *
 * @param pool       Pool handle.
 * @param out_policy Output buffer for null-terminated policy name.
 * @param max_len    Size of `out_policy` in bytes.
 * @return Bytes written (excl. null), or 0 if tiering is disabled.
 */
uint32_t kcmm_get_eviction_policy(kcmm_pool_t *pool, char *out_policy,
                                  uint32_t max_len);

/**
 * Check whether tiering is enabled.
 *
 * @param pool  Pool handle.
 * @return 1 if tiering is enabled, 0 otherwise.
 */
int kcmm_is_tiering_enabled(kcmm_pool_t *pool);

/* ===========================================================================
 * Hint API
 * =========================================================================== */

/**
 * Apply a hint to a sequence, influencing its eviction priority.
 *
 * Protection hints (MULTI_TURN, SYSTEM_PROMPT, HIGH_PRIORITY, ATTENTION_SINK,
 * HEAVY_HITTER) update the access timestamp to make blocks appear recently used,
 * protecting them from LRU-based eviction.
 *
 * Eviction hints (NEAR_END, LOW_PRIORITY, EVICTABLE) mark the sequence as
 * inactive (cool), making its blocks preferred eviction candidates.
 *
 * @param pool     Pool handle.
 * @param seq_idx  Sequence index.
 * @param hint     Hint type to apply.
 * @return 0 on success, -1 if seq_idx is invalid.
 */
int kcmm_hint(kcmm_pool_t *pool, uint32_t seq_idx, kcmm_hint_t hint);

/**
 * Set protection level for specific blocks within a sequence.
 *
 * This is a more precise version of kcmm_hint() operating at the individual
 * block level rather than the whole sequence.
 *
 * PROTECT_NEVER_EVICT / PROTECT_PREFERRED: refresh access timestamp (protect).
 * EVICT_PREFERRED: do not refresh timestamp (makes them appear older).
 *
 * @param pool       Pool handle.
 * @param seq_idx    Sequence index (currently informational).
 * @param block_ids  Array of block indices to protect.
 * @param num_blocks Number of entries in `block_ids`.
 * @param level      Protection level to apply.
 * @return 0 on success, -1 on error.
 */
int kcmm_protect(kcmm_pool_t *pool, uint32_t seq_idx,
                 const uint32_t *block_ids, uint32_t num_blocks,
                 kcmm_protection_t level);

/* ===========================================================================
 * Prefix Sharing
 * =========================================================================== */

/**
 * Share prefix blocks from a source sequence to a destination sequence.
 *
 * NOTE: Prefix sharing is a step-4 feature.  In the current implementation,
 * this function returns 0 but all output blocks are set to 0 (no sharing).
 *
 * @param pool       Pool handle.
 * @param src_seq    Source sequence index (owner of prefix blocks).
 * @param dst_seq    Destination sequence index (receiver).
 * @param num_blocks Number of prefix blocks to attempt to share.
 * @param out_blocks Output buffer for shared block indices
 *                   (≥ num_blocks × sizeof(u32)).
 * @return 0 on success, -1 if sharing is unavailable.
 */
int kcmm_share_prefix(kcmm_pool_t *pool, uint32_t src_seq, uint32_t dst_seq,
                      uint32_t num_blocks, uint32_t *out_blocks);

/* ===========================================================================
 * Utilities
 * =========================================================================== */

/**
 * Check whether the free block ratio is below a low watermark.
 *
 * When this returns true, the caller should trigger eviction
 * (via kcmm_evict_blocks) to free up GPU blocks before OOM.
 *
 * @param pool       Pool handle.
 * @param threshold  Free ratio threshold (e.g. 0.2 = 20% free).
 * @return 1 if below the threshold, 0 otherwise.
 */
int kcmm_below_low_watermark(kcmm_pool_t *pool, float threshold);

/**
 * Synchronize all CUDA streams (evict, restore, prefetch).
 *
 * Blocks the calling CPU thread until all pending GPU operations complete.
 *
 * @param pool  Pool handle.
 * @return 0 on success, -1 on error.
 */
int kcmm_synchronize(kcmm_pool_t *pool);

/* ===========================================================================
 * Pool Accessors (read-only)
 * =========================================================================== */

/** @return Block size in tokens. */
uint32_t kcmm_get_block_size(kcmm_pool_t *pool);

/** @return Maximum blocks per sequence. */
uint32_t kcmm_get_max_blocks_per_seq(kcmm_pool_t *pool);

/** @return Byte size of each block. */
uint32_t kcmm_get_block_bytes(kcmm_pool_t *pool);

/** @return Number of transformer layers. */
uint32_t kcmm_get_num_layers(kcmm_pool_t *pool);

/** @return Maximum batch size. */
uint32_t kcmm_get_max_batch(kcmm_pool_t *pool);

/** @return Maximum sequence length in tokens. */
uint32_t kcmm_get_max_seq_len(kcmm_pool_t *pool);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* KCMM_H */
