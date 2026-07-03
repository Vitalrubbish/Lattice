/**
 * kcmm_c_integration.c — C integration test for libkcmm.so
 *
 * Exercises the full KCMM C API surface.  Requires a CUDA-capable GPU.
 *
 * Build (from project root):
 *   cargo build --release --features kcmm
 *   gcc -Wall -Wextra -O2 -o kcmm_c_test examples/kcmm_c_integration.c \
 *       -Iinclude -Ltarget/release -lkcmm -lcuda \
 *       -Wl,-rpath,target/release
 *
 * Run:
 *   LD_LIBRARY_PATH=target/release ./kcmm_c_test
 *
 * Expected output on success:
 *   [PASS] pool_create
 *   [PASS] pool_destroy_null_safety
 *   [PASS] alloc_blocks
 *   ...
 *   All 20 tests passed.
 */

#include "kcmm.h"
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#if __SIZEOF_SIZE_T__ == 8
_Static_assert(sizeof(kcmm_config_t) == 376, "kcmm_config_t ABI size drift");
_Static_assert(offsetof(kcmm_config_t, max_seq_len) == 336, "max_seq_len offset drift");
_Static_assert(offsetof(kcmm_config_t, low_watermark_threshold) == 344,
               "low_watermark_threshold offset drift");
_Static_assert(offsetof(kcmm_config_t, background_evict_interval_ms) == 352,
               "background_evict_interval_ms offset drift");
_Static_assert(offsetof(kcmm_config_t, attention_sink_blocks) == 360,
               "attention_sink_blocks offset drift");
_Static_assert(offsetof(kcmm_config_t, recent_window_blocks) == 368,
               "recent_window_blocks offset drift");
#endif

/* ---------------------------------------------------------------------------
 * Minimal test harness
 * --------------------------------------------------------------------------- */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                             \
    static void test_##name(void);                                             \
    static void test_##name(void)

#define RUN_TEST(name)                                                         \
    do {                                                                       \
        printf("  %-45s ", #name);                                           \
        fflush(stdout);                                                        \
        test_##name();                                                         \
    } while (0)

static void pass(void) {
    printf("[PASS]\n");
    tests_passed++;
}

static void fail(const char *msg) {
    printf("[FAIL] %s\n", msg);
    tests_failed++;
}

static void check(int cond, const char *msg) {
    if (cond) pass(); else fail(msg);
}

/* ---------------------------------------------------------------------------
 * Test data
 * --------------------------------------------------------------------------- */

static kcmm_pool_t *g_pool = NULL;

static kcmm_config_t default_config(void) {
    kcmm_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    /* Model parameters for TinyLlama-like setup */
    cfg.num_layers = 22;
    cfg.kv_heads   = 4;
    cfg.head_dim   = 64;
    cfg.max_batch  = 8;
    cfg.max_seq_len = 128;
    /* Use defaults for the rest */
    cfg.block_size   = 16;
    cfg.max_blocks   = 1024;
    cfg.tiering      = 0;  /* tiering off for simplicity */
    cfg.device_ordinal = 0;
    cfg.low_watermark_threshold = 0.2f;
    cfg.background_evict_interval_ms = 100;
    cfg.attention_sink_blocks = 1;
    cfg.recent_window_blocks = 4;
    return cfg;
}

/* ===========================================================================
 * Tests
 * =========================================================================== */

/* 1. Pool creation */
TEST(pool_create) {
    kcmm_config_t cfg = default_config();
    g_pool = kcmm_pool_create(&cfg);
    check(g_pool != NULL, "kcmm_pool_create returned NULL");
}

/* 2. Null-pointer safety: pool_destroy(NULL) must not crash */
TEST(pool_destroy_null_safety) {
    kcmm_pool_destroy(NULL);
    pass(); /* if we get here, it didn't crash */
}

/* 3. Error handling: get_last_error on a fresh pool */
TEST(get_last_error_empty) {
    char buf[256] = {0};
    size_t n = kcmm_get_last_error(g_pool, buf, sizeof(buf));
    check(n == 0 && buf[0] == '\0', "expected empty error on fresh pool");
}

/* 4. Error handling: get_last_error with null buffers */
TEST(get_last_error_null_safety) {
    size_t n = kcmm_get_last_error(g_pool, NULL, 0);
    check(n == 0, "expected 0 for null buffer");

    n = kcmm_get_last_error(NULL, (char*)"dummy", 10);
    check(n == 0, "expected 0 for null pool");
}

/* 5. Allocate blocks */
TEST(alloc_blocks) {
    uint32_t blocks[8];
    int rc = kcmm_alloc_blocks(g_pool, 8, blocks);
    check(rc == 0, "kcmm_alloc_blocks failed");

    /* Verify distinct block indices */
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            if (blocks[i] == blocks[j]) {
                fail("duplicate block index in allocation");
                return;
            }
        }
    }
    pass();
}

/* 6. Block queries: VA offset */
TEST(block_va_offset) {
    uint32_t blocks[4];
    kcmm_alloc_blocks(g_pool, 4, blocks);

    for (int i = 0; i < 4; i++) {
        uint64_t va = kcmm_get_block_va_offset(g_pool, blocks[i]);
        if (va == 0) { fail("va offset is 0 for valid block"); return; }
    }
    /* Invalid block index should return 0 */
    uint64_t va = kcmm_get_block_va_offset(g_pool, 999999);
    if (va != 0) { fail("expected 0 va for invalid block index"); return; }
    pass();
}

/* 7. Sequence registration and management */
TEST(sequence_lifecycle) {
    uint32_t blocks[6];
    kcmm_alloc_blocks(g_pool, 6, blocks);

    uint32_t seq_idx = 0;
    int rc = kcmm_register_sequence(g_pool, blocks, 6, &seq_idx);
    if (rc != 0) { fail("register_sequence failed"); return; }

    /* Update and query sequence length */
    kcmm_update_seq_len(g_pool, seq_idx, 100);
    uint32_t len = kcmm_get_seq_len(g_pool, seq_idx);
    if (len != 100) { fail("get_seq_len returned wrong value"); return; }

    /* Touch (mark hot) */
    kcmm_touch(g_pool, seq_idx);
    /* Cool (mark cold) */
    kcmm_cool(g_pool, seq_idx);

    /* Append block */
    uint32_t extra_blocks[1];
    kcmm_alloc_blocks(g_pool, 1, extra_blocks);
    rc = kcmm_append_block_to_sequence(g_pool, seq_idx, extra_blocks[0]);
    if (rc != 0) { fail("append_block_to_sequence failed"); return; }

    /* Get block table */
    uint32_t table[16];
    uint32_t count = 0;
    rc = kcmm_get_block_table(g_pool, seq_idx, table, 16, &count);
    if (rc != 0 || count != 7) { fail("get_block_table wrong count"); return; }

    /* Validate appended block is in the table */
    int found = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (table[i] == extra_blocks[0]) found = 1;
    }
    if (!found) { fail("appended block not in block table"); return; }

    pass();
}

/* 8. Out-of-bounds safety */
TEST(out_of_bounds_safety) {
    /* These must not crash or assert */
    kcmm_touch(g_pool, 99999);
    kcmm_cool(g_pool, 99999);
    kcmm_update_seq_len(g_pool, 99999, 42);

    uint32_t len = kcmm_get_seq_len(g_pool, 99999);
    if (len != 0) { fail("expected 0 for invalid seq_len"); return; }

    uint64_t va = kcmm_get_block_va_offset(g_pool, 99999);
    if (va != 0) { fail("expected 0 va for invalid block"); return; }

    int rc = kcmm_append_block_to_sequence(g_pool, 99999, 0);
    if (rc != -1) { fail("expected -1 for invalid seq in append"); return; }

    pass();
}

/* 9. Get VA K and V bases */
TEST(va_accessors) {
    uint64_t va_k_0 = kcmm_get_va_k(g_pool, 0);
    uint64_t va_v_0 = kcmm_get_va_v(g_pool, 0);
    if (va_k_0 == 0 || va_v_0 == 0) { fail("VA base is 0 for layer 0"); return; }

    /* Out-of-bounds layer should return 0 */
    uint64_t invalid = kcmm_get_va_k(g_pool, 999);
    if (invalid != 0) { fail("expected 0 for out-of-bounds layer"); return; }

    pass();
}

/* 10. Get all block offsets f16 */
TEST(all_block_offsets_f16) {
    uint64_t offsets[32];
    uint32_t count = 0;
    int rc = kcmm_get_all_block_offsets_f16(g_pool, offsets, 32, &count);
    if (rc != 0) { fail("get_all_block_offsets_f16 failed"); return; }
    if (count == 0) { fail("expected non-zero block count"); return; }

    /* At least one block should have a non-zero offset */
    int has_nonzero = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] != 0) has_nonzero = 1;
    }
    if (!has_nonzero) { fail("all f16 offsets are zero"); return; }

    pass();
}

/* 11. Block location */
TEST(block_location) {
    uint32_t blocks[2];
    kcmm_alloc_blocks(g_pool, 2, blocks);

    kcmm_block_location_t loc;
    int rc = kcmm_get_block_location(g_pool, blocks[0], &loc);
    if (rc != 0) { fail("get_block_location failed"); return; }
    if (loc != KCMM_LOC_GPU_RESIDENT) { fail("new block should be GPU resident"); return; }
    pass();
}

/* 12. Block table VA offsets */
TEST(block_table_va_offsets) {
    uint32_t blocks[3];
    kcmm_alloc_blocks(g_pool, 3, blocks);

    uint32_t seq_idx;
    kcmm_register_sequence(g_pool, blocks, 3, &seq_idx);

    uint64_t offsets[16];
    uint32_t count = 0;
    int rc = kcmm_get_block_table_va_offsets(g_pool, seq_idx, offsets, 16, &count);
    if (rc != 0 || count != 3) { fail("wrong count for block_table_va_offsets"); return; }

    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] == 0) { fail("zero offset in block table"); return; }
    }
    pass();
}

/* 13. Pool statistics after allocations */
TEST(pool_stats) {
    kcmm_pool_stats_t stats;
    int rc = kcmm_get_pool_stats(g_pool, &stats);
    if (rc != 0) { fail("get_pool_stats failed"); return; }

    if (stats.num_layers != 22)        { fail("wrong num_layers"); return; }
    if (stats.block_size != 16)        { fail("wrong block_size"); return; }
    if (stats.max_blocks_per_seq != 8) { fail("wrong max_blocks_per_seq"); return; }
    if (stats.tiering_enabled != 0)    { fail("tiering should be disabled"); return; }
    if (stats.blocks_in_use == 0)      { fail("expected blocks in use"); return; }

    pass();
}

/* 14. Quick stats accessors */
TEST(quick_stats) {
    uint32_t used  = kcmm_blocks_in_use(g_pool);
    uint32_t total = kcmm_total_blocks(g_pool);
    int has_free   = kcmm_has_free_blocks(g_pool);
    uint32_t active = kcmm_active_sequences(g_pool);

    if (used == 0)  { fail("expected non-zero blocks_in_use"); return; }
    if (total == 0) { fail("expected non-zero total_blocks"); return; }

    (void)has_free;
    (void)active;

    pass();
}

/* 15. Metrics */
TEST(metrics) {
    kcmm_metrics_t m;
    int rc = kcmm_get_metrics(g_pool, &m);
    if (rc != 0) { fail("get_metrics failed"); return; }

    if (m.gpu_blocks == 0)      { fail("expected non-zero gpu_blocks"); return; }
    if (m.ifr < 0.0)            { fail("ifr should be >= 0"); return; }
    if (m.bu < 0.0 || m.bu > 1.0) { fail("bu out of range"); return; }

    pass();
}

/* 16. Tiering disabled — operations should be safe no-ops */
TEST(tiering_disabled) {
    int enabled = kcmm_is_tiering_enabled(g_pool);
    if (enabled != 0) { fail("tiering should be disabled"); return; }

    /* evict should return 0 (no tiering) */
    uint32_t evicted = kcmm_evict_blocks(g_pool, 10);
    if (evicted != 0) { fail("expected 0 evictions when tiering disabled"); return; }

    /* restore should return 0 (no tiering) */
    uint64_t va = kcmm_restore_evicted_block(g_pool, 0);
    if (va != 0) { fail("expected 0 va from restore when tiering disabled"); return; }

    /* Policy get/set should fail or return 0 when tiering disabled */
    int rc = kcmm_set_eviction_policy(g_pool, "lru");
    /* Setting policy without tiering is an error */
    if (rc != -1) { fail("expected -1 from set_eviction_policy with tiering off"); return; }

    pass();
}

/* 17. Null-pointer safety for all functions */
TEST(null_safety_all) {
    /* kcmm_pool_create returns NULL on NULL config */
    kcmm_pool_t *p = kcmm_pool_create(NULL);
    if (p != NULL) fail("expected NULL from kcmm_pool_create(NULL)");

    /* Various functions with NULL pool must return safe sentinel values */
    if (kcmm_blocks_in_use(NULL) != 0)        fail("blocks_in_use(NULL) != 0");
    if (kcmm_total_blocks(NULL) != 0)         fail("total_blocks(NULL) != 0");
    if (kcmm_free_physical_blocks(NULL) != 0) fail("free_physical_blocks(NULL) != 0");
    if (kcmm_has_free_blocks(NULL) != 0)      fail("has_free_blocks(NULL) != 0");
    if (kcmm_active_sequences(NULL) != 0)     fail("active_sequences(NULL) != 0");
    if (kcmm_get_block_va_offset(NULL, 0) != 0) fail("get_block_va_offset(NULL) != 0");
    if (kcmm_get_va_k(NULL, 0) != 0)          fail("get_va_k(NULL) != 0");
    if (kcmm_get_va_v(NULL, 0) != 0)          fail("get_va_v(NULL) != 0");
    if (kcmm_get_seq_len(NULL, 0) != 0)       fail("get_seq_len(NULL) != 0");
    if (kcmm_is_tiering_enabled(NULL) != 0)   fail("is_tiering_enabled(NULL) != 0");
    if (kcmm_is_gpu_resident(NULL, 0) != 0)   fail("is_gpu_resident(NULL) != 0");
    if (kcmm_get_block_size(NULL) != 0)       fail("get_block_size(NULL) != 0");
    if (kcmm_get_max_blocks_per_seq(NULL) != 0) fail("max_blocks_per_seq(NULL) != 0");
    if (kcmm_get_block_bytes(NULL) != 0)      fail("get_block_bytes(NULL) != 0");
    if (kcmm_get_num_layers(NULL) != 0)       fail("get_num_layers(NULL) != 0");
    if (kcmm_get_max_batch(NULL) != 0)        fail("get_max_batch(NULL) != 0");
    if (kcmm_get_max_seq_len(NULL) != 0)      fail("get_max_seq_len(NULL) != 0");

    /* Void functions with NULL must not crash */
    kcmm_pool_destroy(NULL);
    kcmm_touch(NULL, 0);
    kcmm_cool(NULL, 0);
    kcmm_update_seq_len(NULL, 0, 0);
    kcmm_clear_error(NULL);

    /* Functions returning int should return -1 for NULL pool */
    if (kcmm_alloc_blocks(NULL, 1, (uint32_t*)1) != -1) fail("alloc_blocks(NULL) != -1");
    if (kcmm_free_blocks(NULL, NULL, 0) != 0) fail("free_blocks(NULL,0,0) != 0");
    if (kcmm_unregister_sequence(NULL, 0) != -1) fail("unregister_sequence(NULL) != -1");
    if (kcmm_get_metrics(NULL, (kcmm_metrics_t*)1) != -1) fail("get_metrics(NULL) != -1");
    if (kcmm_get_pool_stats(NULL, (kcmm_pool_stats_t*)1) != -1) fail("get_pool_stats(NULL) != -1");
    if (kcmm_synchronize(NULL) != -1) fail("synchronize(NULL) != -1");

    pass();
}

/* 18. Hint API */
TEST(hint_api) {
    uint32_t blocks[4];
    kcmm_alloc_blocks(g_pool, 4, blocks);

    uint32_t seq_idx;
    kcmm_register_sequence(g_pool, blocks, 4, &seq_idx);

    /* Apply MULTI_TURN hint — should protect */
    int rc = kcmm_hint(g_pool, seq_idx, KCMM_HINT_MULTI_TURN);
    if (rc != 0) { fail("hint MULTI_TURN failed"); return; }

    /* Apply NEAR_END hint — should mark for eviction */
    rc = kcmm_hint(g_pool, seq_idx, KCMM_HINT_NEAR_END);
    if (rc != 0) { fail("hint NEAR_END failed"); return; }

    /* Hint on invalid sequence should fail */
    rc = kcmm_hint(g_pool, 99999, KCMM_HINT_MULTI_TURN);
    if (rc != -1) { fail("expected -1 for invalid seq in hint"); return; }

    pass();
}

/* 19. Protection API */
TEST(protection_api) {
    uint32_t blocks[4];
    kcmm_alloc_blocks(g_pool, 4, blocks);

    uint32_t seq_idx;
    kcmm_register_sequence(g_pool, blocks, 4, &seq_idx);

    /* Protect specific blocks */
    int rc = kcmm_protect(g_pool, seq_idx, blocks, 4, KCMM_PROTECT_NEVER_EVICT);
    if (rc != 0) { fail("protect NEVER_EVICT failed"); return; }

    if (kcmm_is_tiering_enabled(g_pool)) {
        /* These only work when tiering is enabled */
        /* But they should not crash either way */
    }

    pass();
}

/* 20. Cleanup — destroy pool */
TEST(pool_destroy) {
    if (g_pool) {
        kcmm_pool_destroy(g_pool);
        g_pool = NULL;
    }
    pass();
}

/* ===========================================================================
 * Main
 * =========================================================================== */

int main(void) {
    printf("\nKCMM C API Integration Tests\n");
    printf("==============================\n\n");

    /* Null-safety must run first (no pool needed) */
    RUN_TEST(pool_destroy_null_safety);
    RUN_TEST(null_safety_all);

    /* Pool creation */
    RUN_TEST(pool_create);
    if (!g_pool) {
        printf("\nFATAL: pool creation failed — cannot run remaining tests.\n");
        printf("Is a CUDA GPU available?\n");
        return 1;
    }

    /* Error handling */
    RUN_TEST(get_last_error_empty);
    RUN_TEST(get_last_error_null_safety);

    /* Core operations */
    RUN_TEST(alloc_blocks);
    RUN_TEST(block_va_offset);
    RUN_TEST(sequence_lifecycle);
    RUN_TEST(out_of_bounds_safety);
    RUN_TEST(va_accessors);
    RUN_TEST(all_block_offsets_f16);
    RUN_TEST(block_location);
    RUN_TEST(block_table_va_offsets);

    /* Statistics */
    RUN_TEST(pool_stats);
    RUN_TEST(quick_stats);
    RUN_TEST(metrics);

    /* Tiering */
    RUN_TEST(tiering_disabled);

    /* Hint and Protection */
    RUN_TEST(hint_api);
    RUN_TEST(protection_api);

    /* Cleanup */
    RUN_TEST(pool_destroy);

    printf("\n==============================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

    return (tests_failed > 0) ? 1 : 0;
}
