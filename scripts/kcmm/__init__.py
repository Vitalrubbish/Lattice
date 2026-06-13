# KCMM Python Integration Layer
#
# Directory layout:
#   bindings.py        — ctypes loader for libkcmm.so, wraps all C API functions
#   config.py          — kcmm_config_t construction from vLLM ModelConfig
#   patch_vllm.py      — monkey-patch logic (8 interception points, phased by flags)
#   launcher.py        — entry point: import vllm → apply patches → start server
#
# Phased implementation (see docs/adr/0001-vllm-integration-architecture.md):
#   Phase I.C  — observer only (pool create/destroy, single alloc, metrics sampling)
#   Phase II.A — allocator replacement (intercepts 1, 8)
#   Phase II.B — KV write path (intercept 2, reshape_and_cache → kcmm_append_kv_step)
#   Phase II.C — KV read path (intercept 3, block_tables → KCMM VA offsets, A1)
#   Phase III  — tiering + hints + metrics (intercepts 4, 6, 7)
