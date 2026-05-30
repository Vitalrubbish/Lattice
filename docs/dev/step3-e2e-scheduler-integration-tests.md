# E2E Scheduler Integration Tests

**Date**: 2026-05-29

## Summary

Added end-to-end integration tests that exercise the full inference request lifecycle through both schedulers (static and continuous) with dummy (zero) weights. These tests verify:
- Request submission through the `InferenceQueue`
- Scheduler picks up requests and processes them
- Full prefill → decode → response lifecycle completes
- Generated tokens are returned with correct count
- EOS termination works correctly
- Batching of multiple requests works

## Implementation

Added 5 tests across 2 files:

### `src/batch/static_batch.rs` (3 tests)
- `e2e_single_request_lifecycle` — submit 1 request, verify 5 tokens generated
- `e2e_batch_two_requests` — submit 2 requests concurrently, verify both complete
- `e2e_eos_termination` — verify EOS token stops generation after 1 token

### `src/batch/continuous_scheduler.rs` (2 tests)
- `e2e_continuous_single_request` — paged KV path, verify 3 tokens generated
- `e2e_continuous_eos_termination` — paged KV path, verify EOS stops generation

## Key decision: small non-GQA config

Used a small custom config (hidden_size=512, 8 KV heads, 4 layers) instead of tiny_llama because `KvCache.append_step` asserts `hidden_size == kv_heads * head_dim`, which fails for GQA models like tiny_llama (4 KV heads, 32 attention heads). This is a pre-existing limitation in the cache implementation.

## Dummy weights

All tests use `ModelWeights::empty()` → `NaiveTransformer` allocates zero-filled weight matrices. `greedy_sample` on all-zero logits always returns token 0, so tests use `eos_token_id != 0` (e.g., 2) to avoid premature termination, or `eos_token_id = 0` to test EOS termination.
