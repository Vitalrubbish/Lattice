# Step 3 UFS Measurement Fix Plan

**Date:** 2026-06-01
**Status:** Planned

## Problem

The current UFS comparison between baseline and vLLM has a measurement asymmetry
that makes vLLM's fragmentation metrics appear artificially good (BU≈0.96, RFI<0.01 at concurrency=4).

**Root cause:** vLLM's `total_blocks_allocated` is estimated from `nvidia-smi diff` against a
post-startup baseline, which subtracts away vLLM's pre-allocated block pool (~20 GB).
The result is `total_blocks_allocated ≈ blocks_in_use`, giving a false BU≈1.0.

Baseline's `total_blocks_allocated` correctly counts all `cuMemCreate`d superblocks,
so its BU/PME honestly reflects superblock granularity waste.

See [CONTEXT.md](/CONTEXT.md) for precise definitions of each UFS metric.

## Changes

### 1. `scripts/bench_vllm_comprehensive.py` — Fix `total_blocks_allocated`

**File:** `scripts/bench_vllm_comprehensive.py`
**Estimated:** ~40 lines changed

**`UFSStatsCollector` changes:**

- Add `_query_num_gpu_blocks()` method:
  1. `GET /metrics` → parse `vllm:num_gpu_blocks` or `vllm_num_gpu_blocks_total`
  2. Fallback: grep server log for `GPU blocks`
  3. Fallback: estimate from `gpu_memory_utilization × total_vram / (block_bytes × num_layers × 2)`

- Rename `calibrate_baseline()` to `calibrate()`:
  - Remove `nvidia-smi` baseline recording
  - Call `_query_num_gpu_blocks()` and store result as `self.num_gpu_blocks`

- `_take_snapshot()`:
  - `total_blocks_allocated = self.num_gpu_blocks` (constant, queried once)
  - Remove `kv_cache_bytes` / `nvidia-smi diff` logic
  - Remove `estimated_from_tokens` fallback

After fix, vLLM's `total_blocks_allocated` will equal its actual pre-allocated pool size,
matching baseline's semantics: "all physical memory provisioned for KV cache blocks."

### 2. `tests/step3_benchmarks.rs` — Fix max concurrent test

**File:** `tests/step3_benchmarks.rs`
**Estimated:** ~30 lines changed

**`step3_max_concurrent_requests` changes:**

Replace hardcoded `alloc_sequence(16)` with workload-driven admission:

```rust
// Use same prompt-length distribution as vLLM bench (small prompts for max concurrency test)
let prompt_lens: Vec<usize> = (0..max_batch)
    .map(|_| sample_prompt_len())  // e.g., 8, 16, 32 tokens
    .collect();

for pl in prompt_lens {
    let blocks_needed = (pl + BLOCK_SIZE - 1) / BLOCK_SIZE;
    match cache.alloc_sequence(blocks_needed) {
        Ok(table) => {
            cache.register_sequence(table);
            allocated += 1;
        }
        Err(_) => break,  // OOM
    }
}
```

Then simulate decode growth with `alloc_block()` per step until either
`max_new_tokens` reached or OOM. This matches vLLM's bench_max_concurrency
semantics.

Rename the test conceptually from "max concurrent requests" to **"capacity at workload"**:
the number of sequences that fit under a specific token-length distribution.

### 3. `scripts/ufs_metrics.py` — No changes needed

`compute_metrics_vllm()` already uses the correct formula:

```python
actual_physical = total_blocks_allocated × block_bytes × num_layers × 2
```

The bug was only in the caller passing a wrong `total_blocks_allocated`.

### 4. Report presentation — Improve interpretation

**File:** UFS_REPORT.md generation (in `step3_test_baremetal.sh` or post-processing)

Add these sections:

#### Allocator Semantics Notice

Explain why `total_blocks_allocated` and therefore BU/PME differ between systems:

| | Baseline (CUDA VMM) | vLLM (PyTorch) |
|---|---|---|
| Allocation | `cuMemCreate(2MB)` on demand | `cudaMalloc(20GB)` at startup |
| `total_blocks_allocated` | Grows with load | Fixed at startup |
| Low-concurrency BU | Reflects 2MB granularity waste | Reflects 20GB pre-allocation waste |
| High-concurrency BU | Approaches 1.0 | Always capped by pool oversizing |

#### Per-Concurrency-Level Breakdown

Show UFS metrics stratified by concurrency level rather than a single average:

```
                    concurrency=4    concurrency=29    concurrency=max
                    ─────────────    ──────────────    ──────────────
Baseline IFR         0.50             0.039              ~0
Baseline BU          0.11             0.59               ~1.0
Baseline RFI         0.94             0.32               ~0.03
vLLM IFR             0.004            0.006              ~0
vLLM BU              ~0.001           ~0.001             ~0.001
vLLM RFI             ~0.004           ~0.006             ~0
```

#### How to Read These Numbers

- **IFR:** Should be identical across systems. If not, measurement is broken.
- **BU:** Compare trend, not absolute value. Baseline rising = grow-on-demand working.
- **PME:** Same as BU for now; system-specific formulas may diverge with Step 4 (64KB pages).
- **RFI:** Low is good. At high concurrency, baseline's RFI should approach vLLM's.
- **Capacity-at-Workload:** The hardest metric. Same workload → who fits more sequences.

### 5. Validation

```bash
./scripts/step3_test_baremetal.sh compare
```

Expected post-fix results:
- vLLM BU drops from 0.96 to ~0.001 at concurrency=4
- vLLM PME drops similarly
- Baseline max concurrent now reflects workload-driven admission, not worst-case pre-allocation
- IFR remains consistent across both systems
- Report clearly explains the allocator semantics difference

---

## Files Affected

| File | Change | Lines |
|------|--------|-------|
| `scripts/bench_vllm_comprehensive.py` | Fix `total_blocks_allocated` source | ~40 |
| `tests/step3_benchmarks.rs` | Workload-driven capacity test | ~30 |
| `scripts/ufs_metrics.py` | None | 0 |
| Report generation | Add semantics notice, per-concurrency tables, reading guide | ~50 |
