# IFR Measurement Bug: Prefill seq_len Under-reporting

**Date:** 2026-06-01  
**Found by:** UFS stress test — IFR=0.50 in server throughput vs IFR=0.04 in GPU simulation  
**Fixed in:** `d347701`

## Symptom

Baseline server throughput benchmark reported IFR ≈ 0.50 while the GPU simulation
test (`step3_runtime_fragmentation`) reported IFR ≈ 0.04 — a 12× discrepancy.

```
                    Before fix           After fix (expected)
Server (conc=4):    IFR = 0.50           IFR = 0.06
GPU simulation:     IFR = 0.04           IFR = 0.04
vLLM:               IFR = 0.002          IFR = 0.002
```

Both use identical block_size=16. The difference could not be explained by the
prompt length distribution — it pointed to a measurement bug in the server path.

## Root Cause

`ContinuousScheduler::admit_waiting()` allocates blocks for the full prompt:

```rust
let blocks_needed = ceil(prompt_len / block_size);
cache.alloc_sequence(blocks_needed); // e.g., 13 blocks for 200-token prompt
```

But `seq_len` started at 0 and was only incremented token-by-token during prefill.
The first buggy fix (`d6c96d6`) added `update_seq_len` in the prefill branch but
still used the step-by-step position:

```rust
// Buggy: seq_len goes 1, 2, 3, ... 200 — 200 steps to catch up
self.cache.update_seq_len(r.seq_idx, r.position);
```

At any prefill snapshot point:

```
prompt_len = 200, blocks = 13, slots = 208
prefill step 50:  seq_len = 50,  IFR = (208 - 50) / 208 = 0.76
prefill step 150: seq_len = 150, IFR = (208 - 150) / 208 = 0.28
```

The UFS sampler captured many mid-prefill snapshots where `total_tokens` was
dramatically under-reported relative to `total_slots`, inflating IFR 10-100×.

## Fix

Changed the prefill branch to always report `seq_len = prompt_len`, because
blocks ARE reserved for the full prompt length:

```rust
// Fixed: seq_len = prompt_len immediately, reflecting capacity reservation
self.cache.update_seq_len(r.seq_idx, r.req.prompt_tokens.len());
```

The principle: **fragmentation tracking cares about capacity reservation, not
whether data has been written yet.** Blocks allocated for a 200-token prompt
represent a 200-token capacity commitment — the seq_len should reflect that
from the moment the blocks are assigned.

## Verification

After fix, IFR stabilised at ~0.05 across all concurrency levels:

```
conc    IFR (before)    IFR (after)
 1      0.297           0.062
 4      0.439           0.056
 8      0.469           0.060
16      0.544           0.053
32      0.504           0.053
64      0.489           0.053
```

The remaining ~0.05 IFR is genuine internal fragmentation from the sonnet
prompt distribution, matching the GPU simulation test (0.04 with a different
bimodal distribution).

## Lesson

When a metric like IFR depends on `total_tokens` (from `seq_len`), the
`seq_len` value must reflect **capacity reservation**, not **write progress**.
This is especially important during prefill, where block allocation happens
upfront but data writes happen incrementally. The GPU simulation test was
immune because it called `update_seq_len(seq_idx, prompt_len)` immediately
after `alloc_sequence()` — the server scheduler should do the same.
