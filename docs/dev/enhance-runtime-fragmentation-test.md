# Enhance Runtime Fragmentation Test Workload

**2026-5-30** written by **Vitalrubbish**

## Problem

The `step3_runtime_fragmentation` test recorded `memory_allocated_not_free` as always 2 MiB — it never varied across
time steps. This happened because:

- 1 superblock = 256 blocks = 4,096 token slots (2 MiB / 8 KiB per block)
- The original workload (prompts 8–289 tokens, MAX_NEW_TOKENS=64, MAX_BATCH=32) never exceeded ~200 blocks in use
  concurrently
- Since `blocks_not_free` never crossed the 256-block boundary, `ceil(blocks_not_free / 256)` was always 1
- `memory_allocated_not_free` was stuck at `1 × 2 MiB` for every sample

## Solution

### 1. Bimodal prompt length distribution

Replaced the 147-entry `PROMPT_LENS` with a 200-entry bimodal distribution:

| Category | Range | Entries | Simulates |
|----------|-------|--------|-----------|
| Short | 10–60 tokens | ~120 | Simple queries |
| Medium | 100–250 tokens | ~62 | Context-heavy queries |
| Long | 260–500 tokens | ~20 | Long-context documents |

This creates natural load variation: when many long prompts are sampled together, block usage spikes; periods
dominated by short prompts form valleys.

### 2. Increased decode length and request count

- `MAX_NEW_TOKENS`: 64 → 128 (longer decode phase → more blocks per sequence)
- `TOTAL_REQUESTS`: 80 → 200 (more admission/completion cycles)

Expected `memory_allocated_not_free` variation:
- Valley (mostly shorts): ~300 blocks → `ceil(300/256) = 2` → **4 MiB**
- Peak (many longs): ~500–650 blocks → `ceil(500/256) = 2` to `ceil(650/256) = 3` → **4–6 MiB**

### 3. Enhanced output reporting

Added:
- `mem_not_free` min/max and unique value count
- Evenly-spaced sample snapshots (showing the metric across the full timeline, not just first 5 and last 3)
- Fragmentation breakdown grouped by `mem_not_free` level

## Files Modified

- `tests/step3_benchmarks.rs`: New prompt distribution, larger parameters, enhanced output
