# Step 3 Next Steps

**Date:** 2026-06-01
**Status:** Planned
**Prerequisite:** commit `23884b1` (UFS measurement fixes)

## Completed So Far

- [x] UFS metrics defined and implemented (IFR, BU, PME, RFI)
- [x] vLLM `total_blocks_allocated` switched from `nvidia-smi` diff to `num_gpu_blocks`
- [x] Baseline `step3_max_concurrent_requests` → Capacity at Workload
- [x] Baseline internal: `CacheStats` source, `fragmentation_ratio` → `physical_idle_ratio`, legacy ratio cleanup, `record()` privatised
- [x] CONTEXT.md glossary (18 terms)

## Remaining Work

### 1. Fix vLLM log parsing reliability

**Problem:** `_parse_num_blocks_from_log()` silently fails during bench runs
(the log file exists and contains the expected line, but `open()` + `re.search()`
return 0 — likely a race with vLLM's still-writing file handle).

**Fix (already applied, needs verification):**
- Added `os.path.exists()` check before attempting `open()`
- Retry up to 5 times with 1-second delays
- Added `errors='replace'` for encoding robustness
- Added error message on final retry failure

**Validation:** Re-run and confirm "vLLM block pool: 53126 blocks" appears
instead of "WARNING: could not query vLLM block pool, estimated ...".

```bash
# Quick verification — skip baseline to save time
./scripts/step3_test_baremetal.sh vllm
```

### 2. Fix vLLM max-concurrency EOS asymmetry

**Problem:** vLLM's `bench_max_concurrency` sends real text prompts
(`"Hello " * N`) which cause EOS generation for some sequences.
These sequences free their KV blocks early, giving vLLM an
unfairly high capacity count compared to baseline (which always
grows every sequence to `prompt_len + max_new_tokens`).

**Fix — in `scripts/bench_vllm_comprehensive.py`:**

a) Add `--eos-token-id` CLI argument to `bench_max_concurrency`:

```python
# In bench_max_concurrency signature
def bench_max_concurrency(port: int, model: str, max_tokens: int = 64,
                          step: int = 4, timeout_per_req: int = 300,
                          eos_token_id: int = 1_000_000) -> dict:
```

b) Pass `eos_token_id` through to completion requests:

```python
# In send_completion_concurrent
body = json.dumps({
    "model": model,
    "prompt": "Hello " * max(1, prompt_len),
    "max_tokens": max_tokens,
    "stop_token_ids": [],       # disable built-in stop tokens
})
```

Or alternatively, use random token IDs instead of "Hello" text so the
model can't organically generate EOS.

c) Also apply to `bench_fragmentation` and `bench_throughput` where
appropriate (throughput should keep natural EOS behavior; max-concurrency
and fragmentation benches should use deterministic full-length generation).

**Estimated:** ~15 lines changed in `bench_vllm_comprehensive.py`.

### 3. Run stress mode for load-dependent UFS curves

**Goal:** Show how BU, PME, and RFI change as concurrency increases,
demonstrating CUDA VMM's grow-on-demand advantage at low load and
convergence at high load.

**Command:**
```bash
# Baseline stress (concurrency ramp 1..64)
cargo build --release --example bench_throughput
timeout 600 target/release/examples/bench_throughput \
    --addr 127.0.0.1:8000 \
    --num-requests 100 \
    --stress-concurrency "1,2,4,8,16,32,64" \
    --max-new-tokens 64 \
    --output-csv results/stress_baseline_results.csv

# vLLM stress
python3 scripts/bench_vllm_comprehensive.py \
    --port 8001 --model /users/Lattice/models/tinyllama \
    --mode stress \
    --num-requests 100 \
    --concurrency-levels "1,2,4,8,16,32,64" \
    --max-new-tokens 64 \
    --output-dir results/stress_vllm/
```

**Expected output:** CSV with one row per concurrency level, columns
for throughput + all 4 UFS metrics. The key chart: BU vs concurrency,
showing baseline rising from ~0.01 to ~0.5+ while vLLM stays flat at ~0.001.

### 4. Regenerate UFS_REPORT.md with corrected data

After fixes 1-3 are applied and validated, re-run the full comparison:

```bash
MODEL_PATH=/users/Lattice/models/tinyllama \
./scripts/step3_test_baremetal.sh compare
```

Update `UFS_REPORT.md` with:
- [ ] Allocator semantics notice (why BU/PME differ)
- [ ] Per-concurrency-level UFS breakdown (from stress results)
- [ ] Capacity-at-workload comparison (aligned EOS behavior)
- [ ] Interpretation guide for each metric

### 5. Optional: clean up remaining code smells

- [ ] Remove or `#[deprecated]` the now-redundant `legacy_ratio_*` fields
      (these are now fully unused in StatsHandle; double-check no consumers)
- [ ] Merge `record()` body into `record_unified()` — no point maintaining
      two parallel sample vecs when `record_unified` always calls `record`
- [ ] Add `--skip-baseline` flag to `step3_test_baremetal.sh` for faster
      vLLM-only iteration

---

## Priority Order

| Priority | Item | Effort | Impact |
|----------|------|:---:|--------|
| P0 | Fix vLLM EOS asymmetry (#2) | Small | High — makes capacity comparison fair |
| P1 | Verify log parsing (#1) | Tiny | High — removes estimation fallback |
| P1 | Stress mode (#3) | Medium | High — shows CUDA VMM benefit curve |
| P2 | Regenerate report (#4) | Small | Medium — documentation |
| P3 | Code cleanup (#5) | Small | Low — hygiene |
