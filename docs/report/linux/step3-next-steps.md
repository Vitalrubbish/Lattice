# Step 3 Next Steps

**Date:** 2026-06-01  
**Status:** Updated after final report audit  
**Prerequisite:** commit `d347701` (IFR fix), commit `d344ada` (EOS fix), commit `23884b1` (UFS measurement fixes)

## Completed So Far

- [x] UFS metrics defined and implemented (IFR, BU, PME, RFI)
- [x] vLLM `total_blocks_allocated` switched from `nvidia-smi` diff to `num_gpu_blocks`
- [x] Baseline `step3_max_concurrent_requests` → Capacity at Workload
- [x] Baseline internal: `CacheStats` source, `fragmentation_ratio` → `physical_idle_ratio`, legacy ratio cleanup, `record()` privatised
- [x] CONTEXT.md glossary (18 terms)
- [x] IFR measurement bug fixed: `update_seq_len(seq_idx, prompt_len)` during prefill (`d6c96d6`, `d347701`)
- [x] vLLM EOS fix: `ignore_eos=True` in max-concurrency bench (`d344ada`)
- [x] Stress test (concurrency ramp 1..64) executed for both baseline and vLLM
- [x] Deleted misleading GPU Simulation fragmentation test and vLLM dedicated fragmentation test (sections 4.5/4.6 of final report)
- [x] Deleted `step3_runtime_fragmentation` test and `bench_fragmentation` function
- [x] Step 3 final report published at `docs/report/linux/step3/step3-final-report.md`

---

## Remaining Work (P0 — must fix before next report)

### 1. 🔴 Fix vLLM UFS measurement: query live server state, not accumulated tokens

**Problem:** `UFSStatsCollector._take_snapshot()` estimates `blocks_in_use` from
accumulated token counts (`_total_prompt_tokens + _total_completion_tokens`).
These accumulators are **never decremented** when requests complete, so:

- `blocks_in_use` is a cumulative total, not an instantaneous live count
- `total_blocks_used_by_seqs` is always equal to `blocks_in_use`, making IFR = RFI by mathematical identity for vLLM
- `total_tokens` grows monotonically; the only thing bringing it down is the benchmark ending
- BU ≈ 0.005 is just `avg(ceil(cumulative_tokens/16) / 53126)` over the test duration — a function of test parameters, not a measurement

**Fix:** Replace the accumulated-token estimation with a query to vLLM's live state.
Options:
- a) Query `/metrics` for `vllm:num_blocks_used` (if available in vLLM v0.22.0 V1 engine)
- b) Implement a custom `/kv_cache_state` endpoint in vLLM
- c) Query vLLM's internal state via `/stats` or `/v1/metrics` endpoint

**Files:** `scripts/bench_vllm_comprehensive.py` → `UFSStatsCollector._take_snapshot()` (line ~361)

### 2. 🔴 Fix vLLM low sample count at high concurrency

**Problem:** At conc=64, vLLM completes 100 requests in ~1.7 seconds. With 0.3s
poll interval, only **6 samples** are collected. At conc=32, only **9 samples**.
These are insufficient for meaningful statistics (avg, stddev, peak).

**Fix options:**
- a) Reduce poll interval to 0.05s at high concurrency levels
- b) Increase `--num-requests` per level (e.g., 500 instead of 100) so the test runs longer
- c) Run multiple iterations per concurrency level and pool samples

**Files:** `scripts/bench_vllm_comprehensive.py`

### 3. 🔴 Fix IFR explanation in report (Section 4.2, line 112-113)

**Current text:**
> "vLLM IFR is lower (<0.03) because many sequences terminate early (EOS),
> generating fewer tokens and filling their last block more completely."

**Problem:** This is wrong. vLLM IFR is a function of the cumulative estimation
model: `IFR = 1 - total_tokens/(ceil(total_tokens/16)*16)`. For large cumulative N,
the mod-16 fluctuation is small, so IFR ≈ 0. It has nothing to do with EOS behavior.

**Fix:** Rewrite this sentence to accurately describe the estimation artifact.
After issue #1 (live state query) is fixed, the explanation may change entirely.

**Files:** `docs/report/linux/step3/step3-final-report.md`

---

## Remaining Work (P1 — important but not blocking)

### 4. 🟡 Capacity at Workload: asymmetric test harnesses

**Problem:** Baseline capacity (1,024) comes from a GPU simulation
(`tests/step3_benchmarks.rs::step3_max_concurrent_requests`), while vLLM capacity
(896) comes from a live HTTP benchmark (`bench_max_concurrency`). Different
harnesses introduce different overhead (HTTP latency, vLLM scheduler overhead vs.
none in baseline simulation).

**Fix:** Either:
- a) Run baseline capacity test as a live HTTP benchmark (matching vLLM setup)
- b) Document the asymmetry clearly in the report
- c) Both

**Files:** `tests/step3_benchmarks.rs`, `scripts/bench_vllm_comprehensive.py`, report

### 5. 🟡 Report sample counts in stress test table

**Problem:** The stress test table (Section 4.1) presents IFR/BU/PME/RFI with
equal apparent precision but the underlying sample counts vary enormously:
baseline conc=1: 999 samples; vLLM conc=64: 6 samples.

**Fix:** Add a `samples` column to the stress test tables, or add a footnote
about sampling methodology.

**Files:** `docs/report/linux/step3/step3-final-report.md`

### 6. 🟡 Executive summary "17× improvement" is misleading

**Problem:** The headline compares baseline BU against itself at different loads
(0.04 → 0.65), not against vLLM. This is measuring grow-on-demand working as designed,
not a "vs vLLM" improvement. vLLM's BU is structurally low because of pre-allocation,
which is a design choice, not a bug.

**Fix:** Rephrase to clarify that the 17× refers to baseline's own BU growth
under increasing load, demonstrating that grow-on-demand adapts to demand.

**Files:** `docs/report/linux/step3/step3-final-report.md`

---

## Optional Cleanup (P2)

### 7. Clean up vLLM log parsing reliability

**Problem:** `_parse_num_blocks_from_log()` may silently fail during bench runs
(log file race). Retry logic was added but needs verification.

**Validation:** Re-run and confirm "vLLM block pool: 53126 blocks" appears.

```bash
./scripts/step3_test_baremetal.sh vllm
```

### 8. Remove or `#[deprecated]` the now-redundant `legacy_ratio_*` fields

These are fully unused in StatsHandle; double-check no consumers remain.

---

## Priority Order

| Priority | Item | Effort | Impact |
|----------|------|:---:|--------|
| P0 | Fix vLLM UFS measurement (live state query) | Medium | High — all vLLM UFS data currently artificial |
| P0 | Fix vLLM low sample count at high concurrency | Small | High — makes high-conc stats meaningless |
| P0 | Fix IFR explanation in report | Tiny | High — currently factually wrong |
| P1 | Capacity test harness symmetry | Medium | Medium — affects one comparison |
| P1 | Add sample counts to stress table | Tiny | Low — transparency |
| P1 | Rephrase "17× improvement" headline | Tiny | Low — clarity |
| P2 | Verify vLLM log parsing | Tiny | Low — fallback already works |
| P2 | Remove legacy ratio fields | Small | Low — hygiene |
