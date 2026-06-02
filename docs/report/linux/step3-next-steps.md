# Step 3 Next Steps

**Date:** 2026-06-02  
**Status:** Updated after test reorganization audit + trivial fixes applied  
**Prerequisite:** commit `d347701` (IFR fix), commit `d344ada` (EOS fix), commit `23884b1` (UFS measurement fixes)

> **2026-06-02 reorganization audit:** The test scripts were reorganized from
> `scripts/bench_*_comprehensive.py` into `scripts/bench/bench_*.py` with shared
> libraries under `scripts/bench/lib/`. The reorganization improved code structure
> but revealed one pre-existing bug (item #3 below) and one design gap (item #6).
> Three additional trivial issues (prompt length mismatch, EOS asymmetry, missing
> random seed) were found and fixed during the audit.

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
- [x] IFR explanation in report fixed: no longer attributes vLLM low IFR to EOS behavior (`37af2cb`)
- [x] Executive summary "17× improvement" rephrased to clarify adaptive BU growth (`37af2cb`)
- [x] Step 3 final report published at `docs/report/linux/step3/step3-final-report.md`
- [x] Test scripts reorganized: `scripts/bench_*_comprehensive.py` → `scripts/bench/bench_*.py` + `scripts/bench/lib/`
- [x] Time budget asymmetry fixed: both targets now use `DEFAULT_TIME_BUDGET = 600`
- [x] Missing `import re` fixed: added to `scripts/bench/lib/protocol_vllm.py`
- [x] Prompt length mismatch fixed: all three bench scripts now use `send_completion_vllm(..., prompt_token_ids=[1]*prompt_len)` for exact token counts matching baseline
- [x] EOS asymmetry fixed: `bench_throughput.py` vLLM path now uses `ignore_eos=True`, matching baseline and the other two bench scripts
- [x] Random seed fixed: `bench_max_concurrency.py` now uses `random.seed(42)` before prompt generation, matching the other bench scripts
- [x] ThreadPoolExecutor cap removed: `bench_max_concurrency.py` now uses `asyncio` + `aiohttp` for true concurrent I/O (no 64-worker cap)

---

## Remaining Work (P0 — must fix before next report)

### 1. 🔴 Fix vLLM UFS measurement: query live server state, not accumulated tokens

**Problem:** `VLLMStatsCollector._take_snapshot()` in `bench_fragmentation.py`
estimates `blocks_in_use` from accumulated token counts
(`_total_prompt_tokens + _total_completion_tokens`). These accumulators are
**never decremented** when requests complete, so:

- `blocks_in_use` is a cumulative total, not an instantaneous live count
- `total_blocks_used_by_seqs` is always equal to `blocks_in_use`, making IFR = RFI by mathematical identity for vLLM
- `total_tokens` grows monotonically; the only thing bringing it down is the benchmark ending
- BU ≈ 0.005 is just `avg(ceil(cumulative_tokens/16) / 53126)` over the test duration — a function of test parameters, not a measurement

**Fix:** Replace the accumulated-token estimation with a query to vLLM's live state.
Options:
- a) Query `/metrics` for `vllm:num_blocks_used` (if available in vLLM v0.22.0 V1 engine)
- b) Implement a custom `/kv_cache_state` endpoint in vLLM
- c) Query vLLM's internal state via `/stats` or `/v1/metrics` endpoint

**Files:** `scripts/bench/bench_fragmentation.py` → `VLLMStatsCollector._take_snapshot()` (line ~197)

### 2. 🔴 Fix vLLM low sample count at high concurrency

**Problem:** At conc=64, vLLM completes 100 requests in ~1.7 seconds. With 0.2s
poll interval, only **~8 samples** are collected. At conc=32, only **~12 samples**.
These are insufficient for meaningful statistics (avg, stddev, peak).

**Fix options:**
- a) Reduce poll interval to 0.05s at high concurrency levels
- b) Increase `--num-requests` per level (e.g., 500 instead of 100) so the test runs longer
- c) Run multiple iterations per concurrency level and pool samples

**Files:** `scripts/bench/bench_fragmentation.py` lines 375, 382 (poll interval)

### 3. ✅ Fix max_concurrency ThreadPoolExecutor cap at 64 workers

**Note:** This bug existed in the pre-reorganization scripts and was carried
forward. It is not new to the reorganization.

**Problem:** `bench_max_concurrency.py` capped actual in-flight requests at 64 via
`ThreadPoolExecutor(max_workers=min(concurrency, 64))` on lines 96 and 188.
`http.client.HTTPConnection` is synchronous blocking — each HTTP request blocked
one thread for the full decode duration. With at most 64 concurrent senders,
the server **never saw more than 64 concurrent sequences**.

At `concurrency=1024`, requests were serialized through the 64-worker pool.
Any `max_concurrent_requests > 64` measured by this script was **invalid**
(the server only handled 64 at a time, not 1024).

**Fix applied (2026-06-02):** Replaced synchronous `ThreadPoolExecutor` +
`http.client` with `asyncio` + `aiohttp` for true concurrent I/O:

- **`protocol_vllm.py`**: Added `send_completion_vllm_async()` using `aiohttp`
  with shared `ClientSession` and `TCPConnector(limit=0, force_close=True)` for
  unlimited concurrent connections.
- **`protocol_baseline.py`**: Added `send_infer_baseline_async()` using
  `asyncio.open_connection()` for non-blocking TCP I/O.
- **`bench_max_concurrency.py`**: Both `bench_max_concurrency_baseline()` and
  `bench_max_concurrency_vllm()` are now `async` functions that use
  `asyncio.gather()` to fire all requests simultaneously. No thread-pool cap.
  All requests at a concurrency level are truly in-flight at once.

Existing synchronous functions (`send_completion_vllm`, `send_infer_baseline`)
are preserved for other bench scripts that don't need high concurrency.

**Files:** `scripts/bench/bench_max_concurrency.py`, `scripts/bench/lib/protocol_vllm.py`, `scripts/bench/lib/protocol_baseline.py`

---

## Remaining Work (P1 — important but not blocking)

### 4. 🟡 Capacity at Workload: asymmetric test harnesses

**Problem:** The original baseline capacity (1,024) came from a GPU simulation
(`tests/step3_benchmarks.rs::step3_max_concurrent_requests`), while vLLM capacity
(896) came from a live HTTP benchmark. Different harnesses introduce different
overhead.

**Partial fix applied:** The reorganization introduced `bench_max_concurrency.py
--target baseline` which runs against a live TCP server, bringing the harnesses
closer together. However, TCP vs HTTP protocol overhead remains.

**Remaining fix:** Either:
- a) Document the remaining protocol overhead (TCP vs HTTP) in the report
- b) Measure and subtract protocol overhead from comparison

**Files:** `tests/step3_benchmarks.rs`, `scripts/bench/bench_max_concurrency.py`, report

### 5. 🟡 Report sample counts in stress test table

**Problem:** The stress test table (Section 4.1) presents IFR/BU/PME/RFI with
equal apparent precision but the underlying sample counts vary enormously:
baseline conc=1: ~999 samples; vLLM conc=64: ~8 samples.

**Fix:** Add a `samples` column to the stress test tables, or add a footnote
about sampling methodology.

**Files:** `docs/report/linux/step3/step3-final-report.md`

### 6. 🟡 bench_throughput.py missing UFS collection

**Problem:** `bench_throughput.py` has no stats collector thread — it only
measures latency and throughput. If someone wants UFS metrics at a single
concurrency level (not a full ramp), they have no tool.
`bench_fragmentation.py` always runs all 7 levels and writes per-level CSVs.

This is a design asymmetry: each bench script should be able to collect UFS
metrics for its target workload pattern.

**Fix:**
- a) Add an optional `--collect-ufs` flag to `bench_throughput.py` that spawns a `BaselineStatsCollector`/`VLLMStatsCollector` background thread
- b) Document that single-level UFS data can be extracted from `bench_fragmentation.py` results by running with `--concurrency-levels "N"`

**Files:** `scripts/bench/bench_throughput.py`

---

## Optional Cleanup (P2)

### 7. Clean up vLLM log parsing reliability

**Problem:** `get_vllm_num_gpu_blocks()` in `protocol_vllm.py` may silently fail
during bench runs (log file race). Retry logic was added but needs verification.

**Validation:** Re-run and confirm "vLLM block pool: N blocks" appears in output.

```bash
python3 scripts/bench/bench_fragmentation.py --target vllm --port 8001 \
    --vllm-log-path results/vllm_server.log
```

**Files:** `scripts/bench/lib/protocol_vllm.py` → `get_vllm_num_gpu_blocks()`

### 8. Document protocol overhead in max_concurrency comparison

**Problem:** Even after fixing prompt lengths and time budgets, baseline uses
raw TCP sockets while vLLM uses HTTP. HTTP adds per-request overhead (header
parsing, connection management) that isn't present in the baseline protocol.
This overhead is part of the systems' real-world characteristics but conflates
"inference engine efficiency" with "protocol overhead" when comparing
`max_concurrent_requests`.

**Fix:** Add a note in the final report acknowledging the protocol difference
and its potential impact on the comparison. Optionally, benchmark the
protocol overhead separately (e.g., measure latency of a no-op vLLM endpoint).

**Files:** `docs/report/linux/step3/step3-final-report.md`

---

## Priority Order

| Priority | Item | Effort | Impact |
|----------|------|:---:|--------|
| P0 | Fix vLLM UFS measurement (live state query) | Medium | High — all vLLM UFS data currently artificial |
| P0 | Fix vLLM low sample count at high concurrency | Small | High — makes high-conc stats meaningless |
| ~~P0~~ | ~~Fix max_concurrency ThreadPoolExecutor cap at 64 workers~~ ✅ | Medium | High — invalidates any max_concurrency > 64 measurement |
| P1 | Capacity test harness symmetry | Medium | Medium — partially addressed by new bench scripts |
| P1 | Add sample counts to stress table | Tiny | Low — transparency |
| P1 | Add optional UFS collection to bench_throughput.py | Small | Low — design gap; single-level UFS not available without full ramp |
| P2 | Verify vLLM log parsing | Tiny | Low — fallback already works |
| P2 | Document protocol overhead in comparison | Tiny | Low — awareness/transparency |
