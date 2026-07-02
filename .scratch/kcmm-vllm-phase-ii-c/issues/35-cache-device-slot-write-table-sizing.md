# Cache device-slot write table sizing

Status: done
Type: AFK

## What to build

Reduce steady-state host overhead in the device-slot KV write path by avoiding
per-write table sizing calls when the cached device offset/valid tables are
still valid for the current KCMM block-state epoch.

## Why

After Issue 34, the latest host-profile run still showed write-side request-path
overhead:

- `write_device_slot_table_lookup=5.424ms` total
- `write_select_stream=4.655ms` total
- `write_ctypes_launch=3.844ms` total

The table lookup path was still calling `pool.total_blocks()` on every
device-slot write before checking the cached device tables, and cache hits also
queried `pool.block_state_epoch()` twice. In steady state, table validity is
already guarded by the cached block-state epoch; total-block sizing is only
needed when the tables are rebuilt.

## Acceptance criteria

- [x] Stop refreshing device-slot table total-block sizing on cache hits.
- [x] Keep block-state epoch invalidation for cached device offset/valid tables.
- [x] Reduce cache-hit block-state epoch checks to one FFI query.
- [x] Report device-slot total-block refreshes and epoch-query counts.
- [x] Include the new fields in the GPU read A/B contract.
- [x] Make the performance-clean gate fail if total-block refreshes return to
  per-write behavior or epoch queries return to the old two-per-hit pattern.
- [x] Validate short vLLM smoke, performance-clean, performance-clean stress,
  and host-profile gates.

## Boundaries

- This does not change the device-slot write kernel ABI.
- This does not weaken epoch invalidation: epoch changes still rebuild the
  device offset/valid tables before launch.
- This only optimizes the device-slot performance-clean path. Host-slot
  correctness paths still use their existing validation flow.

## Verification

- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m py_compile scripts/kcmm/*.py`
- [x] `git diff --check`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_smoke --mode kcmm --backed-allocations --instrument-kv-writes --kv-write-replace-candidate --no-kv-write-verify --kv-write-device-slots --kv-read-gpu-kernel-candidate --kv-read-fast-current-context-launch --kv-read-precompile-gpu-kernel --no-kv-read-validate-block-tables --no-tracker-report-on-update --no-build-kcmm --no-print-seams`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-write-table-sizing-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_host_profile_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-host-profile-write-table-sizing-latest.json`
- [x] `/home/zhuoxiang/miniconda3/envs/vllm-cu118/bin/python -m scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate --no-build-kcmm --no-print-seams --timeout-seconds 420 --shutdown-timeout-seconds 60 --output /tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-write-table-sizing-latest.json`

## Latest local results

- Date: 2026-07-02
- Short vLLM smoke: `device_slot_total_blocks_refreshes=1`,
  `device_slot_block_state_epoch_queries=9` for `8` device-slot writes
- Performance-clean gate: `passed=true`
- Performance-clean device-slot writes/total-block refreshes/epoch queries:
  `384/3/387`
- Performance-clean request latency: stock `1.837s`, KCMM `1.840s`, ratio
  `1.002`
- Performance-clean tokens/s: stock `17.420`, KCMM `17.391`, ratio `0.998`
- Host-profile gate: `passed=true`
- Host-profile `write_device_slot_table_lookup`: `4.561ms` total, `11.878us`
  avg
- Previous host-profile before this issue:
  `write_device_slot_table_lookup=5.424ms` total, `14.125us` avg
- Performance-clean stress gate: `passed=true`
- Stress device-slot writes/total-block refreshes/epoch queries: `300/4/304`
- Stress request latency: stock `1.828s`, KCMM `1.791s`, ratio `0.980`
- Stress tokens/s: stock `26.258`, KCMM `26.801`, ratio `1.021`

## Follow-up

The remaining write-side steady-state host sections are stream selection around
`4.7ms` total and ctypes launch around `4.0ms` total. The next useful issue
should look for a safe way to reduce `KcmmStreamProvider.select(...)` overhead or
to batch/fuse more launch-side work.
