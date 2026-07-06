# KCMM vLLM performance report

数据日期：2026-07-03
分支：`kcmm-vllm-phase-ii-c-issues`
数据来源：
[`docs/dev/kcmm-vllm-cu118-env.md`](../dev/kcmm-vllm-cu118-env.md),
[`Issue 40`](../../.scratch/kcmm-vllm-phase-ii-c/issues/40-fast-path-canonical-kv-write-rows.md),
[`Issue 22`](../../.scratch/kcmm-vllm-phase-ii-c/issues/22-add-performance-clean-gpu-read-gate.md)

## 1. 总结

| 项 | 结果 |
|---|---|
| 当前状态 | KCMM 已接入 vLLM eager 路径的 KV write + KV read |
| 单请求性能 | latency `1.010x` stock，tokens/s `0.990x` stock |
| 并发 stress 性能 | latency `0.994x` stock，tokens/s `1.006x` stock |
| 正确性 | stock/KCMM completion text matched |
| read path | GPU read kernel 生效，CPU-staged reference read bytes `0` |
| write path | device-slot write 生效，host-slot write calls `0` |
| 显存 | peak GPU memory delta ratio `1.028x` |

## 2. 测试口径

| 项 | 配置 |
|---|---|
| vLLM | `0.6.1` eager path |
| CUDA stack | CUDA 11.8 conda environment |
| GPU | dual RTX 3080 |
| 主要模型 | `facebook/opt-125m` |
| 单请求 gate | `scripts.kcmm.vllm_gpu_read_perf_clean_gate` |
| 并发 gate | `scripts.kcmm.vllm_gpu_read_perf_clean_stress_gate` |
| host profile gate | `scripts.kcmm.vllm_gpu_read_host_profile_gate` |
| performance-clean 关闭项 | read trace、write D2H verify、GPU read profile、per-update reports、read block-table host validation |
| performance-clean 保留项 | vLLM server、scheduler、Python monkey patch、ctypes launch |

## 3. 单请求 performance-clean

| 项 | 值 |
|---|---|
| Model | `facebook/opt-125m` |
| Coverage case | `long_decode` |
| Generated tokens | `32` |
| Result | `passed=true` |
| Correctness failures | `[]` |
| Performance warnings | `[]` |
| Report | `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-row-fastpath-latest.json` |

| Metric | stock vLLM | KCMM | Ratio |
|---|---:|---:|---:|
| Request latency seconds | `1.785` | `1.803` | `1.010x` |
| Tokens per second | `17.927` | `17.748` | `0.990x` |
| Peak GPU memory delta MiB | `5441` | `5591` | `1.028x` |

| Path evidence | Value |
|---|---:|
| GPU read kernel calls | `372` |
| Stream-aware read kernel calls | `372` |
| Reference KCMM read bytes | `0` |
| Device-slot write calls | `384` |
| Host-slot write calls | `0` |
| KCMM write verified rows | `0` |
| Write verification synchronizations | `0` |
| Device-slot status checks/errors | `384/0` |

| Read hot path | Value |
|---|---:|
| Compact plan metadata enabled/calls | `true/372` |
| Detailed plan metadata calls | `0` |
| Fast current-context launch | `true` |
| GPU kernel precompile requested/succeeded/calls | `true/true/1` |
| GPU kernel precompile elapsed | `95.495ms` |
| Stream select/current/cache hits/cache misses | `372/372/371/1` |
| Offset table cache hits/rebuilds | `369/3` |

| Write hot path | Value |
|---|---:|
| Pool shape cached/refreshes | `true/1` |
| Cached pool shape | `block_size=16`, `block_bytes=24576`, `step_elements=768`, `num_layers=12` |
| Device-slot kernel precompile requested/succeeded/calls | `true/true/1` |
| Device-slot kernel precompile elapsed | `79.432ms` |
| Device-slot prepare direct/reshape/dtype/copy | `384/0/0/0` |
| Row prepare direct/fallback/contiguous-copy | `372/12/24` |
| Device-slot offset table cache hits/rebuilds | `381/3` |
| Device-slot valid table cache hits/rebuilds | `381/3` |
| Stream select/current/cache hits/cache misses | `384/384/383/1` |

## 4. 并发 stress performance-clean

| 项 | 值 |
|---|---|
| Coverage cases | `stress_history`, `stress_memory` |
| Completion concurrency | `2` |
| `max_num_seqs` | `2` |
| `max_num_batched_tokens` | `192` |
| Result | `passed=true` |
| Correctness failures | `[]` |
| Performance warnings | `[]` |
| Report | `/tmp/kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-row-fastpath-latest.json` |

| Metric | stock vLLM | KCMM | Ratio |
|---|---:|---:|---:|
| Request latency seconds | `1.787` | `1.776` | `0.994x` |
| Tokens per second | `26.861` | `27.027` | `1.006x` |
| Peak GPU memory delta MiB | `5443` | `5593` | `1.028x` |

| Path evidence | Value |
|---|---:|
| Observed max read batch | `2` |
| Observed max write batch | `17` |
| GPU read kernel calls | `276` |
| Stream-aware read kernel calls | `276` |
| Reference KCMM read bytes | `0` |
| Device-slot write calls | `288` |
| Host-slot write calls | `0` |
| Device-slot status checks/errors | `288/0` |
| Device-slot prepare direct/reshape/dtype/copy | `288/0/0/0` |
| Row prepare direct/fallback/contiguous-copy | `0/288/576` |

## 5. 性能演进

| 阶段 | Latency ratio | Tokens/s ratio | 说明 |
|---|---:|---:|---|
| Issue 22 初版 performance-clean | `1.771x` | `0.565x` | 功能闭合，但 hot path 开销明显 |
| Issue 40 row fast path 后 | `1.010x` | `0.990x` | 单请求 clean gate 基本贴近 stock |
| Issue 40 stress | `0.994x` | `1.006x` | 并发 clean gate 略优于 stock |

| 优化方向 | 代表效果 |
|---|---|
| 关闭 correctness-only 开销 | read trace、write D2H verify、profile timing off |
| read offset table cache | offset table cache hits/rebuilds `369/3` |
| read current-context launch | fast current-context launch `true` |
| read kernel precompile | precompile `95.495ms` 从请求热路径移出 |
| compact read plan metadata | detailed plan metadata calls `0` |
| write pool shape cache | pool shape refreshes `1` |
| device-slot write | host-slot write calls `0` |
| write kernel precompile | precompile `79.432ms` 从请求热路径移出 |
| slot tensor fast path | device-slot prepare `384/0/0/0` |
| row fast path | row prepare `372/12/24` |

## 6. Host-profile

| Metric | stock vLLM | KCMM | Ratio |
|---|---:|---:|---:|
| Request latency seconds | `1.852` | `1.823` | `0.984x` |
| Tokens per second | `17.279` | `17.553` | `1.016x` |

| Top read host section | Time |
|---|---:|
| `read_gpu_kernel_precompile` | `94.257ms` |
| `read_replace_call_total` | `33.352ms` |
| `read_replace_gpu_kernel_host` | `18.259ms` |
| `read_gpu_kernel_host_total` | `17.255ms` |
| `read_replace_build_plan` | `12.060ms` |
| `read_build_plan_total` | `11.123ms` |
| `read_gpu_kernel_ctypes_launch` | `6.086ms` |
| `read_gpu_kernel_select_stream` | `3.233ms` |

| Top write host section | Time |
|---|---:|
| `write_device_slot_kernel_precompile` | `76.782ms` |
| `write_mirror_call_total` | `34.539ms` |
| `write_select_stream` | `4.103ms` |
| `write_ctypes_launch` | `3.906ms` |
| `write_prepare_rows` | `2.394ms` |
| `write_device_slot_table_lookup` | `2.210ms` |
| `write_layer_for_cache` | `0.981ms` |
| `write_validate_dtype` | `0.971ms` |

| Recent write-side change | Before | After | Delta |
|---|---:|---:|---:|
| `write_prepare_rows` | `2.705ms` | `2.394ms` | `-0.311ms` |
| `write_mirror_call_total` | `35.141ms` | `34.539ms` | `-0.602ms` |

## 7. 显存

| 场景 | stock peak delta | KCMM peak delta | Ratio |
|---|---:|---:|---:|
| Single performance-clean | `5441 MiB` | `5591 MiB` | `1.028x` |
| Stress performance-clean | `5443 MiB` | `5593 MiB` | `1.028x` |

## 8. 边界

| 项 | 当前口径 |
|---|---|
| Benchmark type | request-level performance-clean gate，不是纯 kernel microbenchmark |
| vLLM version | `0.6.1` eager path |
| Model coverage | 最新性能主数据来自 `facebook/opt-125m` |
| CUDA graph | 未覆盖，当前使用 eager path |
| Phase III tiering | 不在本报告范围 |
| Long-running workload | 尚未做长时间生产型负载 |

## 9. 下一步

| 优先级 | 下一步 | 目标 |
|---|---|---|
| P0 | combined launch / stronger stream-launch ABI | 减少 Python/ctypes launch 次数 |
| P0 | 更真实长时间 workload | 验证稳定性和尾延迟 |
| P1 | 更大 real-model matrix | 扩展模型覆盖 |
| P1 | 更高并发 stress sweep | 观察 batch/latency/tokens/s 曲线 |
| P2 | Phase III tiering benchmark | 验证 capacity/PME/UFS 收益 |
