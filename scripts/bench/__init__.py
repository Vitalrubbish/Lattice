# scripts/bench/__init__.py
# Unified benchmark scripts for Step 3 baseline vs vLLM comparison.
#
# Three benchmark dimensions:
#   bench_max_concurrency.py  — max concurrent requests (capacity at workload)
#   bench_throughput.py       — throughput at fixed concurrency
#   bench_fragmentation.py    — fragmentation (UFS) via concurrency ramp
#
# Each accepts --target baseline|vllm for direct comparison.
