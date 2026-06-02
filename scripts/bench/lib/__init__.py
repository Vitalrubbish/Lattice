# scripts/bench/lib/__init__.py
# Shared benchmark library for Step 3 baseline vs vLLM comparison.
#
# Usage:
#   from scripts.bench.lib.workload import SONNET_PROMPT_LENS
#   from scripts.bench.lib.ufs import UnifiedFragMetrics
#   from scripts.bench.lib.protocol_baseline import send_infer_baseline
#   from scripts.bench.lib.protocol_vllm import send_completion_vllm
#   from scripts.bench.lib.summary import compute_latency_stats
