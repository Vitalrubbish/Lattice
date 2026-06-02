#!/usr/bin/env python3
"""bench_fragmentation.py — Fragmentation (UFS) benchmark via concurrency ramp.

Runs a stress test: throughput benchmark at multiple concurrency levels while
collecting UFS fragmentation metrics (IFR, BU, PME, RFI) in the background.
Produces directly comparable results between baseline and vLLM.

Usage:
  python3 -m scripts.bench.bench_fragmentation \\
    --target baseline --host 127.0.0.1 --port 8000

  python3 -m scripts.bench.bench_fragmentation \\
    --target vllm --port 8001 --model /path/to/model

  python3 -m scripts.bench.bench_fragmentation \\
    --target baseline --concurrency-levels "1,2,4,8,16,32,64"
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path so imports work from any directory
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from scripts.bench.lib.workload import (
    SONNET_PROMPT_LENS,
    TINYLLAMA_PARAMS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_REQUESTS,
    DEFAULT_TIMEOUT,
    DEFAULT_EOS_TOKEN_ID,
)
from scripts.bench.lib.ufs import (
    UnifiedFragMetrics,
    compute_metrics_vllm,
    compute_summary,
    print_summary,
)
from scripts.bench.lib.protocol_baseline import (
    send_infer_baseline,
    query_baseline_stats,
    wait_for_baseline_server,
)
from scripts.bench.lib.protocol_vllm import (
    send_completion_vllm,
    wait_for_vllm_server,
    warmup_vllm,
    get_vllm_num_gpu_blocks,
)
from scripts.bench.lib.summary import (
    compute_latency_stats,
    compute_ufs_summary,
    write_results_csv,
    write_frag_csv,
    write_stress_summary_csv,
    print_stress_comparison,
)


# ── UFS Stats Collector (background thread) ──

class BaselineStatsCollector:
    """Background stats collector for baseline benchmarks.

    Polls the baseline server's TCP stats endpoint at regular intervals
    to collect live UFS metrics.
    """

    def __init__(self, host: str, port: int, poll_interval_s: float = 0.2):
        self.host = host
        self.port = port
        self.poll_interval_s = poll_interval_s
        self.samples: List[UnifiedFragMetrics] = []
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[UnifiedFragMetrics]:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        return list(self.samples)

    def _poll_loop(self):
        while self._running.is_set():
            try:
                stats = query_baseline_stats(self.host, self.port, timeout=2.0)
                if stats and stats.get("sample_count", 0) > 0:
                    sample = UnifiedFragMetrics(
                        internal_frag_rate=stats.get("internal_frag_rate", 0.0),
                        block_utilization=stats.get("block_utilization", 0.0),
                        physical_memory_efficiency=stats.get("physical_memory_efficiency", 0.0),
                        runtime_frag_index=stats.get("runtime_frag_index", 0.0),
                        active_sequences=stats.get("active_sequences", 0),
                        blocks_in_use=stats.get("blocks_in_use", 0),
                        total_blocks_allocated=stats.get("total_blocks_allocated", 0),
                        total_tokens=stats.get("total_tokens", 0),
                    )
                    with self._lock:
                        self.samples.append(sample)
            except Exception:
                pass
            time.sleep(self.poll_interval_s)


class VLLMStatsCollector:
    """Background stats collector for vLLM benchmarks.

    Calibrates total_blocks_allocated from the vLLM server log, then estimates
    blocks_in_use from accumulated token counts during the benchmark.
    """

    def __init__(self, port: int, poll_interval_s: float = 0.2,
                 server_log_path: Optional[str] = None):
        self.port = port
        self.poll_interval_s = poll_interval_s
        self.samples: List[UnifiedFragMetrics] = []
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Accumulated stats (updated by benchmark thread)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._active_requests = 0

        # Model parameters
        self.kv_heads = TINYLLAMA_PARAMS["kv_heads"]
        self.head_dim = TINYLLAMA_PARAMS["head_dim"]
        self.num_layers = TINYLLAMA_PARAMS["num_layers"]
        self.block_size = TINYLLAMA_PARAMS["block_size"]
        self.block_bytes = TINYLLAMA_PARAMS["block_bytes"]

        # vLLM pool capacity
        self.num_gpu_blocks = 0
        self._server_log_path = server_log_path

    def calibrate(self):
        """Query vLLM's block-pool capacity."""
        self.num_gpu_blocks = get_vllm_num_gpu_blocks(
            port=self.port,
            server_log_path=self._server_log_path,
            block_size=self.block_size,
        )
        if self.num_gpu_blocks > 0:
            print(f"   vLLM block pool: {self.num_gpu_blocks} blocks "
                  f"({self.num_gpu_blocks * self.block_bytes * self.num_layers * 2 / (1024**3):.2f} GiB "
                  f"for all layers K+V)", file=sys.stderr)
        else:
            print(f"   WARNING: could not query vLLM block pool, "
                  f"UFS metrics unavailable", file=sys.stderr)

    def update_request_stats(self, prompt_tokens: int = 0,
                              completion_tokens: int = 0,
                              active_delta: int = 0):
        """Update accumulated stats from the benchmark thread."""
        with self._lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._active_requests += active_delta

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[UnifiedFragMetrics]:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        return list(self.samples)

    def _poll_loop(self):
        while self._running.is_set():
            try:
                snapshot = self._take_snapshot()
                if snapshot is not None:
                    self.samples.append(snapshot)
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def _take_snapshot(self) -> Optional[UnifiedFragMetrics]:
        if self.num_gpu_blocks == 0:
            return None

        with self._lock:
            total_tokens = self._total_prompt_tokens + self._total_completion_tokens
            active_requests = self._active_requests

        estimated_blocks = (total_tokens + self.block_size - 1) // self.block_size
        blocks_in_use = min(estimated_blocks, self.num_gpu_blocks)

        return compute_metrics_vllm(
            block_size=self.block_size,
            blocks_in_use=blocks_in_use,
            total_blocks_allocated=self.num_gpu_blocks,
            total_blocks_used_by_seqs=estimated_blocks,
            total_tokens=total_tokens,
            block_bytes=self.block_bytes,
            num_layers=self.num_layers,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            active_sequences=active_requests,
        )


# ── Benchmark functions ──

def _build_prompts(num_requests: int) -> List[int]:
    """Generate prompt lengths from the sonnet distribution."""
    random.seed(42)
    return [random.choice(SONNET_PROMPT_LENS) for _ in range(num_requests)]


def _run_bench_level_baseline(
    host: str, port: int, num_requests: int, concurrency: int,
    max_new_tokens: int, collector: BaselineStatsCollector,
) -> dict:
    """Run one concurrency level against baseline, collecting UFS samples."""
    prompts = _build_prompts(num_requests)
    results = []
    completed = 0
    lock = threading.Lock()
    t_start = time.time()

    collector.start()

    def send_one(idx: int) -> dict:
        nonlocal completed
        pl = prompts[idx]
        rec = send_infer_baseline(host, port, pl, max_new_tokens,
                                  DEFAULT_EOS_TOKEN_ID, DEFAULT_TIMEOUT)
        with lock:
            completed += 1
        pt = rec.get("prompt_tokens", 0)
        ct = rec.get("completion_tokens", 0)
        ms = rec.get("total_ms", 0)
        status = "OK" if rec.get("success") else "FAIL"
        print(f"  [{completed}/{num_requests}] pt={pt} ct={ct} {ms:.0f}ms {status}",
              file=sys.stderr)
        return rec

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_one, i) for i in range(num_requests)]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"success": False, "error": str(e),
                                "prompt_tokens": 0, "completion_tokens": 0,
                                "total_ms": 0})

    elapsed = time.time() - t_start
    ufs_samples = collector.stop()

    latency_stats = compute_latency_stats(results, elapsed)
    ufs_summary = compute_ufs_summary(ufs_samples)

    print_summary(compute_summary(ufs_samples), prefix="  ", file=sys.stderr)

    return {
        "concurrency": concurrency,
        **latency_stats,
        "ufs_summary": ufs_summary,
        "ufs_samples": ufs_samples,
        "per_request": results,
    }


def _run_bench_level_vllm(
    port: int, model: str, num_requests: int, concurrency: int,
    max_new_tokens: int, collector: VLLMStatsCollector,
) -> dict:
    """Run one concurrency level against vLLM, collecting UFS samples."""
    prompts = _build_prompts(num_requests)
    results = []
    completed = 0
    lock = threading.Lock()
    t_start = time.time()

    collector.update_request_stats(active_delta=concurrency)
    collector.start()

    def send_one(idx: int) -> dict:
        nonlocal completed
        pl = prompts[idx]
        rec = send_completion_vllm(port, model, max_new_tokens,
                                   prompt_token_ids=[1] * pl,
                                   ignore_eos=True,
                                   timeout=DEFAULT_TIMEOUT)
        with lock:
            completed += 1
        pt = rec.get("prompt_tokens", 0)
        ct = rec.get("completion_tokens", 0)
        ms = rec.get("total_ms", 0)
        status = "OK" if rec.get("success") else "FAIL"
        print(f"  [{completed}/{num_requests}] pt={pt} ct={ct} {ms:.0f}ms {status}",
              file=sys.stderr)
        collector.update_request_stats(prompt_tokens=pt, completion_tokens=ct)
        return rec

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_one, i) for i in range(num_requests)]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"success": False, "error": str(e),
                                "prompt_tokens": 0, "completion_tokens": 0,
                                "total_ms": 0})

    elapsed = time.time() - t_start
    collector.update_request_stats(active_delta=-concurrency)
    ufs_samples = collector.stop()

    latency_stats = compute_latency_stats(results, elapsed)
    ufs_summary = compute_ufs_summary(ufs_samples)

    # Only print summary if we have samples
    if ufs_samples:
        print_summary(compute_summary(ufs_samples), prefix="  ", file=sys.stderr)
    else:
        print("  (no UFS samples collected)", file=sys.stderr)

    return {
        "concurrency": concurrency,
        **latency_stats,
        "ufs_summary": ufs_summary,
        "ufs_samples": ufs_samples,
        "per_request": results,
    }


def bench_fragmentation(
    target: str, host: str, port: int, model: str = "",
    num_requests: int = DEFAULT_NUM_REQUESTS,
    concurrency_levels: Optional[List[int]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    output_dir: str = "./results/fragmentation",
    vllm_log_path: str = "",
) -> list:
    """Run fragmentation benchmark across multiple concurrency levels."""
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64]

    label = target.upper()
    print("=" * 60, file=sys.stderr)
    print(f" Fragmentation Benchmark: Concurrency Ramp ({label})", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  levels: {concurrency_levels}", file=sys.stderr)
    print(f"  num_requests per level: {num_requests}", file=sys.stderr)
    print(f"  max_new_tokens: {max_new_tokens}", file=sys.stderr)

    all_results = []

    for concurrency in concurrency_levels:
        print(f"\n{'='*40}", file=sys.stderr)
        print(f"  Level: concurrency={concurrency}", file=sys.stderr)
        print(f"{'='*40}", file=sys.stderr)

        if target == "baseline":
            collector = BaselineStatsCollector(host, port, poll_interval_s=0.2)
            level_result = _run_bench_level_baseline(
                host, port, num_requests, concurrency,
                max_new_tokens, collector,
            )
        else:
            collector = VLLMStatsCollector(
                port, poll_interval_s=0.2,
                server_log_path=vllm_log_path if vllm_log_path else None,
            )
            collector.calibrate()
            level_result = _run_bench_level_vllm(
                port, model, num_requests, concurrency,
                max_new_tokens, collector,
            )

        all_results.append(level_result)

        # Write per-level frag CSV
        ufs_samples = level_result.get("ufs_samples", [])
        if ufs_samples:
            frag_path = os.path.join(
                output_dir,
                f"{target}_stress_c{concurrency}.frag.csv"
            )
            write_frag_csv(ufs_samples, frag_path)

        # Write per-level results CSV
        per_request = level_result.get("per_request", [])
        if per_request:
            csv_path = os.path.join(
                output_dir,
                f"{target}_stress_c{concurrency}.csv"
            )
            write_results_csv(per_request, csv_path, max_new_tokens)

        time.sleep(1.0)

    return all_results


def main():
    ap = argparse.ArgumentParser(
        description="Fragmentation (UFS) Benchmark — Concurrency Ramp"
    )
    ap.add_argument("--target", type=str, choices=["baseline", "vllm"],
                    required=True, help="Which server to benchmark")
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="Server host (baseline only)")
    ap.add_argument("--port", type=int, default=8000,
                    help="Server port (baseline default: 8000, vLLM default: 8001)")
    ap.add_argument("--model", type=str,
                    default="/home/vitalrubbish/models/tinyllama",
                    help="Model name/path (vLLM only)")
    ap.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
                    help="Number of requests per concurrency level")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="Max tokens to generate per request")
    ap.add_argument("--concurrency-levels", type=str, default="1,2,4,8,16,32,64",
                    help="Comma-separated concurrency levels")
    ap.add_argument("--output-dir", type=str,
                    default="./results/fragmentation")
    ap.add_argument("--vllm-log-path", type=str, default="",
                    help="Path to vLLM server log (for block pool calibration)")
    ap.add_argument("--server-ready", action="store_true",
                    help="Server is already running; don't wait for it")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Default port per target
    if args.target == "vllm" and args.port == 8000:
        args.port = 8001

    # Check server is up
    if not args.server_ready:
        if args.target == "baseline":
            print(">>> Checking baseline server...", file=sys.stderr)
            if not wait_for_baseline_server(args.host, args.port, timeout_s=10):
                print("ERROR: Baseline server not reachable.", file=sys.stderr)
                sys.exit(1)
        else:
            print(">>> Checking vLLM server...", file=sys.stderr)
            if not wait_for_vllm_server(args.port, timeout_s=10):
                print("ERROR: vLLM server not reachable.", file=sys.stderr)
                sys.exit(1)
        print("   Server is up.", file=sys.stderr)

    # Warmup vLLM
    if args.target == "vllm":
        print(">>> Warming up vLLM...", file=sys.stderr)
        warmup_vllm(args.port, args.model)

    # Parse concurrency levels
    levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]

    # Run benchmark
    results = bench_fragmentation(
        target=args.target,
        host=args.host,
        port=args.port,
        model=args.model,
        num_requests=args.num_requests,
        concurrency_levels=levels,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        vllm_log_path=args.vllm_log_path,
    )

    # Write stress summary CSV
    summary_path = os.path.join(args.output_dir,
                                f"{args.target}_stress_summary.csv")
    write_stress_summary_csv(results, summary_path)

    # Write full JSON
    json_path = os.path.join(args.output_dir,
                             f"fragmentation_{args.target}.json")
    # Convert UFS samples to serializable format
    serializable = []
    for r in results:
        d = {k: v for k, v in r.items() if k not in ("ufs_samples", "per_request")}
        d["ufs_sample_count"] = len(r.get("ufs_samples", []))
        serializable.append(d)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*80}", file=sys.stderr)
    print(f" {args.target.upper()} Fragmentation UFS Comparison", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print_stress_comparison(results, file=sys.stderr)

    print(f"\n>>> Results written to: {args.output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
