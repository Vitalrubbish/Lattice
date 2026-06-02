#!/usr/bin/env python3
"""bench_throughput.py — Throughput benchmark at fixed concurrency.

Measures request throughput, token throughput, and latency percentiles under
a fixed number of concurrent workers.

Usage:
  python3 -m scripts.bench.bench_throughput \\
    --target baseline --host 127.0.0.1 --port 8000 --num-requests 100

  python3 -m scripts.bench.bench_throughput \\
    --target vllm --port 8001 --model /path/to/model --num-requests 100
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Ensure project root is on sys.path so imports work from any directory
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from scripts.bench.lib.workload import (
    SONNET_PROMPT_LENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_REQUESTS,
    DEFAULT_CONCURRENCY,
    DEFAULT_TIMEOUT,
    DEFAULT_EOS_TOKEN_ID,
)
from scripts.bench.lib.protocol_baseline import (
    send_infer_baseline,
    wait_for_baseline_server,
)
from scripts.bench.lib.protocol_vllm import (
    send_completion_vllm,
    wait_for_vllm_server,
    warmup_vllm,
)
from scripts.bench.lib.summary import compute_latency_stats, write_results_csv


def _build_prompts(num_requests: int) -> List[int]:
    """Generate prompt lengths from the sonnet distribution."""
    random.seed(42)
    return [random.choice(SONNET_PROMPT_LENS) for _ in range(num_requests)]


def bench_throughput_baseline(
    host: str, port: int, num_requests: int = DEFAULT_NUM_REQUESTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict:
    """Run throughput benchmark against baseline server."""
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Throughput (Baseline)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  num_requests: {num_requests}", file=sys.stderr)
    print(f"  concurrency:  {concurrency}", file=sys.stderr)
    print(f"  max_new_tok:  {max_new_tokens}", file=sys.stderr)

    prompts = _build_prompts(num_requests)
    results = []
    completed = 0
    lock = threading.Lock()
    t_start = time.time()

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
    stats = compute_latency_stats(results, elapsed)

    print(f"\n  Throughput: {stats['request_throughput_req_s']:.2f} req/s", file=sys.stderr)
    print(f"  Output:     {stats['output_throughput_tok_s']:.2f} tok/s", file=sys.stderr)
    print(f"  Mean lat:   {stats['total_mean_ms']:.0f} ms", file=sys.stderr)
    print(f"  P95 lat:    {stats['total_p95_ms']:.0f} ms", file=sys.stderr)

    return {
        "target": "baseline",
        "benchmark_duration_s": elapsed,
        **stats,
        "per_request": [{
            "prompt_len": r.get("prompt_tokens", 0),
            "max_new_tokens": max_new_tokens,
            "status": "ok" if r.get("success") else "fail",
            "total_ms": r.get("total_ms", 0),
            "generated_tokens": r.get("completion_tokens", 0),
        } for r in results],
    }


def bench_throughput_vllm(
    port: int, model: str, num_requests: int = DEFAULT_NUM_REQUESTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict:
    """Run throughput benchmark against vLLM server."""
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Throughput (vLLM)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  num_requests: {num_requests}", file=sys.stderr)
    print(f"  concurrency:  {concurrency}", file=sys.stderr)
    print(f"  max_new_tok:  {max_new_tokens}", file=sys.stderr)

    prompts = _build_prompts(num_requests)
    results = []
    completed = 0
    lock = threading.Lock()
    t_start = time.time()

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
    stats = compute_latency_stats(results, elapsed)

    print(f"\n  Throughput: {stats['request_throughput_req_s']:.2f} req/s", file=sys.stderr)
    print(f"  Output:     {stats['output_throughput_tok_s']:.2f} tok/s", file=sys.stderr)
    print(f"  Mean lat:   {stats['total_mean_ms']:.0f} ms", file=sys.stderr)
    print(f"  P95 lat:    {stats['total_p95_ms']:.0f} ms", file=sys.stderr)

    return {
        "target": "vllm",
        "benchmark_duration_s": elapsed,
        **stats,
        "per_request": [{
            "prompt_len": r.get("prompt_tokens", 0),
            "max_new_tokens": max_new_tokens,
            "status": "ok" if r.get("success") else "fail",
            "total_ms": r.get("total_ms", 0),
            "generated_tokens": r.get("completion_tokens", 0),
        } for r in results],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Throughput Benchmark"
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
                    help="Number of requests to send")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="Number of concurrent workers")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="Max tokens to generate per request")
    ap.add_argument("--output-dir", type=str,
                    default="./results/throughput")
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

    # Run benchmark
    if args.target == "baseline":
        result = bench_throughput_baseline(
            args.host, args.port,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = bench_throughput_vllm(
            args.port, args.model,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
        )

    # Write results
    out_path = os.path.join(args.output_dir,
                            f"throughput_{args.target}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Write per-request CSV
    csv_path = os.path.join(args.output_dir,
                            f"throughput_{args.target}.csv")
    write_results_csv(result["per_request"], csv_path, args.max_new_tokens)

    print(f"\n>>> {args.target.upper()} THROUGHPUT: "
          f"{result['request_throughput_req_s']:.2f} req/s", file=sys.stderr)
    print(f"    Results written to: {args.output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
