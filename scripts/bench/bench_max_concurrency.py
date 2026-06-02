#!/usr/bin/env python3
"""bench_max_concurrency.py — Maximum concurrent requests benchmark.

Ramps up concurrency until failures occur, measuring the maximum number of
concurrent requests (capacity at workload). Short prompts are used to maximise
admitted sequences.

Uses asyncio + aiohttp for true concurrent I/O — no ThreadPoolExecutor cap.
All requests at a given concurrency level are in-flight simultaneously.

Usage:
  python3 -m scripts.bench.bench_max_concurrency \\
    --target baseline --host 127.0.0.1 --port 8000

  python3 -m scripts.bench.bench_max_concurrency \\
    --target vllm --port 8001 --model /path/to/model

The benchmark parameters (prompt lengths, ramp schedule, max_tokens, EOS
handling, time budget) are IDENTICAL for both targets, ensuring fair comparison.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from typing import List, Optional

# Ensure project root is on sys.path so imports work from any directory
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from scripts.bench.lib.workload import (
    SHORT_PROMPT_LENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TIMEOUT,
    DEFAULT_TIME_BUDGET,
    DEFAULT_EOS_TOKEN_ID,
)
from scripts.bench.lib.protocol_baseline import (
    send_infer_baseline_async,
    query_baseline_stats,
    wait_for_baseline_server,
)
from scripts.bench.lib.protocol_vllm import (
    send_completion_vllm_async,
    wait_for_vllm_server,
    warmup_vllm,
    get_gpu_memory,
)


def _concurrency_ramp(step: int = 4) -> List[int]:
    """Build the concurrency ramp schedule.

    Fine-grained steps at low concurrency, coarser at high concurrency.
    Identical to the original scripts for backward compatibility.
    """
    levels = (
        list(range(step, min(64, 1024), step)) +
        list(range(64, 128, 16)) +
        list(range(128, 256, 32)) +
        list(range(256, 512, 64)) +
        list(range(512, 1024, 128))
    )
    return sorted(set(levels))


async def bench_max_concurrency_baseline(
    host: str, port: int, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    step: int = 4, time_budget_s: float = DEFAULT_TIME_BUDGET,
) -> dict:
    """Ramp up concurrent requests against baseline server until failures occur.

    Uses asyncio for true concurrent I/O — all requests at a concurrency level
    are in-flight simultaneously, without any thread-pool bottleneck.
    """
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Maximum Concurrent Requests (Baseline)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    max_ok = 0
    concurrency_levels = _concurrency_ramp(step)
    bench_start = time.time()
    concurrent_levels_tested = []

    for concurrency in concurrency_levels:
        if time.time() - bench_start > time_budget_s:
            print(f"    Stopping: time budget ({time_budget_s}s) exceeded",
                  file=sys.stderr)
            break

        print(f"\n  Testing concurrency={concurrency}...", file=sys.stderr)
        random.seed(42)
        prompt_lens = [random.choice(SHORT_PROMPT_LENS)
                       for _ in range(concurrency)]

        t_start = time.time()

        # Fire all requests concurrently via asyncio — no thread-pool cap
        tasks = [
            send_infer_baseline_async(
                host, port, pl, max_new_tokens,
                DEFAULT_EOS_TOKEN_ID, DEFAULT_TIMEOUT,
            )
            for pl in prompt_lens
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Normalise results: exceptions become failure dicts
        results = []
        for r in raw_results:
            if isinstance(r, Exception):
                results.append({"success": False, "error": str(r)})
            else:
                results.append(r)

        elapsed = time.time() - t_start
        ok = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        n_ok = len(ok)
        n_failed = len(failed)

        print(f"    ok={n_ok}  failed={n_failed}  elapsed={elapsed:.1f}s",
              file=sys.stderr)
        if ok:
            latencies = [r.get("total_ms", 0) for r in ok]
            print(f"    latency min/avg/max: "
                  f"{min(latencies):.0f}/{sum(latencies)/len(latencies):.0f}/"
                  f"{max(latencies):.0f} ms",
                  file=sys.stderr)

        concurrent_levels_tested.append({
            "concurrency": concurrency, "ok": n_ok, "failed": n_failed,
            "elapsed_s": elapsed,
            "ok_ratio": n_ok / max(concurrency, 1),
        })

        if n_ok > max_ok:
            max_ok = n_ok

        if n_failed > concurrency * 0.2:
            print(f"    Stopping: failure rate > 20%", file=sys.stderr)
            break

        await asyncio.sleep(0.5)

    # Query server stats after the benchmark
    stats = query_baseline_stats(host, port)
    if stats:
        print(f"\n  Server stats after benchmark:", file=sys.stderr)
        print(f"    active_sequences:      {stats.get('active_sequences', 0)}",
              file=sys.stderr)
        print(f"    blocks_in_use:         {stats.get('blocks_in_use', 0)}",
              file=sys.stderr)
        print(f"    total_blocks_allocated: {stats.get('total_blocks_allocated', 0)}",
              file=sys.stderr)
        print(f"    block_utilization:     {stats.get('block_utilization', 0):.4f}",
              file=sys.stderr)

    return {
        "target": "baseline",
        "max_concurrent_requests": max_ok,
        "concurrent_levels_tested": concurrent_levels_tested,
        "server_stats": stats,
    }


async def bench_max_concurrency_vllm(
    port: int, model: str, max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    step: int = 4, time_budget_s: float = DEFAULT_TIME_BUDGET,
) -> dict:
    """Ramp up concurrent requests against vLLM server until failures occur.

    Uses aiohttp with an unlimited-connection connector so all requests at a
    concurrency level are truly in-flight simultaneously — no 64-worker cap.
    """
    import aiohttp

    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Maximum Concurrent Requests (vLLM)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    max_ok = 0
    concurrency_levels = _concurrency_ramp(step)
    bench_start = time.time()
    concurrent_levels_tested = []

    for concurrency in concurrency_levels:
        if time.time() - bench_start > time_budget_s:
            print(f"    Stopping: time budget ({time_budget_s}s) exceeded",
                  file=sys.stderr)
            break

        print(f"\n  Testing concurrency={concurrency}...", file=sys.stderr)
        random.seed(42)
        prompt_lens = [random.choice(SHORT_PROMPT_LENS)
                       for _ in range(concurrency)]

        t_start = time.time()

        # Shared session with unlimited connections — no artificial cap
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        async with aiohttp.ClientSession(connector=connector,
                                         timeout=timeout) as session:
            tasks = [
                send_completion_vllm_async(
                    port, model, max_tokens,
                    prompt_token_ids=[1] * pl,
                    ignore_eos=True,
                    timeout=DEFAULT_TIMEOUT,
                    session=session,
                )
                for pl in prompt_lens
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Normalise results: exceptions become failure dicts
        results = []
        for r in raw_results:
            if isinstance(r, Exception):
                results.append({"success": False, "error": str(r)})
            else:
                results.append(r)

        elapsed = time.time() - t_start
        ok = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        n_ok = len(ok)
        n_failed = len(failed)

        print(f"    ok={n_ok}  failed={n_failed}  elapsed={elapsed:.1f}s",
              file=sys.stderr)

        concurrent_levels_tested.append({
            "concurrency": concurrency, "ok": n_ok, "failed": n_failed,
            "elapsed_s": elapsed,
            "ok_ratio": n_ok / max(concurrency, 1),
        })

        if n_ok > max_ok:
            max_ok = n_ok

        if n_failed > concurrency * 0.2:
            print(f"    Stopping: failure rate > 20%", file=sys.stderr)
            break

        await asyncio.sleep(0.5)

    gpu_mem = get_gpu_memory()

    return {
        "target": "vllm",
        "max_concurrent_requests": max_ok,
        "gpu_memory_limit_mib": gpu_mem.get("total_mib", 0),
        "gpu_memory_used_mib": gpu_mem.get("used_mib", 0),
        "concurrent_levels_tested": concurrent_levels_tested,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Maximum Concurrent Requests Benchmark"
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
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="Max tokens to generate per request")
    ap.add_argument("--step", type=int, default=4,
                    help="Concurrency ramp step size")
    ap.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET,
                    help="Time budget in seconds")
    ap.add_argument("--output-dir", type=str,
                    default="./results/max_concurrency")
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

    # Run benchmark (async)
    if args.target == "baseline":
        result = asyncio.run(bench_max_concurrency_baseline(
            args.host, args.port,
            max_new_tokens=args.max_new_tokens,
            step=args.step,
            time_budget_s=args.time_budget,
        ))
    else:
        result = asyncio.run(bench_max_concurrency_vllm(
            args.port, args.model,
            max_tokens=args.max_new_tokens,
            step=args.step,
            time_budget_s=args.time_budget,
        ))

    # Write results
    out_path = os.path.join(args.output_dir,
                            f"max_concurrency_{args.target}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n>>> {args.target.upper()} MAX CONCURRENT REQUESTS: "
          f"{result['max_concurrent_requests']}", file=sys.stderr)
    print(f"    Results written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
