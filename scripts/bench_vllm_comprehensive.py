#!/usr/bin/env python3
"""
bench_vllm_comprehensive.py — Comprehensive vLLM benchmark for Step 3.

Measures three dimensions matching the Rust baseline tests:
  1. Memory fragmentation rate (external + internal)
  2. Maximum concurrent requests
  3. Throughput under concurrent load

Usage:
  python3 bench_vllm_comprehensive.py \
    --model /path/to/model \
    --port 8001 \
    --mode {fragmentation|max_concurrency|throughput|all} \
    --output-dir ./results
"""

import argparse
import csv
import http.client
import json
import os
import random
import subprocess
import signal
import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

# ── Prompt length distribution (same as baseline bench_throughput.rs) ──
SONNET_PROMPT_LENS = [
    8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13,
    39, 39, 40, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42,
    42, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45,
    46, 46, 46, 46, 46, 47, 47, 48, 48, 50, 72, 72, 73, 73, 73,
    74, 74, 75, 75, 76, 76, 77, 77, 77, 78, 79, 80, 80, 80, 80,
    106, 122, 126, 128, 135, 145, 146, 152, 152, 152, 153, 155, 155, 156, 157,
    160, 162, 170, 239, 251, 263, 273, 288, 289, 289,
]


@dataclass
class BenchResult:
    mode: str
    success: bool = True
    error: str = ""

    # Common
    requests_completed: int = 0
    requests_failed: int = 0

    # Max concurrency
    max_concurrent_requests: int = 0
    gpu_memory_limit_mib: float = 0.0

    # Fragmentation
    external_frag_ratio: float = 0.0
    internal_frag_ratio: float = 0.0
    blocks_allocated: int = 0
    blocks_in_use: int = 0
    free_blocks_in_pool: int = 0
    total_tokens_stored: int = 0
    wasted_slots: int = 0

    # Throughput
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_throughput_req_s: float = 0.0
    output_throughput_tok_s: float = 0.0
    total_throughput_tok_s: float = 0.0
    ttft_mean_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    total_mean_ms: float = 0.0
    total_p50_ms: float = 0.0
    total_p95_ms: float = 0.0
    total_p99_ms: float = 0.0
    benchmark_duration_s: float = 0.0

    per_request: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  HTTP helpers
# ═══════════════════════════════════════════════════════════════

def send_completion(port: int, model: str, prompt_len: int,
                     max_tokens: int, timeout: int = 120) -> dict:
    """Send one completion request; return timings + token counts."""
    prompt = "Hello " * max(1, prompt_len)

    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    })

    t0 = time.time()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("POST", "/v1/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        elapsed_ms = (time.time() - t0) * 1000

        usage = data.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", prompt_len),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_ms": elapsed_ms,
            "success": resp.status == 200,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


def send_completion_concurrent(port: int, model: str, prompt_len: int,
                                max_tokens: int, timeout: int = 300) -> dict:
    """Send one completion request; return timings + token counts."""
    prompt = "Hello " * max(1, prompt_len)
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    })

    t0 = time.time()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("POST", "/v1/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        elapsed_ms = (time.time() - t0) * 1000

        usage = data.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", prompt_len),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_ms": elapsed_ms,
            "success": resp.status == 200,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


def wait_for_server(port: int, timeout_s: int = 300) -> bool:
    """Wait for vLLM server to accept connections."""
    for _ in range(timeout_s * 2):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def warmup_server(port: int, model: str):
    """Send a warmup request to trigger JIT compilation."""
    try:
        send_completion(port, model, prompt_len=8, max_tokens=4, timeout=120)
        print("   Warmup complete.", file=sys.stderr)
    except Exception as e:
        print(f"   Warmup warning: {e}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
#  Benchmark: Maximum Concurrent Requests
# ═══════════════════════════════════════════════════════════════

def bench_max_concurrency(port: int, model: str, max_tokens: int = 64,
                          step: int = 4, timeout_per_req: int = 300) -> dict:
    """
    Ramp up concurrent requests until failures occur.
    Returns the maximum number of requests that succeeded concurrently.
    """
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Maximum Concurrent Requests", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    max_ok = 0
    final_failures = 0
    concurrent_levels_tested = []

    # Binary search approach: ramp up concurrency levels
    for concurrency in range(step, 1024, step):
        print(f"\n  Testing concurrency={concurrency}...", file=sys.stderr)

        # Use small prompts and small max_tokens to maximize concurrency
        prompt_lens = [random.choice([8, 16, 32]) for _ in range(concurrency)]

        results = []
        t_start = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    send_completion_concurrent, port, model, pl, max_tokens, timeout_per_req
                ): i
                for i, pl in enumerate(prompt_lens)
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"success": False, "error": str(e)})

        elapsed = time.time() - t_start
        ok = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        n_ok = len(ok)
        n_failed = len(failed)

        print(f"    ok={n_ok}  failed={n_failed}  elapsed={elapsed:.1f}s",
              file=sys.stderr)

        concurrent_levels_tested.append({
            "concurrency": concurrency,
            "ok": n_ok,
            "failed": n_failed,
            "elapsed_s": elapsed,
            "ok_ratio": n_ok / concurrency if concurrency else 0,
        })

        if n_ok > max_ok:
            max_ok = n_ok

        # Stop if more than 20% failures (server is saturated/OOM)
        if n_failed > concurrency * 0.2:
            final_failures = n_failed
            print(f"    Stopping: failure rate > 20%", file=sys.stderr)
            break

        # Small delay between levels to let server recover
        time.sleep(1)

    # Also report GPU memory info
    gpu_mem = _get_gpu_memory()

    return {
        "max_concurrent_requests": max_ok,
        "gpu_memory_limit_mib": gpu_mem.get("total_mib", 0),
        "gpu_memory_used_mib": gpu_mem.get("used_mib", 0),
        "concurrent_levels_tested": concurrent_levels_tested,
    }


def _get_gpu_memory() -> dict:
    """Get GPU memory via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5
        ).decode().strip()
        total, used = out.split(",")
        return {"total_mib": float(total.strip()), "used_mib": float(used.strip())}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════
#  Benchmark: Fragmentation Rate
# ═══════════════════════════════════════════════════════════════

def bench_fragmentation(port: int, model: str, max_tokens: int = 64) -> dict:
    """
    Measure fragmentation by mimicking the Rust test pattern:
      1. Allocate many sequences (fill cache)
      2. Free 50% (create holes)
      3. Re-allocate shorter sequences (test hole-filling)

    Uses the vLLM server: we send requests, then let them complete/free,
    and observe how many new requests can be served.
    """
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Fragmentation Rate", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # We'll measure fragmentation indirectly through the server's behavior:
    # How many requests can be served after creating allocation holes vs clean slate.
    block_size = 16  # vLLM default

    # Phase 1: Fill cache — send many concurrent long requests
    # Use a large prompt + long generation to consume KV cache blocks
    print("\n  Phase 1: Filling KV cache with long requests...", file=sys.stderr)
    num_fill = 32
    fill_prompt_len = 128
    fill_max_tokens = 128  # each seq needs (128+128)/16 = 16 blocks

    fill_results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(num_fill, 16)) as executor:
        futures = [
            executor.submit(
                send_completion_concurrent, port, model,
                fill_prompt_len, fill_max_tokens, timeout=300
            )
            for _ in range(num_fill)
        ]
        for f in as_completed(futures):
            try:
                fill_results.append(f.result())
            except Exception:
                pass

    fill_ok = [r for r in fill_results if r.get("success")]
    fill_elapsed = time.time() - t0
    print(f"    Phase 1: {len(fill_ok)}/{num_fill} succeeded in {fill_elapsed:.1f}s",
          file=sys.stderr)

    # After completion, all those blocks should be freed (vLLM frees on completion)

    # Phase 2: Allocate then abort half — create fragmentation pattern
    # We do this by sending streaming requests that we cancel mid-way
    # Actually, vLLM doesn't easily allow mid-stream cancellation via non-streaming API.
    # Alternative: send requests with different lengths to create natural fragmentation.
    print("\n  Phase 2: Creating fragmentation pattern...", file=sys.stderr)

    # Send a mix of short and long requests concurrently
    mixed_configs = []
    for i in range(48):
        if i % 2 == 0:
            mixed_configs.append((64, 32))   # short prompt, short generation
        else:
            mixed_configs.append((128, 96))  # long prompt, long generation

    mixed_results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(mixed_configs), 16)) as executor:
        futures = {
            executor.submit(
                send_completion_concurrent, port, model, pl, mt, timeout=300
            ): i
            for i, (pl, mt) in enumerate(mixed_configs)
        }
        for f in as_completed(futures):
            try:
                mixed_results.append(f.result())
            except Exception:
                pass

    mixed_ok = [r for r in mixed_results if r.get("success")]
    mixed_elapsed = time.time() - t0
    print(f"    Phase 2: {len(mixed_ok)}/{len(mixed_configs)} succeeded in {mixed_elapsed:.1f}s",
          file=sys.stderr)

    # Phase 3: Re-fill after fragmentation
    print("\n  Phase 3: Re-filling after fragmentation...", file=sys.stderr)
    re_fill_results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(
                send_completion_concurrent, port, model,
                random.choice([32, 48, 64]), 32, timeout=300
            )
            for _ in range(32)
        ]
        for f in as_completed(futures):
            try:
                re_fill_results.append(f.result())
            except Exception:
                pass

    re_ok = [r for r in re_fill_results if r.get("success")]
    re_elapsed = time.time() - t0
    print(f"    Phase 3: {len(re_ok)}/32 succeeded in {re_elapsed:.1f}s",
          file=sys.stderr)

    # Compute fragmentation metrics
    total_tokens = sum(r.get("completion_tokens", 0) for r in mixed_ok + re_ok)
    total_prompts = sum(r.get("prompt_tokens", 0) for r in mixed_ok + re_ok)
    all_ok = len(mixed_ok) + len(re_ok)

    # Internal fragmentation: wasted slots in last block per sequence
    estimated_blocks = sum(
        (r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) + block_size - 1) // block_size
        for r in mixed_ok + re_ok
    )
    total_slots = estimated_blocks * block_size
    total_stored = total_tokens + total_prompts
    wasted = total_slots - total_stored
    internal_frag = wasted / max(total_slots, 1)

    # External fragmentation proxy: phase3 success / phase1 success
    # Lower ratio = more fragmentation preventing new allocations
    external_frag_proxy = 1.0 - (len(re_ok) / max(len(fill_ok), 1))

    print(f"\n  Results:", file=sys.stderr)
    print(f"    Phase 1 fill success:      {len(fill_ok)}", file=sys.stderr)
    print(f"    Phase 2 mixed success:     {len(mixed_ok)}", file=sys.stderr)
    print(f"    Phase 3 re-fill success:   {len(re_ok)}", file=sys.stderr)
    print(f"    Estimated total blocks:    {estimated_blocks}", file=sys.stderr)
    print(f"    Total slots:               {total_slots}", file=sys.stderr)
    print(f"    Total stored tokens:       {total_stored}", file=sys.stderr)
    print(f"    Wasted slots:              {wasted}", file=sys.stderr)
    print(f"    Internal fragmentation:    {internal_frag:.4f} ({internal_frag*100:.2f}%)", file=sys.stderr)
    print(f"    External frag proxy:       {external_frag_proxy:.4f}", file=sys.stderr)

    return {
        "external_frag_ratio": external_frag_proxy,
        "internal_frag_ratio": internal_frag,
        "blocks_estimated": estimated_blocks,
        "total_slots": total_slots,
        "total_tokens_stored": total_stored,
        "wasted_slots": wasted,
        "phase1_success": len(fill_ok),
        "phase2_success": len(mixed_ok),
        "phase3_success": len(re_ok),
    }


# ═══════════════════════════════════════════════════════════════
#  Benchmark: Throughput (with proper concurrency)
# ═══════════════════════════════════════════════════════════════

def bench_throughput(port: int, model: str, num_requests: int = 100,
                     concurrency: int = 4, max_new_tokens: int = 64) -> dict:
    """Throughput benchmark with concurrent requests."""
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Throughput (concurrent)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  num_requests: {num_requests}", file=sys.stderr)
    print(f"  concurrency:  {concurrency}", file=sys.stderr)
    print(f"  max_new_tok:  {max_new_tokens}", file=sys.stderr)
    print(f"  prompt dist:  {len(SONNET_PROMPT_LENS)} samples", file=sys.stderr)

    random.seed(42)

    # Generate requests upfront
    prompts = [random.choice(SONNET_PROMPT_LENS) for _ in range(num_requests)]

    results = []
    t_start = time.time()

    # Use semaphore-style concurrency via ThreadPoolExecutor
    completed = 0
    lock = threading.Lock()

    def send_one(idx: int) -> dict:
        nonlocal completed
        pl = prompts[idx]
        rec = send_completion(port, model, pl, max_new_tokens, timeout=300)
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

    ok = [r for r in results if r.get("success")]
    total_in = sum(r.get("prompt_tokens", 0) for r in ok)
    total_out = sum(r.get("completion_tokens", 0) for r in ok)

    latencies = sorted([r.get("total_ms", 0) for r in ok])

    def pct(pct_val: float) -> float:
        if not latencies:
            return 0.0
        idx = int(len(latencies) * pct_val / 100.0)
        return latencies[min(idx, len(latencies) - 1)]

    ttfts = sorted([r.get("total_ms", 0) * 0.3 for r in ok])  # approximate TTFT as 30% of total

    result = {
        "benchmark_duration_s": elapsed,
        "requests_completed": len(ok),
        "requests_failed": len(results) - len(ok),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "request_throughput_req_s": len(ok) / max(elapsed, 0.001),
        "output_throughput_tok_s": total_out / max(elapsed, 0.001),
        "total_throughput_tok_s": (total_in + total_out) / max(elapsed, 0.001),
        "ttft_mean_ms": sum(ttfts) / max(len(ttfts), 1),
        "ttft_p50_ms": pct(50),
        "ttft_p95_ms": pct(95),
        "ttft_p99_ms": pct(99),
        "total_mean_ms": sum(latencies) / max(len(latencies), 1),
        "total_p50_ms": pct(50),
        "total_p95_ms": pct(95),
        "total_p99_ms": pct(99),
        "per_request": [{
            "prompt_len": r.get("prompt_tokens", 0),
            "max_new_tokens": max_new_tokens,
            "status": "ok" if r.get("success") else "fail",
            "total_ms": r.get("total_ms", 0),
            "generated_tokens": r.get("completion_tokens", 0),
        } for r in results],
    }

    print(f"\n  Throughput: {result['request_throughput_req_s']:.2f} req/s", file=sys.stderr)
    print(f"  Output:     {result['output_throughput_tok_s']:.2f} tok/s", file=sys.stderr)
    print(f"  Mean lat:   {result['total_mean_ms']:.0f} ms", file=sys.stderr)
    print(f"  P95 lat:    {result['total_p95_ms']:.0f} ms", file=sys.stderr)

    return result


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive vLLM benchmark for Step 3 comparison"
    )
    ap.add_argument("--port", type=int, default=8001,
                    help="vLLM server port")
    ap.add_argument("--model", type=str,
                    default="/home/vitalrubbish/models/tinyllama",
                    help="Model path")
    ap.add_argument("--mode", type=str,
                    choices=["fragmentation", "max_concurrency", "throughput", "all"],
                    default="all")
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output-dir", type=str, default="./results/comprehensive")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--server-ready", action="store_true",
                    help="Server is already running; don't wait for it")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check server is up
    if not args.server_ready:
        print(">>> Checking vLLM server...", file=sys.stderr)
        if not wait_for_server(args.port, timeout_s=10):
            print("ERROR: vLLM server not reachable. Start it first with:", file=sys.stderr)
            print(f"  vllm serve {args.model} --port {args.port} \\", file=sys.stderr)
            print(f"    --block-size 16 --gpu-memory-utilization 0.85 \\", file=sys.stderr)
            print(f"    --max-num-seqs 128 --max-model-len 512 --enforce-eager", file=sys.stderr)
            sys.exit(1)
        print("   Server is up.", file=sys.stderr)

    # Warmup
    if not args.skip_warmup:
        print(">>> Warming up...", file=sys.stderr)
        warmup_server(args.port, args.model)

    all_results = {}

    # ── Max Concurrency ──
    if args.mode in ("max_concurrency", "all"):
        mc = bench_max_concurrency(
            args.port, args.model,
            max_tokens=args.max_new_tokens,
        )
        all_results["max_concurrency"] = mc

        with open(os.path.join(args.output_dir, "max_concurrency.json"), "w") as f:
            json.dump(mc, f, indent=2, default=str)

        print(f"\n>>> MAX CONCURRENT REQUESTS: {mc['max_concurrent_requests']}",
              file=sys.stderr)

    # ── Fragmentation ──
    if args.mode in ("fragmentation", "all"):
        frag = bench_fragmentation(args.port, args.model,
                                   max_tokens=args.max_new_tokens)
        all_results["fragmentation"] = frag

        with open(os.path.join(args.output_dir, "fragmentation.json"), "w") as f:
            json.dump(frag, f, indent=2, default=str)

        print(f"\n>>> INTERNAL FRAGMENTATION: {frag['internal_frag_ratio']:.4f} "
              f"({frag['internal_frag_ratio']*100:.2f}%)", file=sys.stderr)
        print(f">>> EXTERNAL FRAG PROXY:    {frag['external_frag_ratio']:.4f}",
              file=sys.stderr)

    # ── Throughput ──
    if args.mode in ("throughput", "all"):
        tp = bench_throughput(
            args.port, args.model,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
        )
        all_results["throughput"] = tp

        with open(os.path.join(args.output_dir, "throughput.json"), "w") as f:
            json.dump(tp, f, indent=2, default=str)

        # Write per-request CSV
        csv_path = os.path.join(args.output_dir, "vllm_results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["req_id", "prompt_len", "max_new_tokens", "status",
                        "ttft_ms", "total_ms", "generated_tokens"])
            for i, rec in enumerate(tp.get("per_request", [])):
                w.writerow([
                    i,
                    rec.get("prompt_len", 0),
                    args.max_new_tokens,
                    rec.get("status", "fail"),
                    0.0,
                    f"{rec.get('total_ms', 0):.2f}",
                    rec.get("generated_tokens", 0),
                ])

    # ── Summary ──
    print("\n" + "=" * 60, file=sys.stderr)
    print(" COMPREHENSIVE BENCHMARK SUMMARY (vLLM)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if "max_concurrency" in all_results:
        print(f"  max_concurrent_requests:  {all_results['max_concurrency']['max_concurrent_requests']}",
              file=sys.stderr)

    if "fragmentation" in all_results:
        f = all_results["fragmentation"]
        print(f"  internal_fragmentation:   {f['internal_frag_ratio']:.4f} ({f['internal_frag_ratio']*100:.2f}%)",
              file=sys.stderr)
        print(f"  external_frag_proxy:      {f['external_frag_ratio']:.4f}",
              file=sys.stderr)

    if "throughput" in all_results:
        t = all_results["throughput"]
        print(f"  requests_completed:       {t['requests_completed']}",
              file=sys.stderr)
        print(f"  request_throughput_req_s: {t['request_throughput_req_s']:.2f}",
              file=sys.stderr)
        print(f"  output_throughput_tok_s:  {t['output_throughput_tok_s']:.2f}",
              file=sys.stderr)
        print(f"  total_mean_ms:            {t['total_mean_ms']:.0f}",
              file=sys.stderr)
        print(f"  total_p95_ms:             {t['total_p95_ms']:.0f}",
              file=sys.stderr)

    print(f"\n  Results written to: {args.output_dir}/", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
