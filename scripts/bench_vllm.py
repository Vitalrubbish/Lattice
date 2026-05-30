#!/usr/bin/env python3
"""
bench_vllm.py — Send concurrent requests to vLLM using the same prompt-length
distribution as the baseline benchmark (bench_throughput.rs).

Usage:
  python3 bench_vllm.py \
    --port 8001 \
    --model /path/to/model \
    --num-requests 100 \
    --concurrency 4 \
    --max-new-tokens 64 \
    --output-csv vllm_results.csv
"""

import argparse
import csv
import http.client
import json
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Same 145-sample sonnet-derived distribution used by examples/bench_throughput.rs
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


def send_request(port: int, model: str, max_tokens: int,
                 prompt_len: int = None, timeout: int = 300) -> dict:
    """Send one completion request; return timings + token counts."""
    if prompt_len is None:
        pl = random.choice(SONNET_PROMPT_LENS)
    else:
        pl = prompt_len
    # Build approximate-length prompt from repeated filler
    prompt = "Hello " * max(1, pl)

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
            "prompt_tokens": usage.get("prompt_tokens", pl),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_ms": elapsed_ms,
            "success": resp.status == 200,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "prompt_tokens": pl,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="vLLM Throughput Benchmark (matching baseline prompt distribution)"
    )
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", type=str, default="/home/vitalrubbish/models/tinyllama")
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Number of concurrent connections (default: 4)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output-csv", type=str, default="vllm_results.csv")
    ap.add_argument("--sequential", action="store_true",
                    help="Send requests one at a time (legacy mode)")
    args = ap.parse_args()

    random.seed(42)

    print(f"=== vLLM Throughput Benchmark ===", file=sys.stderr)
    print(f"server:       127.0.0.1:{args.port}", file=sys.stderr)
    print(f"num_requests: {args.num_requests}", file=sys.stderr)
    print(f"concurrency:  {args.concurrency}" +
          (" (sequential)" if args.sequential else ""), file=sys.stderr)
    print(f"max_new_tok:  {args.max_new_tokens}", file=sys.stderr)
    print(f"prompt dist:  {len(SONNET_PROMPT_LENS)} samples, median 42, range 8-289",
          file=sys.stderr)
    print("", file=sys.stderr)

    # Generate prompt lengths upfront for reproducibility
    prompt_lengths = [random.choice(SONNET_PROMPT_LENS)
                      for _ in range(args.num_requests)]

    if args.sequential:
        # ── Sequential mode (legacy) ──
        records = []
        t_start = time.time()

        for i in range(args.num_requests):
            rec = send_request(args.port, args.model, args.max_new_tokens,
                               prompt_len=prompt_lengths[i])
            records.append(rec)
            pt = rec["prompt_tokens"]
            ct = rec["completion_tokens"]
            ms = rec["total_ms"]
            print(f"  [{i+1}/{args.num_requests}] pt={pt} ct={ct} {ms:.0f}ms",
                  file=sys.stderr)
    else:
        # ── Concurrent mode (matches baseline bench_throughput.rs) ──
        records = []
        completed = 0
        lock = threading.Lock()
        t_start = time.time()

        def send_one(idx: int) -> dict:
            nonlocal completed
            rec = send_request(args.port, args.model, args.max_new_tokens,
                               prompt_len=prompt_lengths[idx])
            with lock:
                completed += 1
            pt = rec["prompt_tokens"]
            ct = rec["completion_tokens"]
            ms = rec["total_ms"]
            status = "OK" if rec["success"] else "FAIL"
            print(f"  [{completed}/{args.num_requests}] pt={pt} ct={ct} {ms:.0f}ms {status}",
                  file=sys.stderr)
            return rec

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(send_one, i)
                       for i in range(args.num_requests)]
            for f in as_completed(futures):
                try:
                    records.append(f.result())
                except Exception as e:
                    records.append({
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_ms": 0,
                        "success": False,
                        "error": str(e),
                    })

    elapsed = time.time() - t_start
    ok = [r for r in records if r["success"]]
    total_in = sum(r["prompt_tokens"] for r in ok)
    total_out = sum(r["completion_tokens"] for r in ok)

    latencies = sorted([r["total_ms"] for r in ok])

    def percentile(pct: float) -> float:
        if not latencies:
            return 0.0
        idx = int(len(latencies) * pct / 100.0)
        return latencies[min(idx, len(latencies) - 1)]

    print()
    print("=== Results ===")
    print(f"benchmark_duration_s:         {elapsed:.2f}")
    print(f"requests_completed:           {len(ok)}")
    print(f"requests_failed:              {len(records) - len(ok)}")
    print(f"total_input_tokens:           {total_in}")
    print(f"total_output_tokens:          {total_out}")
    print(f"request_throughput_req_s:     {len(ok) / max(elapsed, 0.001):.2f}")
    print(f"output_throughput_tok_s:      {total_out / max(elapsed, 0.001):.2f}")
    print(f"total_throughput_tok_s:       {(total_in + total_out) / max(elapsed, 0.001):.2f}")
    print("--- latency ---")
    print(f"total_mean_ms:                {sum(latencies) / max(len(latencies), 1):.2f}")
    print(f"total_p50_ms:                 {percentile(50):.2f}")
    print(f"total_p95_ms:                 {percentile(95):.2f}")
    print(f"total_p99_ms:                 {percentile(99):.2f}")

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "prompt_len", "max_new_tokens", "status",
                    "ttft_ms", "total_ms", "generated_tokens"])
        for i, rec in enumerate(records):
            w.writerow([
                i,
                rec["prompt_tokens"],
                args.max_new_tokens,
                "ok" if rec["success"] else "fail",
                0.0,  # vLLM doesn't expose TTFT via non-streaming completions
                f"{rec['total_ms']:.2f}",
                rec["completion_tokens"],
            ])

    print(f"Wrote {len(records)} records to {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
