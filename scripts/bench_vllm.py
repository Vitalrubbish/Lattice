#!/usr/bin/env python3
"""bench_vllm.py — Send requests to vLLM using the same prompt-length distribution as the baseline benchmark."""

import argparse
import csv
import http.client
import json
import random
import sys
import time

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


def send_request(port: int, model: str, max_tokens: int) -> dict:
    """Send one completion request; return timings + token counts."""
    pl = random.choice(SONNET_PROMPT_LENS)
    # Build approximate-length prompt from repeated filler
    prompt = "Hello " * max(1, pl)

    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    })

    t0 = time.time()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", type=str, default="/home/vitalrubbish/models/tinyllama")
    ap.add_argument("--num-requests", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output-csv", type=str, default="vllm_results.csv")
    args = ap.parse_args()

    random.seed(42)

    print(f"=== vLLM Throughput Benchmark ===", file=sys.stderr)
    print(f"server:       127.0.0.1:{args.port}", file=sys.stderr)
    print(f"num_requests: {args.num_requests}", file=sys.stderr)
    print(f"max_new_tok:  {args.max_new_tokens}", file=sys.stderr)
    print(f"prompt dist:  {len(SONNET_PROMPT_LENS)} samples, median 42, range 8-289", file=sys.stderr)
    print("", file=sys.stderr)

    records = []
    t_start = time.time()

    for i in range(args.num_requests):
        rec = send_request(args.port, args.model, args.max_new_tokens)
        records.append(rec)
        pt = rec["prompt_tokens"]
        ct = rec["completion_tokens"]
        ms = rec["total_ms"]
        print(f"  [{i+1}/{args.num_requests}] pt={pt} ct={ct} {ms:.0f}ms", file=sys.stderr)

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
    print(f"request_throughput_req_s:     {len(ok) / elapsed:.2f}")
    print(f"output_throughput_tok_s:      {total_out / elapsed:.2f}")
    print(f"total_throughput_tok_s:       {(total_in + total_out) / elapsed:.2f}")
    print("--- latency ---")
    print(f"total_mean_ms:                {sum(latencies) / max(len(latencies), 1):.2f}")
    print(f"total_p50_ms:                 {percentile(50):.2f}")
    print(f"total_p95_ms:                 {percentile(95):.2f}")
    print(f"total_p99_ms:                 {percentile(99):.2f}")

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "prompt_len", "max_new_tokens", "status", "ttft_ms", "total_ms", "generated_tokens"])
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
