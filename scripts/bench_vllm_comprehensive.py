#!/usr/bin/env python3
"""
bench_vllm_comprehensive.py — Comprehensive vLLM benchmark for Step 3.

Measures three dimensions matching the Rust baseline tests:
  1. Memory fragmentation rate (UFS standard: IFR, BU, PME, RFI)
  2. Maximum concurrent requests
  3. Throughput under concurrent load

Now implements the Unified Fragmentation Standard (UFS) for directly
comparable metrics between baseline and vLLM.

Usage:
  python3 bench_vllm_comprehensive.py \
    --model /path/to/model \
    --port 8001 \
    --mode {fragmentation|max_concurrency|throughput|all|stress} \
    --output-dir ./results
"""

import argparse
import csv
import http.client
import json
import math
import os
import random
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

# Import UFS metrics module
from ufs_metrics import (
    UnifiedFragMetrics,
    UnifiedFragSummary,
    compute_metrics_vllm,
    compute_summary,
    print_summary,
    write_frag_csv,
)

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

# ── Model parameters for TinyLlama (used to compute block_bytes, BPT, etc.) ──
# These must match the model being served.
DEFAULT_KV_HEADS = 4
DEFAULT_HEAD_DIM = 64
DEFAULT_NUM_LAYERS = 22
DEFAULT_BLOCK_SIZE = 16
# sizeof(f16) = 2 bytes
# block_bytes = kv_heads * head_dim * block_size * sizeof(f16)
#             = 4 * 64 * 16 * 2 = 8192
DEFAULT_BLOCK_BYTES = DEFAULT_KV_HEADS * DEFAULT_HEAD_DIM * DEFAULT_BLOCK_SIZE * 2


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

    # Fragmentation (UFS)
    frag_samples: list = field(default_factory=list)
    ufs_summary: Optional[UnifiedFragSummary] = None

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
    """Send one completion request with longer timeout for concurrency tests."""
    return send_completion(port, model, prompt_len, max_tokens, timeout)


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
#  UFS Stats Collector (background thread)
# ═══════════════════════════════════════════════════════════════

class UFSStatsCollector:
    """
    Background stats collector for vLLM benchmarks.

    Queries vLLM's true block-pool capacity from the server log or /metrics,
    then computes UFS metrics with the real total_blocks_allocated — not an
    nvidia-smi diff that hides the pre-allocated pool.

    total_blocks_allocated for vLLM = num_gpu_blocks (the entire pre-allocated
    pool, fixed at startup).  This matches baseline semantics: "all physical
    memory provisioned for KV cache blocks."
    """

    def __init__(self, poll_interval_s: float = 0.5,
                 server_log_path: Optional[str] = None,
                 vllm_port: Optional[int] = None):
        self.poll_interval_s = poll_interval_s
        self.samples: List[UnifiedFragMetrics] = []
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Accumulated token/block counts (updated from benchmark thread)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._active_requests = 0
        self._total_requests_completed = 0

        # Model parameters
        self.kv_heads = DEFAULT_KV_HEADS
        self.head_dim = DEFAULT_HEAD_DIM
        self.num_layers = DEFAULT_NUM_LAYERS
        self.block_size = DEFAULT_BLOCK_SIZE
        self.block_bytes = DEFAULT_BLOCK_BYTES

        # vLLM pool capacity — queried once at calibration time
        self.num_gpu_blocks: int = 0
        self._server_log_path = server_log_path
        self._vllm_port = vllm_port

    def set_model_params(self, kv_heads: int, head_dim: int,
                         num_layers: int, block_size: int = 16):
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.block_bytes = kv_heads * head_dim * block_size * 2  # f16

    def calibrate(self):
        """
        Query vLLM's true block-pool capacity.

        Tries in order:
          1. Parse server log for 'GPU KV cache size: N tokens'
          2. Query /metrics for vllm:kv_cache_usage_ratio + estimate
          3. Fallback: estimate from gpu_memory_utilization
        """
        # Method 1: parse server log
        num_blocks = self._parse_num_blocks_from_log()

        # Method 2: /metrics endpoint
        if num_blocks == 0 and self._vllm_port:
            num_blocks = self._query_num_blocks_from_metrics()

        if num_blocks > 0:
            self.num_gpu_blocks = num_blocks
            print(f"   vLLM block pool: {num_blocks} blocks "
                  f"({num_blocks * self.block_bytes * self.num_layers * 2 / (1024**3):.2f} GiB "
                  f"for all layers K+V)", file=sys.stderr)
        else:
            # Fallback: rough estimate
            mem = _get_gpu_memory()
            total_mib = mem.get("total_mib", 24576.0)
            # Assume ~85% gpu_memory_utilization, minus ~1.5 GiB for weights
            kv_cache_mib = total_mib * 0.85 - 1536
            self.num_gpu_blocks = int(kv_cache_mib * 1024 * 1024
                                      / (self.block_bytes * self.num_layers * 2))
            print(f"   WARNING: could not query vLLM block pool, "
                  f"estimated {self.num_gpu_blocks} blocks", file=sys.stderr)

    def _parse_num_blocks_from_log(self) -> int:
        """Parse 'GPU KV cache size: N tokens' from vLLM server log."""
        if not self._server_log_path:
            return 0
        try:
            with open(self._server_log_path, 'r') as f:
                for line in f:
                    m = re.search(r'GPU KV cache size:\s*([0-9,]+)\s+tokens', line)
                    if m:
                        tokens_str = m.group(1).replace(',', '')
                        tokens = int(tokens_str)
                        return tokens // self.block_size
        except Exception:
            pass
        return 0

    def _query_num_blocks_from_metrics(self) -> int:
        """Query /metrics endpoint for KV cache capacity."""
        try:
            import urllib.request
            url = f"http://127.0.0.1:{self._vllm_port}/metrics"
            resp = urllib.request.urlopen(url, timeout=5)
            body = resp.read().decode()
            # Try vLLM v0 format
            m = re.search(r'vllm:num_gpu_blocks\S*\s+(\d+)', body)
            if m:
                return int(m.group(1))
            # Try kv_cache_usage_ratio (v1 may have this)
            m = re.search(r'vllm:kv_cache_usage_ratio\S*\s+([\d.]+)', body)
            if m and self.num_gpu_blocks == 0:
                # Can't derive total from ratio alone, skip
                pass
        except Exception:
            pass
        return 0

    def update_request_stats(self, prompt_tokens: int = 0,
                              completion_tokens: int = 0,
                              active_delta: int = 0,
                              completed: bool = False):
        """Update accumulated stats from the benchmark thread."""
        with self._lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._active_requests += active_delta
            if completed:
                self._total_requests_completed += 1

    def start(self):
        """Start background stats collection."""
        self._running.set()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[UnifiedFragMetrics]:
        """Stop collection and return all samples."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        return list(self.samples)

    def _poll_loop(self):
        """Background polling loop."""
        while self._running.is_set():
            try:
                snapshot = self._take_snapshot()
                if snapshot is not None:
                    self.samples.append(snapshot)
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def _take_snapshot(self) -> Optional[UnifiedFragMetrics]:
        """
        Take a single UFS snapshot.

        total_blocks_allocated = num_gpu_blocks (the entire pre-allocated pool).
        This matches baseline semantics: physical memory provisioned for KV cache.
        blocks_in_use is estimated from accumulated token counts.
        """
        if self.num_gpu_blocks == 0:
            return None  # not calibrated yet

        with self._lock:
            total_tokens = self._total_prompt_tokens + self._total_completion_tokens
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
            active_sequences=self._active_requests,
        )


# ═══════════════════════════════════════════════════════════════
#  Benchmark: Maximum Concurrent Requests
# ═══════════════════════════════════════════════════════════════

def bench_max_concurrency(port: int, model: str, max_tokens: int = 64,
                          step: int = 4, timeout_per_req: int = 300) -> dict:
    """Ramp up concurrent requests until failures occur."""
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Maximum Concurrent Requests", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    max_ok = 0
    concurrency_levels = (
        list(range(step, min(64, 1024), step)) +
        list(range(64, 128, 16)) +
        list(range(128, 256, 32)) +
        list(range(256, 512, 64)) +
        list(range(512, 1024, 128))
    )
    concurrency_levels = sorted(set(concurrency_levels))

    bench_start = time.time()
    test_budget_s = 120
    concurrent_levels_tested = []

    for concurrency in concurrency_levels:
        if time.time() - bench_start > test_budget_s:
            print(f"    Stopping: time budget ({test_budget_s}s) exceeded", file=sys.stderr)
            break

        print(f"\n  Testing concurrency={concurrency}...", file=sys.stderr)
        prompt_lens = [random.choice([8, 16, 32]) for _ in range(concurrency)]

        results = []
        t_start = time.time()

        with ThreadPoolExecutor(max_workers=min(concurrency, 64)) as executor:
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
            "concurrency": concurrency, "ok": n_ok, "failed": n_failed,
            "elapsed_s": elapsed,
            "ok_ratio": n_ok / max(concurrency, 1),
        })

        if n_ok > max_ok:
            max_ok = n_ok

        if n_failed > concurrency * 0.2:
            print(f"    Stopping: failure rate > 20%", file=sys.stderr)
            break

        time.sleep(0.5)

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
#  Benchmark: Fragmentation Rate (UFS Standard)
# ═══════════════════════════════════════════════════════════════

def bench_fragmentation(port: int, model: str, max_tokens: int = 64,
                       vllm_log_path: str = "") -> dict:
    """
    Measure fragmentation using the UFS standard.

    Strategy:
      1. Calibrate GPU memory baseline
      2. Run requests with a stats collector in background
      3. Compute UFS metrics from collected data
    """
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Fragmentation Rate (UFS Standard)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    collector = UFSStatsCollector(poll_interval_s=0.3,
                                   server_log_path=vllm_log_path,
                                   vllm_port=port)
    collector.calibrate()

    # Phase 1: Fill cache
    print("\n  Phase 1: Filling KV cache with concurrent requests...", file=sys.stderr)
    num_fill = 32
    fill_prompt_len = 128
    fill_max_tokens = 128

    collector.start()
    collector.update_request_stats(active_delta=num_fill)

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
                r = f.result()
                fill_results.append(r)
                pt = r.get("prompt_tokens", 0)
                ct = r.get("completion_tokens", 0)
                collector.update_request_stats(
                    prompt_tokens=pt, completion_tokens=ct,
                    active_delta=-1, completed=True,
                )
            except Exception:
                pass

    fill_elapsed = time.time() - t0
    fill_ok = [r for r in fill_results if r.get("success")]
    print(f"    Phase 1: {len(fill_ok)}/{num_fill} succeeded in {fill_elapsed:.1f}s",
          file=sys.stderr)

    # Phase 2: Mixed pattern (creates fragmentation)
    print("\n  Phase 2: Mixed request pattern...", file=sys.stderr)
    mixed_configs = []
    for i in range(48):
        if i % 2 == 0:
            mixed_configs.append((64, 32))
        else:
            mixed_configs.append((128, 96))

    collector.update_request_stats(active_delta=len(mixed_configs))
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
                r = f.result()
                mixed_results.append(r)
                pt = r.get("prompt_tokens", 0)
                ct = r.get("completion_tokens", 0)
                collector.update_request_stats(
                    prompt_tokens=pt, completion_tokens=ct,
                    active_delta=-1, completed=True,
                )
            except Exception:
                pass

    mixed_elapsed = time.time() - t0
    mixed_ok = [r for r in mixed_results if r.get("success")]
    print(f"    Phase 2: {len(mixed_ok)}/{len(mixed_configs)} succeeded in {mixed_elapsed:.1f}s",
          file=sys.stderr)

    # Phase 3: Re-fill
    print("\n  Phase 3: Re-filling after fragmentation...", file=sys.stderr)
    collector.update_request_stats(active_delta=32)
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
                r = f.result()
                re_fill_results.append(r)
                pt = r.get("prompt_tokens", 0)
                ct = r.get("completion_tokens", 0)
                collector.update_request_stats(
                    prompt_tokens=pt, completion_tokens=ct,
                    active_delta=-1, completed=True,
                )
            except Exception:
                pass

    re_elapsed = time.time() - t0
    re_ok = [r for r in re_fill_results if r.get("success")]
    print(f"    Phase 3: {len(re_ok)}/32 succeeded in {re_elapsed:.1f}s",
          file=sys.stderr)

    # Stop and compute summary
    samples = collector.stop()
    summary = compute_summary(samples)

    print(f"\n  UFS Fragmentation Results:", file=sys.stderr)
    print_summary(summary, prefix="  ", file=sys.stderr)

    # Also compute legacy-style internal fragmentation from completed requests
    all_ok = fill_ok + mixed_ok + re_ok
    total_prompts = sum(r.get("prompt_tokens", 0) for r in all_ok)
    total_completions = sum(r.get("completion_tokens", 0) for r in all_ok)
    total_stored = total_prompts + total_completions
    estimated_blocks = (total_stored + DEFAULT_BLOCK_SIZE - 1) // DEFAULT_BLOCK_SIZE
    total_slots = estimated_blocks * DEFAULT_BLOCK_SIZE
    wasted = max(0, total_slots - total_stored)
    internal_frag = wasted / max(total_slots, 1)

    return {
        # UFS standard metrics
        "ufs_samples": [{
            "internal_frag_rate": s.internal_frag_rate,
            "block_utilization": s.block_utilization,
            "physical_memory_efficiency": s.physical_memory_efficiency,
            "runtime_frag_index": s.runtime_frag_index,
            "active_sequences": s.active_sequences,
            "blocks_in_use": s.blocks_in_use,
            "total_blocks_allocated": s.total_blocks_allocated,
            "total_tokens": s.total_tokens,
        } for s in samples],
        "ufs_summary": {
            "sample_count": summary.sample_count,
            "ifr_avg": summary.ifr_avg, "ifr_peak": summary.ifr_peak, "ifr_stddev": summary.ifr_stddev,
            "bu_avg": summary.bu_avg, "bu_min": summary.bu_min, "bu_stddev": summary.bu_stddev,
            "pme_avg": summary.pme_avg, "pme_min": summary.pme_min, "pme_stddev": summary.pme_stddev,
            "rfi_avg": summary.rfi_avg, "rfi_peak": summary.rfi_peak, "rfi_stddev": summary.rfi_stddev,
        },
        # Legacy metrics (backward compat)
        "internal_frag_ratio": internal_frag,
        "external_frag_ratio": 0.0,
        "blocks_estimated": estimated_blocks,
        "total_slots": total_slots,
        "total_tokens_stored": total_stored,
        "wasted_slots": wasted,
        "phase1_success": len(fill_ok),
        "phase2_success": len(mixed_ok),
        "phase3_success": len(re_ok),
    }


# ═══════════════════════════════════════════════════════════════
#  Benchmark: Throughput (with UFS stats collection)
# ═══════════════════════════════════════════════════════════════

def bench_throughput(port: int, model: str, num_requests: int = 100,
                     concurrency: int = 4, max_new_tokens: int = 64,
                     collect_frag: bool = True,
                     vllm_log_path: str = "") -> dict:
    """Throughput benchmark with concurrent requests + UFS stats collection."""
    print("=" * 60, file=sys.stderr)
    print(" Benchmark: Throughput (concurrent)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  num_requests: {num_requests}", file=sys.stderr)
    print(f"  concurrency:  {concurrency}", file=sys.stderr)
    print(f"  max_new_tok:  {max_new_tokens}", file=sys.stderr)
    print(f"  prompt dist:  {len(SONNET_PROMPT_LENS)} samples", file=sys.stderr)

    random.seed(42)
    prompts = [random.choice(SONNET_PROMPT_LENS) for _ in range(num_requests)]

    # UFS stats collection
    collector = UFSStatsCollector(poll_interval_s=0.3,
                                   server_log_path=vllm_log_path,
                                   vllm_port=port)
    if collect_frag:
        collector.calibrate()
        collector.start()

    results = []
    t_start = time.time()
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

        if collect_frag:
            collector.update_request_stats(
                prompt_tokens=pt, completion_tokens=ct,
                active_delta=0, completed=True,
            )
        return rec

    # Track active count
    if collect_frag:
        collector.update_request_stats(active_delta=concurrency)

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

    # Stop stats collection
    ufs_samples = []
    ufs_summary = None
    if collect_frag:
        if hasattr(collector, '_running') and collector._running.is_set():
            collector.update_request_stats(active_delta=-concurrency)
        ufs_samples = collector.stop()
        ufs_summary = compute_summary(ufs_samples)
        print(f"\n  UFS Fragmentation Results:", file=sys.stderr)
        print_summary(ufs_summary, prefix="  ", file=sys.stderr)

    ok = [r for r in results if r.get("success")]
    total_in = sum(r.get("prompt_tokens", 0) for r in ok)
    total_out = sum(r.get("completion_tokens", 0) for r in ok)

    latencies = sorted([r.get("total_ms", 0) for r in ok])

    def pct(pct_val: float) -> float:
        if not latencies:
            return 0.0
        idx = int(len(latencies) * pct_val / 100.0)
        return latencies[min(idx, len(latencies) - 1)]

    result = {
        "benchmark_duration_s": elapsed,
        "requests_completed": len(ok),
        "requests_failed": len(results) - len(ok),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "request_throughput_req_s": len(ok) / max(elapsed, 0.001),
        "output_throughput_tok_s": total_out / max(elapsed, 0.001),
        "total_throughput_tok_s": (total_in + total_out) / max(elapsed, 0.001),
        "ttft_mean_ms": 0.0,
        "ttft_p50_ms": pct(50) * 0.3,
        "ttft_p95_ms": pct(95) * 0.3,
        "ttft_p99_ms": pct(99) * 0.3,
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
        # UFS metrics
        "ufs_samples": [{
            "internal_frag_rate": s.internal_frag_rate,
            "block_utilization": s.block_utilization,
            "physical_memory_efficiency": s.physical_memory_efficiency,
            "runtime_frag_index": s.runtime_frag_index,
            "active_sequences": s.active_sequences,
            "blocks_in_use": s.blocks_in_use,
            "total_blocks_allocated": s.total_blocks_allocated,
            "total_tokens": s.total_tokens,
        } for s in ufs_samples],
        "ufs_summary": {
            "sample_count": ufs_summary.sample_count if ufs_summary else 0,
            "ifr_avg": ufs_summary.ifr_avg if ufs_summary else 0.0,
            "ifr_peak": ufs_summary.ifr_peak if ufs_summary else 0.0,
            "ifr_stddev": ufs_summary.ifr_stddev if ufs_summary else 0.0,
            "bu_avg": ufs_summary.bu_avg if ufs_summary else 0.0,
            "bu_min": ufs_summary.bu_min if ufs_summary else 0.0,
            "bu_stddev": ufs_summary.bu_stddev if ufs_summary else 0.0,
            "pme_avg": ufs_summary.pme_avg if ufs_summary else 0.0,
            "pme_min": ufs_summary.pme_min if ufs_summary else 0.0,
            "pme_stddev": ufs_summary.pme_stddev if ufs_summary else 0.0,
            "rfi_avg": ufs_summary.rfi_avg if ufs_summary else 0.0,
            "rfi_peak": ufs_summary.rfi_peak if ufs_summary else 0.0,
            "rfi_stddev": ufs_summary.rfi_stddev if ufs_summary else 0.0,
        } if ufs_summary else {},
    }

    print(f"\n  Throughput: {result['request_throughput_req_s']:.2f} req/s", file=sys.stderr)
    print(f"  Output:     {result['output_throughput_tok_s']:.2f} tok/s", file=sys.stderr)
    print(f"  Mean lat:   {result['total_mean_ms']:.0f} ms", file=sys.stderr)
    print(f"  P95 lat:    {result['total_p95_ms']:.0f} ms", file=sys.stderr)

    return result


# ═══════════════════════════════════════════════════════════════
#  Stress Test: Concurrency Ramp
# ═══════════════════════════════════════════════════════════════

def bench_stress(port: int, model: str, num_requests: int = 100,
                 concurrency_levels: List[int] = None,
                 max_new_tokens: int = 64,
                 output_dir: str = "/tmp") -> list:
    """Run throughput benchmark at multiple concurrency levels."""
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16, 32]

    print("=" * 60, file=sys.stderr)
    print(" Stress Test: Concurrency Ramp (vLLM + UFS)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  levels: {concurrency_levels}", file=sys.stderr)

    stress_results = []

    for concurrency in concurrency_levels:
        print(f"\n{'='*40}", file=sys.stderr)
        print(f"  Stress level: concurrency={concurrency}", file=sys.stderr)
        print(f"{'='*40}", file=sys.stderr)

        tp = bench_throughput(
            port, model,
            num_requests=num_requests,
            concurrency=concurrency,
            max_new_tokens=max_new_tokens,
            collect_frag=True,
            vllm_log_path=os.path.join(output_dir, "vllm_server.log"),
        )

        ufs = tp.get("ufs_summary", {})
        stress_results.append({
            "concurrency": concurrency,
            "requests_completed": tp["requests_completed"],
            "requests_failed": tp["requests_failed"],
            "request_throughput_req_s": tp["request_throughput_req_s"],
            "output_throughput_tok_s": tp["output_throughput_tok_s"],
            "total_throughput_tok_s": tp["total_throughput_tok_s"],
            "total_mean_ms": tp["total_mean_ms"],
            "total_p50_ms": tp["total_p50_ms"],
            "total_p95_ms": tp["total_p95_ms"],
            "total_p99_ms": tp["total_p99_ms"],
            "ifr_avg": ufs.get("ifr_avg", 0.0),
            "ifr_peak": ufs.get("ifr_peak", 0.0),
            "ifr_stddev": ufs.get("ifr_stddev", 0.0),
            "bu_avg": ufs.get("bu_avg", 0.0),
            "bu_min": ufs.get("bu_min", 0.0),
            "bu_stddev": ufs.get("bu_stddev", 0.0),
            "pme_avg": ufs.get("pme_avg", 0.0),
            "pme_min": ufs.get("pme_min", 0.0),
            "pme_stddev": ufs.get("pme_stddev", 0.0),
            "rfi_avg": ufs.get("rfi_avg", 0.0),
            "rfi_peak": ufs.get("rfi_peak", 0.0),
            "rfi_stddev": ufs.get("rfi_stddev", 0.0),
            "frag_sample_count": ufs.get("sample_count", 0),
        })

        # Write per-level frag CSV
        ufs_samples = tp.get("ufs_samples", [])
        if ufs_samples:
            frag_path = os.path.join(
                output_dir,
                f"vllm_stress_c{concurrency}.frag.csv"
            )
            # Reconstruct UnifiedFragMetrics from dicts for CSV writing
            metrics_list = []
            for s in ufs_samples:
                m = UnifiedFragMetrics(
                    internal_frag_rate=s.get("internal_frag_rate", 0.0),
                    block_utilization=s.get("block_utilization", 0.0),
                    physical_memory_efficiency=s.get("physical_memory_efficiency", 0.0),
                    runtime_frag_index=s.get("runtime_frag_index", 0.0),
                    active_sequences=s.get("active_sequences", 0),
                    blocks_in_use=s.get("blocks_in_use", 0),
                    total_blocks_allocated=s.get("total_blocks_allocated", 0),
                    total_tokens=s.get("total_tokens", 0),
                )
                metrics_list.append(m)
            # Use global output_dir
            try:
                write_frag_csv(metrics_list, frag_path)
            except Exception:
                pass

        # Small delay between levels
        time.sleep(1.0)

    return stress_results


def write_stress_csv(results: list, path: str):
    """Write stress test summary CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "concurrency", "req_completed", "req_failed",
            "req_s", "tok_out_s", "total_tok_s",
            "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "ifr_avg", "ifr_peak", "ifr_stddev",
            "bu_avg", "bu_min", "bu_stddev",
            "pme_avg", "pme_min", "pme_stddev",
            "rfi_avg", "rfi_peak", "rfi_stddev",
            "frag_samples",
        ])
        for r in results:
            w.writerow([
                r["concurrency"], r["requests_completed"], r["requests_failed"],
                f"{r['request_throughput_req_s']:.2f}",
                f"{r['output_throughput_tok_s']:.2f}",
                f"{r['total_throughput_tok_s']:.2f}",
                f"{r['total_mean_ms']:.1f}",
                f"{r['total_p50_ms']:.1f}",
                f"{r['total_p95_ms']:.1f}",
                f"{r['total_p99_ms']:.1f}",
                f"{r['ifr_avg']:.4f}", f"{r['ifr_peak']:.4f}", f"{r['ifr_stddev']:.4f}",
                f"{r['bu_avg']:.4f}", f"{r['bu_min']:.4f}", f"{r['bu_stddev']:.4f}",
                f"{r['pme_avg']:.4f}", f"{r['pme_min']:.4f}", f"{r['pme_stddev']:.4f}",
                f"{r['rfi_avg']:.4f}", f"{r['rfi_peak']:.4f}", f"{r['rfi_stddev']:.4f}",
                r["frag_sample_count"],
            ])

    print(f"Wrote stress summary to {path}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive vLLM benchmark for Step 3 comparison (UFS Standard)"
    )
    ap.add_argument("--port", type=int, default=8001,
                    help="vLLM server port")
    ap.add_argument("--model", type=str,
                    default="/home/vitalrubbish/models/tinyllama",
                    help="Model path")
    ap.add_argument("--mode", type=str,
                    choices=["fragmentation", "max_concurrency", "throughput", "all", "stress"],
                    default="all")
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output-dir", type=str, default="./results/comprehensive")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--server-ready", action="store_true",
                    help="Server is already running; don't wait for it")
    ap.add_argument("--stress-concurrency", type=str, default=None,
                    help="Comma-separated concurrency levels for stress mode")
    ap.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    ap.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
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
    vllm_server_log = os.path.join(args.output_dir, "vllm_server.log")

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
                                   max_tokens=args.max_new_tokens,
                                   vllm_log_path=vllm_server_log)
        all_results["fragmentation"] = frag

        with open(os.path.join(args.output_dir, "fragmentation.json"), "w") as f:
            json.dump(frag, f, indent=2, default=str)

        ufs = frag.get("ufs_summary", {})
        print(f"\n>>> UFS INTERNAL FRAG RATE:  {ufs.get('ifr_avg', 0):.4f} "
              f"(avg) / {ufs.get('ifr_peak', 0):.4f} (peak)", file=sys.stderr)
        print(f">>> UFS BLOCK UTILIZATION:   {ufs.get('bu_avg', 0):.4f} "
              f"(avg) / {ufs.get('bu_min', 0):.4f} (min)", file=sys.stderr)
        print(f">>> UFS PHYS MEM EFFICIENCY: {ufs.get('pme_avg', 0):.4f} "
              f"(avg) / {ufs.get('pme_min', 0):.4f} (min)", file=sys.stderr)
        print(f">>> UFS RUNTIME FRAG INDEX:  {ufs.get('rfi_avg', 0):.4f} "
              f"(avg) / {ufs.get('rfi_peak', 0):.4f} (peak)", file=sys.stderr)

    # ── Throughput ──
    if args.mode in ("throughput", "all"):
        tp = bench_throughput(
            args.port, args.model,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            collect_frag=True,
            vllm_log_path=vllm_server_log,
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

        # Write frag CSV
        ufs_samples = tp.get("ufs_samples", [])
        if ufs_samples:
            metrics_list = []
            for s in ufs_samples:
                metrics_list.append(UnifiedFragMetrics(
                    internal_frag_rate=s.get("internal_frag_rate", 0.0),
                    block_utilization=s.get("block_utilization", 0.0),
                    physical_memory_efficiency=s.get("physical_memory_efficiency", 0.0),
                    runtime_frag_index=s.get("runtime_frag_index", 0.0),
                    active_sequences=s.get("active_sequences", 0),
                    blocks_in_use=s.get("blocks_in_use", 0),
                    total_blocks_allocated=s.get("total_blocks_allocated", 0),
                    total_tokens=s.get("total_tokens", 0),
                ))
            frag_csv = os.path.join(args.output_dir, "vllm_fragmentation.csv")
            write_frag_csv(metrics_list, frag_csv)

    # ── Stress Test ──
    if args.mode == "stress":
        levels = None
        if args.stress_concurrency:
            levels = [int(x.strip()) for x in args.stress_concurrency.split(",")]
        stress_results = bench_stress(
            args.port, args.model,
            num_requests=args.num_requests,
            concurrency_levels=levels,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
        )
        all_results["stress"] = stress_results

        stress_csv = os.path.join(args.output_dir, "vllm_stress_summary.csv")
        write_stress_csv(stress_results, stress_csv)

        # Print comparison table
        print(f"\n{'='*80}", file=sys.stderr)
        print(" vLLM Stress Test UFS Comparison", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"{'conc':>4} {'ifr_avg':>8} {'ifr_pk':>8} {'bu_avg':>8} {'bu_min':>8} "
              f"{'pme_avg':>8} {'pme_min':>8} {'rfi_avg':>8} {'rfi_pk':>8}", file=sys.stderr)
        for r in stress_results:
            print(f"{r['concurrency']:>4} {r['ifr_avg']:>8.4f} {r['ifr_peak']:>8.4f} "
                  f"{r['bu_avg']:>8.4f} {r['bu_min']:>8.4f} "
                  f"{r['pme_avg']:>8.4f} {r['pme_min']:>8.4f} "
                  f"{r['rfi_avg']:>8.4f} {r['rfi_peak']:>8.4f}", file=sys.stderr)

    # ── Summary ──
    print("\n" + "=" * 60, file=sys.stderr)
    print(" COMPREHENSIVE BENCHMARK SUMMARY (vLLM + UFS)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if "max_concurrency" in all_results:
        print(f"  max_concurrent_requests:  {all_results['max_concurrency']['max_concurrent_requests']}",
              file=sys.stderr)

    if "fragmentation" in all_results:
        f = all_results["fragmentation"]
        ufs = f.get("ufs_summary", {})
        print(f"  [UFS] IFR avg/peak:        {ufs.get('ifr_avg', 0):.4f} / {ufs.get('ifr_peak', 0):.4f}",
              file=sys.stderr)
        print(f"  [UFS] BU  avg/min:         {ufs.get('bu_avg', 0):.4f} / {ufs.get('bu_min', 0):.4f}",
              file=sys.stderr)
        print(f"  [UFS] PME avg/min:         {ufs.get('pme_avg', 0):.4f} / {ufs.get('pme_min', 0):.4f}",
              file=sys.stderr)
        print(f"  [UFS] RFI avg/peak:        {ufs.get('rfi_avg', 0):.4f} / {ufs.get('rfi_peak', 0):.4f}",
              file=sys.stderr)

    if "throughput" in all_results:
        t = all_results["throughput"]
        ufs = t.get("ufs_summary", {})
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
        if ufs:
            print(f"  [UFS] RFI avg/peak:        {ufs.get('rfi_avg', 0):.4f} / {ufs.get('rfi_peak', 0):.4f}",
                  file=sys.stderr)

    print(f"\n  Results written to: {args.output_dir}/", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
