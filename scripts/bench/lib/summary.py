"""summary.py — Unified result computation: latency stats, UFS aggregation, CSV output."""

import csv
import math
import sys
from typing import List, Dict, Any, Optional

from .ufs import UnifiedFragMetrics, UnifiedFragSummary, compute_summary


# ── Latency statistics ──

def compute_latency_stats(records: List[Dict[str, Any]],
                          elapsed_s: float) -> Dict[str, Any]:
    """Compute throughput and latency statistics from a list of per-request records.

    Args:
        records: list of dicts with keys: prompt_tokens, completion_tokens,
                 total_ms, success
        elapsed_s: total benchmark duration in seconds

    Returns:
        dict with req_s, tok_out_s, total_tok_s, mean_ms, p50_ms, p95_ms, p99_ms,
        total_input_tokens, total_output_tokens, requests_completed, requests_failed
    """
    ok = [r for r in records if r.get("success")]
    total_in = sum(r.get("prompt_tokens", 0) for r in ok)
    total_out = sum(r.get("completion_tokens", 0) for r in ok)
    latencies = sorted([r.get("total_ms", 0) for r in ok])

    def pct(pct_val: float) -> float:
        if not latencies:
            return 0.0
        idx = int(len(latencies) * pct_val / 100.0)
        return latencies[min(idx, len(latencies) - 1)]

    return {
        "requests_completed": len(ok),
        "requests_failed": len(records) - len(ok),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "request_throughput_req_s": len(ok) / max(elapsed_s, 0.001),
        "output_throughput_tok_s": total_out / max(elapsed_s, 0.001),
        "total_throughput_tok_s": (total_in + total_out) / max(elapsed_s, 0.001),
        "total_mean_ms": sum(latencies) / max(len(latencies), 1),
        "total_p50_ms": pct(50),
        "total_p95_ms": pct(95),
        "total_p99_ms": pct(99),
    }


# ── UFS summary ──

def compute_ufs_summary(samples: List[UnifiedFragMetrics]) -> Dict[str, Any]:
    """Compute UFS summary from a list of UnifiedFragMetrics samples.

    Returns a JSON-serializable dict with ifr_avg, ifr_peak, ifr_stddev, etc.
    """
    summary = compute_summary(samples)
    return {
        "sample_count": summary.sample_count,
        "ifr_avg": summary.ifr_avg,
        "ifr_peak": summary.ifr_peak,
        "ifr_stddev": summary.ifr_stddev,
        "bu_avg": summary.bu_avg,
        "bu_min": summary.bu_min,
        "bu_stddev": summary.bu_stddev,
        "pme_avg": summary.pme_avg,
        "pme_min": summary.pme_min,
        "pme_stddev": summary.pme_stddev,
        "rfi_avg": summary.rfi_avg,
        "rfi_peak": summary.rfi_peak,
        "rfi_stddev": summary.rfi_stddev,
    }


def frag_samples_to_dicts(samples: List[UnifiedFragMetrics]) -> List[Dict[str, Any]]:
    """Convert UnifiedFragMetrics list to JSON-serializable dicts."""
    return [{
        "internal_frag_rate": s.internal_frag_rate,
        "block_utilization": s.block_utilization,
        "physical_memory_efficiency": s.physical_memory_efficiency,
        "runtime_frag_index": s.runtime_frag_index,
        "active_sequences": s.active_sequences,
        "blocks_in_use": s.blocks_in_use,
        "total_blocks_allocated": s.total_blocks_allocated,
        "total_tokens": s.total_tokens,
    } for s in samples]


# ── CSV output ──

def write_results_csv(records: List[Dict[str, Any]], path: str,
                      max_new_tokens: int = 64):
    """Write per-request latency results CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "prompt_len", "max_new_tokens", "status",
                     "ttft_ms", "total_ms", "generated_tokens"])
        for i, rec in enumerate(records):
            w.writerow([
                i,
                rec.get("prompt_tokens", 0),
                max_new_tokens,
                "ok" if rec.get("success") else "fail",
                0.0,
                f"{rec.get('total_ms', 0):.2f}",
                rec.get("completion_tokens", 0),
            ])
    print(f"Wrote {len(records)} records to {path}", file=sys.stderr)


def write_frag_csv(samples: List[UnifiedFragMetrics], path: str):
    """Write fragmentation time-series CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "internal_frag_rate", "block_utilization",
            "physical_memory_efficiency", "runtime_frag_index",
            "active_sequences", "blocks_in_use", "total_blocks_allocated",
            "total_tokens", "rfi_avg", "rfi_peak", "rfi_stddev", "sample_count",
        ])
        for i, s in enumerate(samples):
            w.writerow([
                i, f"{s.internal_frag_rate:.6f}", f"{s.block_utilization:.6f}",
                f"{s.physical_memory_efficiency:.6f}", f"{s.runtime_frag_index:.6f}",
                s.active_sequences, s.blocks_in_use, s.total_blocks_allocated,
                s.total_tokens, 0.0, 0.0, 0.0, 1,
            ])
    print(f"Wrote {len(samples)} frag samples to {path}", file=sys.stderr)


def write_stress_summary_csv(results: List[Dict[str, Any]], path: str):
    """Write stress test aggregate summary CSV."""
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
            ufs = r.get("ufs_summary", {})
            w.writerow([
                r["concurrency"], r["requests_completed"], r["requests_failed"],
                f"{r['request_throughput_req_s']:.2f}",
                f"{r['output_throughput_tok_s']:.2f}",
                f"{r['total_throughput_tok_s']:.2f}",
                f"{r['total_mean_ms']:.1f}",
                f"{r['total_p50_ms']:.1f}",
                f"{r['total_p95_ms']:.1f}",
                f"{r['total_p99_ms']:.1f}",
                f"{ufs.get('ifr_avg', 0):.4f}",
                f"{ufs.get('ifr_peak', 0):.4f}",
                f"{ufs.get('ifr_stddev', 0):.4f}",
                f"{ufs.get('bu_avg', 0):.4f}",
                f"{ufs.get('bu_min', 0):.4f}",
                f"{ufs.get('bu_stddev', 0):.4f}",
                f"{ufs.get('pme_avg', 0):.4f}",
                f"{ufs.get('pme_min', 0):.4f}",
                f"{ufs.get('pme_stddev', 0):.4f}",
                f"{ufs.get('rfi_avg', 0):.4f}",
                f"{ufs.get('rfi_peak', 0):.4f}",
                f"{ufs.get('rfi_stddev', 0):.4f}",
                ufs.get("sample_count", 0),
            ])
    print(f"Wrote stress summary to {path}", file=sys.stderr)


def print_stress_comparison(results: List[Dict[str, Any]], file=None):
    """Print a formatted UFS comparison table for stress test results."""
    header = (f"{'conc':>4} {'ifr_avg':>8} {'ifr_pk':>8} {'bu_avg':>8} "
              f"{'bu_min':>8} {'pme_avg':>8} {'pme_min':>8} "
              f"{'rfi_avg':>8} {'rfi_pk':>8} {'req_s':>8} {'p95_ms':>8}")
    print(header, file=file)
    print("-" * len(header), file=file)
    for r in results:
        ufs = r.get("ufs_summary", {})
        print(f"{r['concurrency']:>4} "
              f"{ufs.get('ifr_avg', 0):>8.4f} {ufs.get('ifr_peak', 0):>8.4f} "
              f"{ufs.get('bu_avg', 0):>8.4f} {ufs.get('bu_min', 0):>8.4f} "
              f"{ufs.get('pme_avg', 0):>8.4f} {ufs.get('pme_min', 0):>8.4f} "
              f"{ufs.get('rfi_avg', 0):>8.4f} {ufs.get('rfi_peak', 0):>8.4f} "
              f"{r.get('request_throughput_req_s', 0):>8.2f} "
              f"{r.get('total_p95_ms', 0):>8.0f}", file=file)
