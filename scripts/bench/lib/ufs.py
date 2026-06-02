"""ufs.py — Unified Fragmentation Standard (UFS) for KV cache benchmarks.

Provides the same 4 fragmentation metrics used by the Rust baseline
(src/cache/unified_frag.rs), so both baseline and vLLM benchmarks produce
directly comparable results.

## Core Metrics

1. IFR — Internal Fragmentation Rate
   (total_slots - total_tokens) / total_slots
   Waste within partially-filled blocks (last-block problem).
   Range: [0, 1).  Directly comparable across systems.

2. BU  — Block Utilization
   blocks_in_use / total_blocks_allocated
   How efficiently the block pool is utilized.
   Range: [0, 1].  Directly comparable across systems.

3. PME — Physical Memory Efficiency
   total_tokens × BPT / actual_physical_bytes
   Physical memory waste from allocator granularity + pool underutilization.
   Range: (0, 1].  System-specific actual_physical_bytes formula.

4. RFI — Runtime Fragmentation Index
   1 - (total_tokens × BPT / actual_active_bytes)
   Combined internal+external waste during active processing.
   Range: [0, 1).  System-specific actual_active_bytes formula.

## System-specific actual bytes

  vLLM (PyTorch allocator — no superblocks):
    actual_physical_bytes = total_blocks_allocated × block_bytes × num_layers × 2
    actual_active_bytes   = blocks_in_use × block_bytes × num_layers × 2

  Baseline (CUDA VMM with 2 MiB superblocks):
    actual_physical_bytes = superblock_count × 2 MiB × num_layers × 2
    actual_active_bytes   = ceil(blocks_in_use / blocks_per_sb) × 2 MiB × num_layers × 2
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UnifiedFragMetrics:
    """A single snapshot of unified fragmentation metrics at one point in time."""
    # Directly comparable (formula identical across systems)
    internal_frag_rate: float = 0.0
    block_utilization: float = 0.0

    # System-specific (formula documented, same metric name)
    physical_memory_efficiency: float = 0.0
    runtime_frag_index: float = 0.0

    # Raw counts for verification / debugging
    active_sequences: int = 0
    blocks_in_use: int = 0
    total_blocks_allocated: int = 0
    total_tokens: int = 0
    ideal_physical_bytes: int = 0
    actual_physical_bytes: int = 0


@dataclass
class UnifiedFragSummary:
    """Time-series aggregation of unified fragmentation metrics."""
    sample_count: int = 0

    # IFR
    ifr_avg: float = 0.0
    ifr_peak: float = 0.0
    ifr_stddev: float = 0.0

    # BU
    bu_avg: float = 0.0
    bu_min: float = 0.0
    bu_stddev: float = 0.0

    # PME
    pme_avg: float = 0.0
    pme_min: float = 0.0
    pme_stddev: float = 0.0

    # RFI
    rfi_avg: float = 0.0
    rfi_peak: float = 0.0
    rfi_stddev: float = 0.0


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _stddev(values: List[float], mean: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


def compute_metrics(
    *,
    block_size: int,
    blocks_in_use: int,
    total_blocks_allocated: int,
    total_blocks_used_by_seqs: int,
    total_tokens: int,
    block_bytes: int,
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    actual_physical_bytes: int,
    actual_active_bytes: int,
    active_sequences: int = 0,
) -> UnifiedFragMetrics:
    """Compute unified fragmentation metrics from raw values.

    For vLLM (PyTorch allocator — no superblocks):
      use compute_metrics_vllm() instead.

    For baseline (CUDA VMM):
      use compute_metrics_baseline() instead.
    """
    total_slots = total_blocks_used_by_seqs * block_size

    # IFR: internal fragmentation
    internal_frag_rate = (total_slots - total_tokens) / max(total_slots, 1)

    # BU: block utilization
    block_utilization = blocks_in_use / max(total_blocks_allocated, 1)

    # PME: physical memory efficiency
    bpt_all = kv_heads * head_dim * 2 * num_layers * 2
    ideal_physical_bytes = total_tokens * bpt_all
    physical_memory_efficiency = ideal_physical_bytes / max(actual_physical_bytes, 1)

    # RFI: runtime fragmentation index
    ideal_active_bytes = total_tokens * bpt_all
    runtime_frag_index = 0.0
    if actual_active_bytes > 0:
        runtime_frag_index = 1.0 - (ideal_active_bytes / actual_active_bytes)

    return UnifiedFragMetrics(
        internal_frag_rate=internal_frag_rate,
        block_utilization=block_utilization,
        physical_memory_efficiency=physical_memory_efficiency,
        runtime_frag_index=runtime_frag_index,
        active_sequences=active_sequences,
        blocks_in_use=blocks_in_use,
        total_blocks_allocated=total_blocks_allocated,
        total_tokens=total_tokens,
        ideal_physical_bytes=ideal_physical_bytes,
        actual_physical_bytes=actual_physical_bytes,
    )


def compute_metrics_vllm(
    *,
    block_size: int,
    blocks_in_use: int,
    total_blocks_allocated: int,
    total_blocks_used_by_seqs: int,
    total_tokens: int,
    block_bytes: int,
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    active_sequences: int = 0,
) -> UnifiedFragMetrics:
    """Compute unified fragmentation metrics for vLLM.
    vLLM uses PyTorch's CUDA allocator — no superblock granularity.
    """
    actual_physical = total_blocks_allocated * block_bytes * num_layers * 2
    actual_active = blocks_in_use * block_bytes * num_layers * 2

    return compute_metrics(
        block_size=block_size,
        blocks_in_use=blocks_in_use,
        total_blocks_allocated=total_blocks_allocated,
        total_blocks_used_by_seqs=total_blocks_used_by_seqs,
        total_tokens=total_tokens,
        block_bytes=block_bytes,
        num_layers=num_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        actual_physical_bytes=actual_physical,
        actual_active_bytes=actual_active,
        active_sequences=active_sequences,
    )


def compute_metrics_baseline(
    *,
    block_size: int,
    blocks_in_use: int,
    total_blocks_allocated: int,
    total_blocks_used_by_seqs: int,
    total_tokens: int,
    block_bytes: int,
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    superblock_count: int,
    blocks_per_superblock: int,
    active_sequences: int = 0,
) -> UnifiedFragMetrics:
    """Compute unified fragmentation metrics for baseline (CUDA VMM).

    The baseline uses 2 MiB superblocks. Physical memory is allocated at
    superblock granularity via cuMemCreate. actual_active_bytes rounds
    blocks_in_use up to superblock boundaries.

    Mirrors Rust: src/cache/unified_frag.rs:96-176
    """
    SUPERBLOCK_SIZE = 2 * 1024 * 1024  # 2 MiB

    # actual_physical_bytes = all superblocks (reserved VA regions)
    actual_physical = superblock_count * SUPERBLOCK_SIZE * num_layers * 2

    # actual_active_bytes = superblocks that contain at least one used block
    active_superblocks = _round_up(blocks_in_use, blocks_per_superblock) // blocks_per_superblock
    actual_active = active_superblocks * SUPERBLOCK_SIZE * num_layers * 2

    return compute_metrics(
        block_size=block_size,
        blocks_in_use=blocks_in_use,
        total_blocks_allocated=total_blocks_allocated,
        total_blocks_used_by_seqs=total_blocks_used_by_seqs,
        total_tokens=total_tokens,
        block_bytes=block_bytes,
        num_layers=num_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        actual_physical_bytes=actual_physical,
        actual_active_bytes=actual_active,
        active_sequences=active_sequences,
    )


def compute_summary(samples: List[UnifiedFragMetrics]) -> UnifiedFragSummary:
    """Compute summary statistics from a time series of UnifiedFragMetrics samples."""
    if not samples:
        return UnifiedFragSummary()

    n = len(samples)

    # IFR
    ifr_values = [s.internal_frag_rate for s in samples]
    ifr_avg = _mean(ifr_values)
    ifr_peak = max(ifr_values)
    ifr_stddev = _stddev(ifr_values, ifr_avg)

    # BU
    bu_values = [s.block_utilization for s in samples]
    bu_avg = _mean(bu_values)
    bu_min = min(bu_values)
    bu_stddev = _stddev(bu_values, bu_avg)

    # PME
    pme_values = [s.physical_memory_efficiency for s in samples]
    pme_avg = _mean(pme_values)
    pme_min = min(pme_values)
    pme_stddev = _stddev(pme_values, pme_avg)

    # RFI
    rfi_values = [s.runtime_frag_index for s in samples]
    rfi_avg = _mean(rfi_values)
    rfi_peak = max(rfi_values)
    rfi_stddev = _stddev(rfi_values, rfi_avg)

    return UnifiedFragSummary(
        sample_count=n,
        ifr_avg=ifr_avg, ifr_peak=ifr_peak, ifr_stddev=ifr_stddev,
        bu_avg=bu_avg, bu_min=bu_min, bu_stddev=bu_stddev,
        pme_avg=pme_avg, pme_min=pme_min, pme_stddev=pme_stddev,
        rfi_avg=rfi_avg, rfi_peak=rfi_peak, rfi_stddev=rfi_stddev,
    )


def print_summary(summary: UnifiedFragSummary, prefix: str = "", file=None):
    """Print a human-readable UFS summary."""
    print(f"{prefix}--- unified fragmentation (UFS) ---", file=file)
    print(f"{prefix}frag_sample_count:          {summary.sample_count}", file=file)
    print(f"{prefix}ifr_avg:                    {summary.ifr_avg:.4f}", file=file)
    print(f"{prefix}ifr_peak:                   {summary.ifr_peak:.4f}", file=file)
    print(f"{prefix}ifr_stddev:                 {summary.ifr_stddev:.4f}", file=file)
    print(f"{prefix}bu_avg:                     {summary.bu_avg:.4f}", file=file)
    print(f"{prefix}bu_min:                     {summary.bu_min:.4f}", file=file)
    print(f"{prefix}bu_stddev:                  {summary.bu_stddev:.4f}", file=file)
    print(f"{prefix}pme_avg:                    {summary.pme_avg:.4f}", file=file)
    print(f"{prefix}pme_min:                    {summary.pme_min:.4f}", file=file)
    print(f"{prefix}pme_stddev:                 {summary.pme_stddev:.4f}", file=file)
    print(f"{prefix}rfi_avg:                    {summary.rfi_avg:.4f}", file=file)
    print(f"{prefix}rfi_peak:                   {summary.rfi_peak:.4f}", file=file)
    print(f"{prefix}rfi_stddev:                 {summary.rfi_stddev:.4f}", file=file)
