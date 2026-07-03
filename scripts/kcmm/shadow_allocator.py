"""KCMM shadow allocator state for vLLM Phase II.A.

The shadow allocator mirrors vLLM GPU block lifetimes into KCMM while leaving
vLLM's native block IDs and KV tensors untouched.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bindings import KcmmError, KcmmPool


@dataclass
class ShadowBlock:
    vllm_block_id: int
    kcmm_block_id: int
    refcount: int
    source: str


class ShadowAllocationTracker:
    """Mirror vLLM GPU block allocation/free events into KCMM."""

    def __init__(self, report_path: str | None = None):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._lock = threading.RLock()
        self._blocks: dict[int, ShadowBlock] = {}
        self._native_allocations = 0
        self._native_frees = 0
        self._kcmm_allocations = 0
        self._kcmm_frees = 0
        self._ref_increments = 0
        self._ref_decrements = 0
        self._error_count = 0
        self._last_error: str | None = None

    def attach_pool(self, pool: KcmmPool) -> None:
        with self._lock:
            self._pool = pool
            self.write_report()

    def _require_pool(self) -> KcmmPool:
        if self._pool is None:
            raise KcmmError("KCMM shadow allocator saw a vLLM GPU block before pool creation")
        return self._pool

    def _record_error(self, exc: BaseException) -> None:
        self._error_count += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self.write_report()

    def mirror_allocated_blocks(self, blocks: list[Any], source: str) -> None:
        for block in blocks:
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                raise KcmmError(f"{source}: vLLM returned a block without block_id")
            self.mirror_allocated_block(int(block_id), source)

    def mirror_allocated_block(self, vllm_block_id: int, source: str) -> None:
        with self._lock:
            try:
                self._native_allocations += 1
                existing = self._blocks.get(vllm_block_id)
                if existing is not None:
                    existing.refcount += 1
                    self._ref_increments += 1
                    self.write_report()
                    return

                pool = self._require_pool()
                allocated = pool.alloc_blocks(1)
                if len(allocated) != 1:
                    raise KcmmError(
                        f"{source}: kcmm_alloc_blocks returned {len(allocated)} blocks"
                    )
                self._blocks[vllm_block_id] = ShadowBlock(
                    vllm_block_id=vllm_block_id,
                    kcmm_block_id=allocated[0],
                    refcount=1,
                    source=source,
                )
                self._kcmm_allocations += 1
                self.write_report()
            except BaseException as exc:
                self._record_error(exc)
                raise

    def mirror_freed_block(self, vllm_block_id: int, source: str) -> None:
        with self._lock:
            try:
                self._native_frees += 1
                existing = self._blocks.get(vllm_block_id)
                if existing is None:
                    raise KcmmError(
                        f"{source}: vLLM freed unknown GPU block {vllm_block_id}"
                    )
                if existing.refcount <= 0:
                    raise KcmmError(
                        f"{source}: invalid shadow refcount {existing.refcount} "
                        f"for vLLM block {vllm_block_id}"
                    )
                existing.refcount -= 1
                self._ref_decrements += 1
                if existing.refcount == 0:
                    pool = self._require_pool()
                    pool.free_blocks([existing.kcmm_block_id])
                    self._kcmm_frees += 1
                    del self._blocks[vllm_block_id]
                self.write_report()
            except BaseException as exc:
                self._record_error(exc)
                raise

    def report(self) -> dict[str, Any]:
        with self._lock:
            pool_stats: dict[str, int | float] | None = None
            if self._pool is not None:
                try:
                    pool_stats = self._pool.stats()
                except Exception as exc:
                    pool_stats = {"error": str(exc)}
            outstanding = [
                asdict(block)
                for block in sorted(
                    self._blocks.values(), key=lambda item: item.vllm_block_id
                )
            ]
            return {
                "phase": "II.A",
                "mode": "shadow_allocator",
                "native_gpu_allocations": self._native_allocations,
                "native_gpu_frees": self._native_frees,
                "kcmm_allocations": self._kcmm_allocations,
                "kcmm_frees": self._kcmm_frees,
                "shadow_ref_increments": self._ref_increments,
                "shadow_ref_decrements": self._ref_decrements,
                "outstanding_mappings": len(outstanding),
                "outstanding": outstanding[:32],
                "error_count": self._error_count,
                "last_error": self._last_error,
                "pool_attached": self._pool is not None,
                "pool_stats": pool_stats,
            }

    def write_report(self) -> None:
        if self._report_path is None:
            return
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
