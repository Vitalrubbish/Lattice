"""KCMM-backed allocator state for vLLM Phase II.A.

This mode lets KCMM choose GPU block IDs while vLLM native KV tensors remain the
storage of record. A KCMM block ID is only accepted if it is also a free native
vLLM GPU block ID.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bindings import KcmmError, KcmmPool


@dataclass
class BackedBlock:
    vllm_block_id: int
    kcmm_block_id: int
    source: str


class KcmmBackedAllocationTracker:
    """Delegate vLLM GPU block ID selection to KCMM."""

    def __init__(self, report_path: str | None = None):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._lock = threading.RLock()
        self._blocks: dict[int, BackedBlock] = {}
        self._native_allocations = 0
        self._native_frees = 0
        self._kcmm_allocations = 0
        self._kcmm_frees = 0
        self._error_count = 0
        self._last_error: str | None = None
        self._stop_condition: str | None = None
        self._allocator_total_blocks: int | None = None

    def attach_pool(self, pool: KcmmPool) -> None:
        with self._lock:
            self._pool = pool
            self.write_report()

    def validate_runtime(self, sizing: Any) -> None:
        if getattr(sizing, "vllm_version", None) != "0.6.1.post1":
            self.fail_closed(
                "unsupported_vllm_version",
                f"expected vLLM 0.6.1.post1, got {getattr(sizing, 'vllm_version', None)}",
            )
        if getattr(sizing, "enable_prefix_caching", False):
            self.fail_closed(
                "prefix_caching_unsupported",
                "Phase II.A KCMM-backed allocator only supports prefix caching disabled",
            )

    def bind_vllm_gpu_allocator(self, allocator: Any) -> None:
        class_path = f"{type(allocator).__module__}.{type(allocator).__qualname__}"
        if class_path != "vllm.core.block.naive_block.NaiveBlockAllocator":
            self.fail_closed(
                "unsupported_vllm_allocator",
                f"expected NaiveBlockAllocator GPU path, got {class_path}",
            )
        total_blocks = len(getattr(allocator, "_all_block_indices", ()))
        if total_blocks <= 0:
            self.fail_closed(
                "invalid_vllm_allocator_capacity",
                "vLLM GPU allocator has no native KV block IDs",
            )
        self._allocator_total_blocks = total_blocks
        setattr(allocator, "_kcmm_backed_tracker", self)
        self.write_report()

    def fail_closed(self, reason: str, message: str) -> None:
        exc = KcmmError(f"{reason}: {message}")
        self._record_error(exc, stop_condition=reason)
        raise exc

    def _require_pool(self) -> KcmmPool:
        if self._pool is None:
            self.fail_closed(
                "pool_not_attached",
                "vLLM requested a KCMM-backed GPU block before KCMM pool creation",
            )
        assert self._pool is not None
        return self._pool

    def _record_error(
        self,
        exc: BaseException,
        *,
        stop_condition: str | None = None,
    ) -> None:
        self._error_count += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._stop_condition = stop_condition or self._stop_condition
        self.write_report()

    def allocate_block_id(self, allocator: Any, source: str) -> int:
        with self._lock:
            allocated: list[int] = []
            try:
                free_ids: deque[int] = getattr(allocator, "_free_block_indices")
                all_ids = getattr(allocator, "_all_block_indices")
                if not free_ids:
                    from vllm.core.block.interfaces import BlockAllocator

                    raise BlockAllocator.NoFreeBlocksError()

                pool = self._require_pool()
                allocated = pool.alloc_blocks(1)
                if len(allocated) != 1:
                    self.fail_closed(
                        "kcmm_allocation_shape_mismatch",
                        f"kcmm_alloc_blocks returned {len(allocated)} blocks",
                    )
                kcmm_block_id = int(allocated[0])

                if kcmm_block_id not in all_ids:
                    pool.free_blocks([kcmm_block_id])
                    allocated = []
                    self.fail_closed(
                        "kcmm_block_id_outside_vllm_native_kv",
                        "KCMM chose block id "
                        f"{kcmm_block_id}, but vLLM native KV tensors only "
                        "accept IDs from the GPU allocator block set",
                    )

                try:
                    free_ids.remove(kcmm_block_id)
                except ValueError:
                    pool.free_blocks([kcmm_block_id])
                    allocated = []
                    self.fail_closed(
                        "kcmm_block_id_not_free_in_vllm",
                        "KCMM chose block id "
                        f"{kcmm_block_id}, but that native vLLM block is not free",
                    )

                allocator._refcounter.incr(kcmm_block_id)
                self._blocks[kcmm_block_id] = BackedBlock(
                    vllm_block_id=kcmm_block_id,
                    kcmm_block_id=kcmm_block_id,
                    source=source,
                )
                self._native_allocations += 1
                self._kcmm_allocations += 1
                self.write_report()
                return kcmm_block_id
            except BaseException as exc:
                if allocated:
                    try:
                        self._require_pool().free_blocks([int(allocated[0])])
                    except Exception:
                        pass
                if not isinstance(exc, KcmmError) or self._last_error is None:
                    self._record_error(exc)
                raise

    def free_block_id(self, block_id: int, released: bool, source: str) -> None:
        with self._lock:
            try:
                self._native_frees += 1
                if not released:
                    self.write_report()
                    return
                block = self._blocks.get(block_id)
                if block is None:
                    self.fail_closed(
                        "free_unknown_kcmm_backed_block",
                        f"vLLM freed unknown KCMM-backed GPU block {block_id}",
                    )
                pool = self._require_pool()
                pool.free_blocks([block.kcmm_block_id])
                self._kcmm_frees += 1
                del self._blocks[block_id]
                self.write_report()
            except BaseException as exc:
                if not isinstance(exc, KcmmError) or self._last_error is None:
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
                "mode": "kcmm_backed_allocator",
                "decision_source": "kcmm_alloc_blocks",
                "storage_of_record": "native_vllm_kv_tensors",
                "native_gpu_allocations": self._native_allocations,
                "native_gpu_frees": self._native_frees,
                "kcmm_allocations": self._kcmm_allocations,
                "kcmm_frees": self._kcmm_frees,
                "outstanding_mappings": len(outstanding),
                "outstanding": outstanding[:32],
                "error_count": self._error_count,
                "last_error": self._last_error,
                "stop_condition": self._stop_condition,
                "pool_attached": self._pool is not None,
                "vllm_gpu_allocator_total_blocks": self._allocator_total_blocks,
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
