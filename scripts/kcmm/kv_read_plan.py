"""KCMM KV read offset-table planner for vLLM Phase II.C.

This mode does not replace vLLM's attention kernel. It proves the A2 seam can
materialize a side table indexed by native vLLM/KCMM block_id at every
`paged_attention` read call.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bindings import KcmmError, KcmmPool


@dataclass
class ReadPlanCall:
    function: str
    block_ids_sample: list[int]
    unique_block_ids: int
    max_block_id: int | None
    offset_table_entries: int
    offset_table_dtype: str
    offset_table_device: str
    offset_table_data_ptr: int
    missing_block_ids: list[int]
    block_locations_sample: dict[str, str]
    offset_f16_sample: dict[str, int]


def _shape(value: Any) -> list[int] | None:
    try:
        return [int(dim) for dim in value.shape]
    except Exception:
        return None


def _tensor_block_ids(block_tables: Any) -> list[int]:
    import torch

    tensor = block_tables.detach().to(device="cpu", dtype=torch.int64).flatten()
    return [int(item) for item in tensor.tolist() if int(item) >= 0]


class KcmmKvReadOffsetTableTracker:
    """Build and validate a KCMM A2 read offset table at vLLM read seams."""

    def __init__(self, report_path: str | None = None):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._lock = threading.RLock()
        self._read_calls = 0
        self._planned_calls = 0
        self._offset_table_builds = 0
        self._total_block_table_entries = 0
        self._unique_block_ids_seen: set[int] = set()
        self._max_block_id_seen: int | None = None
        self._counts_by_function: dict[str, int] = {}
        self._recent_calls: list[ReadPlanCall] = []
        self._error_count = 0
        self._last_error: str | None = None
        self._last_offset_table: Any | None = None

    def attach_pool(self, pool: KcmmPool) -> None:
        with self._lock:
            self._pool = pool
            self.write_report()

    def validate_runtime(self, sizing: Any) -> None:
        if getattr(sizing, "vllm_version", None) != "0.6.1.post1":
            raise KcmmError(
                "unsupported_vllm_version: expected vLLM 0.6.1.post1, got "
                f"{getattr(sizing, 'vllm_version', None)}"
            )
        if getattr(sizing, "enable_prefix_caching", False):
            raise KcmmError(
                "prefix_caching_unsupported: Phase II.C read offset-table "
                "planning requires prefix caching disabled"
            )

    def _require_pool(self) -> KcmmPool:
        if self._pool is None:
            raise KcmmError("KCMM KV read offset planner has no attached pool")
        return self._pool

    def _record_error(self, exc: BaseException) -> None:
        self._error_count += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self.write_report()

    def plan_call(
        self,
        call_key: str,
        function_name: str,
        arguments: dict[str, Any],
    ) -> None:
        with self._lock:
            self._read_calls += 1
            self._counts_by_function[call_key] = (
                self._counts_by_function.get(call_key, 0) + 1
            )
            try:
                pool = self._require_pool()
                block_tables = arguments["block_tables"]
                block_ids = _tensor_block_ids(block_tables)
                unique_ids = sorted(set(block_ids))
                max_block_id = max(unique_ids) if unique_ids else None
                min_entries = (max_block_id + 1) if max_block_id is not None else 1

                offsets_f16 = pool.all_block_offsets_f16(min_entries=min_entries)
                missing_block_ids: list[int] = []
                locations: dict[int, str] = {}
                for block_id in unique_ids:
                    if block_id >= len(offsets_f16):
                        missing_block_ids.append(block_id)
                        continue
                    try:
                        locations[block_id] = pool.block_location(block_id)
                    except Exception:
                        missing_block_ids.append(block_id)

                if missing_block_ids:
                    raise KcmmError(
                        "KCMM read offset table is missing block ids observed "
                        f"in vLLM block_tables: {missing_block_ids[:16]}"
                    )

                import torch

                device = getattr(block_tables, "device", "cpu")
                offset_table = torch.tensor(
                    offsets_f16,
                    dtype=torch.int64,
                    device=device,
                )
                self._last_offset_table = offset_table
                self._planned_calls += 1
                self._offset_table_builds += 1
                self._total_block_table_entries += len(block_ids)
                self._unique_block_ids_seen.update(unique_ids)
                if max_block_id is not None:
                    self._max_block_id_seen = max(
                        max_block_id,
                        self._max_block_id_seen
                        if self._max_block_id_seen is not None
                        else max_block_id,
                    )

                sample_ids = unique_ids[:16]
                call = ReadPlanCall(
                    function=function_name,
                    block_ids_sample=sample_ids,
                    unique_block_ids=len(unique_ids),
                    max_block_id=max_block_id,
                    offset_table_entries=len(offsets_f16),
                    offset_table_dtype=str(offset_table.dtype),
                    offset_table_device=str(offset_table.device),
                    offset_table_data_ptr=int(offset_table.data_ptr()),
                    missing_block_ids=[],
                    block_locations_sample={
                        str(block_id): locations[block_id] for block_id in sample_ids
                    },
                    offset_f16_sample={
                        str(block_id): int(offsets_f16[block_id])
                        for block_id in sample_ids
                    },
                )
                self._recent_calls.append(call)
                self._recent_calls = self._recent_calls[-16:]
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
            last_table = self._last_offset_table
            return {
                "phase": "II.C",
                "mode": "kv_read_offset_table_plan",
                "candidate": "A2",
                "kernel_replaced": False,
                "read_path": "native_vllm_paged_attention",
                "offset_table_contract": "torch.int64[f16_va_offset_by_block_id]",
                "required_allocator_mode": "kcmm_backed_allocator",
                "pool_attached": self._pool is not None,
                "read_calls": self._read_calls,
                "planned_calls": self._planned_calls,
                "offset_table_builds": self._offset_table_builds,
                "total_block_table_entries": self._total_block_table_entries,
                "unique_block_ids_seen": len(self._unique_block_ids_seen),
                "max_block_id_seen": self._max_block_id_seen,
                "counts_by_function": dict(sorted(self._counts_by_function.items())),
                "recent_calls": [asdict(call) for call in self._recent_calls],
                "last_offset_table_shape": _shape(last_table),
                "error_count": self._error_count,
                "last_error": self._last_error,
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
