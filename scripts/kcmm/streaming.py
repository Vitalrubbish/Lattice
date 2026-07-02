"""CUDA stream helpers for KCMM raw-pointer launches from PyTorch seams."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .bindings import KcmmError


@dataclass(frozen=True)
class KcmmStreamSelection:
    stream: Any
    stream_ptr: int
    original_stream: Any
    original_stream_ptr: int
    default_stream_ptr: int
    forced_non_default: bool


class KcmmStreamProvider:
    """Selects the CUDA stream used by KCMM `_on_stream` FFI calls."""

    def __init__(self, *, force_non_default: bool = False) -> None:
        self._force_non_default = bool(force_non_default)
        self._streams: dict[int, Any] = {}
        self._default_stream_ptrs: dict[int, int] = {}
        self._select_calls = 0
        self._current_stream_queries = 0
        self._default_stream_ptr_cache_hits = 0
        self._default_stream_ptr_cache_misses = 0
        self._lock = threading.RLock()

    @property
    def force_non_default(self) -> bool:
        return self._force_non_default

    def _stream_for_device(self, device_index: int) -> Any:
        with self._lock:
            existing = self._streams.get(device_index)
            if existing is not None:
                return existing

            import torch

            with torch.cuda.device(device_index):
                stream = torch.cuda.Stream()
            self._streams[device_index] = stream
            return stream

    def _default_stream_ptr_for_device(self, device_index: int) -> int:
        with self._lock:
            existing = self._default_stream_ptrs.get(device_index)
            if existing is not None:
                self._default_stream_ptr_cache_hits += 1
                return existing

        import torch

        default_stream_ptr = int(torch.cuda.default_stream(device_index).cuda_stream)
        with self._lock:
            existing = self._default_stream_ptrs.get(device_index)
            if existing is not None:
                self._default_stream_ptr_cache_hits += 1
                return existing
            self._default_stream_ptrs[device_index] = default_stream_ptr
            self._default_stream_ptr_cache_misses += 1
        return default_stream_ptr

    def select(self, device_index: int) -> KcmmStreamSelection:
        with self._lock:
            self._select_calls += 1
        import torch

        current_stream = torch.cuda.current_stream(device_index)
        with self._lock:
            self._current_stream_queries += 1
        current_stream_ptr = int(current_stream.cuda_stream)
        default_stream_ptr = self._default_stream_ptr_for_device(device_index)
        if not self._force_non_default:
            return KcmmStreamSelection(
                stream=current_stream,
                stream_ptr=current_stream_ptr,
                original_stream=current_stream,
                original_stream_ptr=current_stream_ptr,
                default_stream_ptr=default_stream_ptr,
                forced_non_default=False,
            )

        stream = self._stream_for_device(device_index)
        stream_ptr = int(stream.cuda_stream)
        if stream_ptr == 0 or stream_ptr == default_stream_ptr:
            raise KcmmError(
                "KCMM forced non-default stream resolved to the default stream: "
                f"stream_ptr={stream_ptr} default_stream_ptr={default_stream_ptr}"
            )
        stream.wait_stream(current_stream)
        return KcmmStreamSelection(
            stream=stream,
            stream_ptr=stream_ptr,
            original_stream=current_stream,
            original_stream_ptr=current_stream_ptr,
            default_stream_ptr=default_stream_ptr,
            forced_non_default=True,
        )

    @staticmethod
    def record_tensors(selection: KcmmStreamSelection, *tensors: Any) -> None:
        if not selection.forced_non_default:
            return
        for tensor in tensors:
            record_stream = getattr(tensor, "record_stream", None)
            if callable(record_stream):
                record_stream(selection.stream)

    @staticmethod
    def complete(selection: KcmmStreamSelection) -> None:
        if selection.forced_non_default:
            selection.original_stream.wait_stream(selection.stream)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "select_calls": self._select_calls,
                "current_stream_queries": self._current_stream_queries,
                "default_stream_ptr_cache_hits": (
                    self._default_stream_ptr_cache_hits
                ),
                "default_stream_ptr_cache_misses": (
                    self._default_stream_ptr_cache_misses
                ),
                "default_stream_devices": sorted(self._default_stream_ptrs),
            }
