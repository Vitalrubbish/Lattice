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

    def select(self, device_index: int) -> KcmmStreamSelection:
        import torch

        current_stream = torch.cuda.current_stream(device_index)
        current_stream_ptr = int(current_stream.cuda_stream)
        default_stream_ptr = int(
            torch.cuda.default_stream(device_index).cuda_stream
        )
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
