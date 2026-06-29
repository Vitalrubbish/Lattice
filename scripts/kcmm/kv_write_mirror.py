"""KCMM KV write mirror state for vLLM Phase II.B.

This mode leaves vLLM's native KV cache as the storage of record and mirrors
successful `reshape_and_cache` writes into KCMM-managed memory for validation.
"""

from __future__ import annotations

import ctypes
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bindings import KcmmError, KcmmPool
from .streaming import KcmmStreamProvider


@dataclass
class CacheLayer:
    layer_idx: int
    key_cache_ptr: int
    value_cache_ptr: int
    key_cache_shape: list[int] | None
    value_cache_shape: list[int] | None


class _CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
        self._check(self.lib.cuInit(0), "cuInit")

    @staticmethod
    def _check(rc: int, operation: str) -> None:
        if rc != 0:
            raise KcmmError(f"{operation} failed with CUDA rc={rc}")

    def memcpy_dtoh(self, src_device_ptr: int, byte_count: int) -> bytes:
        buffer = (ctypes.c_ubyte * byte_count)()
        rc = self.lib.cuMemcpyDtoH_v2(
            ctypes.c_void_p(ctypes.addressof(buffer)),
            ctypes.c_uint64(src_device_ptr),
            ctypes.c_size_t(byte_count),
        )
        self._check(rc, "cuMemcpyDtoH_v2")
        return bytes(buffer)


def _shape(value: Any) -> list[int] | None:
    try:
        return [int(dim) for dim in value.shape]
    except Exception:
        return None


def _data_ptr(value: Any, name: str) -> int:
    method = getattr(value, "data_ptr", None)
    if not callable(method):
        raise KcmmError(f"{name} has no data_ptr()")
    return int(method())


def _device_index(value: Any, name: str) -> int:
    if not bool(getattr(value, "is_cuda", False)):
        raise KcmmError(f"{name} must be a CUDA tensor")
    method = getattr(value, "get_device", None)
    if callable(method):
        device = int(method())
        if device >= 0:
            return device
    index = getattr(getattr(value, "device", None), "index", None)
    if index is not None:
        return int(index)
    raise KcmmError(f"{name} CUDA device index is unavailable")


def _tensor_bytes(value: Any) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes()


def _first_mismatch(actual: bytes, expected: bytes) -> int:
    for index, (left, right) in enumerate(zip(actual, expected, strict=False)):
        if left != right:
            return index
    return min(len(actual), len(expected))


class KcmmKvWriteMirrorTracker:
    """Mirror vLLM KV write custom-op inputs into KCMM."""

    def __init__(
        self,
        report_path: str | None = None,
        *,
        verify_rows_per_call: int = 4,
        report_on_update: bool = True,
        replace_native: bool = False,
        force_non_default_stream: bool = False,
    ):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._verify_rows_per_call = max(0, int(verify_rows_per_call))
        self._report_on_update = bool(report_on_update)
        self._replace_native = bool(replace_native)
        self._stream_provider = KcmmStreamProvider(
            force_non_default=force_non_default_stream
        )
        self._lock = threading.RLock()
        self._cache_layers: dict[tuple[int, int], CacheLayer] = {}
        self._driver: _CudaDriver | None = None
        self._write_calls = 0
        self._native_passthrough_calls = 0
        self._native_skipped_calls = 0
        self._skipped_no_pool_calls = 0
        self._skipped_empty_batches = 0
        self._mirror_calls = 0
        self._mirrored_rows = 0
        self._padding_slots = 0
        self._external_block_ensure_calls = 0
        self._external_blocks_allocated = 0
        self._verified_rows = 0
        self._verification_bytes = 0
        self._stream_aware_write_calls = 0
        self._forced_non_default_stream_calls = 0
        self._stream_synchronize_for_verification_calls = 0
        self._last_stream_ptr: int | None = None
        self._last_original_stream_ptr: int | None = None
        self._last_default_stream_ptr: int | None = None
        self._max_batch_seen = 0
        self._counts_by_function: dict[str, int] = {}
        self._recent_calls: list[dict[str, Any]] = []
        self._report_write_count = 0
        self._error_count = 0
        self._last_error: str | None = None

    @property
    def replace_native(self) -> bool:
        return self._replace_native

    @property
    def native_write_mode(self) -> str:
        if self._replace_native:
            return "replace_native_write"
        return "mirror_after_native"

    def attach_pool(self, pool: KcmmPool) -> None:
        with self._lock:
            self._pool = pool
            self._write_report_on_update()

    def _require_pool(self) -> KcmmPool:
        if self._pool is None:
            raise KcmmError("KCMM KV write mirror has no attached pool")
        return self._pool

    def _cuda_driver(self) -> _CudaDriver:
        if self._driver is None:
            self._driver = _CudaDriver()
        return self._driver

    def _ensure_slot_blocks(self, pool: KcmmPool, slots: list[int]) -> None:
        if not self._replace_native:
            return
        stats = pool.stats()
        block_size = int(stats["block_size"])
        block_ids = sorted({slot // block_size for slot in slots if slot >= 0})
        if not block_ids:
            return

        allocated_blocks = 0
        for block_id in block_ids:
            for _attempt in range(block_id + 2):
                try:
                    pool.block_location(block_id)
                    break
                except KcmmError:
                    total_blocks = int(pool.stats().get("total_blocks", 0))
                    needed = max(1, block_id + 1 - total_blocks)
                    allocated_blocks += len(pool.alloc_blocks(needed))
            else:
                raise KcmmError(
                    "KCMM KV write replacement could not ensure local block "
                    f"{block_id} before appending slot-mapped KV rows"
                )

        if allocated_blocks:
            self._external_block_ensure_calls += 1
            self._external_blocks_allocated += allocated_blocks

    def _layer_for_cache(
        self,
        key_cache: Any,
        value_cache: Any,
    ) -> int:
        pool = self._require_pool()
        key_ptr = _data_ptr(key_cache, "key_cache")
        value_ptr = _data_ptr(value_cache, "value_cache")
        cache_key = (key_ptr, value_ptr)
        existing = self._cache_layers.get(cache_key)
        if existing is not None:
            return existing.layer_idx

        pool_stats = pool.stats()
        num_layers = int(pool_stats.get("num_layers", 0))
        layer_idx = len(self._cache_layers)
        if layer_idx >= num_layers:
            raise KcmmError(
                "KCMM KV write mirror saw more cache tensors than KCMM layers: "
                f"next_layer={layer_idx} num_layers={num_layers}"
            )

        self._cache_layers[cache_key] = CacheLayer(
            layer_idx=layer_idx,
            key_cache_ptr=key_ptr,
            value_cache_ptr=value_ptr,
            key_cache_shape=_shape(key_cache),
            value_cache_shape=_shape(value_cache),
        )
        return layer_idx

    @staticmethod
    def _slot_mapping_to_list(slot_mapping: Any) -> list[int]:
        import torch

        tensor = slot_mapping.detach().to(device="cpu", dtype=torch.int64).flatten()
        return [int(item) for item in tensor.tolist()]

    @staticmethod
    def _validate_dtype(key: Any, value: Any) -> None:
        if str(getattr(key, "dtype", "")) != "torch.float16":
            raise KcmmError(f"KCMM KV write mirror requires FP16 key, got {key.dtype}")
        if str(getattr(value, "dtype", "")) != "torch.float16":
            raise KcmmError(f"KCMM KV write mirror requires FP16 value, got {value.dtype}")

    @staticmethod
    def _prepare_rows(key: Any, value: Any, batch: int) -> tuple[Any, Any]:
        if _shape(key) is None or _shape(value) is None:
            raise KcmmError("KCMM KV write mirror requires tensor-shaped key/value")
        if int(key.shape[0]) != batch:
            raise KcmmError(
                f"key batch {int(key.shape[0])} != slot_mapping batch {batch}"
            )
        if int(value.shape[0]) != batch:
            raise KcmmError(
                f"value batch {int(value.shape[0])} != slot_mapping batch {batch}"
            )
        k_rows = key.contiguous().view(batch, -1)
        v_rows = value.contiguous().view(batch, -1)
        if int(k_rows.shape[1]) != int(v_rows.shape[1]):
            raise KcmmError(
                "key/value row widths differ: "
                f"key={int(k_rows.shape[1])} value={int(v_rows.shape[1])}"
            )
        return k_rows, v_rows

    def _verify_rows(
        self,
        *,
        pool: KcmmPool,
        layer_idx: int,
        slots: list[int],
        k_rows: Any,
        v_rows: Any,
        block_size: int,
        step_elements: int,
    ) -> tuple[int, int]:
        if self._verify_rows_per_call <= 0:
            return 0, 0

        driver = self._cuda_driver()
        va_k = pool.va_k(layer_idx)
        va_v = pool.va_v(layer_idx)
        byte_count = step_elements * 2
        verified = 0
        verified_bytes = 0
        for row, slot in enumerate(slots):
            if slot < 0:
                continue
            if verified >= self._verify_rows_per_call:
                break

            block_id = slot // block_size
            offset_in_block = slot % block_size
            block_offset = pool.block_va_offset(block_id)
            token_offset_bytes = offset_in_block * byte_count
            k_addr = va_k + block_offset + token_offset_bytes
            v_addr = va_v + block_offset + token_offset_bytes
            expected_k = _tensor_bytes(k_rows[row])
            expected_v = _tensor_bytes(v_rows[row])
            actual_k = driver.memcpy_dtoh(k_addr, byte_count)
            actual_v = driver.memcpy_dtoh(v_addr, byte_count)
            if actual_k != expected_k:
                mismatch = _first_mismatch(actual_k, expected_k)
                raise KcmmError(
                    "KCMM mirrored key bytes differ: "
                    f"layer={layer_idx} row={row} slot={slot} mismatch_at={mismatch}"
                )
            if actual_v != expected_v:
                mismatch = _first_mismatch(actual_v, expected_v)
                raise KcmmError(
                    "KCMM mirrored value bytes differ: "
                    f"layer={layer_idx} row={row} slot={slot} mismatch_at={mismatch}"
                )
            verified += 1
            verified_bytes += byte_count * 2
        return verified, verified_bytes

    def _record_error(self, exc: BaseException) -> None:
        self._error_count += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self.write_report()

    def _write_report_on_update(self) -> None:
        if self._report_on_update:
            self.write_report()

    def mirror_call(
        self,
        call_key: str,
        key: Any,
        value: Any,
        key_cache: Any,
        value_cache: Any,
        slot_mapping: Any,
        *,
        native_written: bool,
    ) -> None:
        with self._lock:
            self._write_calls += 1
            if native_written:
                self._native_passthrough_calls += 1
            else:
                self._native_skipped_calls += 1
            self._counts_by_function[call_key] = (
                self._counts_by_function.get(call_key, 0) + 1
            )

            if self._pool is None:
                if native_written:
                    self._skipped_no_pool_calls += 1
                    self._write_report_on_update()
                    return
                exc = KcmmError(
                    "KCMM KV write replacement cannot skip native write before "
                    "a KCMM pool is attached"
                )
                self._record_error(exc)
                raise exc

            try:
                self._validate_dtype(key, value)
                slots = self._slot_mapping_to_list(slot_mapping)
                batch = len(slots)
                self._max_batch_seen = max(self._max_batch_seen, batch)
                if batch == 0:
                    self._skipped_empty_batches += 1
                    self._write_report_on_update()
                    return

                pool = self._require_pool()
                layer_idx = self._layer_for_cache(key_cache, value_cache)
                k_rows, v_rows = self._prepare_rows(key, value, batch)
                row_width = int(k_rows.shape[1])
                stats = pool.stats()
                block_size = int(stats["block_size"])
                block_bytes = int(stats["block_bytes"])
                step_elements = block_bytes // block_size // 2
                if row_width != step_elements:
                    raise KcmmError(
                        "key/value row width does not match KCMM pool shape: "
                        f"row_width={row_width} step_elements={step_elements}"
                    )

                device_index = _device_index(k_rows, "key")
                if _device_index(v_rows, "value") != device_index:
                    raise KcmmError("key and value are on different CUDA devices")

                self._ensure_slot_blocks(pool, slots)

                stream_selection = self._stream_provider.select(device_index)
                self._stream_provider.record_tensors(
                    stream_selection,
                    k_rows,
                    v_rows,
                )
                pool.append_kv_slots(
                    layer_idx=layer_idx,
                    slot_mapping=slots,
                    k_src_ptr=int(k_rows.data_ptr()),
                    v_src_ptr=int(v_rows.data_ptr()),
                    stream_ptr=stream_selection.stream_ptr,
                )
                self._stream_provider.complete(stream_selection)

                padding_slots = sum(1 for slot in slots if slot < 0)
                mirrored_rows = batch - padding_slots
                self._stream_aware_write_calls += 1
                if stream_selection.forced_non_default:
                    self._forced_non_default_stream_calls += 1
                self._last_stream_ptr = stream_selection.stream_ptr
                self._last_original_stream_ptr = stream_selection.original_stream_ptr
                self._last_default_stream_ptr = stream_selection.default_stream_ptr

                verification_synchronized = False
                if self._verify_rows_per_call > 0 and mirrored_rows > 0:
                    stream_selection.stream.synchronize()
                    verification_synchronized = True
                    self._stream_synchronize_for_verification_calls += 1

                verified_rows, verified_bytes = self._verify_rows(
                    pool=pool,
                    layer_idx=layer_idx,
                    slots=slots,
                    k_rows=k_rows,
                    v_rows=v_rows,
                    block_size=block_size,
                    step_elements=step_elements,
                )

                self._mirror_calls += 1
                self._mirrored_rows += mirrored_rows
                self._padding_slots += padding_slots
                self._verified_rows += verified_rows
                self._verification_bytes += verified_bytes
                self._recent_calls.append(
                    {
                        "function": call_key,
                        "layer_idx": layer_idx,
                        "batch": batch,
                        "native_written": native_written,
                        "mirrored_rows": mirrored_rows,
                        "padding_slots": padding_slots,
                        "verified_rows": verified_rows,
                        "stream_aware_write": True,
                        "stream_ptr": stream_selection.stream_ptr,
                        "original_stream_ptr": (
                            stream_selection.original_stream_ptr
                        ),
                        "default_stream_ptr": stream_selection.default_stream_ptr,
                        "forced_non_default_stream": (
                            stream_selection.forced_non_default
                        ),
                        "verification_synchronized": verification_synchronized,
                        "slot_sample": slots[:16],
                        "key_shape": _shape(key),
                        "value_shape": _shape(value),
                    }
                )
                self._recent_calls = self._recent_calls[-16:]
                self._write_report_on_update()
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
            return {
                "phase": "II.B",
                "mode": (
                    "kv_write_replace_candidate"
                    if self._replace_native
                    else "kv_write_mirror"
                ),
                "native_write_mode": self.native_write_mode,
                "storage_of_record": (
                    "kcmm_kv_storage_candidate"
                    if self._replace_native
                    else "native_vllm_kv_tensors"
                ),
                "write_path": "kcmm_append_kv_slots",
                "stream_aware_write_path": "kcmm_append_kv_slots_on_stream",
                "force_non_default_stream": (
                    self._stream_provider.force_non_default
                ),
                "write_verification_enabled": self._verify_rows_per_call > 0,
                "verify_rows_per_call": self._verify_rows_per_call,
                "report_on_update": self._report_on_update,
                "report_write_count": self._report_write_count,
                "slot_formula": "slot = block_id * block_size + offset_in_block",
                "pool_attached": self._pool is not None,
                "write_calls": self._write_calls,
                "native_calls": self._native_passthrough_calls,
                "native_passthrough_calls": self._native_passthrough_calls,
                "native_skipped_calls": self._native_skipped_calls,
                "skipped_no_pool_calls": self._skipped_no_pool_calls,
                "skipped_empty_batches": self._skipped_empty_batches,
                "mirror_calls": self._mirror_calls,
                "mirrored_rows": self._mirrored_rows,
                "padding_slots": self._padding_slots,
                "external_block_ensure_calls": self._external_block_ensure_calls,
                "external_blocks_allocated": self._external_blocks_allocated,
                "verified_rows": self._verified_rows,
                "verification_bytes": self._verification_bytes,
                "stream_aware_write_calls": self._stream_aware_write_calls,
                "forced_non_default_stream_calls": (
                    self._forced_non_default_stream_calls
                ),
                "stream_synchronize_for_verification_calls": (
                    self._stream_synchronize_for_verification_calls
                ),
                "last_stream_ptr": self._last_stream_ptr,
                "last_original_stream_ptr": self._last_original_stream_ptr,
                "last_default_stream_ptr": self._last_default_stream_ptr,
                "max_batch_seen": self._max_batch_seen,
                "counts_by_function": dict(sorted(self._counts_by_function.items())),
                "cache_layers": [
                    asdict(layer)
                    for layer in sorted(
                        self._cache_layers.values(),
                        key=lambda item: item.layer_idx,
                    )
                ],
                "recent_calls": list(self._recent_calls),
                "error_count": self._error_count,
                "last_error": self._last_error,
                "pool_stats": pool_stats,
            }

    def write_report(self) -> None:
        if self._report_path is None:
            return
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_write_count += 1
        self._report_path.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
