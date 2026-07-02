"""KCMM KV write mirror state for vLLM Phase II.B.

This mode leaves vLLM's native KV cache as the storage of record and mirrors
successful `reshape_and_cache` writes into KCMM-managed memory for validation.
"""

from __future__ import annotations

import ctypes
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bindings import KcmmError, KcmmPool
from .host_profile import HostSectionProfiler
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
        profile_host_sections: bool = False,
        replace_native: bool = False,
        force_non_default_stream: bool = False,
        use_device_slot_write: bool = False,
    ):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._verify_rows_per_call = max(0, int(verify_rows_per_call))
        self._report_on_update = bool(report_on_update)
        self._host_profiler = HostSectionProfiler(profile_host_sections)
        self._replace_native = bool(replace_native)
        self._use_device_slot_write = bool(use_device_slot_write)
        self._stream_provider = KcmmStreamProvider(
            force_non_default=force_non_default_stream
        )
        self._lock = threading.RLock()
        self._cache_layers: dict[tuple[int, int], CacheLayer] = {}
        self._pool_block_size: int | None = None
        self._pool_block_bytes: int | None = None
        self._pool_step_elements: int | None = None
        self._pool_num_layers: int | None = None
        self._pool_shape_refreshes = 0
        self._known_slot_blocks: set[int] = set()
        self._slot_block_ensure_cache_hits = 0
        self._slot_block_ensure_cache_misses = 0
        self._last_device_slot_offsets: list[int] = []
        self._last_device_slot_offset_table: Any | None = None
        self._last_device_slot_valid_flags: list[int] = []
        self._last_device_slot_valid_table: Any | None = None
        self._last_device_slot_table_epoch: int | None = None
        self._last_device_slot_table_device_index: int | None = None
        self._device_slot_total_blocks: int | None = None
        self._device_slot_total_blocks_refreshes = 0
        self._device_slot_block_state_epoch_queries = 0
        self._recent_device_slot_tables: list[Any] = []
        self._pending_device_slot_statuses: list[Any] = []
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
        self._host_slot_write_calls = 0
        self._device_slot_write_calls = 0
        self._device_slot_status_checks = 0
        self._device_slot_status_error_count = 0
        self._device_slot_status_codes: dict[int, int] = {}
        self._last_device_slot_status: int | None = None
        self._device_slot_offset_table_cache_hits = 0
        self._device_slot_offset_table_cache_rebuilds = 0
        self._device_slot_valid_table_cache_hits = 0
        self._device_slot_valid_table_cache_rebuilds = 0
        self._device_slot_padding_slots_unknown_calls = 0
        self._device_slot_prepare_direct_calls = 0
        self._device_slot_prepare_reshape_calls = 0
        self._device_slot_prepare_dtype_conversions = 0
        self._device_slot_prepare_contiguous_copies = 0
        self._device_slot_kernel_precompile_requested = self._should_use_device_slot_write()
        self._device_slot_kernel_precompile_calls = 0
        self._device_slot_kernel_precompile_succeeded = False
        self._device_slot_kernel_precompile_elapsed_ms: float | None = None
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
            self._refresh_pool_shape(pool)
            if self._should_use_device_slot_write():
                started_ns = self._host_profiler.start()
                precompile_started_ns = time.perf_counter_ns()
                self._device_slot_kernel_precompile_calls += 1
                try:
                    pool.precompile_vllm_kv_write_f16()
                except BaseException as exc:
                    self._record_error(exc)
                    raise
                finally:
                    elapsed_ns = time.perf_counter_ns() - precompile_started_ns
                    self._device_slot_kernel_precompile_elapsed_ms = round(
                        elapsed_ns / 1_000_000,
                        6,
                    )
                    self._host_profiler.stop(
                        "write_device_slot_kernel_precompile",
                        started_ns,
                    )
                self._device_slot_kernel_precompile_succeeded = True
            self._write_report_on_update()

    def _require_pool(self) -> KcmmPool:
        if self._pool is None:
            raise KcmmError("KCMM KV write mirror has no attached pool")
        return self._pool

    def _refresh_pool_shape(self, pool: KcmmPool) -> None:
        stats = pool.stats()
        block_size = int(stats["block_size"])
        block_bytes = int(stats["block_bytes"])
        if block_size <= 0:
            raise KcmmError(f"KCMM pool has invalid block_size={block_size}")
        if block_bytes <= 0:
            raise KcmmError(f"KCMM pool has invalid block_bytes={block_bytes}")

        self._pool_block_size = block_size
        self._pool_block_bytes = block_bytes
        self._pool_step_elements = block_bytes // block_size // 2
        self._pool_num_layers = int(stats.get("num_layers", 0))
        self._pool_shape_refreshes += 1

    def _pool_shape(self, pool: KcmmPool) -> tuple[int, int, int, int]:
        if (
            self._pool_block_size is None
            or self._pool_block_bytes is None
            or self._pool_step_elements is None
            or self._pool_num_layers is None
        ):
            self._refresh_pool_shape(pool)

        block_size = self._pool_block_size
        block_bytes = self._pool_block_bytes
        step_elements = self._pool_step_elements
        num_layers = self._pool_num_layers
        if (
            block_size is None
            or block_bytes is None
            or step_elements is None
            or num_layers is None
        ):
            raise KcmmError("KCMM pool shape cache is unavailable")
        return block_size, block_bytes, step_elements, num_layers

    def _cuda_driver(self) -> _CudaDriver:
        if self._driver is None:
            self._driver = _CudaDriver()
        return self._driver

    def _ensure_slot_blocks(
        self,
        pool: KcmmPool,
        slots: list[int],
        *,
        block_size: int,
    ) -> None:
        started_ns = self._host_profiler.start()
        if not self._replace_native:
            self._host_profiler.stop("write_ensure_slot_blocks", started_ns)
            return
        observed_block_ids = sorted({slot // block_size for slot in slots if slot >= 0})
        if not observed_block_ids:
            self._host_profiler.stop("write_ensure_slot_blocks", started_ns)
            return
        block_ids = [
            block_id
            for block_id in observed_block_ids
            if block_id not in self._known_slot_blocks
        ]
        self._slot_block_ensure_cache_hits += (
            len(observed_block_ids) - len(block_ids)
        )
        self._slot_block_ensure_cache_misses += len(block_ids)
        if not block_ids:
            self._host_profiler.stop("write_ensure_slot_blocks", started_ns)
            return

        allocated_blocks = 0
        for block_id in block_ids:
            for _attempt in range(block_id + 2):
                try:
                    pool.block_location(block_id)
                    self._known_slot_blocks.add(block_id)
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
        self._host_profiler.stop("write_ensure_slot_blocks", started_ns)

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

        _, _, _, num_layers = self._pool_shape(pool)
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
        return tensor.tolist()

    @staticmethod
    def _slot_mapping_numel(slot_mapping: Any) -> int:
        method = getattr(slot_mapping, "numel", None)
        if callable(method):
            return int(method())
        shape = _shape(slot_mapping)
        if shape is None:
            raise KcmmError("KCMM KV write mirror requires tensor-shaped slot_mapping")
        count = 1
        for dim in shape:
            count *= int(dim)
        return count

    def _should_use_device_slot_write(self) -> bool:
        return self._use_device_slot_write and self._verify_rows_per_call == 0

    def _prepare_device_slot_tensor(self, slot_mapping: Any) -> Any:
        import torch

        dtype = getattr(slot_mapping, "dtype", None)
        dim_method = getattr(slot_mapping, "dim", None)
        if callable(dim_method):
            ndim = int(dim_method())
        else:
            shape = _shape(slot_mapping)
            ndim = len(shape) if shape is not None else None
        contiguous_method = getattr(slot_mapping, "is_contiguous", None)
        is_contiguous = (
            bool(contiguous_method()) if callable(contiguous_method) else False
        )
        if (
            bool(getattr(slot_mapping, "is_cuda", False))
            and dtype == torch.int64
            and ndim == 1
            and is_contiguous
        ):
            self._device_slot_prepare_direct_calls += 1
            return slot_mapping

        self._device_slot_prepare_reshape_calls += 1
        slot_tensor = slot_mapping.detach().reshape(-1)
        if getattr(slot_tensor, "dtype", None) != torch.int64:
            slot_tensor = slot_tensor.to(dtype=torch.int64)
            self._device_slot_prepare_dtype_conversions += 1
        contiguous_method = getattr(slot_tensor, "is_contiguous", None)
        if callable(contiguous_method) and not bool(contiguous_method()):
            slot_tensor = slot_tensor.contiguous()
            self._device_slot_prepare_contiguous_copies += 1
        return slot_tensor

    def _device_slot_block_state_epoch(self, pool: KcmmPool) -> int:
        self._device_slot_block_state_epoch_queries += 1
        return pool.block_state_epoch()

    def _refresh_device_slot_total_blocks(self, pool: KcmmPool) -> int:
        total_blocks = max(int(pool.total_blocks()), 1)
        self._device_slot_total_blocks = total_blocks
        self._device_slot_total_blocks_refreshes += 1
        return total_blocks

    def _device_slot_tables_for_device(
        self,
        *,
        pool: KcmmPool,
        device: Any,
        device_index: int,
    ) -> tuple[Any, Any]:
        import torch

        epoch = self._device_slot_block_state_epoch(pool)
        cached_offsets = self._last_device_slot_offset_table
        cached_valid = self._last_device_slot_valid_table
        if (
            cached_offsets is not None
            and cached_valid is not None
            and self._last_device_slot_offsets
            and self._last_device_slot_valid_flags
            and self._last_device_slot_table_device_index == device_index
            and self._last_device_slot_table_epoch == epoch
        ):
            self._device_slot_offset_table_cache_hits += 1
            self._device_slot_valid_table_cache_hits += 1
            return cached_offsets, cached_valid

        for attempt in range(3):
            if attempt:
                epoch = self._device_slot_block_state_epoch(pool)
            min_entries = self._refresh_device_slot_total_blocks(pool)
            offsets_started_ns = self._host_profiler.start()
            offsets_f16 = pool.all_block_offsets_f16(min_entries=min_entries)
            offset_table = torch.tensor(
                offsets_f16,
                dtype=torch.int64,
                device=device,
            )
            self._host_profiler.stop(
                "write_device_slot_offset_table_rebuild",
                offsets_started_ns,
            )

            valid_started_ns = self._host_profiler.start()
            valid_flags = pool.all_block_valid_flags(min_entries=min_entries)
            valid_table = torch.tensor(
                valid_flags,
                dtype=torch.uint8,
                device=device,
            )
            self._host_profiler.stop(
                "write_device_slot_valid_table_rebuild",
                valid_started_ns,
            )
            if self._device_slot_block_state_epoch(pool) == epoch:
                break
        else:
            raise KcmmError(
                "KCMM block-state epoch changed while building device-slot tables"
            )

        self._last_device_slot_offsets = offsets_f16
        self._last_device_slot_offset_table = offset_table
        self._last_device_slot_valid_flags = valid_flags
        self._last_device_slot_valid_table = valid_table
        self._last_device_slot_table_epoch = epoch
        self._last_device_slot_table_device_index = device_index
        self._recent_device_slot_tables.append(offset_table)
        self._recent_device_slot_tables.append(valid_table)
        self._recent_device_slot_tables = self._recent_device_slot_tables[-16:]
        self._device_slot_offset_table_cache_rebuilds += 1
        self._device_slot_valid_table_cache_rebuilds += 1

        if int(offset_table.numel()) != int(valid_table.numel()):
            raise KcmmError(
                "KCMM device-slot write table length mismatch: "
                f"offsets={int(offset_table.numel())} "
                f"valid={int(valid_table.numel())}"
            )
        return offset_table, valid_table

    def _collect_pending_device_slot_statuses(self) -> None:
        if not self._pending_device_slot_statuses:
            return
        statuses = self._pending_device_slot_statuses
        self._pending_device_slot_statuses = []
        for status in statuses:
            value = int(status.detach().cpu().item())
            self._device_slot_status_checks += 1
            self._last_device_slot_status = value
            self._device_slot_status_codes[value] = (
                self._device_slot_status_codes.get(value, 0) + 1
            )
            if value != 0:
                self._device_slot_status_error_count += 1
                self._error_count += 1
                self._last_error = (
                    "KCMM device-slot write reported invalid slot status "
                    f"{value}"
                )

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
        call_started_ns = self._host_profiler.start()
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
                validate_started_ns = self._host_profiler.start()
                self._validate_dtype(key, value)
                self._host_profiler.stop("write_validate_dtype", validate_started_ns)
                use_device_slot_write = self._should_use_device_slot_write()
                slots: list[int] | None = None
                if use_device_slot_write:
                    batch = self._slot_mapping_numel(slot_mapping)
                else:
                    slot_started_ns = self._host_profiler.start()
                    slots = self._slot_mapping_to_list(slot_mapping)
                    self._host_profiler.stop(
                        "write_slot_mapping_to_host",
                        slot_started_ns,
                    )
                    batch = len(slots)
                self._max_batch_seen = max(self._max_batch_seen, batch)
                if batch == 0:
                    self._skipped_empty_batches += 1
                    self._write_report_on_update()
                    return

                pool = self._require_pool()
                layer_started_ns = self._host_profiler.start()
                layer_idx = self._layer_for_cache(key_cache, value_cache)
                self._host_profiler.stop("write_layer_for_cache", layer_started_ns)
                rows_started_ns = self._host_profiler.start()
                k_rows, v_rows = self._prepare_rows(key, value, batch)
                self._host_profiler.stop("write_prepare_rows", rows_started_ns)
                row_width = int(k_rows.shape[1])
                stats_started_ns = self._host_profiler.start()
                block_size, _, step_elements, _ = self._pool_shape(pool)
                self._host_profiler.stop("write_pool_stats_shape_check", stats_started_ns)
                if row_width != step_elements:
                    raise KcmmError(
                        "key/value row width does not match KCMM pool shape: "
                        f"row_width={row_width} step_elements={step_elements}"
                    )

                device_index = _device_index(k_rows, "key")
                if _device_index(v_rows, "value") != device_index:
                    raise KcmmError("key and value are on different CUDA devices")

                if use_device_slot_write:
                    if not bool(getattr(slot_mapping, "is_cuda", False)):
                        raise KcmmError(
                            "KCMM device-slot write requires CUDA slot_mapping"
                        )
                    slot_device_index = _device_index(slot_mapping, "slot_mapping")
                    if slot_device_index != device_index:
                        raise KcmmError(
                            "slot_mapping and key/value are on different CUDA devices"
                        )
                    import torch

                    slot_prepare_started_ns = self._host_profiler.start()
                    slot_tensor = self._prepare_device_slot_tensor(slot_mapping)
                    self._host_profiler.stop(
                        "write_device_slot_prepare_tensor",
                        slot_prepare_started_ns,
                    )
                    table_started_ns = self._host_profiler.start()
                    offset_table, valid_table = self._device_slot_tables_for_device(
                        pool=pool,
                        device=slot_tensor.device,
                        device_index=device_index,
                    )
                    self._host_profiler.stop(
                        "write_device_slot_table_lookup",
                        table_started_ns,
                    )
                    status_tensor = torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=slot_tensor.device,
                    )
                else:
                    if slots is None:
                        raise KcmmError("host slot list was not materialized")
                    self._ensure_slot_blocks(pool, slots, block_size=block_size)

                stream_started_ns = self._host_profiler.start()
                stream_selection = self._stream_provider.select(device_index)
                self._host_profiler.stop("write_select_stream", stream_started_ns)
                record_started_ns = self._host_profiler.start()
                if use_device_slot_write:
                    self._stream_provider.record_tensors(
                        stream_selection,
                        k_rows,
                        v_rows,
                        slot_tensor,
                        offset_table,
                        valid_table,
                        status_tensor,
                    )
                else:
                    self._stream_provider.record_tensors(
                        stream_selection,
                        k_rows,
                        v_rows,
                    )
                self._host_profiler.stop("write_record_tensors", record_started_ns)
                ptr_started_ns = self._host_profiler.start()
                k_src_ptr = int(k_rows.data_ptr())
                v_src_ptr = int(v_rows.data_ptr())
                self._host_profiler.stop("write_data_ptrs", ptr_started_ns)
                launch_started_ns = self._host_profiler.start()
                if use_device_slot_write:
                    pool.append_kv_device_slots_on_stream(
                        layer_idx=layer_idx,
                        slot_mapping_ptr=int(slot_tensor.data_ptr()),
                        block_offsets_f16_ptr=int(offset_table.data_ptr()),
                        valid_blocks_ptr=int(valid_table.data_ptr()),
                        block_offsets_f16_len=int(offset_table.numel()),
                        batch=batch,
                        k_src_ptr=k_src_ptr,
                        v_src_ptr=v_src_ptr,
                        status_ptr=int(status_tensor.data_ptr()),
                        stream_ptr=stream_selection.stream_ptr,
                    )
                else:
                    if slots is None:
                        raise KcmmError("host slot list was not materialized")
                    pool.append_kv_slots(
                        layer_idx=layer_idx,
                        slot_mapping=slots,
                        k_src_ptr=k_src_ptr,
                        v_src_ptr=v_src_ptr,
                        stream_ptr=stream_selection.stream_ptr,
                    )
                self._host_profiler.stop("write_ctypes_launch", launch_started_ns)
                complete_started_ns = self._host_profiler.start()
                self._stream_provider.complete(stream_selection)
                self._host_profiler.stop("write_complete_stream", complete_started_ns)

                if use_device_slot_write:
                    padding_slots = 0
                    mirrored_rows = batch
                    self._device_slot_padding_slots_unknown_calls += 1
                    self._device_slot_write_calls += 1
                    self._pending_device_slot_statuses.append(status_tensor)
                else:
                    if slots is None:
                        raise KcmmError("host slot list was not materialized")
                    padding_slots = sum(1 for slot in slots if slot < 0)
                    mirrored_rows = batch - padding_slots
                    self._host_slot_write_calls += 1
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
                    slots=slots or [],
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
                        "padding_slots_known": not use_device_slot_write,
                        "verified_rows": verified_rows,
                        "write_path": (
                            "kcmm_append_kv_device_slots_on_stream"
                            if use_device_slot_write
                            else "kcmm_append_kv_slots_on_stream"
                        ),
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
                        "slot_sample": [] if slots is None else slots[:16],
                        "slot_sample_available": slots is not None,
                        "key_shape": _shape(key),
                        "value_shape": _shape(value),
                    }
                )
                self._recent_calls = self._recent_calls[-16:]
                self._write_report_on_update()
                self._host_profiler.stop("write_mirror_call_total", call_started_ns)
            except BaseException as exc:
                self._record_error(exc)
                raise

    def report(self) -> dict[str, Any]:
        with self._lock:
            self._collect_pending_device_slot_statuses()
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
                "device_slot_write_enabled": self._use_device_slot_write,
                "device_slot_write_active": self._should_use_device_slot_write(),
                "device_slot_write_path": (
                    "kcmm_append_kv_device_slots_on_stream"
                ),
                "force_non_default_stream": (
                    self._stream_provider.force_non_default
                ),
                "stream_provider": self._stream_provider.report(),
                "write_verification_enabled": self._verify_rows_per_call > 0,
                "verify_rows_per_call": self._verify_rows_per_call,
                "report_on_update": self._report_on_update,
                "report_write_count": self._report_write_count,
                "host_profile": self._host_profiler.report(),
                "slot_formula": "slot = block_id * block_size + offset_in_block",
                "pool_attached": self._pool is not None,
                "pool_shape_cached": self._pool_step_elements is not None,
                "pool_shape_refreshes": self._pool_shape_refreshes,
                "pool_block_size": self._pool_block_size,
                "pool_block_bytes": self._pool_block_bytes,
                "pool_step_elements": self._pool_step_elements,
                "pool_num_layers": self._pool_num_layers,
                "known_slot_blocks": len(self._known_slot_blocks),
                "slot_block_ensure_cache_hits": self._slot_block_ensure_cache_hits,
                "slot_block_ensure_cache_misses": (
                    self._slot_block_ensure_cache_misses
                ),
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
                "host_slot_write_calls": self._host_slot_write_calls,
                "device_slot_write_calls": self._device_slot_write_calls,
                "device_slot_status_checks": self._device_slot_status_checks,
                "device_slot_status_error_count": (
                    self._device_slot_status_error_count
                ),
                "device_slot_status_codes": {
                    str(key): value
                    for key, value in sorted(
                        self._device_slot_status_codes.items()
                    )
                },
                "last_device_slot_status": self._last_device_slot_status,
                "device_slot_kernel_precompile_requested": (
                    self._device_slot_kernel_precompile_requested
                ),
                "device_slot_kernel_precompile_calls": (
                    self._device_slot_kernel_precompile_calls
                ),
                "device_slot_kernel_precompile_succeeded": (
                    self._device_slot_kernel_precompile_succeeded
                ),
                "device_slot_kernel_precompile_elapsed_ms": (
                    self._device_slot_kernel_precompile_elapsed_ms
                ),
                "device_slot_table_epoch": self._last_device_slot_table_epoch,
                "device_slot_table_device_index": (
                    self._last_device_slot_table_device_index
                ),
                "device_slot_total_blocks": self._device_slot_total_blocks,
                "device_slot_total_blocks_refreshes": (
                    self._device_slot_total_blocks_refreshes
                ),
                "device_slot_block_state_epoch_queries": (
                    self._device_slot_block_state_epoch_queries
                ),
                "device_slot_offset_table_cache_hits": (
                    self._device_slot_offset_table_cache_hits
                ),
                "device_slot_offset_table_cache_rebuilds": (
                    self._device_slot_offset_table_cache_rebuilds
                ),
                "device_slot_valid_table_cache_hits": (
                    self._device_slot_valid_table_cache_hits
                ),
                "device_slot_valid_table_cache_rebuilds": (
                    self._device_slot_valid_table_cache_rebuilds
                ),
                "device_slot_padding_slots_unknown_calls": (
                    self._device_slot_padding_slots_unknown_calls
                ),
                "device_slot_prepare_direct_calls": (
                    self._device_slot_prepare_direct_calls
                ),
                "device_slot_prepare_reshape_calls": (
                    self._device_slot_prepare_reshape_calls
                ),
                "device_slot_prepare_dtype_conversions": (
                    self._device_slot_prepare_dtype_conversions
                ),
                "device_slot_prepare_contiguous_copies": (
                    self._device_slot_prepare_contiguous_copies
                ),
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
