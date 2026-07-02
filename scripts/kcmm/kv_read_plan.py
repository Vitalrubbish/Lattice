"""KCMM KV read offset-table planner for vLLM Phase II.C.

This mode does not replace vLLM's attention kernel. It proves the A2 seam can
materialize a side table indexed by native vLLM/KCMM block_id at every
`paged_attention` read call.
"""

from __future__ import annotations

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
class ReadPlanCall:
    function: str
    layer_idx: int
    batch: int | None
    query_shape: list[int] | None
    query_stride: list[int] | None
    query_is_contiguous: bool | None
    out_shape: list[int] | None
    out_stride: list[int] | None
    out_is_contiguous: bool | None
    block_tables_shape: list[int] | None
    block_tables_stride: list[int] | None
    block_tables_is_contiguous: bool | None
    seq_lens_shape: list[int] | None
    seq_lens_stride: list[int] | None
    seq_lens_is_contiguous: bool | None
    block_table_entries: int
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
    block_table_validation_enabled: bool
    offset_table_reused: bool
    native_replaced: bool
    replacement_backend: str
    reference_read_bytes: int
    gpu_kernel_launched: bool
    stream_ptr: int | None
    stream_aware_launch: bool
    forced_non_default_stream: bool
    original_stream_ptr: int | None
    default_stream_ptr: int | None
    gpu_kernel_elapsed_ms: float | None


@dataclass
class CacheLayer:
    layer_idx: int
    key_cache_ptr: int
    value_cache_ptr: int
    key_cache_shape: list[int] | None
    value_cache_shape: list[int] | None


class _CudaDriver:
    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
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
        ctypes = self._ctypes
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


def _first_dim(value: Any) -> int | None:
    try:
        return int(value.shape[0])
    except Exception:
        return None


def _stride(value: Any) -> list[int] | None:
    method = getattr(value, "stride", None)
    if not callable(method):
        return None
    try:
        return [int(dim) for dim in method()]
    except Exception:
        return None


def _is_contiguous(value: Any) -> bool | None:
    method = getattr(value, "is_contiguous", None)
    if not callable(method):
        return None
    try:
        return bool(method())
    except Exception:
        return None


def _data_ptr(value: Any, name: str) -> int:
    method = getattr(value, "data_ptr", None)
    if not callable(method):
        raise KcmmError(f"{name} has no data_ptr()")
    return int(method())


def _tensor_block_ids(block_tables: Any) -> list[int]:
    import torch

    tensor = block_tables.detach().to(device="cpu", dtype=torch.int64).flatten()
    return [int(item) for item in tensor.tolist() if int(item) >= 0]


class KcmmKvReadOffsetTableTracker:
    """Build and validate a KCMM A2 read offset table at vLLM read seams."""

    def __init__(
        self,
        report_path: str | None = None,
        *,
        replace_native: bool = False,
        replacement_backend: str = "reference",
        force_non_default_stream: bool = False,
        profile_gpu_kernel: bool = False,
        report_on_update: bool = True,
        validate_block_tables: bool = True,
        profile_host_sections: bool = False,
        fast_current_context_launch: bool = False,
        precompile_gpu_kernel: bool = False,
    ):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._replace_native = bool(replace_native)
        if replacement_backend not in {"reference", "gpu_kernel"}:
            raise ValueError(f"unsupported replacement backend: {replacement_backend}")
        self._replacement_backend = replacement_backend
        self._profile_gpu_kernel = bool(profile_gpu_kernel)
        self._report_on_update = bool(report_on_update)
        self._validate_block_tables = bool(validate_block_tables)
        self._host_profiler = HostSectionProfiler(profile_host_sections)
        self._fast_current_context_launch = bool(fast_current_context_launch)
        self._precompile_gpu_kernel = bool(precompile_gpu_kernel)
        self._stream_provider = KcmmStreamProvider(
            force_non_default=force_non_default_stream
        )
        self._lock = threading.RLock()
        self._cache_layers: dict[tuple[int, int], CacheLayer] = {}
        self._driver: _CudaDriver | None = None
        self._read_calls = 0
        self._planned_calls = 0
        self._replacement_calls = 0
        self._gpu_kernel_calls = 0
        self._stream_aware_kernel_calls = 0
        self._forced_non_default_stream_calls = 0
        self._offset_table_builds = 0
        self._reference_read_bytes = 0
        self._total_block_table_entries = 0
        self._unique_block_ids_seen: set[int] = set()
        self._max_block_id_seen: int | None = None
        self._max_batch_seen = 0
        self._last_stream_ptr: int | None = None
        self._last_original_stream_ptr: int | None = None
        self._last_default_stream_ptr: int | None = None
        self._gpu_kernel_profile_samples_ms: list[float] = []
        self._counts_by_function: dict[str, int] = {}
        self._recent_calls: list[ReadPlanCall] = []
        self._report_write_count = 0
        self._error_count = 0
        self._last_error: str | None = None
        self._last_offset_table: Any | None = None
        self._last_offsets_f16: list[int] = []
        self._recent_offset_tables: list[Any] = []
        self._offset_table_cache_hits = 0
        self._offset_table_cache_rebuilds = 0
        self._min_entries_total_blocks_calls = 0
        self._gpu_kernel_precompile_requested = bool(precompile_gpu_kernel)
        self._gpu_kernel_precompile_calls = 0
        self._gpu_kernel_precompile_succeeded = False
        self._gpu_kernel_precompile_elapsed_ms: float | None = None
        self._compact_plan_metadata = self._should_use_compact_plan_metadata()
        self._compact_plan_metadata_calls = 0
        self._detailed_plan_metadata_calls = 0

    @property
    def replace_native(self) -> bool:
        return self._replace_native

    def attach_pool(self, pool: KcmmPool) -> None:
        with self._lock:
            self._pool = pool
            if self._precompile_gpu_kernel and self._replacement_backend == "gpu_kernel":
                started_ns = self._host_profiler.start()
                precompile_started_ns = time.perf_counter_ns()
                self._gpu_kernel_precompile_calls += 1
                try:
                    pool.precompile_paged_attn_decode_f16()
                except BaseException as exc:
                    self._record_error(exc)
                    raise
                finally:
                    elapsed_ns = time.perf_counter_ns() - precompile_started_ns
                    self._gpu_kernel_precompile_elapsed_ms = round(
                        elapsed_ns / 1_000_000,
                        6,
                    )
                    self._host_profiler.stop(
                        "read_gpu_kernel_precompile",
                        started_ns,
                    )
                self._gpu_kernel_precompile_succeeded = True
            self._write_report_on_update()

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

    def _should_use_compact_plan_metadata(self) -> bool:
        return (
            self._replace_native
            and self._replacement_backend == "gpu_kernel"
            and not self._validate_block_tables
            and not self._report_on_update
            and not self._profile_gpu_kernel
        )

    def _write_report_on_update(self) -> None:
        if self._report_on_update:
            self.write_report()

    def _cuda_driver(self) -> _CudaDriver:
        if self._driver is None:
            self._driver = _CudaDriver()
        return self._driver

    def _layer_for_cache(self, key_cache: Any, value_cache: Any) -> int:
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
                "KCMM KV read planner saw more cache tensors than KCMM layers: "
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

    def _offset_table_for_device(
        self,
        *,
        pool: KcmmPool,
        device: Any,
        min_entries: int,
    ) -> tuple[list[int], Any, bool]:
        started_ns = self._host_profiler.start()
        cached_table = self._last_offset_table
        if (
            cached_table is not None
            and self._last_offsets_f16
            and len(self._last_offsets_f16) >= min_entries
            and str(getattr(cached_table, "device", None)) == str(device)
        ):
            self._offset_table_cache_hits += 1
            self._host_profiler.stop("read_offset_table_cache_hit", started_ns)
            return self._last_offsets_f16, cached_table, True

        import torch

        offsets_f16 = pool.all_block_offsets_f16(min_entries=min_entries)
        offset_table = torch.tensor(
            offsets_f16,
            dtype=torch.int64,
            device=device,
        )
        self._last_offsets_f16 = offsets_f16
        self._last_offset_table = offset_table
        self._recent_offset_tables.append(offset_table)
        self._recent_offset_tables = self._recent_offset_tables[-16:]
        self._offset_table_cache_rebuilds += 1
        self._host_profiler.stop("read_offset_table_rebuild", started_ns)
        return offsets_f16, offset_table, False

    def _build_plan(
        self,
        function_name: str,
        arguments: dict[str, Any],
    ) -> tuple[ReadPlanCall, list[int], list[int]]:
        build_started_ns = self._host_profiler.start()
        pool = self._require_pool()
        block_tables = arguments["block_tables"]
        query = arguments.get("query")
        out = arguments.get("out")
        seq_lens = arguments.get("seq_lens")
        key_cache = arguments["key_cache"]
        value_cache = arguments["value_cache"]
        layer_started_ns = self._host_profiler.start()
        layer_idx = self._layer_for_cache(key_cache, value_cache)
        self._host_profiler.stop("read_layer_for_cache", layer_started_ns)
        compact_metadata = self._compact_plan_metadata
        if compact_metadata:
            query_shape = None
            out_shape = None
            block_tables_shape = None
            seq_lens_shape = None
            batch = _first_dim(query)
        else:
            shape_started_ns = self._host_profiler.start()
            query_shape = _shape(query)
            out_shape = _shape(out)
            block_tables_shape = _shape(block_tables)
            seq_lens_shape = _shape(seq_lens)
            batch = query_shape[0] if query_shape else None
            self._host_profiler.stop("read_tensor_shape_capture", shape_started_ns)
        device = getattr(block_tables, "device", "cpu")
        if self._validate_block_tables:
            block_ids_started_ns = self._host_profiler.start()
            block_ids = _tensor_block_ids(block_tables)
            unique_ids = sorted(set(block_ids))
            max_block_id = max(unique_ids) if unique_ids else None
            min_entries = (max_block_id + 1) if max_block_id is not None else 1
            self._host_profiler.stop(
                "read_block_tables_to_host",
                block_ids_started_ns,
            )
        else:
            stats_started_ns = self._host_profiler.start()
            block_ids = []
            unique_ids = []
            max_block_id = None
            min_entries = max(pool.total_blocks(), 1)
            self._min_entries_total_blocks_calls += 1
            self._host_profiler.stop("read_pool_stats_for_min_entries", stats_started_ns)

        offset_started_ns = self._host_profiler.start()
        offsets_f16, offset_table, offset_table_reused = self._offset_table_for_device(
            pool=pool,
            device=device,
            min_entries=min_entries,
        )
        self._host_profiler.stop("read_offset_table_lookup", offset_started_ns)
        missing_block_ids: list[int] = []
        locations: dict[int, str] = {}
        if self._validate_block_tables:
            validate_started_ns = self._host_profiler.start()
            for block_id in unique_ids:
                if block_id >= len(offsets_f16):
                    missing_block_ids.append(block_id)
                    continue
                try:
                    locations[block_id] = pool.block_location(block_id)
                except Exception:
                    missing_block_ids.append(block_id)
            self._host_profiler.stop(
                "read_block_location_validation",
                validate_started_ns,
            )

        if missing_block_ids:
            raise KcmmError(
                "KCMM read offset table is missing block ids observed "
                f"in vLLM block_tables: {missing_block_ids[:16]}"
            )

        sample_ids = unique_ids[:16]
        if compact_metadata:
            query_stride = None
            out_stride = None
            block_tables_stride = None
            seq_lens_stride = None
            query_is_contiguous = None
            out_is_contiguous = None
            block_tables_is_contiguous = None
            seq_lens_is_contiguous = None
            block_locations_sample = {}
            offset_f16_sample = {}
            self._compact_plan_metadata_calls += 1
        else:
            query_stride = _stride(query)
            out_stride = _stride(out)
            block_tables_stride = _stride(block_tables)
            seq_lens_stride = _stride(seq_lens)
            query_is_contiguous = _is_contiguous(query)
            out_is_contiguous = _is_contiguous(out)
            block_tables_is_contiguous = _is_contiguous(block_tables)
            seq_lens_is_contiguous = _is_contiguous(seq_lens)
            block_locations_sample = {
                str(block_id): locations[block_id] for block_id in sample_ids
            }
            offset_f16_sample = {
                str(block_id): int(offsets_f16[block_id]) for block_id in sample_ids
            }
            self._detailed_plan_metadata_calls += 1
        call = ReadPlanCall(
            function=function_name,
            layer_idx=layer_idx,
            batch=batch,
            query_shape=query_shape,
            query_stride=query_stride,
            query_is_contiguous=query_is_contiguous,
            out_shape=out_shape,
            out_stride=out_stride,
            out_is_contiguous=out_is_contiguous,
            block_tables_shape=block_tables_shape,
            block_tables_stride=block_tables_stride,
            block_tables_is_contiguous=block_tables_is_contiguous,
            seq_lens_shape=seq_lens_shape,
            seq_lens_stride=seq_lens_stride,
            seq_lens_is_contiguous=seq_lens_is_contiguous,
            block_table_entries=len(block_ids),
            block_ids_sample=sample_ids,
            unique_block_ids=len(unique_ids),
            max_block_id=max_block_id,
            offset_table_entries=len(offsets_f16),
            offset_table_dtype=str(offset_table.dtype),
            offset_table_device=str(offset_table.device),
            offset_table_data_ptr=int(offset_table.data_ptr()),
            missing_block_ids=[],
            block_locations_sample=block_locations_sample,
            offset_f16_sample=offset_f16_sample,
            block_table_validation_enabled=self._validate_block_tables,
            offset_table_reused=offset_table_reused,
            native_replaced=self._replace_native,
            replacement_backend=self._replacement_backend,
            reference_read_bytes=0,
            gpu_kernel_launched=False,
            stream_ptr=None,
            stream_aware_launch=False,
            forced_non_default_stream=False,
            original_stream_ptr=None,
            default_stream_ptr=None,
            gpu_kernel_elapsed_ms=None,
        )
        self._host_profiler.stop("read_build_plan_total", build_started_ns)
        return call, offsets_f16, unique_ids

    def _gpu_kernel_profile_summary(self) -> dict[str, Any]:
        samples = list(self._gpu_kernel_profile_samples_ms)

        def rounded(value: float | None) -> float | None:
            return round(value, 6) if value is not None else None

        def sample_stats(values: list[float]) -> dict[str, Any]:
            sorted_values = sorted(values)

            def percentile(percentile_value: int) -> float | None:
                if not sorted_values:
                    return None
                rank = (percentile_value * len(sorted_values) + 99) // 100
                index = min(max(rank - 1, 0), len(sorted_values) - 1)
                return sorted_values[index]

            return {
                "count": len(values),
                "min_ms": rounded(min(values) if values else None),
                "avg_ms": rounded((sum(values) / len(values)) if values else None),
                "p50_ms": rounded(percentile(50)),
                "p95_ms": rounded(percentile(95)),
                "p99_ms": rounded(percentile(99)),
                "max_ms": rounded(max(values) if values else None),
            }

        warmup_excluded_count = 1 if len(samples) > 1 else 0
        steady_state_samples = samples[warmup_excluded_count:]
        stats = sample_stats(samples)

        return {
            "enabled": self._profile_gpu_kernel,
            "unit": "ms",
            **stats,
            "first_call_ms": rounded(samples[0] if samples else None),
            "warmup_excluded_count": warmup_excluded_count,
            "steady_state": sample_stats(steady_state_samples),
            "samples_ms": [rounded(sample) for sample in samples],
        }

    def plan_call(
        self,
        call_key: str,
        function_name: str,
        arguments: dict[str, Any],
    ) -> None:
        call_started_ns = self._host_profiler.start()
        with self._lock:
            self._read_calls += 1
            self._counts_by_function[call_key] = (
                self._counts_by_function.get(call_key, 0) + 1
            )
            try:
                call, _offsets_f16, unique_ids = self._build_plan(
                    function_name,
                    arguments,
                )
                self._planned_calls += 1
                self._offset_table_builds += 1
                self._total_block_table_entries += call.block_table_entries
                if call.batch is not None:
                    self._max_batch_seen = max(self._max_batch_seen, call.batch)
                self._unique_block_ids_seen.update(unique_ids)
                if call.max_block_id is not None:
                    self._max_block_id_seen = max(
                        call.max_block_id,
                        self._max_block_id_seen
                        if self._max_block_id_seen is not None
                        else call.max_block_id,
                    )
                self._recent_calls.append(call)
                self._recent_calls = self._recent_calls[-16:]
                self._write_report_on_update()
                self._host_profiler.stop("read_plan_call_total", call_started_ns)
            except BaseException as exc:
                self._record_error(exc)
                raise

    def replace_call(
        self,
        call_key: str,
        function_name: str,
        arguments: dict[str, Any],
    ) -> None:
        call_started_ns = self._host_profiler.start()
        with self._lock:
            self._read_calls += 1
            self._counts_by_function[call_key] = (
                self._counts_by_function.get(call_key, 0) + 1
            )
            try:
                build_started_ns = self._host_profiler.start()
                call, offsets_f16, unique_ids = self._build_plan(
                    function_name,
                    arguments,
                )
                self._host_profiler.stop(
                    "read_replace_build_plan",
                    build_started_ns,
                )
                if self._replacement_backend == "gpu_kernel":
                    kernel_started_ns = self._host_profiler.start()
                    stream_info = self._run_gpu_kernel_attention(
                        layer_idx=call.layer_idx,
                        arguments=arguments,
                    )
                    self._host_profiler.stop(
                        "read_replace_gpu_kernel_host",
                        kernel_started_ns,
                    )
                    read_bytes = 0
                    call.gpu_kernel_launched = True
                    call.stream_ptr = stream_info["stream_ptr"]
                    call.stream_aware_launch = True
                    call.forced_non_default_stream = stream_info[
                        "forced_non_default_stream"
                    ]
                    call.original_stream_ptr = stream_info["original_stream_ptr"]
                    call.default_stream_ptr = stream_info["default_stream_ptr"]
                    elapsed_ms = stream_info.get("gpu_kernel_elapsed_ms")
                    if isinstance(elapsed_ms, (int, float)):
                        call.gpu_kernel_elapsed_ms = float(elapsed_ms)
                        self._gpu_kernel_profile_samples_ms.append(float(elapsed_ms))
                    self._gpu_kernel_calls += 1
                    self._stream_aware_kernel_calls += 1
                    if call.forced_non_default_stream:
                        self._forced_non_default_stream_calls += 1
                    self._last_stream_ptr = call.stream_ptr
                    self._last_original_stream_ptr = call.original_stream_ptr
                    self._last_default_stream_ptr = call.default_stream_ptr
                else:
                    read_bytes = self._run_reference_attention(
                        layer_idx=call.layer_idx,
                        offsets_f16=offsets_f16,
                        arguments=arguments,
                    )
                    call.reference_read_bytes = read_bytes
                self._planned_calls += 1
                self._replacement_calls += 1
                self._offset_table_builds += 1
                self._reference_read_bytes += read_bytes
                self._total_block_table_entries += call.block_table_entries
                if call.batch is not None:
                    self._max_batch_seen = max(self._max_batch_seen, call.batch)
                self._unique_block_ids_seen.update(unique_ids)
                if call.max_block_id is not None:
                    self._max_block_id_seen = max(
                        call.max_block_id,
                        self._max_block_id_seen
                        if self._max_block_id_seen is not None
                        else call.max_block_id,
                    )
                self._recent_calls.append(call)
                self._recent_calls = self._recent_calls[-16:]
                self._write_report_on_update()
                self._host_profiler.stop("read_replace_call_total", call_started_ns)
            except BaseException as exc:
                self._record_error(exc)
                raise

    def _validate_replacement_args(self, arguments: dict[str, Any]) -> None:
        if function_name := arguments.get("function_name"):
            raise KcmmError(f"unexpected function_name argument: {function_name}")
        if arguments.get("alibi_slopes") is not None:
            raise KcmmError("KCMM read replacement does not support alibi_slopes")
        for name in (
            "blocksparse_local_blocks",
            "blocksparse_vert_stride",
            "blocksparse_head_sliding_step",
        ):
            value = int(arguments.get(name, 0) or 0)
            if value != 0:
                raise KcmmError(f"KCMM read replacement does not support {name}={value}")
        if float(arguments.get("k_scale", 1.0)) != 1.0:
            raise KcmmError("KCMM read replacement only supports k_scale=1.0")
        if float(arguments.get("v_scale", 1.0)) != 1.0:
            raise KcmmError("KCMM read replacement only supports v_scale=1.0")

    def _run_reference_attention(
        self,
        *,
        layer_idx: int,
        offsets_f16: list[int],
        arguments: dict[str, Any],
    ) -> int:
        self._validate_replacement_args(arguments)

        import torch

        pool = self._require_pool()
        out = arguments["out"]
        query = arguments["query"]
        block_tables = arguments["block_tables"]
        seq_lens = arguments["seq_lens"]
        num_kv_heads = int(arguments["num_kv_heads"])
        scale = float(arguments["scale"])
        block_size = int(arguments["block_size"])

        stats = pool.stats()
        step_elements = int(stats["block_bytes"]) // block_size // 2
        head_dim = int(query.shape[-1])
        if step_elements % num_kv_heads != 0:
            raise KcmmError(
                "KCMM read replacement cannot derive head_dim: "
                f"step_elements={step_elements} num_kv_heads={num_kv_heads}"
            )
        if step_elements // num_kv_heads != head_dim:
            raise KcmmError(
                "KCMM read replacement head_dim mismatch: "
                f"kcmm={step_elements // num_kv_heads} query={head_dim}"
            )

        va_k = pool.va_k(layer_idx)
        va_v = pool.va_v(layer_idx)
        byte_count = step_elements * 2
        driver = self._cuda_driver()
        block_tables_cpu = (
            block_tables.detach().to(device="cpu", dtype=torch.int64).tolist()
        )
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.int64).tolist()
        num_seqs = int(query.shape[0])
        num_heads = int(query.shape[1])
        head_indices = torch.arange(num_heads, device=query.device)
        kv_head_indices = (head_indices * num_kv_heads // num_heads).long()
        total_read_bytes = 0

        outputs: list[Any] = []
        for seq_idx in range(num_seqs):
            seq_len = int(seq_lens_cpu[seq_idx])
            if seq_len <= 0:
                outputs.append(torch.zeros_like(query[seq_idx]))
                continue

            k_rows = []
            v_rows = []
            for pos in range(seq_len):
                logical_block = pos // block_size
                offset_in_block = pos % block_size
                block_id = int(block_tables_cpu[seq_idx][logical_block])
                if block_id < 0 or block_id >= len(offsets_f16):
                    raise KcmmError(
                        "KCMM read replacement saw invalid block id "
                        f"{block_id} at seq={seq_idx} logical_block={logical_block}"
                    )
                block_offset_bytes = int(offsets_f16[block_id]) * 2
                token_offset_bytes = offset_in_block * byte_count
                k_addr = va_k + block_offset_bytes + token_offset_bytes
                v_addr = va_v + block_offset_bytes + token_offset_bytes
                k_bytes = driver.memcpy_dtoh(k_addr, byte_count)
                v_bytes = driver.memcpy_dtoh(v_addr, byte_count)
                total_read_bytes += byte_count * 2
                k_rows.append(
                    torch.frombuffer(bytearray(k_bytes), dtype=torch.float16)
                    .view(num_kv_heads, head_dim)
                    .to(device=query.device)
                )
                v_rows.append(
                    torch.frombuffer(bytearray(v_bytes), dtype=torch.float16)
                    .view(num_kv_heads, head_dim)
                    .to(device=query.device)
                )

            k_seq = torch.stack(k_rows, dim=0)
            v_seq = torch.stack(v_rows, dim=0)
            q = query[seq_idx].to(dtype=torch.float32)
            k_for_heads = k_seq[:, kv_head_indices, :].to(dtype=torch.float32)
            logits = torch.einsum("hd,lhd->hl", q, k_for_heads) * scale
            probs = torch.softmax(logits, dim=-1)
            v_for_heads = v_seq[:, kv_head_indices, :].to(dtype=torch.float32)
            output = torch.einsum("hl,lhd->hd", probs, v_for_heads)
            outputs.append(output.to(dtype=out.dtype))

        out.copy_(torch.stack(outputs, dim=0))
        return total_read_bytes

    def _run_gpu_kernel_attention(
        self,
        *,
        layer_idx: int,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        total_started_ns = self._host_profiler.start()
        validate_started_ns = self._host_profiler.start()
        self._validate_replacement_args(arguments)
        self._host_profiler.stop("read_gpu_kernel_validate_args", validate_started_ns)

        import torch

        pool = self._require_pool()
        out = arguments["out"]
        query = arguments["query"]
        block_tables = arguments["block_tables"]
        seq_lens = arguments["seq_lens"]
        offset_table = self._last_offset_table
        if offset_table is None:
            raise KcmmError("KCMM read replacement has no offset table")

        query_shape = _shape(query)
        block_tables_shape = _shape(block_tables)
        if query_shape is None or len(query_shape) != 3:
            raise KcmmError(f"invalid query shape for KCMM read kernel: {query_shape}")
        if block_tables_shape is None or len(block_tables_shape) != 2:
            raise KcmmError(
                f"invalid block_tables shape for KCMM read kernel: {block_tables_shape}"
            )
        if str(getattr(query, "dtype", "")) != "torch.float16":
            raise KcmmError(f"KCMM read kernel requires FP16 query, got {query.dtype}")
        if str(getattr(out, "dtype", "")) != "torch.float16":
            raise KcmmError(f"KCMM read kernel requires FP16 out, got {out.dtype}")
        if str(getattr(block_tables, "dtype", "")) != "torch.int32":
            raise KcmmError(
                f"KCMM read kernel requires int32 block_tables, got {block_tables.dtype}"
            )
        if str(getattr(seq_lens, "dtype", "")) != "torch.int32":
            raise KcmmError(f"KCMM read kernel requires int32 seq_lens, got {seq_lens.dtype}")

        device = getattr(query, "device", None)
        device_index = getattr(device, "index", None)
        if device_index is None:
            device_index = torch.cuda.current_device()

        # The KCMM kernel indexes tensors by raw pointer and therefore requires
        # compact layouts. These copies enqueue on PyTorch's current stream, and
        # forced non-default stream mode waits on that stream before launch.
        prepare_started_ns = self._host_profiler.start()
        query = query.contiguous()
        block_tables = block_tables.contiguous()
        seq_lens = seq_lens.contiguous()
        if not out.is_contiguous():
            out_tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        else:
            out_tmp = out
        self._host_profiler.stop("read_gpu_kernel_prepare_tensors", prepare_started_ns)

        stream_started_ns = self._host_profiler.start()
        stream_selection = self._stream_provider.select(device_index)
        self._host_profiler.stop("read_gpu_kernel_select_stream", stream_started_ns)
        record_started_ns = self._host_profiler.start()
        self._stream_provider.record_tensors(
            stream_selection,
            query,
            block_tables,
            seq_lens,
            offset_table,
            out_tmp,
        )
        self._host_profiler.stop("read_gpu_kernel_record_tensors", record_started_ns)
        start_event = None
        end_event = None
        if self._profile_gpu_kernel:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream_selection.stream)
        ptr_started_ns = self._host_profiler.start()
        query_ptr = _data_ptr(query, "query")
        out_ptr = _data_ptr(out_tmp, "out")
        block_tables_ptr = _data_ptr(block_tables, "block_tables")
        seq_lens_ptr = _data_ptr(seq_lens, "seq_lens")
        block_offsets_f16_ptr = _data_ptr(offset_table, "block_offsets_f16")
        self._host_profiler.stop("read_gpu_kernel_data_ptrs", ptr_started_ns)
        launch_started_ns = self._host_profiler.start()
        batch = int(query_shape[0])
        num_q_heads = int(query_shape[1])
        kv_heads = int(arguments["num_kv_heads"])
        head_dim = int(query_shape[2])
        block_size = int(arguments["block_size"])
        max_blocks_per_seq = int(block_tables_shape[1])
        scale = float(arguments["scale"])
        if self._fast_current_context_launch:
            pool.paged_attn_decode_f16_on_current_context_stream(
                layer_idx,
                query_ptr,
                out_ptr,
                block_tables_ptr,
                seq_lens_ptr,
                block_offsets_f16_ptr,
                batch,
                num_q_heads,
                kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
                scale,
                stream_selection.stream_ptr,
            )
        else:
            pool.paged_attn_decode_f16(
                layer_idx=layer_idx,
                query_ptr=query_ptr,
                out_ptr=out_ptr,
                block_tables_ptr=block_tables_ptr,
                seq_lens_ptr=seq_lens_ptr,
                block_offsets_f16_ptr=block_offsets_f16_ptr,
                batch=batch,
                num_q_heads=num_q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                max_blocks_per_seq=max_blocks_per_seq,
                scale=scale,
                stream_ptr=stream_selection.stream_ptr,
            )
        self._host_profiler.stop("read_gpu_kernel_ctypes_launch", launch_started_ns)
        if end_event is not None:
            end_event.record(stream_selection.stream)
        complete_started_ns = self._host_profiler.start()
        self._stream_provider.complete(stream_selection)
        self._host_profiler.stop("read_gpu_kernel_complete_stream", complete_started_ns)
        if out_tmp is not out:
            copy_started_ns = self._host_profiler.start()
            out.copy_(out_tmp)
            self._host_profiler.stop("read_gpu_kernel_copy_out", copy_started_ns)
        elapsed_ms = None
        if start_event is not None and end_event is not None:
            end_event.synchronize()
            elapsed_ms = float(start_event.elapsed_time(end_event))
        self._host_profiler.stop("read_gpu_kernel_host_total", total_started_ns)
        return {
            "stream_ptr": stream_selection.stream_ptr,
            "original_stream_ptr": stream_selection.original_stream_ptr,
            "default_stream_ptr": stream_selection.default_stream_ptr,
            "forced_non_default_stream": stream_selection.forced_non_default,
            "gpu_kernel_elapsed_ms": elapsed_ms,
        }

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
                "mode": (
                    "kv_read_replace_candidate"
                    if self._replace_native
                    else "kv_read_offset_table_plan"
                ),
                "candidate": "A2",
                "kernel_replaced": self._replace_native,
                "replacement_backend": self._replacement_backend,
                "read_path": (
                    "native_vllm_paged_attention"
                    if not self._replace_native
                    else (
                        "kcmm_paged_attn_decode_f16"
                        if self._replacement_backend == "gpu_kernel"
                        else "kcmm_reference_attention"
                    )
                ),
                "force_non_default_stream": (
                    self._stream_provider.force_non_default
                ),
                "report_on_update": self._report_on_update,
                "report_write_count": self._report_write_count,
                "block_table_validation_enabled": self._validate_block_tables,
                "compact_plan_metadata": self._compact_plan_metadata,
                "compact_plan_metadata_calls": self._compact_plan_metadata_calls,
                "detailed_plan_metadata_calls": self._detailed_plan_metadata_calls,
                "fast_current_context_launch": self._fast_current_context_launch,
                "gpu_kernel_precompile_requested": (
                    self._gpu_kernel_precompile_requested
                ),
                "gpu_kernel_precompile_calls": self._gpu_kernel_precompile_calls,
                "gpu_kernel_precompile_succeeded": (
                    self._gpu_kernel_precompile_succeeded
                ),
                "gpu_kernel_precompile_elapsed_ms": (
                    self._gpu_kernel_precompile_elapsed_ms
                ),
                "offset_table_contract": "torch.int64[f16_va_offset_by_block_id]",
                "required_allocator_mode": "kcmm_backed_allocator",
                "pool_attached": self._pool is not None,
                "read_calls": self._read_calls,
                "planned_calls": self._planned_calls,
                "replacement_calls": self._replacement_calls,
                "gpu_kernel_calls": self._gpu_kernel_calls,
                "stream_aware_kernel_calls": self._stream_aware_kernel_calls,
                "forced_non_default_stream_calls": (
                    self._forced_non_default_stream_calls
                ),
                "offset_table_builds": self._offset_table_builds,
                "offset_table_cache_hits": self._offset_table_cache_hits,
                "offset_table_cache_rebuilds": self._offset_table_cache_rebuilds,
                "min_entries_total_blocks_calls": (
                    self._min_entries_total_blocks_calls
                ),
                "reference_read_bytes": self._reference_read_bytes,
                "total_block_table_entries": self._total_block_table_entries,
                "unique_block_ids_seen": len(self._unique_block_ids_seen),
                "max_block_id_seen": self._max_block_id_seen,
                "max_batch_seen": self._max_batch_seen,
                "last_stream_ptr": self._last_stream_ptr,
                "last_original_stream_ptr": self._last_original_stream_ptr,
                "last_default_stream_ptr": self._last_default_stream_ptr,
                "gpu_kernel_profile": self._gpu_kernel_profile_summary(),
                "host_profile": self._host_profiler.report(),
                "counts_by_function": dict(sorted(self._counts_by_function.items())),
                "cache_layers": [
                    asdict(layer)
                    for layer in sorted(
                        self._cache_layers.values(),
                        key=lambda item: item.layer_idx,
                    )
                ],
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
        self._report_write_count += 1
        self._report_path.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
