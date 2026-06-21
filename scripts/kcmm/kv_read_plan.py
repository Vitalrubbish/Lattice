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
    layer_idx: int
    batch: int | None
    query_shape: list[int] | None
    block_tables_shape: list[int] | None
    seq_lens_shape: list[int] | None
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
    native_replaced: bool
    replacement_backend: str
    reference_read_bytes: int
    gpu_kernel_launched: bool
    stream_ptr: int | None
    stream_aware_launch: bool


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
    ):
        self._pool: KcmmPool | None = None
        self._report_path = Path(report_path) if report_path else None
        self._replace_native = bool(replace_native)
        if replacement_backend not in {"reference", "gpu_kernel"}:
            raise ValueError(f"unsupported replacement backend: {replacement_backend}")
        self._replacement_backend = replacement_backend
        self._lock = threading.RLock()
        self._cache_layers: dict[tuple[int, int], CacheLayer] = {}
        self._driver: _CudaDriver | None = None
        self._read_calls = 0
        self._planned_calls = 0
        self._replacement_calls = 0
        self._gpu_kernel_calls = 0
        self._stream_aware_kernel_calls = 0
        self._offset_table_builds = 0
        self._reference_read_bytes = 0
        self._total_block_table_entries = 0
        self._unique_block_ids_seen: set[int] = set()
        self._max_block_id_seen: int | None = None
        self._max_batch_seen = 0
        self._counts_by_function: dict[str, int] = {}
        self._recent_calls: list[ReadPlanCall] = []
        self._error_count = 0
        self._last_error: str | None = None
        self._last_offset_table: Any | None = None

    @property
    def replace_native(self) -> bool:
        return self._replace_native

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

    def _build_plan(
        self,
        function_name: str,
        arguments: dict[str, Any],
    ) -> tuple[ReadPlanCall, list[int], list[int]]:
        pool = self._require_pool()
        block_tables = arguments["block_tables"]
        query = arguments.get("query")
        seq_lens = arguments.get("seq_lens")
        key_cache = arguments["key_cache"]
        value_cache = arguments["value_cache"]
        layer_idx = self._layer_for_cache(key_cache, value_cache)
        query_shape = _shape(query)
        block_tables_shape = _shape(block_tables)
        seq_lens_shape = _shape(seq_lens)
        batch = query_shape[0] if query_shape else None
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

        sample_ids = unique_ids[:16]
        call = ReadPlanCall(
            function=function_name,
            layer_idx=layer_idx,
            batch=batch,
            query_shape=query_shape,
            block_tables_shape=block_tables_shape,
            seq_lens_shape=seq_lens_shape,
            block_table_entries=len(block_ids),
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
                str(block_id): int(offsets_f16[block_id]) for block_id in sample_ids
            },
            native_replaced=self._replace_native,
            replacement_backend=self._replacement_backend,
            reference_read_bytes=0,
            gpu_kernel_launched=False,
            stream_ptr=None,
            stream_aware_launch=False,
        )
        return call, offsets_f16, unique_ids

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
                self.write_report()
            except BaseException as exc:
                self._record_error(exc)
                raise

    def replace_call(
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
                call, offsets_f16, unique_ids = self._build_plan(
                    function_name,
                    arguments,
                )
                if self._replacement_backend == "gpu_kernel":
                    stream_ptr = self._run_gpu_kernel_attention(
                        layer_idx=call.layer_idx,
                        arguments=arguments,
                    )
                    read_bytes = 0
                    call.gpu_kernel_launched = True
                    call.stream_ptr = stream_ptr
                    call.stream_aware_launch = True
                    self._gpu_kernel_calls += 1
                    self._stream_aware_kernel_calls += 1
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
                self.write_report()
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
    ) -> int:
        self._validate_replacement_args(arguments)

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
        stream_ptr = int(torch.cuda.current_stream(device_index).cuda_stream)
        pool.paged_attn_decode_f16(
            layer_idx=layer_idx,
            query_ptr=_data_ptr(query, "query"),
            out_ptr=_data_ptr(out, "out"),
            block_tables_ptr=_data_ptr(block_tables, "block_tables"),
            seq_lens_ptr=_data_ptr(seq_lens, "seq_lens"),
            block_offsets_f16_ptr=_data_ptr(offset_table, "block_offsets_f16"),
            batch=int(query_shape[0]),
            num_q_heads=int(query_shape[1]),
            kv_heads=int(arguments["num_kv_heads"]),
            head_dim=int(query_shape[2]),
            block_size=int(arguments["block_size"]),
            max_blocks_per_seq=int(block_tables_shape[1]),
            scale=float(arguments["scale"]),
            stream_ptr=stream_ptr,
        )
        return stream_ptr

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
                "offset_table_contract": "torch.int64[f16_va_offset_by_block_id]",
                "required_allocator_mode": "kcmm_backed_allocator",
                "pool_attached": self._pool is not None,
                "read_calls": self._read_calls,
                "planned_calls": self._planned_calls,
                "replacement_calls": self._replacement_calls,
                "gpu_kernel_calls": self._gpu_kernel_calls,
                "stream_aware_kernel_calls": self._stream_aware_kernel_calls,
                "offset_table_builds": self._offset_table_builds,
                "reference_read_bytes": self._reference_read_bytes,
                "total_block_table_entries": self._total_block_table_entries,
                "unique_block_ids_seen": len(self._unique_block_ids_seen),
                "max_block_id_seen": self._max_block_id_seen,
                "max_batch_seen": self._max_batch_seen,
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
        self._report_path.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
