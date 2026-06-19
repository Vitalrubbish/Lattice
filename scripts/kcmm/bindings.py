"""ctypes bindings for the KCMM C ABI.

This module intentionally has no vLLM dependency. It is the narrow seam between
Python launch code and the Rust `cdylib` exported from this repository.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


class KcmmError(RuntimeError):
    """Raised when a KCMM C API call reports failure."""


class KcmmConfig(ctypes.Structure):
    _fields_ = [
        ("block_size", ctypes.c_size_t),
        ("max_blocks", ctypes.c_size_t),
        ("cpu_cache_path", ctypes.c_char * 256),
        ("tiering", ctypes.c_int32),
        ("eviction_policy", ctypes.c_int32),
        ("prefetch_window", ctypes.c_size_t),
        ("max_batch_blocks", ctypes.c_size_t),
        ("device_ordinal", ctypes.c_size_t),
        ("num_layers", ctypes.c_size_t),
        ("kv_heads", ctypes.c_size_t),
        ("head_dim", ctypes.c_size_t),
        ("max_batch", ctypes.c_size_t),
        ("max_seq_len", ctypes.c_size_t),
        ("low_watermark_threshold", ctypes.c_float),
        ("background_evict_interval_ms", ctypes.c_uint64),
        ("attention_sink_blocks", ctypes.c_size_t),
        ("recent_window_blocks", ctypes.c_size_t),
    ]


class KcmmMetrics(ctypes.Structure):
    _fields_ = [
        ("ifr", ctypes.c_double),
        ("pme", ctypes.c_double),
        ("bu", ctypes.c_double),
        ("rfi", ctypes.c_double),
        ("gpu_blocks", ctypes.c_uint64),
        ("cpu_blocks", ctypes.c_uint64),
        ("nvme_blocks", ctypes.c_uint64),
        ("eviction_count", ctypes.c_uint64),
        ("restoration_count", ctypes.c_uint64),
    ]


class KcmmPoolStats(ctypes.Structure):
    _fields_ = [
        ("blocks_in_use", ctypes.c_uint32),
        ("total_blocks", ctypes.c_uint32),
        ("total_physical_blocks", ctypes.c_uint32),
        ("free_physical_blocks", ctypes.c_uint32),
        ("active_sequences", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
        ("blocks_per_superblock", ctypes.c_uint32),
        ("superblock_count", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("max_blocks_per_seq", ctypes.c_uint32),
        ("block_bytes", ctypes.c_uint32),
        ("tiering_enabled", ctypes.c_int32),
        ("sharing_enabled", ctypes.c_int32),
        ("physical_idle_ratio", ctypes.c_float),
    ]


class BlockLocation:
    GPU_RESIDENT = 0
    CPU_RESIDENT = 1
    NVME_RESIDENT = 2
    EVICTING = 3
    RESTORING = 4

    _NAMES = {
        GPU_RESIDENT: "gpu",
        CPU_RESIDENT: "cpu",
        NVME_RESIDENT: "nvme",
        EVICTING: "evicting",
        RESTORING: "restoring",
    }

    @classmethod
    def name(cls, value: int) -> str:
        return cls._NAMES.get(value, f"unknown({value})")


def _assert_layout() -> None:
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        return
    expected_offsets = {
        "max_seq_len": 336,
        "low_watermark_threshold": 344,
        "background_evict_interval_ms": 352,
        "attention_sink_blocks": 360,
        "recent_window_blocks": 368,
    }
    if ctypes.sizeof(KcmmConfig) != 376:
        raise KcmmError(
            f"kcmm_config_t ABI mismatch: expected size 376, "
            f"got {ctypes.sizeof(KcmmConfig)}"
        )
    for field, expected in expected_offsets.items():
        actual = getattr(KcmmConfig, field).offset
        if actual != expected:
            raise KcmmError(
                f"kcmm_config_t ABI mismatch: {field} offset "
                f"expected {expected}, got {actual}"
            )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_library_candidates() -> list[Path | str]:
    root = _repo_root()
    candidates: list[Path | str] = []
    env_path = os.environ.get("KCMM_LIB_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            root / "target" / "release" / "libbaseline_llm_os.so",
            root / "target" / "debug" / "libbaseline_llm_os.so",
            root / "target" / "release" / "deps" / "libbaseline_llm_os.so",
            root / "target" / "debug" / "deps" / "libbaseline_llm_os.so",
        ]
    )
    for name in ("kcmm", "baseline_llm_os"):
        found = ctypes.util.find_library(name)
        if found:
            candidates.append(found)
    return candidates


def resolve_library_path(explicit: str | None = None) -> Path | str:
    candidates: Iterable[Path | str]
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = default_library_candidates()

    checked: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Path):
            checked.append(str(candidate))
            if candidate.exists():
                return candidate
        else:
            checked.append(candidate)
            return candidate

    raise KcmmError(
        "Could not find KCMM shared library. Build it with "
        "`cargo build --release --features kcmm` or set KCMM_LIB_PATH. "
        f"Checked: {', '.join(checked)}"
    )


@dataclass(frozen=True)
class ObserverProbeResult:
    library_path: str
    allocated_blocks: list[int]
    block_va_offsets: list[int]
    block_locations: list[str]
    stats: dict[str, int | float]
    metrics: dict[str, int | float]


class KcmmLibrary:
    def __init__(self, path: str | Path | None = None):
        _assert_layout()
        self.path = resolve_library_path(str(path) if path else None)
        self.lib = ctypes.CDLL(str(self.path))
        self._bind_functions()

    def _bind_functions(self) -> None:
        pool = ctypes.c_void_p
        lib = self.lib

        lib.kcmm_pool_create.argtypes = [ctypes.POINTER(KcmmConfig)]
        lib.kcmm_pool_create.restype = pool
        lib.kcmm_pool_destroy.argtypes = [pool]
        lib.kcmm_pool_destroy.restype = None

        lib.kcmm_get_last_error.argtypes = [
            pool,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.kcmm_get_last_error.restype = ctypes.c_size_t
        lib.kcmm_clear_error.argtypes = [pool]
        lib.kcmm_clear_error.restype = None

        lib.kcmm_alloc_blocks.argtypes = [
            pool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.kcmm_alloc_blocks.restype = ctypes.c_int
        lib.kcmm_free_blocks.argtypes = [
            pool,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
        ]
        lib.kcmm_free_blocks.restype = ctypes.c_int

        lib.kcmm_register_sequence.argtypes = [
            pool,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.kcmm_register_sequence.restype = ctypes.c_int
        lib.kcmm_unregister_sequence.argtypes = [pool, ctypes.c_uint32]
        lib.kcmm_unregister_sequence.restype = ctypes.c_int
        lib.kcmm_append_block_to_sequence.argtypes = [
            pool,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.kcmm_append_block_to_sequence.restype = ctypes.c_int
        lib.kcmm_get_block_table.argtypes = [
            pool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.kcmm_get_block_table.restype = ctypes.c_int
        lib.kcmm_get_block_table_va_offsets.argtypes = [
            pool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.kcmm_get_block_table_va_offsets.restype = ctypes.c_int

        lib.kcmm_get_block_va_offset.argtypes = [pool, ctypes.c_uint32]
        lib.kcmm_get_block_va_offset.restype = ctypes.c_uint64
        lib.kcmm_get_va_k.argtypes = [pool, ctypes.c_uint32]
        lib.kcmm_get_va_k.restype = ctypes.c_uint64
        lib.kcmm_get_va_v.argtypes = [pool, ctypes.c_uint32]
        lib.kcmm_get_va_v.restype = ctypes.c_uint64
        lib.kcmm_append_kv_step.argtypes = [
            pool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.kcmm_append_kv_step.restype = ctypes.c_int
        lib.kcmm_get_block_location.argtypes = [
            pool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.kcmm_get_block_location.restype = ctypes.c_int

        lib.kcmm_get_metrics.argtypes = [pool, ctypes.POINTER(KcmmMetrics)]
        lib.kcmm_get_metrics.restype = ctypes.c_int
        lib.kcmm_get_pool_stats.argtypes = [pool, ctypes.POINTER(KcmmPoolStats)]
        lib.kcmm_get_pool_stats.restype = ctypes.c_int
        lib.kcmm_synchronize.argtypes = [pool]
        lib.kcmm_synchronize.restype = ctypes.c_int

    def create_pool(self, config: KcmmConfig) -> "KcmmPool":
        handle = self.lib.kcmm_pool_create(ctypes.byref(config))
        if not handle:
            raise KcmmError("kcmm_pool_create returned NULL")
        return KcmmPool(self, handle)


class KcmmPool:
    def __init__(self, library: KcmmLibrary, handle: int | ctypes.c_void_p):
        self.library = library
        self.handle = (
            handle if isinstance(handle, ctypes.c_void_p) else ctypes.c_void_p(handle)
        )
        self._destroyed = False

    def __enter__(self) -> "KcmmPool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.destroy()

    def _last_error(self) -> str:
        buf = ctypes.create_string_buffer(4096)
        written = self.library.lib.kcmm_get_last_error(
            self.handle, buf, ctypes.sizeof(buf)
        )
        if written:
            return buf.value.decode("utf-8", errors="replace")
        return "unknown KCMM error"

    def _check(self, rc: int, operation: str) -> None:
        if rc != 0:
            raise KcmmError(f"{operation} failed: {self._last_error()}")

    def destroy(self) -> None:
        if not self._destroyed:
            self.library.lib.kcmm_pool_destroy(self.handle)
            self._destroyed = True

    def alloc_blocks(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        out = (ctypes.c_uint32 * count)()
        rc = self.library.lib.kcmm_alloc_blocks(self.handle, count, out)
        self._check(rc, "kcmm_alloc_blocks")
        return [int(out[i]) for i in range(count)]

    def free_blocks(self, blocks: list[int]) -> None:
        if not blocks:
            return
        arr = (ctypes.c_uint32 * len(blocks))(*blocks)
        rc = self.library.lib.kcmm_free_blocks(self.handle, arr, len(blocks))
        self._check(rc, "kcmm_free_blocks")

    def register_sequence(self, block_table: list[int]) -> int:
        arr = (ctypes.c_uint32 * len(block_table))(*block_table)
        out = ctypes.c_uint32()
        rc = self.library.lib.kcmm_register_sequence(
            self.handle,
            arr,
            len(block_table),
            ctypes.byref(out),
        )
        self._check(rc, "kcmm_register_sequence")
        return int(out.value)

    def unregister_sequence(self, seq_idx: int) -> None:
        rc = self.library.lib.kcmm_unregister_sequence(self.handle, seq_idx)
        self._check(rc, "kcmm_unregister_sequence")

    def append_block_to_sequence(self, seq_idx: int, block_idx: int) -> None:
        rc = self.library.lib.kcmm_append_block_to_sequence(
            self.handle,
            seq_idx,
            block_idx,
        )
        self._check(rc, "kcmm_append_block_to_sequence")

    def block_table(self, seq_idx: int, max_blocks: int) -> list[int]:
        if max_blocks <= 0:
            return []
        out = (ctypes.c_uint32 * max_blocks)()
        count = ctypes.c_uint32()
        rc = self.library.lib.kcmm_get_block_table(
            self.handle,
            seq_idx,
            out,
            max_blocks,
            ctypes.byref(count),
        )
        self._check(rc, "kcmm_get_block_table")
        return [int(out[i]) for i in range(int(count.value))]

    def block_table_va_offsets(self, seq_idx: int, max_blocks: int) -> list[int]:
        if max_blocks <= 0:
            return []
        out = (ctypes.c_uint64 * max_blocks)()
        count = ctypes.c_uint32()
        rc = self.library.lib.kcmm_get_block_table_va_offsets(
            self.handle,
            seq_idx,
            out,
            max_blocks,
            ctypes.byref(count),
        )
        self._check(rc, "kcmm_get_block_table_va_offsets")
        return [int(out[i]) for i in range(int(count.value))]

    def block_va_offset(self, block_idx: int) -> int:
        return int(
            self.library.lib.kcmm_get_block_va_offset(self.handle, block_idx)
        )

    def va_k(self, layer: int) -> int:
        value = int(self.library.lib.kcmm_get_va_k(self.handle, layer))
        if value == 0:
            raise KcmmError(f"kcmm_get_va_k returned 0 for layer {layer}")
        return value

    def va_v(self, layer: int) -> int:
        value = int(self.library.lib.kcmm_get_va_v(self.handle, layer))
        if value == 0:
            raise KcmmError(f"kcmm_get_va_v returned 0 for layer {layer}")
        return value

    def append_kv_step(
        self,
        layer_idx: int,
        seq_indices: list[int],
        positions: list[int],
        k_src_ptr: int,
        v_src_ptr: int,
    ) -> None:
        if len(seq_indices) != len(positions):
            raise ValueError("seq_indices and positions must have equal length")
        batch = len(seq_indices)
        if batch <= 0:
            raise ValueError("batch must be positive")
        seq_arr = (ctypes.c_uint32 * batch)(*seq_indices)
        pos_arr = (ctypes.c_uint32 * batch)(*positions)
        rc = self.library.lib.kcmm_append_kv_step(
            self.handle,
            layer_idx,
            seq_arr,
            pos_arr,
            batch,
            int(k_src_ptr),
            int(v_src_ptr),
        )
        self._check(rc, "kcmm_append_kv_step")

    def block_location(self, block_idx: int) -> str:
        out = ctypes.c_uint32()
        rc = self.library.lib.kcmm_get_block_location(
            self.handle, block_idx, ctypes.byref(out)
        )
        self._check(rc, "kcmm_get_block_location")
        return BlockLocation.name(int(out.value))

    def metrics(self) -> dict[str, int | float]:
        out = KcmmMetrics()
        rc = self.library.lib.kcmm_get_metrics(self.handle, ctypes.byref(out))
        self._check(rc, "kcmm_get_metrics")
        return _structure_to_dict(out)

    def stats(self) -> dict[str, int | float]:
        out = KcmmPoolStats()
        rc = self.library.lib.kcmm_get_pool_stats(self.handle, ctypes.byref(out))
        self._check(rc, "kcmm_get_pool_stats")
        return _structure_to_dict(out)

    def synchronize(self) -> None:
        rc = self.library.lib.kcmm_synchronize(self.handle)
        self._check(rc, "kcmm_synchronize")

    def observer_probe(self, blocks: int = 1) -> ObserverProbeResult:
        allocated = self.alloc_blocks(blocks)
        try:
            offsets = [self.block_va_offset(block) for block in allocated]
            locations = [self.block_location(block) for block in allocated]
            stats = self.stats()
            metrics = self.metrics()
            return ObserverProbeResult(
                library_path=str(self.library.path),
                allocated_blocks=allocated,
                block_va_offsets=offsets,
                block_locations=locations,
                stats=stats,
                metrics=metrics,
            )
        finally:
            self.free_blocks(allocated)
            self.synchronize()


def _structure_to_dict(value: ctypes.Structure) -> dict[str, int | float]:
    return {field: getattr(value, field) for field, _ctype in value._fields_}


def probe_once(
    config: KcmmConfig,
    library_path: str | None = None,
    blocks: int = 1,
) -> ObserverProbeResult:
    library = KcmmLibrary(library_path)
    with library.create_pool(config) as pool:
        return pool.observer_probe(blocks=blocks)


def result_to_dict(result: ObserverProbeResult) -> dict[str, object]:
    return asdict(result)
