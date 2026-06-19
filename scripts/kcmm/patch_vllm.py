"""vLLM seam inspection and observer instrumentation for KCMM.

Phase I.C is deliberately observer-only: this module locates the vLLM classes
that later phases will patch, but does not alter vLLM behavior. Phase II.A uses
the same seam list for observer-only allocator instrumentation.
"""

from __future__ import annotations

import atexit
import importlib
import inspect
import json
import os
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable


SEAMS = (
    (
        "BlockSpaceManagerV2",
        "vllm.core.block_manager_v2",
        "BlockSpaceManagerV2",
    ),
    (
        "NaiveBlockAllocator",
        "vllm.core.block.naive_block",
        "NaiveBlockAllocator",
    ),
    (
        "PrefixCachingBlockAllocator",
        "vllm.core.block.prefix_caching_block",
        "PrefixCachingBlockAllocator",
    ),
    (
        "CpuGpuBlockAllocator",
        "vllm.core.block.cpu_gpu_block_allocator",
        "CpuGpuBlockAllocator",
    ),
)

ALLOCATOR_METHODS = {
    "vllm.core.block_manager_v2.BlockSpaceManagerV2": (
        "__init__",
        "allocate",
        "free",
        "append_slots",
        "can_allocate",
        "can_append_slots",
        "get_num_free_gpu_blocks",
        "get_num_free_cpu_blocks",
    ),
    "vllm.core.block.naive_block.NaiveBlockAllocator": (
        "__init__",
        "allocate_mutable_block",
        "allocate_immutable_block",
        "allocate_immutable_blocks",
        "free",
        "get_num_free_blocks",
    ),
    "vllm.core.block.prefix_caching_block.PrefixCachingBlockAllocator": (
        "__init__",
        "allocate_mutable_block",
        "allocate_immutable_block",
        "allocate_immutable_blocks",
        "free",
        "get_num_free_blocks",
    ),
}

REQUIRED_ALLOCATOR_GROUPS = {
    "block_manager_constructed": (
        "vllm.core.block_manager_v2.BlockSpaceManagerV2.__init__",
    ),
    "block_manager_allocate": (
        "vllm.core.block_manager_v2.BlockSpaceManagerV2.allocate",
    ),
    "block_manager_free": (
        "vllm.core.block_manager_v2.BlockSpaceManagerV2.free",
    ),
    "allocator_constructed": (
        "vllm.core.block.naive_block.NaiveBlockAllocator.__init__",
        "vllm.core.block.prefix_caching_block.PrefixCachingBlockAllocator.__init__",
    ),
    "allocator_allocate": (
        "vllm.core.block.naive_block.NaiveBlockAllocator.allocate_mutable_block",
        "vllm.core.block.naive_block.NaiveBlockAllocator.allocate_immutable_block",
        "vllm.core.block.naive_block.NaiveBlockAllocator.allocate_immutable_blocks",
        "vllm.core.block.prefix_caching_block.PrefixCachingBlockAllocator.allocate_mutable_block",
        "vllm.core.block.prefix_caching_block.PrefixCachingBlockAllocator.allocate_immutable_block",
        "vllm.core.block.prefix_caching_block.PrefixCachingBlockAllocator.allocate_immutable_blocks",
    ),
    "allocator_free": (
        "vllm.core.block.naive_block.NaiveBlockAllocator.free",
        "vllm.core.block.prefix_caching_block.PrefixCachingBlockAllocator.free",
    ),
}

_TRACE_LOCK = threading.RLock()
_TRACE_PATH: Path | None = None
_TRACE_COUNTS: dict[str, int] = {}
_TRACE_SEQUENCE = 0
_INSTRUMENTED = False
_REQUIRE_SEAMS = False
_RUNTIME_POOL_PATCHED = False
_SHADOW_ALLOCATOR_PATCHED = False
_KCMM_BACKED_ALLOCATOR_PATCHED = False
_KV_WRITE_TRACE_PATH: Path | None = None
_KV_WRITE_COUNTS: dict[str, int] = {}
_KV_WRITE_SEQUENCE = 0
_KV_WRITE_INSTRUMENTED = False
_REQUIRE_KV_WRITE_SEAMS = False
_KV_WRITE_MIRROR_PATCHED = False

KV_WRITE_FUNCTIONS = {
    "vllm._custom_ops": (
        "reshape_and_cache",
        "reshape_and_cache_flash",
    )
}

REQUIRED_KV_WRITE_GROUPS = {
    "kv_write_function_called": tuple(
        f"vllm._custom_ops.{name}" for name in KV_WRITE_FUNCTIONS["vllm._custom_ops"]
    ),
}


@dataclass(frozen=True)
class VllmSeam:
    name: str
    module: str
    attribute: str
    available: bool
    object_path: str | None = None
    init_signature: str | None = None
    error: str | None = None


def _object_path(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _method_key(class_path: str, method_name: str) -> str:
    return f"{class_path}.{method_name}"


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _safe_attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return None


def _safe_summary(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 2:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": [_safe_summary(item, depth + 1) for item in value[:5]],
        }
    if isinstance(value, (set, frozenset)):
        items = sorted((_safe_summary(item, depth + 1) for item in value), key=str)
        return {"type": type(value).__name__, "len": len(value), "items": items[:5]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "len": len(value),
            "items": [
                [_safe_summary(key, depth + 1), _safe_summary(val, depth + 1)]
                for key, val in list(value.items())[:5]
            ],
        }
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", "unknown"))
        return {"type": "callable", "path": f"{module}.{qualname}"}
    if hasattr(value, "name") and hasattr(value, "value"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": getattr(value, "name", None),
            "value": getattr(value, "value", None),
        }

    summary: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}"
    }
    for attr in (
        "block_id",
        "block_size",
        "num_blocks",
        "num_total_blocks",
        "device",
        "status",
    ):
        attr_value = _safe_attr(value, attr)
        if attr_value is not None:
            summary[attr] = _safe_summary(attr_value, depth + 1)
    length = _safe_len(value)
    if length is not None:
        summary["len"] = length
    return summary


def _tensor_summary(value: Any, include_values: bool = False) -> Any:
    summary = _safe_summary(value)
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return summary

    try:
        shape = [int(dim) for dim in value.shape]
    except Exception:
        shape = None
    try:
        stride = [int(dim) for dim in value.stride()]
    except Exception:
        stride = None
    result: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "shape": shape,
        "dtype": str(getattr(value, "dtype", "unknown")),
        "device": str(getattr(value, "device", "unknown")),
        "is_cuda": bool(getattr(value, "is_cuda", False)),
        "stride": stride,
    }
    for method_name in ("numel", "element_size", "data_ptr"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                result[method_name] = int(method())
            except Exception:
                pass
    if include_values:
        result["values_sample"] = _tensor_values_sample(value)
    return result


def _tensor_values_sample(value: Any, max_items: int = 32) -> Any:
    try:
        flattened = value.detach().flatten()
        total = int(flattened.numel())
        sample = flattened[:max_items].cpu().tolist()
        return {
            "total": total,
            "sample_count": len(sample),
            "sample": [int(item) for item in sample],
            "truncated": total > len(sample),
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
        }


def _shape_list(value: Any) -> list[int] | None:
    try:
        return [int(dim) for dim in value.shape]
    except Exception:
        return None


def _infer_kv_cache_layout(key_cache: Any, value_cache: Any) -> dict[str, Any]:
    key_shape = _shape_list(key_cache)
    value_shape = _shape_list(value_cache)
    if key_shape is None or value_shape is None:
        return {
            "layout": "unknown",
            "reason": "missing cache tensor shapes",
            "key_cache_shape": key_shape,
            "value_cache_shape": value_shape,
        }

    if len(key_shape) == 5 and len(value_shape) == 4:
        block_size = key_shape[3]
        if value_shape[3] != block_size:
            return {
                "layout": "unknown",
                "reason": "key/value cache block-size dimensions differ",
                "key_cache_shape": key_shape,
                "value_cache_shape": value_shape,
            }
        return {
            "layout": "paged_kv_cache",
            "slot_formula": "slot = block_id * block_size + offset_in_block",
            "num_blocks": key_shape[0],
            "block_size": block_size,
            "key_cache_shape": key_shape,
            "value_cache_shape": value_shape,
        }

    if len(key_shape) == 4 and len(value_shape) == 4:
        block_size = key_shape[1]
        if value_shape[1] != block_size:
            return {
                "layout": "unknown",
                "reason": "flash key/value cache block-size dimensions differ",
                "key_cache_shape": key_shape,
                "value_cache_shape": value_shape,
            }
        return {
            "layout": "flash_kv_cache",
            "slot_formula": "slot = block_id * block_size + offset_in_block",
            "num_blocks": key_shape[0],
            "block_size": block_size,
            "key_cache_shape": key_shape,
            "value_cache_shape": value_shape,
        }

    return {
        "layout": "unknown",
        "reason": "unsupported cache tensor rank",
        "key_cache_shape": key_shape,
        "value_cache_shape": value_shape,
    }


def _slot_mapping_contract(
    slot_mapping: Any,
    key_cache: Any,
    value_cache: Any,
) -> dict[str, Any]:
    layout = _infer_kv_cache_layout(key_cache, value_cache)
    block_size = layout.get("block_size")
    num_blocks = layout.get("num_blocks")
    values = _tensor_values_sample(slot_mapping)
    if not isinstance(block_size, int) or not isinstance(num_blocks, int):
        return {
            **layout,
            "valid": False,
            "reason": "could not infer block_size and num_blocks",
            "slot_sample": values,
        }
    if block_size <= 0 or num_blocks <= 0:
        return {
            **layout,
            "valid": False,
            "reason": "invalid block_size or num_blocks",
            "slot_sample": values,
        }

    decoded: list[dict[str, Any]] = []
    invalid_slots: list[int] = []
    sample = values.get("sample", []) if isinstance(values, dict) else []
    for raw_slot in sample:
        slot = int(raw_slot)
        if slot < 0:
            decoded.append(
                {
                    "slot": slot,
                    "is_padding": True,
                    "valid": True,
                }
            )
            continue
        block_id = slot // block_size
        offset_in_block = slot % block_size
        valid = block_id < num_blocks
        if not valid:
            invalid_slots.append(slot)
        decoded.append(
            {
                "slot": slot,
                "is_padding": False,
                "block_id": block_id,
                "offset_in_block": offset_in_block,
                "valid": valid,
            }
        )

    return {
        **layout,
        "valid": not invalid_slots,
        "slot_sample": values,
        "decoded_sample": decoded,
        "invalid_slots": invalid_slots,
    }


def _kv_write_args_summary(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    kv_cache_dtype: Any,
    k_scale: Any,
    v_scale: Any,
) -> dict[str, Any]:
    return {
        "key": _tensor_summary(key),
        "value": _tensor_summary(value),
        "key_cache": _tensor_summary(key_cache),
        "value_cache": _tensor_summary(value_cache),
        "slot_mapping": _tensor_summary(slot_mapping, include_values=True),
        "slot_mapping_contract": _slot_mapping_contract(
            slot_mapping,
            key_cache,
            value_cache,
        ),
        "kv_cache_dtype": _safe_summary(kv_cache_dtype),
        "k_scale": _safe_summary(k_scale),
        "v_scale": _safe_summary(v_scale),
    }


def _bound_arguments(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(fn)
        bound = signature.bind_partial(None, *args, **kwargs)
        return {
            key: _safe_summary(value)
            for key, value in bound.arguments.items()
            if key != "self"
        }
    except Exception:
        return {
            "args": _safe_summary(list(args)),
            "kwargs": _safe_summary(kwargs),
        }


def _write_trace(event: dict[str, Any]) -> None:
    global _TRACE_SEQUENCE
    path = _TRACE_PATH
    if path is None:
        return
    with _TRACE_LOCK:
        _TRACE_SEQUENCE += 1
        payload = {"seq": _TRACE_SEQUENCE, **event}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_kv_write_trace(event: dict[str, Any]) -> None:
    global _KV_WRITE_SEQUENCE
    path = _KV_WRITE_TRACE_PATH
    if path is None:
        return
    with _TRACE_LOCK:
        _KV_WRITE_SEQUENCE += 1
        payload = {"seq": _KV_WRITE_SEQUENCE, **event}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _record_call(
    class_path: str,
    method_name: str,
    args_summary: dict[str, Any],
    result_summary: Any = None,
    error: BaseException | None = None,
) -> None:
    key = _method_key(class_path, method_name)
    _TRACE_COUNTS[key] = _TRACE_COUNTS.get(key, 0) + 1
    event: dict[str, Any] = {
        "event": "method_call",
        "class": class_path,
        "method": method_name,
        "key": key,
        "count": _TRACE_COUNTS[key],
        "args": args_summary,
    }
    if error is None:
        event["result"] = result_summary
    else:
        event["error"] = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
    _write_trace(event)


def _record_kv_write_call(
    key: str,
    args_summary: dict[str, Any],
    error: BaseException | None = None,
) -> None:
    _KV_WRITE_COUNTS[key] = _KV_WRITE_COUNTS.get(key, 0) + 1
    event: dict[str, Any] = {
        "event": "kv_write_call",
        "key": key,
        "count": _KV_WRITE_COUNTS[key],
        "args": args_summary,
    }
    if error is not None:
        event["error"] = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
    _write_kv_write_trace(event)


def _missing_required_groups() -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for group, keys in REQUIRED_ALLOCATOR_GROUPS.items():
        if not any(_TRACE_COUNTS.get(key, 0) > 0 for key in keys):
            missing[group] = list(keys)
    return missing


def _missing_required_kv_write_groups() -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for group, keys in REQUIRED_KV_WRITE_GROUPS.items():
        if not any(_KV_WRITE_COUNTS.get(key, 0) > 0 for key in keys):
            missing[group] = list(keys)
    return missing


def _write_trace_summary() -> None:
    missing = _missing_required_groups()
    _write_trace(
        {
            "event": "summary",
            "counts": dict(sorted(_TRACE_COUNTS.items())),
            "required_groups": REQUIRED_ALLOCATOR_GROUPS,
            "missing_required_groups": missing,
        }
    )
    if _REQUIRE_SEAMS and missing:
        print(
            "KCMM allocator instrumentation missing required seams: "
            + json.dumps(missing, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )


def _write_kv_write_trace_summary() -> None:
    missing = _missing_required_kv_write_groups()
    _write_kv_write_trace(
        {
            "event": "summary",
            "counts": dict(sorted(_KV_WRITE_COUNTS.items())),
            "required_groups": REQUIRED_KV_WRITE_GROUPS,
            "missing_required_groups": missing,
        }
    )
    if _REQUIRE_KV_WRITE_SEAMS and missing:
        print(
            "KCMM KV write instrumentation missing required seams: "
            + json.dumps(missing, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )


def _wrap_method(class_path: str, cls: type, method_name: str) -> None:
    original = getattr(cls, method_name)
    if getattr(original, "_kcmm_instrumented", False):
        return

    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        args_summary = _bound_arguments(original, args, kwargs)
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            _record_call(class_path, method_name, args_summary, error=exc)
            raise
        _record_call(
            class_path,
            method_name,
            args_summary,
            result_summary=_safe_summary(result),
        )
        return result

    wrapper._kcmm_instrumented = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapper)


def _wrap_kv_write_function(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if getattr(original, "_kcmm_kv_write_instrumented", False):
        return
    call_key = f"{module.__name__}.{function_name}"
    signature = inspect.signature(original)

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        args_summary = _kv_write_args_summary(
            bound.arguments["key"],
            bound.arguments["value"],
            bound.arguments["key_cache"],
            bound.arguments["value_cache"],
            bound.arguments["slot_mapping"],
            bound.arguments["kv_cache_dtype"],
            bound.arguments["k_scale"],
            bound.arguments["v_scale"],
        )
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _record_kv_write_call(call_key, args_summary, error=exc)
            raise
        _record_kv_write_call(call_key, args_summary)
        return result

    wrapper._kcmm_kv_write_instrumented = True  # type: ignore[attr-defined]
    setattr(module, function_name, wrapper)


def _wrap_kv_write_mirror_function(
    module: Any,
    function_name: str,
    mirror: Any,
) -> None:
    original = getattr(module, function_name)
    if getattr(original, "_kcmm_kv_write_mirror_patched", False):
        return
    call_key = f"{module.__name__}.{function_name}"
    signature = inspect.signature(original)

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        result = original(*args, **kwargs)
        mirror.mirror_call(
            call_key,
            bound.arguments["key"],
            bound.arguments["value"],
            bound.arguments["key_cache"],
            bound.arguments["value_cache"],
            bound.arguments["slot_mapping"],
        )
        return result

    wrapper._kcmm_kv_write_mirror_patched = True  # type: ignore[attr-defined]
    setattr(module, function_name, wrapper)


def _wrap_llm_engine_init(callback: Callable[[Any], dict[str, Any]]) -> bool:
    module = importlib.import_module("vllm.engine.llm_engine")
    cls = getattr(module, "LLMEngine")
    original = getattr(cls, "__init__")
    if getattr(original, "_kcmm_runtime_pool_patched", False):
        return False

    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        try:
            report = callback(self)
        except BaseException as exc:
            print(
                "KCMM runtime-derived pool initialization failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        _write_trace(
            {
                "event": "runtime_pool_sized",
                "report": _safe_summary(report),
            }
        )

    wrapper._kcmm_runtime_pool_patched = True  # type: ignore[attr-defined]
    setattr(cls, "__init__", wrapper)
    return True


def _is_gpu_device(device: Any) -> bool:
    try:
        from vllm.core.block.interfaces import Device

        return device == Device.GPU
    except Exception:
        return str(device).endswith("GPU") or getattr(device, "name", None) == "GPU"


def _is_gpu_block(cpu_gpu_allocator: Any, block_id: int) -> bool:
    try:
        from vllm.core.block.interfaces import Device

        allocator = cpu_gpu_allocator._block_ids_to_allocator[block_id]
        return allocator is cpu_gpu_allocator._allocators[Device.GPU]
    except Exception:
        return False


def _block_id(block: Any) -> int:
    value = getattr(block, "block_id", None)
    if value is None:
        raise RuntimeError("vLLM block has no block_id")
    return int(value)


def _wrap_shadow_allocator_method(cls: type, method_name: str, shadow: Any) -> None:
    original = getattr(cls, method_name)
    if getattr(original, "_kcmm_shadow_allocator_patched", False):
        return

    @wraps(original)
    def allocate_one(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            device = args[-1]
        if _is_gpu_device(device):
            shadow.mirror_allocated_block(
                _block_id(result),
                f"{cls.__module__}.{cls.__qualname__}.{method_name}",
            )
        return result

    @wraps(original)
    def allocate_many(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            device = args[-1]
        if _is_gpu_device(device):
            shadow.mirror_allocated_blocks(
                list(result),
                f"{cls.__module__}.{cls.__qualname__}.{method_name}",
            )
        return result

    @wraps(original)
    def free_one(self: Any, block: Any, *args: Any, **kwargs: Any) -> Any:
        module = importlib.import_module("vllm.core.block.cpu_gpu_block_allocator")
        null_block = getattr(module, "NullBlock")
        if isinstance(block, null_block):
            return original(self, block, *args, **kwargs)

        block_id = _block_id(block)
        should_mirror = _is_gpu_block(self, block_id)
        result = original(self, block, *args, **kwargs)
        if should_mirror:
            shadow.mirror_freed_block(
                block_id,
                f"{cls.__module__}.{cls.__qualname__}.{method_name}",
            )
        return result

    @wraps(original)
    def clear_copy_on_writes(self: Any, *args: Any, **kwargs: Any) -> Any:
        mappings = original(self, *args, **kwargs)
        source = f"{cls.__module__}.{cls.__qualname__}.{method_name}"
        for src_block_id, dst_block_id in mappings:
            shadow.mirror_freed_block(int(src_block_id), source)
            shadow.mirror_allocated_block(int(dst_block_id), source)
        return mappings

    @wraps(original)
    def fork(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        for block in result:
            block_id = _block_id(block)
            if _is_gpu_block(self, block_id):
                shadow.mirror_allocated_block(
                    block_id,
                    f"{cls.__module__}.{cls.__qualname__}.{method_name}",
                )
        return result

    wrapper_by_method = {
        "allocate_mutable_block": allocate_one,
        "allocate_immutable_block": allocate_one,
        "allocate_immutable_blocks": allocate_many,
        "free": free_one,
        "clear_copy_on_writes": clear_copy_on_writes,
        "fork": fork,
    }
    wrapper = wrapper_by_method[method_name]
    wrapper._kcmm_shadow_allocator_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapper)


def _wrap_kcmm_backed_cpu_gpu_init(tracker: Any) -> None:
    module = importlib.import_module("vllm.core.block.cpu_gpu_block_allocator")
    cls = getattr(module, "CpuGpuBlockAllocator")
    original = getattr(cls, "__init__")
    if getattr(original, "_kcmm_backed_allocator_patched", False):
        return

    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        from vllm.core.block.interfaces import Device

        tracker.bind_vllm_gpu_allocator(self._allocators[Device.GPU])

    wrapper._kcmm_backed_allocator_patched = True  # type: ignore[attr-defined]
    setattr(cls, "__init__", wrapper)


def _wrap_kcmm_backed_naive_allocator_methods(tracker: Any) -> None:
    module = importlib.import_module("vllm.core.block.naive_block")
    cls = getattr(module, "NaiveBlockAllocator")

    original_alloc = getattr(cls, "_allocate_block_id")
    if not getattr(original_alloc, "_kcmm_backed_allocator_patched", False):

        @wraps(original_alloc)
        def allocate_block_id(self: Any) -> Any:
            active_tracker = getattr(self, "_kcmm_backed_tracker", None)
            if active_tracker is None:
                return original_alloc(self)
            return active_tracker.allocate_block_id(
                self,
                f"{cls.__module__}.{cls.__qualname__}._allocate_block_id",
            )

        allocate_block_id._kcmm_backed_allocator_patched = True  # type: ignore[attr-defined]
        setattr(cls, "_allocate_block_id", allocate_block_id)

    original_free = getattr(cls, "_free_block_id")
    if not getattr(original_free, "_kcmm_backed_allocator_patched", False):

        @wraps(original_free)
        def free_block_id(self: Any, block: Any) -> Any:
            active_tracker = getattr(self, "_kcmm_backed_tracker", None)
            if active_tracker is None:
                return original_free(self, block)

            block_id = _block_id(block)
            refcount_before = int(self._refcounter.get(block_id))
            result = original_free(self, block)
            active_tracker.free_block_id(
                block_id,
                released=refcount_before == 1,
                source=f"{cls.__module__}.{cls.__qualname__}._free_block_id",
            )
            return result

        free_block_id._kcmm_backed_allocator_patched = True  # type: ignore[attr-defined]
        setattr(cls, "_free_block_id", free_block_id)


def inspect_vllm_seams() -> dict[str, object]:
    try:
        import vllm

        version = getattr(vllm, "__version__", "unknown")
    except Exception as exc:
        return {
            "phase": "I.C",
            "patched": False,
            "vllm_version": None,
            "error": repr(exc),
            "seams": [],
        }

    seams: list[VllmSeam] = []
    for name, module_name, attribute in SEAMS:
        try:
            module = importlib.import_module(module_name)
            value = getattr(module, attribute)
            try:
                init_signature = str(inspect.signature(value.__init__))
            except (TypeError, ValueError):
                init_signature = None
            seams.append(
                VllmSeam(
                    name=name,
                    module=module_name,
                    attribute=attribute,
                    available=True,
                    object_path=_object_path(value),
                    init_signature=init_signature,
                )
            )
        except Exception as exc:
            seams.append(
                VllmSeam(
                    name=name,
                    module=module_name,
                    attribute=attribute,
                    available=False,
                    error=repr(exc),
                )
            )

    return {
        "phase": "I.C",
        "patched": False,
        "vllm_version": version,
        "seams": [asdict(seam) for seam in seams],
    }


def apply_observer_patches() -> dict[str, object]:
    report = inspect_vllm_seams()
    report["reason"] = "observer-only phase; no monkey-patching applied"
    return report


def apply_runtime_pool_sizing(
    callback: Callable[[Any], dict[str, Any]],
) -> dict[str, object]:
    """Patch vLLM so KCMM can size its pool from live engine config."""

    global _RUNTIME_POOL_PATCHED

    report = inspect_vllm_seams()
    patched = False
    if not _RUNTIME_POOL_PATCHED:
        patched = _wrap_llm_engine_init(callback)
        _RUNTIME_POOL_PATCHED = True

    return {
        "phase": "II.A",
        "patched": True,
        "observer_only": True,
        "target": "vllm.engine.llm_engine.LLMEngine.__init__",
        "new_patch_installed": patched,
        "seam_report": report,
    }


def apply_shadow_allocator(shadow: Any) -> dict[str, object]:
    """Patch vLLM's device-aware allocator to mirror GPU block lifetimes."""

    global _SHADOW_ALLOCATOR_PATCHED

    module = importlib.import_module("vllm.core.block.cpu_gpu_block_allocator")
    cls = getattr(module, "CpuGpuBlockAllocator")
    patched: list[str] = []
    if not _SHADOW_ALLOCATOR_PATCHED:
        for method_name in (
            "allocate_mutable_block",
            "allocate_immutable_block",
            "allocate_immutable_blocks",
            "free",
            "clear_copy_on_writes",
            "fork",
        ):
            if hasattr(cls, method_name):
                _wrap_shadow_allocator_method(cls, method_name, shadow)
                patched.append(method_name)
        _SHADOW_ALLOCATOR_PATCHED = True

    return {
        "phase": "II.A",
        "patched": True,
        "observer_only": True,
        "target": "vllm.core.block.cpu_gpu_block_allocator.CpuGpuBlockAllocator",
        "patched_methods": patched,
        "seam_report": inspect_vllm_seams(),
    }


def apply_kcmm_backed_allocator(tracker: Any) -> dict[str, object]:
    """Patch vLLM so KCMM chooses GPU block IDs for the V2 allocator."""

    global _KCMM_BACKED_ALLOCATOR_PATCHED

    patched: list[str] = []
    if not _KCMM_BACKED_ALLOCATOR_PATCHED:
        _wrap_kcmm_backed_cpu_gpu_init(tracker)
        _wrap_kcmm_backed_naive_allocator_methods(tracker)
        patched = [
            "vllm.core.block.cpu_gpu_block_allocator.CpuGpuBlockAllocator.__init__",
            "vllm.core.block.naive_block.NaiveBlockAllocator._allocate_block_id",
            "vllm.core.block.naive_block.NaiveBlockAllocator._free_block_id",
        ]
        _KCMM_BACKED_ALLOCATOR_PATCHED = True

    return {
        "phase": "II.A",
        "patched": True,
        "observer_only": False,
        "target": "vLLM V2 GPU NaiveBlockAllocator",
        "patched_methods": patched,
        "storage_of_record": "native_vllm_kv_tensors",
        "seam_report": inspect_vllm_seams(),
    }


def apply_allocator_instrumentation(
    trace_path: str | None = None,
    require_seams: bool = False,
) -> dict[str, object]:
    global _INSTRUMENTED, _REQUIRE_SEAMS, _TRACE_PATH

    _REQUIRE_SEAMS = require_seams
    if trace_path is None:
        trace_path = os.environ.get("KCMM_ALLOCATOR_TRACE_PATH")
    if trace_path is None:
        trace_path = str(Path(tempfile.gettempdir()) / "kcmm-vllm-allocator-trace.jsonl")

    _TRACE_PATH = Path(trace_path)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.write_text("", encoding="utf-8")

    report = inspect_vllm_seams()
    patched: list[dict[str, object]] = []
    if not _INSTRUMENTED:
        for class_path, method_names in ALLOCATOR_METHODS.items():
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            for method_name in method_names:
                if hasattr(cls, method_name):
                    _wrap_method(class_path, cls, method_name)
                    patched.append({"class": class_path, "method": method_name})
        atexit.register(_write_trace_summary)
        _INSTRUMENTED = True

    _write_trace(
        {
            "event": "instrumentation_enabled",
            "trace_path": str(_TRACE_PATH),
            "require_seams": require_seams,
            "patched": patched,
            "seam_report": report,
        }
    )

    return {
        "phase": "II.A",
        "patched": True,
        "observer_only": True,
        "trace_path": str(_TRACE_PATH),
        "require_seams": require_seams,
        "patched_methods": patched,
        "seam_report": report,
    }


def apply_kv_write_instrumentation(
    trace_path: str | None = None,
    require_seams: bool = False,
) -> dict[str, object]:
    """Patch vLLM's KV write custom-op wrappers to record the write contract."""

    global _KV_WRITE_INSTRUMENTED, _REQUIRE_KV_WRITE_SEAMS, _KV_WRITE_TRACE_PATH

    _REQUIRE_KV_WRITE_SEAMS = require_seams
    if trace_path is None:
        trace_path = os.environ.get("KCMM_KV_WRITE_TRACE_PATH")
    if trace_path is None:
        trace_path = str(Path(tempfile.gettempdir()) / "kcmm-vllm-kv-write-trace.jsonl")

    _KV_WRITE_TRACE_PATH = Path(trace_path)
    _KV_WRITE_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _KV_WRITE_TRACE_PATH.write_text("", encoding="utf-8")

    patched: list[str] = []
    if not _KV_WRITE_INSTRUMENTED:
        for module_name, function_names in KV_WRITE_FUNCTIONS.items():
            module = importlib.import_module(module_name)
            for function_name in function_names:
                if hasattr(module, function_name):
                    _wrap_kv_write_function(module, function_name)
                    patched.append(f"{module_name}.{function_name}")
        atexit.register(_write_kv_write_trace_summary)
        _KV_WRITE_INSTRUMENTED = True

    _write_kv_write_trace(
        {
            "event": "kv_write_instrumentation_enabled",
            "trace_path": str(_KV_WRITE_TRACE_PATH),
            "require_seams": require_seams,
            "patched": patched,
            "required_groups": REQUIRED_KV_WRITE_GROUPS,
        }
    )

    return {
        "phase": "II.B",
        "patched": True,
        "observer_only": True,
        "trace_path": str(_KV_WRITE_TRACE_PATH),
        "require_seams": require_seams,
        "patched_functions": patched,
        "required_groups": REQUIRED_KV_WRITE_GROUPS,
    }


def apply_kv_write_mirror(mirror: Any) -> dict[str, object]:
    """Patch vLLM's KV write custom-op wrappers to mirror writes into KCMM."""

    global _KV_WRITE_MIRROR_PATCHED

    patched: list[str] = []
    if not _KV_WRITE_MIRROR_PATCHED:
        for module_name, function_names in KV_WRITE_FUNCTIONS.items():
            module = importlib.import_module(module_name)
            for function_name in function_names:
                if hasattr(module, function_name):
                    _wrap_kv_write_mirror_function(module, function_name, mirror)
                    patched.append(f"{module_name}.{function_name}")
        _KV_WRITE_MIRROR_PATCHED = True

    return {
        "phase": "II.B",
        "patched": True,
        "observer_only": False,
        "target": "vLLM KV write custom ops",
        "write_path": "kcmm_append_kv_slots",
        "storage_of_record": "native_vllm_kv_tensors",
        "patched_functions": patched,
        "required_allocator_mode": "kcmm_backed_allocator",
    }
