"""vLLM seam inspection for KCMM Phase I.C.

Phase I.C is deliberately observer-only: this module locates the vLLM classes
that later phases will patch, but does not alter vLLM behavior.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass
from typing import Any


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
)


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
