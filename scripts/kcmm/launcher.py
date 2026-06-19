"""Launch vLLM with the KCMM Phase I.C observer pool."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from typing import Sequence

from .bindings import KcmmError, KcmmLibrary, KcmmPool, result_to_dict
from .config import ObserverConfig, add_kcmm_args
from .patch_vllm import apply_allocator_instrumentation, apply_observer_patches


_ACTIVE_POOL: KcmmPool | None = None


def _print_json(payload: object, *, stream: object = sys.stderr) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), file=stream)


def initialize_torch_cuda(device_ordinal: int) -> dict[str, object]:
    import torch

    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not torch.cuda.is_available():
        raise KcmmError("PyTorch reports CUDA unavailable")
    if device_ordinal < 0 or device_ordinal >= torch.cuda.device_count():
        raise KcmmError(
            f"CUDA device ordinal {device_ordinal} outside available range "
            f"0..{torch.cuda.device_count() - 1}"
        )

    torch.cuda.set_device(device_ordinal)
    torch.empty(1, device=f"cuda:{device_ordinal}")
    torch.cuda.synchronize(device_ordinal)
    info["current_device"] = int(torch.cuda.current_device())
    info["device_name"] = torch.cuda.get_device_name(device_ordinal)
    return info


def _destroy_active_pool() -> None:
    global _ACTIVE_POOL
    if _ACTIVE_POOL is not None:
        _ACTIVE_POOL.destroy()
        _ACTIVE_POOL = None


def _create_observer_pool(
    config: ObserverConfig,
) -> tuple[KcmmPool, dict[str, object]]:
    cuda_info = initialize_torch_cuda(config.device_ordinal)
    library = KcmmLibrary(config.library_path)
    pool = library.create_pool(config.to_c_config())
    probe = pool.observer_probe(blocks=config.probe_blocks)
    return pool, {
        "cuda": cuda_info,
        "kcmm": result_to_dict(probe),
    }


def _run_vllm(vllm_args: Sequence[str]) -> int:
    import vllm.scripts

    old_argv = sys.argv
    sys.argv = ["vllm", *vllm_args]
    try:
        return int(vllm.scripts.main() or 0)
    finally:
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_POOL

    args = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m scripts.kcmm.launcher",
        add_help=False,
        description="KCMM Phase I.C observer wrapper for vLLM.",
    )
    add_kcmm_args(parser)
    namespace, vllm_args = parser.parse_known_args(args)

    if namespace.kcmm_help:
        parser.print_help()
        return 0

    config = ObserverConfig.from_namespace(namespace)

    seam_report = None
    allocator_report = None
    if config.instrument_allocators:
        allocator_report = apply_allocator_instrumentation(
            trace_path=config.allocator_trace_path,
            require_seams=config.require_allocator_seams,
        )
        _print_json({"vllm_allocator_instrumentation": allocator_report})

    if config.print_seams:
        seam_report = apply_observer_patches()
        if not config.observer_only:
            _print_json({"vllm_seams": seam_report})

    if not config.observer_only and not vllm_args:
        print(
            "No vLLM command was supplied. Use --kcmm-observer-only for a probe, "
            "or pass vLLM args such as: serve <model>.",
            file=sys.stderr,
        )
        return 2

    observer_report: dict[str, object] | None = None
    if not config.skip_observer:
        try:
            pool, observer_report = _create_observer_pool(config)
        except Exception as exc:
            print(f"KCMM observer initialization failed: {exc}", file=sys.stderr)
            return 1

        if config.observer_only or config.destroy_before_vllm:
            pool.destroy()
        else:
            _ACTIVE_POOL = pool
            atexit.register(_destroy_active_pool)

    if config.observer_only:
        _print_json(
            {
                "observer": observer_report,
                "vllm_seams": seam_report,
                "vllm_allocator_instrumentation": allocator_report,
            },
            stream=sys.stdout,
        )
        return 0

    if observer_report is not None:
        _print_json({"observer": observer_report})

    return _run_vllm(vllm_args)


if __name__ == "__main__":
    raise SystemExit(main())
