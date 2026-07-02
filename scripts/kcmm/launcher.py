"""Launch vLLM with KCMM observer and Phase II.A runtime pool hooks."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from dataclasses import replace
from typing import Any, Sequence

from .backed_allocator import KcmmBackedAllocationTracker
from .bindings import KcmmError, KcmmLibrary, KcmmPool, result_to_dict
from .config import ObserverConfig, VllmRuntimeSizing, add_kcmm_args
from .kv_read_plan import KcmmKvReadOffsetTableTracker
from .kv_write_mirror import KcmmKvWriteMirrorTracker
from .patch_vllm import (
    apply_allocator_instrumentation,
    apply_kcmm_backed_allocator,
    apply_kv_read_offset_table,
    apply_kv_read_instrumentation,
    apply_kv_write_mirror,
    apply_kv_write_instrumentation,
    apply_observer_patches,
    apply_runtime_pool_sizing,
    apply_shadow_allocator,
    apply_worker_runtime_pool_sizing,
)
from .shadow_allocator import ShadowAllocationTracker


_ACTIVE_POOL: KcmmPool | None = None
_SHADOW_TRACKER: ShadowAllocationTracker | None = None
_BACKED_TRACKER: KcmmBackedAllocationTracker | None = None
_KV_WRITE_MIRROR_TRACKER: KcmmKvWriteMirrorTracker | None = None
_KV_READ_OFFSET_TABLE_TRACKER: KcmmKvReadOffsetTableTracker | None = None


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


def _write_shadow_report() -> None:
    if _SHADOW_TRACKER is not None:
        _SHADOW_TRACKER.write_report()
        _print_json({"kcmm_shadow_allocator": _SHADOW_TRACKER.report()})


def _write_backed_report() -> None:
    if _BACKED_TRACKER is not None:
        _BACKED_TRACKER.write_report()
        _print_json({"kcmm_backed_allocator": _BACKED_TRACKER.report()})


def _write_kv_write_mirror_report() -> None:
    if _KV_WRITE_MIRROR_TRACKER is not None:
        _KV_WRITE_MIRROR_TRACKER.write_report()
        _print_json({"kcmm_kv_write_mirror": _KV_WRITE_MIRROR_TRACKER.report()})


def _write_kv_read_offset_table_report() -> None:
    if _KV_READ_OFFSET_TABLE_TRACKER is not None:
        _KV_READ_OFFSET_TABLE_TRACKER.write_report()
        _print_json(
            {"kcmm_kv_read_offset_table": _KV_READ_OFFSET_TABLE_TRACKER.report()}
        )


def _create_observer_pool(
    config: ObserverConfig,
    *,
    source: str = "fixed",
    runtime_sizing: VllmRuntimeSizing | None = None,
) -> tuple[KcmmPool, dict[str, object]]:
    cuda_info = initialize_torch_cuda(config.device_ordinal)
    library = KcmmLibrary(config.library_path)
    pool = library.create_pool(config.to_c_config())
    probe = pool.observer_probe(blocks=config.probe_blocks)
    report: dict[str, object] = {
        "phase": "II.A" if source == "runtime" else "I.C",
        "pool_source": source,
        "kcmm_config": config.pool_shape_dict(),
        "cuda": cuda_info,
        "kcmm": result_to_dict(probe),
    }
    if runtime_sizing is not None:
        report["vllm_runtime"] = runtime_sizing.to_dict()
        report["alignment"] = {
            "kcmm_max_blocks": config.max_blocks,
            "vllm_effective_num_gpu_blocks": runtime_sizing.effective_num_gpu_blocks,
            "max_blocks_match": (
                config.max_blocks == runtime_sizing.effective_num_gpu_blocks
            ),
            "tiering_disabled": not config.enable_tiering,
        }
    return pool, report


def _required_positive_int(value: object, name: str) -> int:
    if value is None:
        raise KcmmError(f"missing vLLM runtime sizing field: {name}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise KcmmError(f"invalid vLLM runtime sizing field {name}: {value!r}") from exc
    if integer <= 0:
        raise KcmmError(f"vLLM runtime sizing field {name} must be positive")
    return integer


def _runtime_sizing_from_engine(engine: Any) -> VllmRuntimeSizing:
    try:
        import vllm
    except Exception:
        vllm = None

    model_config = getattr(engine, "model_config", None)
    cache_config = getattr(engine, "cache_config", None)
    parallel_config = getattr(engine, "parallel_config", None)
    scheduler_config = getattr(engine, "scheduler_config", None)
    if model_config is None:
        raise KcmmError("vLLM engine has no model_config")
    if cache_config is None:
        raise KcmmError("vLLM engine has no cache_config")
    if parallel_config is None:
        raise KcmmError("vLLM engine has no parallel_config")
    if scheduler_config is None:
        raise KcmmError("vLLM engine has no scheduler_config")

    pipeline_parallel_size = _required_positive_int(
        getattr(parallel_config, "pipeline_parallel_size", None),
        "parallel_config.pipeline_parallel_size",
    )
    num_gpu_blocks = _required_positive_int(
        getattr(cache_config, "num_gpu_blocks", None),
        "cache_config.num_gpu_blocks",
    )
    effective_num_gpu_blocks = max(1, num_gpu_blocks // pipeline_parallel_size)

    try:
        num_layers = model_config.get_num_attention_layers(parallel_config)
        kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
    except Exception as exc:
        raise KcmmError(f"failed to derive vLLM model KV shape: {exc}") from exc

    max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", 0)
    if max_num_batched_tokens is None:
        max_num_batched_tokens = 0

    return VllmRuntimeSizing(
        vllm_version=getattr(vllm, "__version__", "unknown"),
        block_size=_required_positive_int(
            getattr(cache_config, "block_size", None),
            "cache_config.block_size",
        ),
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=int(getattr(cache_config, "num_cpu_blocks", 0) or 0),
        effective_num_gpu_blocks=effective_num_gpu_blocks,
        num_layers=_required_positive_int(num_layers, "model_config.num_attention_layers"),
        kv_heads=_required_positive_int(kv_heads, "model_config.num_kv_heads"),
        head_dim=_required_positive_int(head_dim, "model_config.head_size"),
        max_model_len=_required_positive_int(
            getattr(model_config, "max_model_len", None),
            "model_config.max_model_len",
        ),
        max_num_seqs=_required_positive_int(
            getattr(scheduler_config, "max_num_seqs", None),
            "scheduler_config.max_num_seqs",
        ),
        max_num_batched_tokens=int(max_num_batched_tokens),
        tensor_parallel_size=_required_positive_int(
            getattr(parallel_config, "tensor_parallel_size", None),
            "parallel_config.tensor_parallel_size",
        ),
        pipeline_parallel_size=pipeline_parallel_size,
        cache_dtype=str(getattr(cache_config, "cache_dtype", "unknown")),
        model_dtype=str(getattr(model_config, "dtype", "unknown")),
        use_v2_block_manager=bool(
            getattr(scheduler_config, "use_v2_block_manager", False)
        ),
        enforce_eager=bool(getattr(model_config, "enforce_eager", False)),
        enable_prefix_caching=bool(
            getattr(cache_config, "enable_prefix_caching", False)
        ),
    )


def _runtime_sizing_from_worker(
    worker: Any,
    *,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
) -> VllmRuntimeSizing:
    try:
        import vllm
    except Exception:
        vllm = None

    model_config = getattr(worker, "model_config", None)
    cache_config = getattr(worker, "cache_config", None)
    parallel_config = getattr(worker, "parallel_config", None)
    scheduler_config = getattr(worker, "scheduler_config", None)
    if model_config is None:
        raise KcmmError("vLLM worker has no model_config")
    if cache_config is None:
        raise KcmmError("vLLM worker has no cache_config")
    if parallel_config is None:
        raise KcmmError("vLLM worker has no parallel_config")
    if scheduler_config is None:
        raise KcmmError("vLLM worker has no scheduler_config")

    pipeline_parallel_size = _required_positive_int(
        getattr(parallel_config, "pipeline_parallel_size", None),
        "parallel_config.pipeline_parallel_size",
    )
    effective_num_gpu_blocks = max(1, int(num_gpu_blocks) // pipeline_parallel_size)

    try:
        num_layers = model_config.get_num_attention_layers(parallel_config)
        kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
    except Exception as exc:
        raise KcmmError(f"failed to derive vLLM worker KV shape: {exc}") from exc

    max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", 0)
    if max_num_batched_tokens is None:
        max_num_batched_tokens = 0

    return VllmRuntimeSizing(
        vllm_version=getattr(vllm, "__version__", "unknown"),
        block_size=_required_positive_int(
            getattr(cache_config, "block_size", None),
            "cache_config.block_size",
        ),
        num_gpu_blocks=_required_positive_int(num_gpu_blocks, "worker.num_gpu_blocks"),
        num_cpu_blocks=int(num_cpu_blocks or 0),
        effective_num_gpu_blocks=effective_num_gpu_blocks,
        num_layers=_required_positive_int(num_layers, "model_config.num_attention_layers"),
        kv_heads=_required_positive_int(kv_heads, "model_config.num_kv_heads"),
        head_dim=_required_positive_int(head_dim, "model_config.head_size"),
        max_model_len=_required_positive_int(
            getattr(model_config, "max_model_len", None),
            "model_config.max_model_len",
        ),
        max_num_seqs=_required_positive_int(
            getattr(scheduler_config, "max_num_seqs", None),
            "scheduler_config.max_num_seqs",
        ),
        max_num_batched_tokens=int(max_num_batched_tokens),
        tensor_parallel_size=_required_positive_int(
            getattr(parallel_config, "tensor_parallel_size", None),
            "parallel_config.tensor_parallel_size",
        ),
        pipeline_parallel_size=pipeline_parallel_size,
        cache_dtype=str(getattr(cache_config, "cache_dtype", "unknown")),
        model_dtype=str(getattr(model_config, "dtype", "unknown")),
        use_v2_block_manager=bool(
            getattr(scheduler_config, "use_v2_block_manager", False)
        ),
        enforce_eager=bool(getattr(model_config, "enforce_eager", False)),
        enable_prefix_caching=bool(
            getattr(cache_config, "enable_prefix_caching", False)
        ),
    )


def _run_vllm(vllm_args: Sequence[str]) -> int:
    import vllm.scripts

    old_argv = sys.argv
    sys.argv = ["vllm", *vllm_args]
    try:
        return int(vllm.scripts.main() or 0)
    finally:
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_POOL, _SHADOW_TRACKER, _BACKED_TRACKER
    global _KV_WRITE_MIRROR_TRACKER, _KV_READ_OFFSET_TABLE_TRACKER

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
    try:
        config.validate()
    except Exception as exc:
        print(f"Invalid KCMM launcher configuration: {exc}", file=sys.stderr)
        return 2

    if config.pool_mode == "runtime":
        if config.observer_only:
            print(
                "KCMM runtime pool mode requires a vLLM engine; "
                "use --kcmm-pool-mode fixed with --kcmm-observer-only.",
                file=sys.stderr,
            )
            return 2
        if config.destroy_before_vllm:
            print(
                "KCMM runtime pool mode is incompatible with "
                "--kcmm-destroy-before-vllm.",
                file=sys.stderr,
            )
            return 2
        if "--disable-frontend-multiprocessing" not in vllm_args:
            print(
                "KCMM runtime pool mode requires vLLM "
                "--disable-frontend-multiprocessing so monkey-patches run in "
                "the engine process.",
                file=sys.stderr,
            )
            return 2

    seam_report = None
    allocator_report = None
    kv_write_report = None
    kv_read_report = None
    kv_read_offset_table_report = None
    kv_write_mirror_report = None
    runtime_pool_report = None
    shadow_report = None
    backed_report = None
    if config.shadow_allocations:
        _SHADOW_TRACKER = ShadowAllocationTracker(config.shadow_report_path)
        shadow_report = apply_shadow_allocator(_SHADOW_TRACKER)
        _print_json({"kcmm_shadow_allocator_patch": shadow_report})

    if config.backed_allocations:
        _BACKED_TRACKER = KcmmBackedAllocationTracker(config.backed_report_path)
        backed_report = apply_kcmm_backed_allocator(_BACKED_TRACKER)
        _print_json({"kcmm_backed_allocator_patch": backed_report})

    if config.instrument_allocators:
        allocator_report = apply_allocator_instrumentation(
            trace_path=config.allocator_trace_path,
            require_seams=config.require_allocator_seams,
        )
        _print_json({"vllm_allocator_instrumentation": allocator_report})

    if config.kv_write_mirror or config.kv_write_replace_candidate:
        _KV_WRITE_MIRROR_TRACKER = KcmmKvWriteMirrorTracker(
            config.kv_write_mirror_report_path,
            verify_rows_per_call=4 if config.kv_write_verify else 0,
            report_on_update=config.tracker_report_on_update,
            profile_host_sections=config.tracker_host_profile,
            replace_native=config.kv_write_replace_candidate,
            force_non_default_stream=config.kv_force_non_default_stream,
            use_device_slot_write=config.kv_write_device_slots,
        )
        kv_write_mirror_report = apply_kv_write_mirror(_KV_WRITE_MIRROR_TRACKER)
        _print_json({"kcmm_kv_write_mirror_patch": kv_write_mirror_report})

    if config.instrument_kv_writes:
        kv_write_report = apply_kv_write_instrumentation(
            trace_path=config.kv_write_trace_path,
            require_seams=config.require_kv_write_seams,
        )
        _print_json({"vllm_kv_write_instrumentation": kv_write_report})

    if (
        config.kv_read_offset_table
        or config.kv_read_replace_candidate
        or config.kv_read_gpu_kernel_candidate
    ):
        _KV_READ_OFFSET_TABLE_TRACKER = KcmmKvReadOffsetTableTracker(
            config.kv_read_offset_table_report_path,
            replace_native=(
                config.kv_read_replace_candidate
                or config.kv_read_gpu_kernel_candidate
            ),
            replacement_backend=(
                "gpu_kernel"
                if config.kv_read_gpu_kernel_candidate
                else "reference"
            ),
            force_non_default_stream=config.kv_force_non_default_stream,
            profile_gpu_kernel=config.kv_read_profile,
            report_on_update=config.tracker_report_on_update,
            validate_block_tables=config.kv_read_validate_block_tables,
            profile_host_sections=config.tracker_host_profile,
            fast_current_context_launch=config.kv_read_fast_current_context_launch,
            precompile_gpu_kernel=config.kv_read_precompile_gpu_kernel,
        )
        kv_read_offset_table_report = apply_kv_read_offset_table(
            _KV_READ_OFFSET_TABLE_TRACKER
        )
        _print_json({"kcmm_kv_read_offset_table_patch": kv_read_offset_table_report})

    if config.instrument_kv_reads:
        kv_read_report = apply_kv_read_instrumentation(
            trace_path=config.kv_read_trace_path,
            require_seams=config.require_kv_read_seams,
        )
        _print_json({"vllm_kv_read_instrumentation": kv_read_report})

    if not config.skip_observer and config.pool_mode == "runtime":

        def attach_runtime_pool(
            runtime_sizing: VllmRuntimeSizing,
            *,
            device_ordinal: int,
        ) -> dict[str, Any]:
            global _ACTIVE_POOL
            if _ACTIVE_POOL is not None:
                return {"skipped": "KCMM runtime pool already active"}
            runtime_config = replace(
                config.with_runtime_sizing(runtime_sizing),
                device_ordinal=device_ordinal,
            )
            pool, report = _create_observer_pool(
                runtime_config,
                source="runtime",
                runtime_sizing=runtime_sizing,
            )
            _ACTIVE_POOL = pool
            atexit.register(_destroy_active_pool)
            if _SHADOW_TRACKER is not None:
                _SHADOW_TRACKER.attach_pool(pool)
                atexit.register(_write_shadow_report)
            if _BACKED_TRACKER is not None:
                _BACKED_TRACKER.validate_runtime(runtime_sizing)
                _BACKED_TRACKER.attach_pool(pool)
                atexit.register(_write_backed_report)
            if _KV_WRITE_MIRROR_TRACKER is not None:
                _KV_WRITE_MIRROR_TRACKER.attach_pool(pool)
                atexit.register(_write_kv_write_mirror_report)
            if _KV_READ_OFFSET_TABLE_TRACKER is not None:
                _KV_READ_OFFSET_TABLE_TRACKER.validate_runtime(runtime_sizing)
                _KV_READ_OFFSET_TABLE_TRACKER.attach_pool(pool)
                atexit.register(_write_kv_read_offset_table_report)
            _print_json({"observer": report})
            return report

        def create_runtime_pool(engine: Any) -> dict[str, Any]:
            runtime_sizing = _runtime_sizing_from_engine(engine)
            return attach_runtime_pool(
                runtime_sizing,
                device_ordinal=config.device_ordinal,
            )

        def create_worker_runtime_pool(
            worker: Any,
            num_gpu_blocks: int,
            num_cpu_blocks: int,
        ) -> dict[str, Any]:
            runtime_sizing = _runtime_sizing_from_worker(
                worker,
                num_gpu_blocks=num_gpu_blocks,
                num_cpu_blocks=num_cpu_blocks,
            )
            device_ordinal = _required_positive_int(
                int(getattr(worker, "local_rank", config.device_ordinal)) + 1,
                "worker.local_rank_plus_one",
            ) - 1
            return attach_runtime_pool(
                runtime_sizing,
                device_ordinal=device_ordinal,
            )

        runtime_pool_report = apply_runtime_pool_sizing(create_runtime_pool)
        _print_json({"kcmm_runtime_pool_sizing": runtime_pool_report})
        worker_runtime_pool_report = apply_worker_runtime_pool_sizing(
            create_worker_runtime_pool
        )
        _print_json({"kcmm_worker_runtime_pool_sizing": worker_runtime_pool_report})

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
    if not config.skip_observer and config.pool_mode == "fixed":
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
                "vllm_kv_write_instrumentation": kv_write_report,
                "vllm_kv_read_instrumentation": kv_read_report,
                "kcmm_kv_read_offset_table_patch": kv_read_offset_table_report,
                "kcmm_kv_write_mirror_patch": kv_write_mirror_report,
                "kcmm_runtime_pool_sizing": runtime_pool_report,
                "kcmm_shadow_allocator_patch": shadow_report,
                "kcmm_backed_allocator_patch": backed_report,
            },
            stream=sys.stdout,
        )
        return 0

    if observer_report is not None:
        _print_json({"observer": observer_report})

    return _run_vllm(vllm_args)


if __name__ == "__main__":
    raise SystemExit(main())
