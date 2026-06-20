"""Phase II.C stock-vs-KCMM GPU read-kernel vLLM A/B gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_smoke import (
    DEFAULT_KCMM_LIB_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    SmokeConfig,
    SmokeFailure,
    repo_root,
    resolve_repo_path,
    run_smoke,
    tail_file,
)


MODE_ORDER = ("stock", "kcmm_gpu_read")


@dataclass(frozen=True)
class GateConfig:
    host: str
    port: int
    model_path: Path
    model_name: str
    kcmm_lib_path: Path
    timeout_seconds: float
    shutdown_timeout_seconds: float
    prompt: str
    max_tokens: int
    build_kcmm: bool
    keep_model: bool
    print_seams: bool
    output_path: Path
    latency_warning_ratio: float
    throughput_warning_ratio: float
    memory_warning_ratio: float
    memory_warning_min_delta_mib: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--build-kcmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the KCMM shared library before the KCMM mode if needed.",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Keep the generated tiny model after the gate run.",
    )
    parser.add_argument(
        "--print-seams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the KCMM launcher to print vLLM seam inspection output.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="A/B gate JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument(
        "--latency-warning-ratio",
        type=float,
        default=2.0,
        help="Warn when KCMM startup or request latency exceeds stock by this ratio.",
    )
    parser.add_argument(
        "--throughput-warning-ratio",
        type=float,
        default=0.5,
        help="Warn when KCMM token throughput falls below stock times this ratio.",
    )
    parser.add_argument(
        "--memory-warning-ratio",
        type=float,
        default=1.5,
        help="Warn when KCMM peak GPU memory delta exceeds stock by this ratio.",
    )
    parser.add_argument(
        "--memory-warning-min-delta-mib",
        type=int,
        default=256,
        help="Minimum extra GPU memory delta before memory ratio warnings are emitted.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> GateConfig:
    args = build_parser().parse_args(argv)
    timestamp_ms = int(time.time() * 1000)
    output_path = (
        Path(args.output)
        if args.output
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-ab-{timestamp_ms}.json"
    )
    return GateConfig(
        host=args.host,
        port=args.port,
        model_path=resolve_repo_path(args.model_path),
        model_name=args.model_name,
        kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
        timeout_seconds=args.timeout_seconds,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        build_kcmm=args.build_kcmm,
        keep_model=args.keep_model,
        print_seams=args.print_seams,
        output_path=output_path,
        latency_warning_ratio=args.latency_warning_ratio,
        throughput_warning_ratio=args.throughput_warning_ratio,
        memory_warning_ratio=args.memory_warning_ratio,
        memory_warning_min_delta_mib=args.memory_warning_min_delta_mib,
    )


def tiny_model_exists(model_path: Path) -> bool:
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    return all((model_path / name).exists() for name in required)


def smoke_config_for_mode(
    config: GateConfig,
    mode_name: str,
    run_dir: Path,
) -> SmokeConfig:
    if mode_name not in MODE_ORDER:
        raise ValueError(f"unknown A/B mode: {mode_name}")
    is_stock = mode_name == "stock"
    is_gpu_read = mode_name == "kcmm_gpu_read"
    return SmokeConfig(
        mode="stock" if is_stock else "kcmm",
        host=config.host,
        port=config.port,
        model_path=config.model_path,
        model_name=config.model_name,
        kcmm_lib_path=config.kcmm_lib_path,
        timeout_seconds=config.timeout_seconds,
        shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        build_kcmm=(config.build_kcmm and not is_stock),
        keep_model=True,
        print_seams=(config.print_seams and not is_stock),
        instrument_allocators=False,
        instrument_kv_writes=False,
        instrument_kv_reads=is_gpu_read,
        kv_read_offset_table=False,
        kv_read_replace_candidate=False,
        kv_read_gpu_kernel_candidate=is_gpu_read,
        kv_write_mirror=False,
        kv_write_replace_candidate=is_gpu_read,
        runtime_derived_pool=is_gpu_read,
        shadow_allocations=False,
        backed_allocations=is_gpu_read,
        allocator_trace_path=run_dir / f"{mode_name}-allocator-trace.jsonl",
        kv_write_trace_path=run_dir / f"{mode_name}-kv-write-trace.jsonl",
        kv_read_trace_path=run_dir / f"{mode_name}-kv-read-trace.jsonl",
        kv_read_offset_table_report_path=(
            run_dir / f"{mode_name}-kv-read-offset-table-report.json"
        ),
        kv_write_mirror_report_path=(
            run_dir / f"{mode_name}-kv-write-mirror-report.json"
        ),
        shadow_report_path=run_dir / f"{mode_name}-shadow-report.json",
        backed_report_path=run_dir / f"{mode_name}-backed-report.json",
        require_allocator_seams=True,
        require_kv_write_seams=True,
        require_kv_read_seams=True,
        log_path=run_dir / f"{mode_name}.log",
    )


def completion_text(result: dict[str, Any]) -> str | None:
    choices = (result.get("completion") or {}).get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    text = choices[0].get("text")
    return text if isinstance(text, str) else None


def finish_reason(result: dict[str, Any]) -> str | None:
    choices = (result.get("completion") or {}).get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None


def usage_value(result: dict[str, Any], key: str) -> int | None:
    usage = (result.get("completion") or {}).get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) else None


def token_throughput(result: dict[str, Any]) -> float | None:
    generated_tokens = usage_value(result, "completion_tokens")
    latency = result.get("completion_seconds")
    if not isinstance(generated_tokens, int) or not isinstance(latency, (int, float)):
        return None
    if latency <= 0:
        return None
    return round(generated_tokens / latency, 3)


def summarize_gpu_read_contract(result: dict[str, Any]) -> dict[str, Any]:
    read_report = result.get("kv_read_offset_table_report") or {}
    write_report = result.get("kv_write_replace_candidate_report") or {}
    backed_report = result.get("backed_allocator") or {}
    backed_pool_stats = backed_report.get("pool_stats") or {}
    return {
        "read_path": read_report.get("read_path"),
        "replacement_backend": read_report.get("replacement_backend"),
        "gpu_kernel_calls": read_report.get("gpu_kernel_calls"),
        "stream_aware_kernel_calls": read_report.get("stream_aware_kernel_calls"),
        "reference_read_bytes": read_report.get("reference_read_bytes"),
        "replacement_calls": read_report.get("replacement_calls"),
        "offset_table_builds": read_report.get("offset_table_builds"),
        "native_write_skipped_calls": write_report.get("native_skipped_calls"),
        "kcmm_write_verified_rows": write_report.get("verified_rows"),
        "storage_of_record": write_report.get("storage_of_record"),
        "blocks_in_use_after_shutdown": backed_pool_stats.get("blocks_in_use"),
    }


def summarize_success(mode_name: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "success": True,
        "mode": mode_name,
        "server_mode": result.get("mode"),
        "startup_seconds": result.get("startup_seconds"),
        "request_latency_seconds": result.get("completion_seconds"),
        "tokens_per_second": token_throughput(result),
        "completion_text": completion_text(result),
        "finish_reason": finish_reason(result),
        "completion_tokens": usage_value(result, "completion_tokens"),
        "total_tokens": usage_value(result, "total_tokens"),
        "gpu_memory": result.get("gpu_memory"),
        "generated_model": result.get("generated_model"),
        "log_path": result.get("log_path"),
    }
    if mode_name == "kcmm_gpu_read":
        summary["kcmm_gpu_read_contract"] = summarize_gpu_read_contract(result)
    return summary


def summarize_failure(
    mode_name: str,
    error: Exception,
    smoke_config: SmokeConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "success": False,
        "mode": mode_name,
        "server_mode": smoke_config.mode,
        "error": str(error),
        "log_path": str(smoke_config.log_path),
    }
    if smoke_config.log_path.exists():
        summary["log_tail"] = tail_file(smoke_config.log_path)
    for key, path in (
        ("backed_report_path", smoke_config.backed_report_path),
        ("kv_write_mirror_report_path", smoke_config.kv_write_mirror_report_path),
        (
            "kv_read_offset_table_report_path",
            smoke_config.kv_read_offset_table_report_path,
        ),
        ("kv_read_trace_path", smoke_config.kv_read_trace_path),
    ):
        if path.exists():
            summary[key] = str(path)
    return summary


def run_mode(config: GateConfig, mode_name: str, run_dir: Path) -> dict[str, Any]:
    smoke_config = smoke_config_for_mode(config, mode_name, run_dir)
    try:
        result = run_smoke(smoke_config)
    except SmokeFailure as exc:
        return summarize_failure(mode_name, exc, smoke_config)
    return summarize_success(mode_name, result)


def add_correctness_failures(modes: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    stock = modes.get("stock", {})
    gpu_read = modes.get("kcmm_gpu_read", {})
    if not stock.get("success"):
        failures.append(
            {
                "mode": "stock",
                "reason": "stock_failed",
                "detail": stock.get("error", "stock vLLM did not complete"),
            }
        )
        return failures
    if not gpu_read.get("success"):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "kcmm_gpu_read_failed_after_stock_passed",
                "detail": gpu_read.get(
                    "error",
                    "KCMM GPU read-kernel mode did not complete",
                ),
            }
        )
        return failures

    for key in ("completion_text", "finish_reason", "completion_tokens", "total_tokens"):
        if stock.get(key) != gpu_read.get(key):
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{key}_mismatch",
                    "stock_value": stock.get(key),
                    "kcmm_value": gpu_read.get(key),
                }
            )
    return failures


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def nested_number(value: Any, *keys: str) -> float | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return number(current)


def performance_comparison(modes: dict[str, Any]) -> dict[str, Any]:
    stock = modes.get("stock", {})
    gpu_read = modes.get("kcmm_gpu_read", {})
    if not stock.get("success") or not gpu_read.get("success"):
        return {"available": False, "reason": "both modes must pass first"}

    metrics = [
        (
            "startup_seconds",
            number(stock.get("startup_seconds")),
            number(gpu_read.get("startup_seconds")),
            "lower_is_better",
        ),
        (
            "request_latency_seconds",
            number(stock.get("request_latency_seconds")),
            number(gpu_read.get("request_latency_seconds")),
            "lower_is_better",
        ),
        (
            "tokens_per_second",
            number(stock.get("tokens_per_second")),
            number(gpu_read.get("tokens_per_second")),
            "higher_is_better",
        ),
        (
            "gpu_memory_peak_delta_mib",
            nested_number(stock, "gpu_memory", "peak_delta_mib"),
            nested_number(gpu_read, "gpu_memory", "peak_delta_mib"),
            "lower_is_better",
        ),
    ]
    return {
        "available": True,
        "metrics": {
            name: {
                "stock": stock_value,
                "kcmm_gpu_read": kcmm_value,
                "kcmm_to_stock_ratio": ratio(kcmm_value, stock_value),
                "direction": direction,
            }
            for name, stock_value, kcmm_value, direction in metrics
        },
    }


def add_ratio_warning(
    warnings: list[dict[str, Any]],
    *,
    metric: str,
    stock_value: float | None,
    mode_value: float | None,
    ratio_threshold: float,
    higher_is_worse: bool,
) -> None:
    if stock_value is None or mode_value is None:
        return
    if stock_value <= 0:
        return
    threshold = stock_value * ratio_threshold
    if higher_is_worse and mode_value > threshold:
        warnings.append(
            {
                "mode": "kcmm_gpu_read",
                "metric": metric,
                "stock_value": round(stock_value, 3),
                "mode_value": round(mode_value, 3),
                "threshold": round(threshold, 3),
                "classification": "performance_warning",
            }
        )
    if not higher_is_worse and mode_value < threshold:
        warnings.append(
            {
                "mode": "kcmm_gpu_read",
                "metric": metric,
                "stock_value": round(stock_value, 3),
                "mode_value": round(mode_value, 3),
                "threshold": round(threshold, 3),
                "classification": "performance_warning",
            }
        )


def add_memory_warning(
    warnings: list[dict[str, Any]],
    *,
    config: GateConfig,
    stock_delta: float | None,
    mode_delta: float | None,
) -> None:
    if stock_delta is None or mode_delta is None:
        return
    threshold = max(
        stock_delta * config.memory_warning_ratio,
        stock_delta + config.memory_warning_min_delta_mib,
    )
    if mode_delta > threshold:
        warnings.append(
            {
                "mode": "kcmm_gpu_read",
                "metric": "gpu_memory_peak_delta_mib",
                "stock_value": round(stock_delta, 3),
                "mode_value": round(mode_delta, 3),
                "threshold": round(threshold, 3),
                "classification": "performance_warning",
            }
        )


def add_performance_warnings(
    config: GateConfig,
    modes: dict[str, Any],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    stock = modes.get("stock", {})
    gpu_read = modes.get("kcmm_gpu_read", {})
    if not stock.get("success") or not gpu_read.get("success"):
        return warnings

    add_ratio_warning(
        warnings,
        metric="startup_seconds",
        stock_value=number(stock.get("startup_seconds")),
        mode_value=number(gpu_read.get("startup_seconds")),
        ratio_threshold=config.latency_warning_ratio,
        higher_is_worse=True,
    )
    add_ratio_warning(
        warnings,
        metric="request_latency_seconds",
        stock_value=number(stock.get("request_latency_seconds")),
        mode_value=number(gpu_read.get("request_latency_seconds")),
        ratio_threshold=config.latency_warning_ratio,
        higher_is_worse=True,
    )
    add_ratio_warning(
        warnings,
        metric="tokens_per_second",
        stock_value=number(stock.get("tokens_per_second")),
        mode_value=number(gpu_read.get("tokens_per_second")),
        ratio_threshold=config.throughput_warning_ratio,
        higher_is_worse=False,
    )
    add_memory_warning(
        warnings,
        config=config,
        stock_delta=nested_number(stock, "gpu_memory", "peak_delta_mib"),
        mode_delta=nested_number(gpu_read, "gpu_memory", "peak_delta_mib"),
    )
    return warnings


def run_gate(config: GateConfig) -> dict[str, Any]:
    run_id = int(time.time() * 1000)
    run_dir = Path(tempfile.gettempdir()) / f"kcmm-vllm-phase-ii-c-gpu-read-ab-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    created_model_dir = not config.model_path.exists()
    model_existed = tiny_model_exists(config.model_path)
    modes: dict[str, Any] = {}
    try:
        for mode_name in MODE_ORDER:
            print(f"run GPU read A/B mode: {mode_name}", flush=True)
            modes[mode_name] = run_mode(config, mode_name, run_dir)
    finally:
        if created_model_dir and not config.keep_model:
            shutil.rmtree(config.model_path, ignore_errors=True)

    correctness_failures = add_correctness_failures(modes)
    performance_warnings = add_performance_warnings(config, modes)
    report = {
        "phase": "II.C",
        "gate": "stock-vs-kcmm-gpu-read-kernel-ab",
        "passed": not correctness_failures,
        "started_at_unix_ms": run_id,
        "repo_root": str(repo_root()),
        "run_dir": str(run_dir),
        "model_path": str(config.model_path),
        "model_name": config.model_name,
        "model_existed_before_gate": model_existed,
        "prompt": config.prompt,
        "max_tokens": config.max_tokens,
        "mode_order": list(MODE_ORDER),
        "modes": modes,
        "correctness_failures": correctness_failures,
        "performance_comparison": performance_comparison(modes),
        "performance_warnings": performance_warnings,
        "output_path": str(config.output_path),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_gate(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
