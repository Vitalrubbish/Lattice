"""Phase II.A stock-vs-KCMM vLLM A/B gate."""

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


MODE_ORDER = ("stock", "observer", "shadow", "backed")


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
        help="Build the KCMM shared library before each KCMM mode if needed.",
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
        help="Ask KCMM launcher modes to print vLLM seam inspection output.",
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
        / f"kcmm-vllm-phase-ii-a-ab-{timestamp_ms}.json"
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
    is_shadow = mode_name == "shadow"
    is_backed = mode_name == "backed"
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
        runtime_derived_pool=(is_shadow or is_backed),
        shadow_allocations=is_shadow,
        backed_allocations=is_backed,
        allocator_trace_path=run_dir / f"{mode_name}-allocator-trace.jsonl",
        kv_write_trace_path=run_dir / f"{mode_name}-kv-write-trace.jsonl",
        shadow_report_path=run_dir / f"{mode_name}-shadow-report.json",
        backed_report_path=run_dir / f"{mode_name}-backed-report.json",
        require_allocator_seams=True,
        require_kv_write_seams=True,
        log_path=run_dir / f"{mode_name}.log",
    )


def token_throughput(
    generated_tokens: int | None,
    request_latency_seconds: float | None,
) -> float | None:
    if generated_tokens is None or request_latency_seconds is None:
        return None
    if request_latency_seconds <= 0:
        return None
    return round(generated_tokens / request_latency_seconds, 3)


def extract_generated_tokens(completion: dict[str, Any]) -> int | None:
    usage = completion.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int):
            return value
    return None


def extract_total_tokens(completion: dict[str, Any]) -> int | None:
    usage = completion.get("usage")
    if isinstance(usage, dict):
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return value
    return None


def kcmm_stats_for_result(mode_name: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if mode_name == "stock":
        return None
    if mode_name == "shadow":
        return result.get("shadow_allocator")
    if mode_name == "backed":
        return result.get("backed_allocator")
    return {
        "mode": "observer",
        "runtime_derived_pool": result.get("runtime_derived_pool", False),
        "shadow_allocations": False,
        "backed_allocations": False,
    }


def summarize_success(mode_name: str, result: dict[str, Any]) -> dict[str, Any]:
    completion = result.get("completion", {})
    generated_tokens = extract_generated_tokens(completion)
    total_tokens = extract_total_tokens(completion)
    request_latency_seconds = result.get("completion_seconds")
    return {
        "success": True,
        "mode": mode_name,
        "server_mode": result.get("mode"),
        "startup_seconds": result.get("startup_seconds"),
        "request_latency_seconds": request_latency_seconds,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": token_throughput(
            generated_tokens,
            request_latency_seconds,
        ),
        "gpu_memory": result.get("gpu_memory"),
        "kcmm_stats": kcmm_stats_for_result(mode_name, result),
        "log_path": result.get("log_path"),
    }


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
        ("shadow_report_path", smoke_config.shadow_report_path),
        ("backed_report_path", smoke_config.backed_report_path),
        ("allocator_trace_path", smoke_config.allocator_trace_path),
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


def add_correctness_failures(modes: dict[str, Any]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    stock = modes.get("stock", {})
    if not stock.get("success"):
        failures.append(
            {
                "mode": "stock",
                "reason": "stock_failed",
                "detail": stock.get("error", "stock vLLM did not complete"),
            }
        )
    if stock.get("success") and not modes.get("observer", {}).get("success"):
        failures.append(
            {
                "mode": "observer",
                "reason": "observer_failed_after_stock_passed",
                "detail": modes.get("observer", {}).get(
                    "error",
                    "KCMM observer mode did not complete",
                ),
            }
        )
    if stock.get("success") and not modes.get("shadow", {}).get("success"):
        failures.append(
            {
                "mode": "shadow",
                "reason": "shadow_failed_after_stock_passed",
                "detail": modes.get("shadow", {}).get(
                    "error",
                    "KCMM shadow allocator mode did not complete",
                ),
            }
        )
    if stock.get("success") and not modes.get("backed", {}).get("success"):
        failures.append(
            {
                "mode": "backed",
                "reason": "backed_failed_after_stock_passed",
                "detail": modes.get("backed", {}).get(
                    "error",
                    "KCMM-backed allocator mode did not complete cleanly",
                ),
            }
        )
    return failures


def number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def add_ratio_warning(
    warnings: list[dict[str, Any]],
    *,
    mode_name: str,
    metric: str,
    stock_value: float | None,
    mode_value: float | None,
    ratio: float,
    higher_is_worse: bool,
) -> None:
    if stock_value is None or mode_value is None:
        return
    if stock_value <= 0:
        return
    if higher_is_worse:
        threshold = stock_value * ratio
        if mode_value > threshold:
            warnings.append(
                {
                    "mode": mode_name,
                    "metric": metric,
                    "stock_value": round(stock_value, 3),
                    "mode_value": round(mode_value, 3),
                    "threshold": round(threshold, 3),
                    "classification": "performance_warning",
                }
            )
    else:
        threshold = stock_value * ratio
        if mode_value < threshold:
            warnings.append(
                {
                    "mode": mode_name,
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
    mode_name: str,
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
                "mode": mode_name,
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
    if not stock.get("success"):
        return warnings
    stock_startup = number(stock.get("startup_seconds"))
    stock_latency = number(stock.get("request_latency_seconds"))
    stock_tps = number(stock.get("tokens_per_second"))
    stock_memory = stock.get("gpu_memory") or {}
    stock_memory_delta = number(stock_memory.get("peak_delta_mib"))

    for mode_name in MODE_ORDER:
        if mode_name == "stock":
            continue
        mode = modes.get(mode_name, {})
        if not mode.get("success"):
            continue
        add_ratio_warning(
            warnings,
            mode_name=mode_name,
            metric="startup_seconds",
            stock_value=stock_startup,
            mode_value=number(mode.get("startup_seconds")),
            ratio=config.latency_warning_ratio,
            higher_is_worse=True,
        )
        add_ratio_warning(
            warnings,
            mode_name=mode_name,
            metric="request_latency_seconds",
            stock_value=stock_latency,
            mode_value=number(mode.get("request_latency_seconds")),
            ratio=config.latency_warning_ratio,
            higher_is_worse=True,
        )
        add_ratio_warning(
            warnings,
            mode_name=mode_name,
            metric="tokens_per_second",
            stock_value=stock_tps,
            mode_value=number(mode.get("tokens_per_second")),
            ratio=config.throughput_warning_ratio,
            higher_is_worse=False,
        )
        mode_memory = mode.get("gpu_memory") or {}
        add_memory_warning(
            warnings,
            config=config,
            mode_name=mode_name,
            stock_delta=stock_memory_delta,
            mode_delta=number(mode_memory.get("peak_delta_mib")),
        )
    return warnings


def run_gate(config: GateConfig) -> dict[str, Any]:
    run_id = int(time.time() * 1000)
    run_dir = Path(tempfile.gettempdir()) / f"kcmm-vllm-phase-ii-a-ab-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    created_model_dir = not config.model_path.exists()
    model_existed = tiny_model_exists(config.model_path)
    modes: dict[str, Any] = {}
    try:
        for mode_name in MODE_ORDER:
            print(f"run A/B mode: {mode_name}", flush=True)
            modes[mode_name] = run_mode(config, mode_name, run_dir)
    finally:
        if created_model_dir and not config.keep_model:
            shutil.rmtree(config.model_path, ignore_errors=True)

    correctness_failures = add_correctness_failures(modes)
    performance_warnings = add_performance_warnings(config, modes)
    report = {
        "phase": "II.A",
        "gate": "stock-vs-kcmm-ab",
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
