"""Phase II.C GPU read-kernel gate across multiple real vLLM models."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_gpu_read_ab_gate import GateConfig, parse_coverage_case, run_gate
from scripts.kcmm.vllm_gpu_read_real_model_gate import (
    DEFAULT_MODEL_ID,
    real_model_failures,
    resolve_real_model_path,
)
from scripts.kcmm.vllm_smoke import (
    CompletionCase,
    DEFAULT_KCMM_LIB_PATH,
    repo_root,
    resolve_repo_path,
)


DEFAULT_MODEL_IDS = (DEFAULT_MODEL_ID, "distilgpt2")
DEFAULT_MATRIX_COVERAGE_CASES = (
    CompletionCase(name="hello", prompt="Hello", max_tokens=2),
    CompletionCase(name="math", prompt="Question: 2 + 2 =", max_tokens=2),
    CompletionCase(
        name="long_context",
        prompt=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron pi rho sigma tau"
        ),
        max_tokens=2,
    ),
)


@dataclass(frozen=True)
class MatrixModel:
    model_id: str
    model_path: Path
    model_name: str
    downloaded_model: bool


@dataclass(frozen=True)
class RealModelMatrixConfig:
    base_gate: GateConfig
    models: tuple[MatrixModel, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help=(
            "Hugging Face model id to include. May be repeated. Defaults to "
            f"{', '.join(DEFAULT_MODEL_IDS)}."
        ),
    )
    parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download missing model ids into .scratch before running the gate.",
    )
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--coverage-case",
        action="append",
        default=None,
        metavar="NAME:MAX_TOKENS:PROMPT",
        help=(
            "Completion case to compare for every model. May be repeated. "
            "Defaults to hello, math, and a longer prompt that spans multiple "
            "KV blocks."
        ),
    )
    parser.add_argument(
        "--build-kcmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the KCMM shared library before the KCMM mode if needed.",
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
        help="Real-model matrix JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument("--latency-warning-ratio", type=float, default=2.5)
    parser.add_argument("--throughput-warning-ratio", type=float, default=0.4)
    parser.add_argument("--memory-warning-ratio", type=float, default=1.5)
    parser.add_argument("--memory-warning-min-delta-mib", type=int, default=512)
    return parser


def safe_model_name(model_id: str) -> str:
    return "real-" + model_id.replace("/", "-").replace("_", "-")


def parse_cases(values: list[str] | None) -> tuple[CompletionCase, ...]:
    cases = (
        tuple(parse_coverage_case(value) for value in values)
        if values
        else DEFAULT_MATRIX_COVERAGE_CASES
    )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate coverage case names: {names}")
    return cases


def parse_models(
    *,
    model_ids: list[str] | None,
    download_model: bool,
) -> tuple[MatrixModel, ...]:
    ids = tuple(model_ids or DEFAULT_MODEL_IDS)
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"duplicate model ids: {ids}")

    models: list[MatrixModel] = []
    for model_id in ids:
        model_path, downloaded = resolve_real_model_path(
            model_id=model_id,
            model_path=None,
            download_model=download_model,
        )
        models.append(
            MatrixModel(
                model_id=model_id,
                model_path=model_path,
                model_name=safe_model_name(model_id),
                downloaded_model=downloaded,
            )
        )
    return tuple(models)


def parse_config(argv: list[str] | None = None) -> RealModelMatrixConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        parser.error("--gpu-memory-utilization must be in the range (0, 1]")
    try:
        coverage_cases = parse_cases(args.coverage_case)
        models = parse_models(
            model_ids=args.model_id,
            download_model=args.download_model,
        )
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    timestamp_ms = int(time.time() * 1000)
    output_path = (
        Path(args.output)
        if args.output
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-real-model-matrix-{timestamp_ms}.json"
    )
    first_model = models[0]
    return RealModelMatrixConfig(
        base_gate=GateConfig(
            host=args.host,
            port=args.port,
            model_path=first_model.model_path,
            model_name=first_model.model_name,
            kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
            timeout_seconds=args.timeout_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
            generate_tiny_model=False,
            prompt=coverage_cases[0].prompt,
            max_tokens=coverage_cases[0].max_tokens,
            coverage_cases=coverage_cases,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1,
            completion_concurrency=1,
            kv_force_non_default_stream=False,
            kv_read_profile=False,
            kv_read_validate_block_tables=True,
            kv_read_fast_current_context_launch=False,
            kv_read_precompile_gpu_kernel=False,
            instrument_kv_reads=True,
            kv_write_verify=True,
            kv_write_device_slots=False,
            tracker_report_on_update=True,
            tracker_host_profile=False,
            build_kcmm=args.build_kcmm,
            keep_model=True,
            print_seams=args.print_seams,
            output_path=output_path,
            latency_warning_ratio=args.latency_warning_ratio,
            throughput_warning_ratio=args.throughput_warning_ratio,
            memory_warning_ratio=args.memory_warning_ratio,
            memory_warning_min_delta_mib=args.memory_warning_min_delta_mib,
        ),
        models=models,
    )


def gate_for_model(
    config: RealModelMatrixConfig,
    model: MatrixModel,
    output_path: Path,
) -> GateConfig:
    return replace(
        config.base_gate,
        model_path=model.model_path,
        model_name=model.model_name,
        output_path=output_path,
    )


def run_real_model_matrix_gate(config: RealModelMatrixConfig) -> dict[str, Any]:
    started_at_ms = int(time.time() * 1000)
    report_dir = config.base_gate.output_path.parent / (
        f"{config.base_gate.output_path.stem}-reports"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    model_reports: dict[str, Any] = {}

    for model in config.models:
        print(f"run real model matrix entry: {model.model_id}", flush=True)
        output_path = report_dir / f"{model.model_name}.json"
        model_report = run_gate(gate_for_model(config, model, output_path))
        model_report["gate"] = "stock-vs-kcmm-gpu-read-kernel-real-model-entry"
        model_report["real_model"] = {
            "model_id": model.model_id,
            "model_path": str(model.model_path),
            "downloaded_model": model.downloaded_model,
        }
        model_failures = real_model_failures(model_report)
        model_report["correctness_failures"].extend(model_failures)
        model_report["passed"] = not model_report["correctness_failures"]
        output_path.write_text(
            json.dumps(model_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        model_reports[model.model_id] = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "model_path": str(model.model_path),
            "downloaded_model": model.downloaded_model,
            "output_path": str(output_path),
            "report": model_report,
        }

    failed_models = [
        model_id
        for model_id, entry in model_reports.items()
        if not entry["report"].get("passed", False)
    ]
    correctness_failures = [
        {
            "model_id": model_id,
            "failures": entry["report"].get("correctness_failures", []),
        }
        for model_id, entry in model_reports.items()
        if entry["report"].get("correctness_failures")
    ]
    performance_warnings = [
        {
            "model_id": model_id,
            "warnings": entry["report"].get("performance_warnings", []),
        }
        for model_id, entry in model_reports.items()
        if entry["report"].get("performance_warnings")
    ]
    return {
        "phase": "II.C",
        "gate": "stock-vs-kcmm-gpu-read-kernel-real-model-matrix",
        "passed": not failed_models,
        "started_at_unix_ms": started_at_ms,
        "repo_root": str(repo_root()),
        "model_order": [model.model_id for model in config.models],
        "coverage_cases": [
            {
                "name": case.name,
                "prompt": case.prompt,
                "max_tokens": case.max_tokens,
            }
            for case in config.base_gate.coverage_cases
        ],
        "models": model_reports,
        "failed_models": failed_models,
        "correctness_failures": correctness_failures,
        "performance_warnings": performance_warnings,
        "output_path": str(config.base_gate.output_path),
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_real_model_matrix_gate(config)
    config.base_gate.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.base_gate.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
