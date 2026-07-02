"""Phase II.C performance-clean GPU read gate under concurrent real-model load."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_gpu_read_ab_gate import GateConfig, parse_coverage_case
from scripts.kcmm.vllm_gpu_read_perf_clean_gate import (
    PerfCleanGateConfig,
    run_perf_clean_gate,
)
from scripts.kcmm.vllm_gpu_read_real_model_gate import (
    DEFAULT_MODEL_ID,
    resolve_real_model_path,
)
from scripts.kcmm.vllm_smoke import (
    CompletionCase,
    DEFAULT_KCMM_LIB_PATH,
    resolve_repo_path,
)


DEFAULT_STRESS_CASES = (
    CompletionCase(
        name="stress_history",
        prompt="The history of operating systems shows that",
        max_tokens=24,
    ),
    CompletionCase(
        name="stress_memory",
        prompt="In a distributed operating system, memory management",
        max_tokens=24,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Existing local model directory. If omitted, the gate uses "
            ".scratch/kcmm-vllm/real-models/<model-id> and requires "
            "--download-model unless that directory already exists."
        ),
    )
    parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download --model-id into .scratch before running the gate.",
    )
    parser.add_argument("--model-name", default="perf-clean-stress-opt-kcmm")
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--completion-concurrency", type=int, default=2)
    parser.add_argument(
        "--require-min-read-batch",
        type=int,
        default=2,
        help="Fail unless the KCMM read seam observes at least this decode batch.",
    )
    parser.add_argument(
        "--coverage-case",
        action="append",
        default=None,
        metavar="NAME:MAX_TOKENS:PROMPT",
        help=(
            "Completion case to compare. May be repeated. Defaults to two "
            "concurrent real-model prompts sized for a performance-clean stress run."
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
        help="Performance-clean stress gate JSON report path. Defaults to /tmp.",
    )
    parser.add_argument("--latency-warning-ratio", type=float, default=2.5)
    parser.add_argument("--throughput-warning-ratio", type=float, default=0.4)
    parser.add_argument("--memory-warning-ratio", type=float, default=1.5)
    parser.add_argument("--memory-warning-min-delta-mib", type=int, default=512)
    return parser


def parse_cases(values: list[str] | None) -> tuple[CompletionCase, ...]:
    cases = (
        tuple(parse_coverage_case(value) for value in values)
        if values
        else DEFAULT_STRESS_CASES
    )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate coverage case names: {names}")
    return cases


def parse_config(argv: list[str] | None = None) -> tuple[PerfCleanGateConfig, int]:
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "completion_concurrency",
        "require_min_read_batch",
    ):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.require_min_read_batch > args.max_num_seqs:
        parser.error("--require-min-read-batch cannot exceed --max-num-seqs")
    if args.require_min_read_batch > args.completion_concurrency:
        parser.error("--require-min-read-batch cannot exceed --completion-concurrency")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        parser.error("--gpu-memory-utilization must be in the range (0, 1]")
    try:
        coverage_cases = parse_cases(args.coverage_case)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    model_path, downloaded_model = resolve_real_model_path(
        model_id=args.model_id,
        model_path=args.model_path,
        download_model=args.download_model,
    )
    timestamp_ms = int(time.time() * 1000)
    output_path = (
        Path(args.output)
        if args.output
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-perf-clean-stress-{timestamp_ms}.json"
    )
    return (
        PerfCleanGateConfig(
            ab_gate=GateConfig(
                host=args.host,
                port=args.port,
                model_path=model_path,
                model_name=args.model_name,
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
                completion_concurrency=args.completion_concurrency,
                kv_force_non_default_stream=False,
                kv_read_profile=False,
                kv_read_validate_block_tables=False,
                kv_read_fast_current_context_launch=True,
                kv_read_precompile_gpu_kernel=True,
                instrument_kv_reads=False,
                kv_write_verify=False,
                kv_write_device_slots=True,
                tracker_report_on_update=False,
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
            model_id=args.model_id,
            downloaded_model=downloaded_model,
        ),
        args.require_min_read_batch,
    )


def _kcmm_contract(report: dict[str, Any]) -> dict[str, Any]:
    modes = report.get("modes")
    if not isinstance(modes, dict):
        return {}
    kcmm_mode = modes.get("kcmm_gpu_read")
    if not isinstance(kcmm_mode, dict):
        return {}
    contract = kcmm_mode.get("kcmm_gpu_read_contract")
    return contract if isinstance(contract, dict) else {}


def stress_failures(
    *,
    report: dict[str, Any],
    require_min_read_batch: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    contract = _kcmm_contract(report)
    if not contract:
        return failures
    max_read_batch_seen = contract.get("max_read_batch_seen")
    if not isinstance(max_read_batch_seen, int):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_max_read_batch_seen",
                "value": max_read_batch_seen,
            }
        )
    elif max_read_batch_seen < require_min_read_batch:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_batch_requirement_not_met",
                "required_min_read_batch": require_min_read_batch,
                "max_read_batch_seen": max_read_batch_seen,
            }
        )
    return failures


def run_perf_clean_stress_gate(
    config: PerfCleanGateConfig,
    *,
    require_min_read_batch: int,
) -> dict[str, Any]:
    report = run_perf_clean_gate(config)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-performance-clean-stress"
    contract = _kcmm_contract(report)
    report["performance_clean_stress_requirements"] = {
        "max_num_seqs": config.ab_gate.max_num_seqs,
        "max_num_batched_tokens": config.ab_gate.max_num_batched_tokens,
        "completion_concurrency": config.ab_gate.completion_concurrency,
        "coverage_case_count": len(config.ab_gate.coverage_cases),
        "required_min_read_batch": require_min_read_batch,
        "observed_max_read_batch": contract.get("max_read_batch_seen"),
        "observed_max_write_batch": contract.get("max_write_batch_seen"),
    }
    report["correctness_failures"].extend(
        stress_failures(
            report=report,
            require_min_read_batch=require_min_read_batch,
        )
    )
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config, require_min_read_batch = parse_config(argv)
    report = run_perf_clean_stress_gate(
        config,
        require_min_read_batch=require_min_read_batch,
    )
    config.ab_gate.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.ab_gate.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
