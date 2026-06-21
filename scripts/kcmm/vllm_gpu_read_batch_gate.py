"""Phase II.C stock-vs-KCMM GPU read-kernel batch/concurrency gate."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_gpu_read_ab_gate import (
    GateConfig,
    parse_coverage_case,
    run_gate,
)
from scripts.kcmm.vllm_smoke import (
    CompletionCase,
    DEFAULT_KCMM_LIB_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    repo_root,
    resolve_repo_path,
)


DEFAULT_BATCH_CASES = (
    CompletionCase(
        name="parallel_alpha",
        prompt="alpha beta gamma delta epsilon zeta eta theta",
        max_tokens=4,
    ),
    CompletionCase(
        name="parallel_math",
        prompt="Question: 2 + 2 =",
        max_tokens=4,
    ),
)


@dataclass(frozen=True)
class BatchGateConfig:
    ab_gate: GateConfig
    require_min_read_batch: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--coverage-case",
        action="append",
        default=None,
        metavar="NAME:MAX_TOKENS:PROMPT",
        help=(
            "Completion case to compare. May be repeated. Defaults to two "
            "parallel requests sized to exercise vLLM multi-sequence decode."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--completion-concurrency", type=int, default=2)
    parser.add_argument(
        "--require-min-read-batch",
        type=int,
        default=2,
        help="Fail unless the KCMM read seam observes at least this decode batch.",
    )
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
        help="Batch/concurrency gate JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument("--latency-warning-ratio", type=float, default=2.0)
    parser.add_argument("--throughput-warning-ratio", type=float, default=0.5)
    parser.add_argument("--memory-warning-ratio", type=float, default=1.5)
    parser.add_argument("--memory-warning-min-delta-mib", type=int, default=256)
    return parser


def parse_cases(values: list[str] | None) -> tuple[CompletionCase, ...]:
    cases = (
        tuple(parse_coverage_case(value) for value in values)
        if values
        else DEFAULT_BATCH_CASES
    )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate coverage case names: {names}")
    return cases


def parse_config(argv: list[str] | None = None) -> BatchGateConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    positive_fields = (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "completion_concurrency",
        "require_min_read_batch",
    )
    for field in positive_fields:
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.require_min_read_batch > args.max_num_seqs:
        parser.error("--require-min-read-batch cannot exceed --max-num-seqs")
    if args.require_min_read_batch > args.completion_concurrency:
        parser.error("--require-min-read-batch cannot exceed --completion-concurrency")
    try:
        coverage_cases = parse_cases(args.coverage_case)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    timestamp_ms = int(time.time() * 1000)
    output_path = (
        Path(args.output)
        if args.output
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-batch-{timestamp_ms}.json"
    )
    return BatchGateConfig(
        ab_gate=GateConfig(
            host=args.host,
            port=args.port,
            model_path=resolve_repo_path(args.model_path),
            model_name=args.model_name,
            kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
            timeout_seconds=args.timeout_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
            prompt=coverage_cases[0].prompt,
            max_tokens=coverage_cases[0].max_tokens,
            coverage_cases=coverage_cases,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            completion_concurrency=args.completion_concurrency,
            build_kcmm=args.build_kcmm,
            keep_model=args.keep_model,
            print_seams=args.print_seams,
            output_path=output_path,
            latency_warning_ratio=args.latency_warning_ratio,
            throughput_warning_ratio=args.throughput_warning_ratio,
            memory_warning_ratio=args.memory_warning_ratio,
            memory_warning_min_delta_mib=args.memory_warning_min_delta_mib,
        ),
        require_min_read_batch=args.require_min_read_batch,
    )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def batch_observation_failures(
    config: BatchGateConfig,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    modes = report.get("modes")
    if not isinstance(modes, dict):
        return []
    kcmm_mode = modes.get("kcmm_gpu_read")
    if not isinstance(kcmm_mode, dict) or not kcmm_mode.get("success"):
        return []
    contract = kcmm_mode.get("kcmm_gpu_read_contract")
    if not isinstance(contract, dict):
        return [
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_kcmm_gpu_read_contract",
                "detail": "batch gate could not inspect the KCMM read report",
            }
        ]

    max_read_batch_seen = _int_or_none(contract.get("max_read_batch_seen"))
    if max_read_batch_seen is None:
        return [
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_max_read_batch_seen",
                "detail": "KCMM read report did not record a decode batch size",
            }
        ]
    if max_read_batch_seen < config.require_min_read_batch:
        return [
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_batch_requirement_not_met",
                "required_min_read_batch": config.require_min_read_batch,
                "max_read_batch_seen": max_read_batch_seen,
            }
        ]
    return []


def run_batch_gate(config: BatchGateConfig) -> dict[str, Any]:
    report = run_gate(config.ab_gate)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-batch-concurrency"
    modes = report.get("modes", {})
    kcmm_contract = {}
    if isinstance(modes, dict):
        kcmm_mode = modes.get("kcmm_gpu_read", {})
        if isinstance(kcmm_mode, dict):
            maybe_contract = kcmm_mode.get("kcmm_gpu_read_contract", {})
            if isinstance(maybe_contract, dict):
                kcmm_contract = maybe_contract
    report["batch_requirements"] = {
        "max_num_seqs": config.ab_gate.max_num_seqs,
        "max_num_batched_tokens": config.ab_gate.max_num_batched_tokens,
        "completion_concurrency": config.ab_gate.completion_concurrency,
        "required_min_read_batch": config.require_min_read_batch,
        "observed_max_read_batch": kcmm_contract.get("max_read_batch_seen"),
        "observed_max_write_batch": kcmm_contract.get("max_write_batch_seen"),
    }
    failures = batch_observation_failures(config, report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_batch_gate(config)
    config.ab_gate.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.ab_gate.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
