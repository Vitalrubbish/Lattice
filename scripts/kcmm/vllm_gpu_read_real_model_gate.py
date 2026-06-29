"""Phase II.C stock-vs-KCMM GPU read-kernel gate for a real vLLM model."""

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
    repo_root,
    resolve_repo_path,
)


DEFAULT_MODEL_ID = "facebook/opt-125m"
DEFAULT_COVERAGE_CASES = (
    CompletionCase(name="hello", prompt="Hello", max_tokens=2),
    CompletionCase(name="math", prompt="Question: 2 + 2 =", max_tokens=2),
)


@dataclass(frozen=True)
class RealModelGateConfig:
    ab_gate: GateConfig
    model_id: str
    downloaded_model: bool


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
    parser.add_argument("--model-name", default="real-opt-kcmm")
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
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
            "Completion case to compare. May be repeated. Defaults to two "
            "short prompts to keep the first real-model gate small."
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
        help="Real-model gate JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument("--latency-warning-ratio", type=float, default=2.5)
    parser.add_argument("--throughput-warning-ratio", type=float, default=0.4)
    parser.add_argument("--memory-warning-ratio", type=float, default=1.5)
    parser.add_argument("--memory-warning-min-delta-mib", type=int, default=512)
    return parser


def default_model_path(model_id: str) -> Path:
    safe_name = model_id.replace("/", "--")
    return repo_root() / ".scratch" / "kcmm-vllm" / "real-models" / safe_name


def model_path_has_config(model_path: Path) -> bool:
    return (model_path / "config.json").exists()


def resolve_real_model_path(
    *,
    model_id: str,
    model_path: str | None,
    download_model: bool,
) -> tuple[Path, bool]:
    if model_path:
        path = resolve_repo_path(model_path)
        if not model_path_has_config(path):
            raise SystemExit(f"model path does not look complete: {path}")
        return path, False

    path = default_model_path(model_id)
    if model_path_has_config(path):
        return path, False
    if not download_model:
        raise SystemExit(
            "real-model gate needs an existing --model-path or "
            "--download-model to populate "
            f"{path}"
        )

    from huggingface_hub import snapshot_download

    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=path,
        allow_patterns=(
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "*.bin",
        ),
    )
    if not model_path_has_config(path):
        raise SystemExit(f"downloaded model has no config.json: {path}")
    return path, True


def parse_cases(values: list[str] | None) -> tuple[CompletionCase, ...]:
    cases = (
        tuple(parse_coverage_case(value) for value in values)
        if values
        else DEFAULT_COVERAGE_CASES
    )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate coverage case names: {names}")
    return cases


def parse_config(argv: list[str] | None = None) -> RealModelGateConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
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
        / f"kcmm-vllm-phase-ii-c-gpu-read-real-model-{timestamp_ms}.json"
    )
    return RealModelGateConfig(
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
            completion_concurrency=1,
            kv_force_non_default_stream=False,
            kv_read_profile=False,
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
    )


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def real_model_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
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
            }
        ]

    failures: list[dict[str, Any]] = []
    if _positive_int(contract.get("gpu_kernel_calls")) is None:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_gpu_kernel_calls",
                "value": contract.get("gpu_kernel_calls"),
            }
        )
    if contract.get("reference_read_bytes") != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "cpu_staged_reference_reads_seen",
                "reference_read_bytes": contract.get("reference_read_bytes"),
            }
        )
    return failures


def run_real_model_gate(config: RealModelGateConfig) -> dict[str, Any]:
    report = run_gate(config.ab_gate)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-real-model"
    report["real_model"] = {
        "model_id": config.model_id,
        "model_path": str(config.ab_gate.model_path),
        "downloaded_model": config.downloaded_model,
    }
    failures = real_model_failures(report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_real_model_gate(config)
    config.ab_gate.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.ab_gate.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
