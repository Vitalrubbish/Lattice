"""Phase II.C vLLM GPU read-kernel tensor-parallel gate."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_gpu_read_ab_gate import (
    GateConfig,
    parse_config as parse_ab_config,
    run_gate,
)


def parse_config(argv: list[str] | None = None) -> GateConfig:
    args = list(sys.argv[1:] if argv is None else argv)
    explicit_output = any(arg == "--output" or arg.startswith("--output=") for arg in args)
    explicit_tp = any(
        arg == "--tensor-parallel-size" or arg.startswith("--tensor-parallel-size=")
        for arg in args
    )
    if not explicit_tp:
        args.extend(["--tensor-parallel-size", "2"])
    config = parse_ab_config(args)
    if config.tensor_parallel_size < 2:
        raise SystemExit("--tensor-parallel-size must be at least 2 for this gate")
    if explicit_output:
        return config

    timestamp_ms = int(time.time() * 1000)
    return replace(
        config,
        output_path=Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-tensor-parallel-{timestamp_ms}.json",
    )


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def tensor_parallel_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    tensor_parallel_size = report.get("tensor_parallel_size")
    if not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 2:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "tensor_parallel_size_not_enabled",
                "tensor_parallel_size": tensor_parallel_size,
            }
        )

    modes = report.get("modes")
    if not isinstance(modes, dict):
        return failures
    kcmm_mode = modes.get("kcmm_gpu_read")
    if not isinstance(kcmm_mode, dict) or not kcmm_mode.get("success"):
        return failures
    contract = kcmm_mode.get("kcmm_gpu_read_contract")
    if not isinstance(contract, dict):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_kcmm_gpu_read_contract",
            }
        )
        return failures

    if _positive_int(contract.get("gpu_kernel_calls")) is None:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_gpu_kernel_calls",
                "value": contract.get("gpu_kernel_calls"),
            }
        )
    if _positive_int(contract.get("stream_aware_kernel_calls")) is None:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_stream_aware_kernel_calls",
                "value": contract.get("stream_aware_kernel_calls"),
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


def run_tensor_parallel_gate(config: GateConfig) -> dict[str, Any]:
    report = run_gate(config)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-tensor-parallel"
    modes = report.get("modes", {})
    contract: dict[str, Any] = {}
    if isinstance(modes, dict):
        kcmm_mode = modes.get("kcmm_gpu_read", {})
        if isinstance(kcmm_mode, dict):
            maybe_contract = kcmm_mode.get("kcmm_gpu_read_contract", {})
            if isinstance(maybe_contract, dict):
                contract = maybe_contract
    report["tensor_parallel_requirements"] = {
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_kernel_calls": contract.get("gpu_kernel_calls"),
        "stream_aware_kernel_calls": contract.get("stream_aware_kernel_calls"),
        "reference_read_bytes": contract.get("reference_read_bytes"),
    }
    failures = tensor_parallel_failures(report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_tensor_parallel_gate(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
