"""Phase II.C vLLM GPU read-kernel per-call profiling gate."""

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
    explicit_output = any(
        arg == "--output" or arg.startswith("--output=") for arg in args
    )
    if "--kv-read-profile" not in args:
        args.append("--kv-read-profile")
    config = parse_ab_config(args)
    if explicit_output:
        return config

    timestamp_ms = int(time.time() * 1000)
    return replace(
        config,
        output_path=Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-profile-{timestamp_ms}.json",
    )


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def profile_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    modes = report.get("modes")
    if not isinstance(modes, dict):
        return failures
    kcmm_mode = modes.get("kcmm_gpu_read")
    if not isinstance(kcmm_mode, dict) or not kcmm_mode.get("success"):
        return failures
    contract = kcmm_mode.get("kcmm_gpu_read_contract")
    if not isinstance(contract, dict):
        return [
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_kcmm_gpu_read_contract",
            }
        ]

    profile = contract.get("gpu_kernel_profile")
    if not isinstance(profile, dict):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_gpu_kernel_profile",
            }
        )
        return failures
    if not profile.get("enabled", False):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "gpu_kernel_profile_not_enabled",
            }
        )

    profile_count = _positive_int(profile.get("count"))
    kernel_calls = _positive_int(contract.get("gpu_kernel_calls"))
    if profile_count is None:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "gpu_kernel_profile_count_missing",
                "value": profile.get("count"),
            }
        )
    elif kernel_calls is not None and profile_count != kernel_calls:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "gpu_kernel_profile_count_mismatch",
                "profile_count": profile_count,
                "gpu_kernel_calls": kernel_calls,
            }
        )

    for key in ("min_ms", "avg_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
        if _number(profile.get(key)) is None:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"gpu_kernel_profile_{key}_missing",
                    "value": profile.get(key),
                }
            )

    samples = profile.get("samples_ms")
    if not isinstance(samples, list) or not samples:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "gpu_kernel_profile_samples_missing",
            }
        )
    elif profile_count is not None and len(samples) != profile_count:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "gpu_kernel_profile_sample_count_mismatch",
                "profile_count": profile_count,
                "sample_count": len(samples),
            }
        )
    return failures


def run_profile_gate(config: GateConfig) -> dict[str, Any]:
    report = run_gate(config)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-profile"
    modes = report.get("modes", {})
    contract: dict[str, Any] = {}
    if isinstance(modes, dict):
        kcmm_mode = modes.get("kcmm_gpu_read", {})
        if isinstance(kcmm_mode, dict):
            maybe_contract = kcmm_mode.get("kcmm_gpu_read_contract", {})
            if isinstance(maybe_contract, dict):
                contract = maybe_contract
    profile = contract.get("gpu_kernel_profile")
    report["profile_requirements"] = {
        "enabled": config.kv_read_profile,
        "gpu_kernel_calls": contract.get("gpu_kernel_calls"),
        "profile": profile if isinstance(profile, dict) else None,
    }
    failures = profile_failures(report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_profile_gate(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
