"""Phase II.C vLLM GPU read-kernel non-default-stream gate."""

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
    if "--kv-force-non-default-stream" not in args:
        args.append("--kv-force-non-default-stream")
    config = parse_ab_config(args)
    if explicit_output:
        return config

    timestamp_ms = int(time.time() * 1000)
    return replace(
        config,
        output_path=Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-non-default-stream-{timestamp_ms}.json",
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _stream_requirement_failures(contract: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not contract.get("force_non_default_stream", False):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "forced_non_default_stream_not_enabled",
            }
        )

    for prefix in ("read", "write"):
        calls = _positive_int(contract.get(f"{prefix}_forced_non_default_stream_calls"))
        stream_ptr = _positive_int(contract.get(f"{prefix}_last_stream_ptr"))
        default_stream_ptr = contract.get(f"{prefix}_last_default_stream_ptr")
        if calls is None:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{prefix}_forced_non_default_stream_calls_missing",
                    "value": contract.get(f"{prefix}_forced_non_default_stream_calls"),
                }
            )
        if stream_ptr is None:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{prefix}_non_default_stream_ptr_missing",
                    "value": contract.get(f"{prefix}_last_stream_ptr"),
                }
            )
        elif isinstance(default_stream_ptr, int) and stream_ptr == default_stream_ptr:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{prefix}_used_default_stream",
                    "stream_ptr": stream_ptr,
                    "default_stream_ptr": default_stream_ptr,
                }
            )
    return failures


def non_default_stream_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
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
    return _stream_requirement_failures(contract)


def run_non_default_stream_gate(config: GateConfig) -> dict[str, Any]:
    report = run_gate(config)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-non-default-stream"
    modes = report.get("modes", {})
    contract: dict[str, Any] = {}
    if isinstance(modes, dict):
        kcmm_mode = modes.get("kcmm_gpu_read", {})
        if isinstance(kcmm_mode, dict):
            maybe_contract = kcmm_mode.get("kcmm_gpu_read_contract", {})
            if isinstance(maybe_contract, dict):
                contract = maybe_contract
    report["non_default_stream_requirements"] = {
        "forced": config.kv_force_non_default_stream,
        "read_forced_non_default_stream_calls": contract.get(
            "read_forced_non_default_stream_calls"
        ),
        "read_last_stream_ptr": contract.get("read_last_stream_ptr"),
        "read_last_default_stream_ptr": contract.get("read_last_default_stream_ptr"),
        "write_forced_non_default_stream_calls": contract.get(
            "write_forced_non_default_stream_calls"
        ),
        "write_last_stream_ptr": contract.get("write_last_stream_ptr"),
        "write_last_default_stream_ptr": contract.get("write_last_default_stream_ptr"),
    }
    failures = non_default_stream_failures(report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_non_default_stream_gate(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
