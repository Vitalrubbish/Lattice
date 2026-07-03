"""Phase II.C performance-clean GPU read gate with host-side attribution."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_gpu_read_perf_clean_gate import (
    PerfCleanGateConfig,
    parse_config as parse_perf_clean_config,
    run_perf_clean_gate,
)


def parse_config(argv: list[str] | None = None) -> PerfCleanGateConfig:
    args = list(sys.argv[1:] if argv is None else argv)
    explicit_output = any(
        arg == "--output" or arg.startswith("--output=") for arg in args
    )
    config = parse_perf_clean_config(args)
    output_path = config.ab_gate.output_path
    if not explicit_output:
        timestamp_ms = int(time.time() * 1000)
        output_path = (
            Path(tempfile.gettempdir())
            / f"kcmm-vllm-phase-ii-c-gpu-read-host-profile-{timestamp_ms}.json"
        )
    return replace(
        config,
        ab_gate=replace(
            config.ab_gate,
            tracker_host_profile=True,
            output_path=output_path,
        ),
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


def _host_profile_sections(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    sections = profile.get("sections")
    return sections if isinstance(sections, dict) else {}


def _top_sections(sections: dict[str, Any]) -> list[dict[str, Any]]:
    top_sections: list[dict[str, Any]] = []
    for name in sorted(
        sections,
        key=lambda section_name: sections[section_name].get("total_ms", 0),
        reverse=True,
    )[:8]:
        section = sections[name]
        top_sections.append(
            {
                "name": name,
                "count": section.get("count"),
                "total_ms": section.get("total_ms"),
                "avg_us": section.get("avg_us"),
            }
        )
    return top_sections


def host_profile_requirements(report: dict[str, Any]) -> dict[str, Any]:
    contract = _kcmm_contract(report)
    read_profile = contract.get("read_host_profile")
    write_profile = contract.get("write_host_profile")
    read_sections = _host_profile_sections(read_profile)
    write_sections = _host_profile_sections(write_profile)
    return {
        "read_host_profile_enabled": (
            read_profile.get("enabled") if isinstance(read_profile, dict) else None
        ),
        "write_host_profile_enabled": (
            write_profile.get("enabled") if isinstance(write_profile, dict) else None
        ),
        "read_section_count": len(read_sections),
        "write_section_count": len(write_sections),
        "read_top_sections": _top_sections(read_sections),
        "write_top_sections": _top_sections(write_sections),
    }


def host_profile_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    contract = _kcmm_contract(report)
    if not contract:
        return [
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_kcmm_gpu_read_contract",
            }
        ]
    for name, profile in (
        ("read", contract.get("read_host_profile")),
        ("write", contract.get("write_host_profile")),
    ):
        if not isinstance(profile, dict):
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"missing_{name}_host_profile",
                }
            )
            continue
        if profile.get("enabled") is not True:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{name}_host_profile_not_enabled",
                    "value": profile.get("enabled"),
                }
            )
        sections = _host_profile_sections(profile)
        if not sections:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{name}_host_profile_sections_missing",
                }
            )
    return failures


def run_host_profile_gate(config: PerfCleanGateConfig) -> dict[str, Any]:
    report = run_perf_clean_gate(config)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-host-profile"
    report["host_profile_requirements"] = host_profile_requirements(report)
    failures = host_profile_failures(report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_host_profile_gate(config)
    config.ab_gate.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.ab_gate.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
