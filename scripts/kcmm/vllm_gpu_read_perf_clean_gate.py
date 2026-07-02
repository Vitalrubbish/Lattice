"""Phase II.C real-model GPU read-kernel gate without test-only overhead."""

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
from scripts.kcmm.vllm_gpu_read_real_model_gate import (
    DEFAULT_MODEL_ID,
    real_model_failures,
    resolve_real_model_path,
)
from scripts.kcmm.vllm_smoke import (
    CompletionCase,
    DEFAULT_KCMM_LIB_PATH,
    resolve_repo_path,
)


DEFAULT_PERF_CLEAN_CASES = (
    CompletionCase(
        name="long_decode",
        prompt="The history of operating systems shows that",
        max_tokens=32,
    ),
)


@dataclass(frozen=True)
class PerfCleanGateConfig:
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
    parser.add_argument("--model-name", default="perf-clean-opt-kcmm")
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
            "Completion case to compare. May be repeated. Defaults to one "
            "longer decode case that gives the request-level metric more signal."
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
        help="Performance-clean gate JSON report path. Defaults to a /tmp file.",
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
        else DEFAULT_PERF_CLEAN_CASES
    )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate coverage case names: {names}")
    return cases


def parse_config(argv: list[str] | None = None) -> PerfCleanGateConfig:
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
        / f"kcmm-vllm-phase-ii-c-gpu-read-perf-clean-{timestamp_ms}.json"
    )
    return PerfCleanGateConfig(
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


def performance_clean_requirements(report: dict[str, Any]) -> dict[str, Any]:
    modes = report.get("modes")
    kcmm_mode = modes.get("kcmm_gpu_read", {}) if isinstance(modes, dict) else {}
    contract = _kcmm_contract(report)
    return {
        "requested_instrument_kv_reads": report.get("instrument_kv_reads"),
        "requested_kv_write_verify": report.get("kv_write_verify"),
        "requested_kv_write_device_slots": report.get("kv_write_device_slots"),
        "requested_kv_read_fast_current_context_launch": report.get(
            "kv_read_fast_current_context_launch"
        ),
        "requested_kv_read_precompile_gpu_kernel": report.get(
            "kv_read_precompile_gpu_kernel"
        ),
        "kcmm_mode_instrument_kv_reads": (
            kcmm_mode.get("instrument_kv_reads")
            if isinstance(kcmm_mode, dict)
            else None
        ),
        "kcmm_mode_kv_write_verify": (
            kcmm_mode.get("kv_write_verify")
            if isinstance(kcmm_mode, dict)
            else None
        ),
        "kcmm_mode_kv_write_device_slots": (
            kcmm_mode.get("kv_write_device_slots")
            if isinstance(kcmm_mode, dict)
            else None
        ),
        "kcmm_mode_tracker_report_on_update": (
            kcmm_mode.get("tracker_report_on_update")
            if isinstance(kcmm_mode, dict)
            else None
        ),
        "kcmm_mode_kv_read_fast_current_context_launch": (
            kcmm_mode.get("kv_read_fast_current_context_launch")
            if isinstance(kcmm_mode, dict)
            else None
        ),
        "kcmm_mode_kv_read_precompile_gpu_kernel": (
            kcmm_mode.get("kv_read_precompile_gpu_kernel")
            if isinstance(kcmm_mode, dict)
            else None
        ),
        "write_verification_enabled": contract.get("write_verification_enabled"),
        "write_verify_rows_per_call": contract.get("write_verify_rows_per_call"),
        "write_device_slot_enabled": contract.get("write_device_slot_enabled"),
        "write_device_slot_active": contract.get("write_device_slot_active"),
        "write_device_slot_calls": contract.get("write_device_slot_calls"),
        "write_host_slot_calls": contract.get("write_host_slot_calls"),
        "write_device_slot_status_checks": contract.get(
            "write_device_slot_status_checks"
        ),
        "write_device_slot_status_error_count": contract.get(
            "write_device_slot_status_error_count"
        ),
        "write_device_slot_kernel_precompile_requested": contract.get(
            "write_device_slot_kernel_precompile_requested"
        ),
        "write_device_slot_kernel_precompile_succeeded": contract.get(
            "write_device_slot_kernel_precompile_succeeded"
        ),
        "write_device_slot_kernel_precompile_calls": contract.get(
            "write_device_slot_kernel_precompile_calls"
        ),
        "write_device_slot_kernel_precompile_elapsed_ms": contract.get(
            "write_device_slot_kernel_precompile_elapsed_ms"
        ),
        "write_device_slot_total_blocks": contract.get(
            "write_device_slot_total_blocks"
        ),
        "write_device_slot_total_blocks_refreshes": contract.get(
            "write_device_slot_total_blocks_refreshes"
        ),
        "write_device_slot_block_state_epoch_queries": contract.get(
            "write_device_slot_block_state_epoch_queries"
        ),
        "write_device_slot_offset_table_cache_hits": contract.get(
            "write_device_slot_offset_table_cache_hits"
        ),
        "write_device_slot_offset_table_cache_rebuilds": contract.get(
            "write_device_slot_offset_table_cache_rebuilds"
        ),
        "write_device_slot_valid_table_cache_hits": contract.get(
            "write_device_slot_valid_table_cache_hits"
        ),
        "write_device_slot_valid_table_cache_rebuilds": contract.get(
            "write_device_slot_valid_table_cache_rebuilds"
        ),
        "read_fast_current_context_launch": contract.get(
            "read_fast_current_context_launch"
        ),
        "read_gpu_kernel_precompile_requested": contract.get(
            "read_gpu_kernel_precompile_requested"
        ),
        "read_gpu_kernel_precompile_succeeded": contract.get(
            "read_gpu_kernel_precompile_succeeded"
        ),
        "read_gpu_kernel_precompile_calls": contract.get(
            "read_gpu_kernel_precompile_calls"
        ),
        "read_gpu_kernel_precompile_elapsed_ms": contract.get(
            "read_gpu_kernel_precompile_elapsed_ms"
        ),
        "read_block_table_validation_enabled": contract.get(
            "read_block_table_validation_enabled"
        ),
        "read_compact_plan_metadata": contract.get("read_compact_plan_metadata"),
        "read_compact_plan_metadata_calls": contract.get(
            "read_compact_plan_metadata_calls"
        ),
        "read_detailed_plan_metadata_calls": contract.get(
            "read_detailed_plan_metadata_calls"
        ),
        "offset_table_builds": contract.get("offset_table_builds"),
        "offset_table_cache_hits": contract.get("offset_table_cache_hits"),
        "offset_table_cache_rebuilds": contract.get(
            "offset_table_cache_rebuilds"
        ),
        "read_report_on_update": contract.get("read_report_on_update"),
        "read_report_write_count": contract.get("read_report_write_count"),
        "write_report_on_update": contract.get("write_report_on_update"),
        "write_report_write_count": contract.get("write_report_write_count"),
        "kcmm_write_verified_rows": contract.get("kcmm_write_verified_rows"),
        "write_stream_synchronize_for_verification_calls": contract.get(
            "write_stream_synchronize_for_verification_calls"
        ),
        "replacement_calls": contract.get("replacement_calls"),
        "gpu_kernel_calls": contract.get("gpu_kernel_calls"),
        "stream_aware_kernel_calls": contract.get("stream_aware_kernel_calls"),
        "reference_read_bytes": contract.get("reference_read_bytes"),
    }


def performance_clean_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    modes = report.get("modes")
    if not isinstance(modes, dict):
        return failures
    kcmm_mode = modes.get("kcmm_gpu_read")
    if not isinstance(kcmm_mode, dict) or not kcmm_mode.get("success"):
        return failures
    contract = _kcmm_contract(report)
    if not contract:
        return [
            {
                "mode": "kcmm_gpu_read",
                "reason": "missing_kcmm_gpu_read_contract",
            }
        ]

    if report.get("instrument_kv_reads") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_trace_instrumentation_not_disabled_in_config",
                "value": report.get("instrument_kv_reads"),
            }
        )
    if kcmm_mode.get("instrument_kv_reads") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_trace_instrumentation_not_disabled_in_smoke",
                "value": kcmm_mode.get("instrument_kv_reads"),
            }
        )
    if report.get("kv_write_verify") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_verification_not_disabled_in_config",
                "value": report.get("kv_write_verify"),
            }
        )
    if kcmm_mode.get("kv_write_verify") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_verification_not_disabled_in_smoke",
                "value": kcmm_mode.get("kv_write_verify"),
            }
        )
    if report.get("kv_write_device_slots") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_write_not_enabled_in_config",
                "value": report.get("kv_write_device_slots"),
            }
        )
    if kcmm_mode.get("kv_write_device_slots") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_write_not_enabled_in_smoke",
                "value": kcmm_mode.get("kv_write_device_slots"),
            }
        )
    if kcmm_mode.get("tracker_report_on_update") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "tracker_report_on_update_not_disabled_in_smoke",
                "value": kcmm_mode.get("tracker_report_on_update"),
            }
        )
    if report.get("kv_read_fast_current_context_launch") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "fast_current_context_launch_not_enabled_in_config",
                "value": report.get("kv_read_fast_current_context_launch"),
            }
        )
    if kcmm_mode.get("kv_read_fast_current_context_launch") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "fast_current_context_launch_not_enabled_in_smoke",
                "value": kcmm_mode.get("kv_read_fast_current_context_launch"),
            }
        )
    if contract.get("read_fast_current_context_launch") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "fast_current_context_launch_not_enabled_in_report",
                "value": contract.get("read_fast_current_context_launch"),
            }
        )
    if report.get("kv_read_precompile_gpu_kernel") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_kernel_precompile_not_enabled_in_config",
                "value": report.get("kv_read_precompile_gpu_kernel"),
            }
        )
    if kcmm_mode.get("kv_read_precompile_gpu_kernel") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_kernel_precompile_not_enabled_in_smoke",
                "value": kcmm_mode.get("kv_read_precompile_gpu_kernel"),
            }
        )
    if contract.get("read_gpu_kernel_precompile_requested") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_kernel_precompile_not_requested_in_report",
                "value": contract.get("read_gpu_kernel_precompile_requested"),
            }
        )
    if contract.get("read_gpu_kernel_precompile_succeeded") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_kernel_precompile_not_succeeded",
                "value": contract.get("read_gpu_kernel_precompile_succeeded"),
            }
        )
    if contract.get("read_gpu_kernel_precompile_calls") != 1:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_kernel_precompile_call_count_unexpected",
                "value": contract.get("read_gpu_kernel_precompile_calls"),
                "expected": 1,
            }
        )
    if contract.get("read_block_table_validation_enabled") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_block_table_validation_not_disabled",
                "value": contract.get("read_block_table_validation_enabled"),
            }
        )
    if contract.get("read_compact_plan_metadata") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_compact_plan_metadata_not_enabled",
                "value": contract.get("read_compact_plan_metadata"),
            }
        )
    compact_plan_calls = contract.get("read_compact_plan_metadata_calls")
    replacement_calls = contract.get("replacement_calls")
    if not isinstance(compact_plan_calls, int) or compact_plan_calls <= 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_compact_plan_metadata_calls_missing",
                "value": compact_plan_calls,
            }
        )
    elif isinstance(replacement_calls, int) and compact_plan_calls != replacement_calls:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_compact_plan_metadata_call_count_mismatch",
                "value": compact_plan_calls,
                "expected": replacement_calls,
            }
        )
    detailed_plan_calls = contract.get("read_detailed_plan_metadata_calls")
    if detailed_plan_calls != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "read_detailed_plan_metadata_used_in_perf_clean",
                "value": detailed_plan_calls,
                "expected": 0,
            }
        )
    if not isinstance(contract.get("offset_table_cache_hits"), int):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "offset_table_cache_hits_missing",
                "value": contract.get("offset_table_cache_hits"),
            }
        )
    if not isinstance(contract.get("offset_table_cache_rebuilds"), int):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "offset_table_cache_rebuilds_missing",
                "value": contract.get("offset_table_cache_rebuilds"),
            }
        )
    cache_hits = contract.get("offset_table_cache_hits")
    cache_rebuilds = contract.get("offset_table_cache_rebuilds")
    if isinstance(cache_hits, int) and isinstance(cache_rebuilds, int):
        if cache_hits <= 0:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": "offset_table_cache_unused",
                    "offset_table_cache_hits": cache_hits,
                    "offset_table_cache_rebuilds": cache_rebuilds,
                }
            )
    for key in ("read_report_on_update", "write_report_on_update"):
        if contract.get(key) is not False:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{key}_not_disabled",
                    "value": contract.get(key),
                }
            )
    for key in ("read_report_write_count", "write_report_write_count"):
        value = contract.get(key)
        if not isinstance(value, int) or value > 2:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{key}_too_high",
                    "value": value,
                    "threshold": 2,
                }
            )
    if contract.get("write_verification_enabled") is not False:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_verification_enabled_in_report",
                "value": contract.get("write_verification_enabled"),
            }
        )
    if contract.get("write_verify_rows_per_call") != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_verify_rows_per_call_nonzero",
                "value": contract.get("write_verify_rows_per_call"),
            }
        )
    if contract.get("write_device_slot_enabled") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_write_not_enabled_in_report",
                "value": contract.get("write_device_slot_enabled"),
            }
        )
    if contract.get("write_device_slot_active") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_write_not_active_in_report",
                "value": contract.get("write_device_slot_active"),
            }
        )
    device_calls = contract.get("write_device_slot_calls")
    if not isinstance(device_calls, int) or device_calls <= 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_write_calls_missing",
                "value": device_calls,
            }
        )
    if contract.get("write_host_slot_calls") != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "host_slot_writes_used_in_device_slot_mode",
                "value": contract.get("write_host_slot_calls"),
            }
        )
    if contract.get("write_device_slot_status_error_count") != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_status_errors_seen",
                "value": contract.get("write_device_slot_status_error_count"),
            }
        )
    if contract.get("write_device_slot_kernel_precompile_requested") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_kernel_precompile_not_requested_in_report",
                "value": contract.get(
                    "write_device_slot_kernel_precompile_requested"
                ),
            }
        )
    if contract.get("write_device_slot_kernel_precompile_succeeded") is not True:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_kernel_precompile_not_succeeded",
                "value": contract.get(
                    "write_device_slot_kernel_precompile_succeeded"
                ),
            }
        )
    if contract.get("write_device_slot_kernel_precompile_calls") != 1:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_kernel_precompile_call_count_unexpected",
                "value": contract.get("write_device_slot_kernel_precompile_calls"),
                "expected": 1,
            }
        )
    total_block_refreshes = contract.get("write_device_slot_total_blocks_refreshes")
    if not isinstance(total_block_refreshes, int) or total_block_refreshes <= 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_device_slot_total_blocks_refreshes_missing",
                "value": total_block_refreshes,
            }
        )
    elif isinstance(device_calls, int) and device_calls > 0:
        if total_block_refreshes >= device_calls:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": "write_device_slot_total_blocks_refreshed_per_write",
                    "value": total_block_refreshes,
                    "device_slot_write_calls": device_calls,
                }
            )
    epoch_queries = contract.get("write_device_slot_block_state_epoch_queries")
    if not isinstance(epoch_queries, int) or epoch_queries <= 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_device_slot_epoch_queries_missing",
                "value": epoch_queries,
            }
        )
    elif isinstance(device_calls, int) and device_calls > 0:
        max_expected_epoch_queries = device_calls * 2
        if epoch_queries >= max_expected_epoch_queries:
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": "write_device_slot_epoch_queries_too_high",
                    "value": epoch_queries,
                    "threshold_exclusive": max_expected_epoch_queries,
                }
            )
    status_checks = contract.get("write_device_slot_status_checks")
    if not isinstance(status_checks, int) or status_checks <= 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "device_slot_status_checks_missing",
                "value": status_checks,
            }
        )
    if contract.get("kcmm_write_verified_rows") != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_verified_rows_nonzero",
                "value": contract.get("kcmm_write_verified_rows"),
            }
        )
    if contract.get("write_stream_synchronize_for_verification_calls") != 0:
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "write_verification_synchronizations_nonzero",
                "value": contract.get(
                    "write_stream_synchronize_for_verification_calls"
                ),
            }
        )
    return failures


def run_perf_clean_gate(config: PerfCleanGateConfig) -> dict[str, Any]:
    report = run_gate(config.ab_gate)
    report["gate"] = "stock-vs-kcmm-gpu-read-kernel-performance-clean"
    report["real_model"] = {
        "model_id": config.model_id,
        "model_path": str(config.ab_gate.model_path),
        "downloaded_model": config.downloaded_model,
    }
    report["performance_clean_requirements"] = performance_clean_requirements(report)
    failures = real_model_failures(report) + performance_clean_failures(report)
    report["correctness_failures"].extend(failures)
    report["passed"] = not report["correctness_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_perf_clean_gate(config)
    config.ab_gate.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.ab_gate.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
