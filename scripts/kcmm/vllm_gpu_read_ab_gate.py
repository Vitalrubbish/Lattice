"""Phase II.C stock-vs-KCMM GPU read-kernel vLLM A/B gate."""

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
    CompletionCase,
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


MODE_ORDER = ("stock", "kcmm_gpu_read")
DEFAULT_COVERAGE_CASES = (
    CompletionCase(name="hello", prompt="Hello", max_tokens=4),
    CompletionCase(name="math", prompt="Question: 2 + 2 =", max_tokens=3),
    CompletionCase(
        name="long_context",
        prompt=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron pi rho sigma tau"
        ),
        max_tokens=4,
    ),
)


@dataclass(frozen=True)
class GateConfig:
    host: str
    port: int
    model_path: Path
    model_name: str
    kcmm_lib_path: Path
    timeout_seconds: float
    shutdown_timeout_seconds: float
    generate_tiny_model: bool
    prompt: str
    max_tokens: int
    coverage_cases: tuple[CompletionCase, ...]
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    completion_concurrency: int
    kv_force_non_default_stream: bool
    kv_read_profile: bool
    kv_read_validate_block_tables: bool
    kv_read_fast_current_context_launch: bool
    kv_read_precompile_gpu_kernel: bool
    instrument_kv_reads: bool
    kv_write_verify: bool
    kv_write_device_slots: bool
    tracker_report_on_update: bool
    tracker_host_profile: bool
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
    parser.add_argument(
        "--generate-tiny-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate the local tiny OPT model when --model-path is missing. "
            "Disable this for externally supplied real model directories."
        ),
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--completion-concurrency", type=int, default=1)
    parser.add_argument(
        "--kv-force-non-default-stream",
        action="store_true",
        help=(
            "Run KCMM write/read replacement launches through a dedicated "
            "non-default CUDA stream with explicit stream waits."
        ),
    )
    parser.add_argument(
        "--kv-read-profile",
        action="store_true",
        help=(
            "Enable per-call CUDA event profiling for KCMM GPU read kernels "
            "in the KCMM mode."
        ),
    )
    parser.add_argument(
        "--instrument-kv-reads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable observer-only paged_attention read tracing in the KCMM "
            "mode. Disable for performance-clean gates."
        ),
    )
    parser.add_argument(
        "--kv-write-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable bounded D2H verification of KCMM KV writes in the KCMM "
            "mode. Disable for performance-clean gates."
        ),
    )
    parser.add_argument(
        "--kv-write-device-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use device-resident vLLM slot_mapping tensors for KCMM KV writes "
            "in the KCMM mode. Requires --no-kv-write-verify."
        ),
    )
    parser.add_argument(
        "--kv-read-validate-block-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate sampled paged-attention block_tables on the host in the "
            "KCMM mode. Disable for performance-clean gates after correctness "
            "coverage passes."
        ),
    )
    parser.add_argument(
        "--tracker-report-on-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write KCMM tracker reports after every observed seam call in the "
            "KCMM mode. Disable for performance-clean gates that only need "
            "final reports."
        ),
    )
    parser.add_argument(
        "--tracker-host-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect section-level host timings in KCMM tracker final reports.",
    )
    parser.add_argument(
        "--kv-read-fast-current-context-launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the read launch ABI that assumes the vLLM/PyTorch CUDA "
            "context is already current in the KCMM mode."
        ),
    )
    parser.add_argument(
        "--kv-read-precompile-gpu-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Precompile/load the KCMM paged-attention read kernel before "
            "the measured request in the KCMM mode."
        ),
    )
    parser.add_argument(
        "--coverage-case",
        action="append",
        default=None,
        metavar="NAME:MAX_TOKENS:PROMPT",
        help=(
            "Completion case to compare. May be repeated. Defaults to a "
            "short prompt, a math prompt, and a longer-context prompt. "
            "Passing --prompt/--max-tokens without --coverage-case keeps "
            "single-case compatibility."
        ),
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


def parse_coverage_case(value: str) -> CompletionCase:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "coverage cases must use NAME:MAX_TOKENS:PROMPT"
        )
    name, max_tokens_text, prompt = parts
    if not name:
        raise argparse.ArgumentTypeError("coverage case name cannot be empty")
    if not prompt:
        raise argparse.ArgumentTypeError("coverage case prompt cannot be empty")
    try:
        max_tokens = int(max_tokens_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid coverage case max tokens: {max_tokens_text}"
        ) from exc
    if max_tokens <= 0:
        raise argparse.ArgumentTypeError("coverage case max tokens must be positive")
    return CompletionCase(name=name, prompt=prompt, max_tokens=max_tokens)


def coverage_cases_from_args(args: argparse.Namespace) -> tuple[CompletionCase, ...]:
    if args.coverage_case:
        cases = tuple(parse_coverage_case(value) for value in args.coverage_case)
    elif args.prompt != "Hello" or args.max_tokens != 4:
        cases = (
            CompletionCase(
                name="cli",
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            ),
        )
    else:
        cases = DEFAULT_COVERAGE_CASES
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate coverage case names: {names}")
    return cases


def parse_config(argv: list[str] | None = None) -> GateConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tensor_parallel_size",
        "completion_concurrency",
    ):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        parser.error("--gpu-memory-utilization must be in the range (0, 1]")
    try:
        coverage_cases = coverage_cases_from_args(args)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    timestamp_ms = int(time.time() * 1000)
    output_path = (
        Path(args.output)
        if args.output
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-ab-{timestamp_ms}.json"
    )
    return GateConfig(
        host=args.host,
        port=args.port,
        model_path=resolve_repo_path(args.model_path),
        model_name=args.model_name,
        kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
        timeout_seconds=args.timeout_seconds,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
        generate_tiny_model=args.generate_tiny_model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        coverage_cases=coverage_cases,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        completion_concurrency=args.completion_concurrency,
        kv_force_non_default_stream=args.kv_force_non_default_stream,
        kv_read_profile=args.kv_read_profile,
        kv_read_validate_block_tables=args.kv_read_validate_block_tables,
        kv_read_fast_current_context_launch=(
            args.kv_read_fast_current_context_launch
        ),
        kv_read_precompile_gpu_kernel=args.kv_read_precompile_gpu_kernel,
        instrument_kv_reads=args.instrument_kv_reads,
        kv_write_verify=args.kv_write_verify,
        kv_write_device_slots=args.kv_write_device_slots,
        tracker_report_on_update=args.tracker_report_on_update,
        tracker_host_profile=args.tracker_host_profile,
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
    is_gpu_read = mode_name == "kcmm_gpu_read"
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
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
        completion_concurrency=config.completion_concurrency,
        build_kcmm=(config.build_kcmm and not is_stock),
        keep_model=True,
        generate_tiny_model=config.generate_tiny_model,
        print_seams=(config.print_seams and not is_stock),
        instrument_allocators=False,
        instrument_kv_writes=False,
        instrument_kv_reads=(is_gpu_read and config.instrument_kv_reads),
        kv_read_offset_table=False,
        kv_read_replace_candidate=False,
        kv_read_gpu_kernel_candidate=is_gpu_read,
        kv_read_profile=(is_gpu_read and config.kv_read_profile),
        kv_read_validate_block_tables=config.kv_read_validate_block_tables,
        kv_read_fast_current_context_launch=(
            is_gpu_read and config.kv_read_fast_current_context_launch
        ),
        kv_read_precompile_gpu_kernel=(
            is_gpu_read and config.kv_read_precompile_gpu_kernel
        ),
        tracker_report_on_update=config.tracker_report_on_update,
        tracker_host_profile=config.tracker_host_profile,
        kv_write_mirror=False,
        kv_write_replace_candidate=is_gpu_read,
        kv_write_verify=config.kv_write_verify,
        kv_write_device_slots=(is_gpu_read and config.kv_write_device_slots),
        kv_force_non_default_stream=(
            is_gpu_read and config.kv_force_non_default_stream
        ),
        runtime_derived_pool=is_gpu_read,
        shadow_allocations=False,
        backed_allocations=is_gpu_read,
        allocator_trace_path=run_dir / f"{mode_name}-allocator-trace.jsonl",
        kv_write_trace_path=run_dir / f"{mode_name}-kv-write-trace.jsonl",
        kv_read_trace_path=run_dir / f"{mode_name}-kv-read-trace.jsonl",
        kv_read_offset_table_report_path=(
            run_dir / f"{mode_name}-kv-read-offset-table-report.json"
        ),
        kv_write_mirror_report_path=(
            run_dir / f"{mode_name}-kv-write-mirror-report.json"
        ),
        shadow_report_path=run_dir / f"{mode_name}-shadow-report.json",
        backed_report_path=run_dir / f"{mode_name}-backed-report.json",
        require_allocator_seams=True,
        require_kv_write_seams=True,
        require_kv_read_seams=True,
        log_path=run_dir / f"{mode_name}.log",
        completion_cases=config.coverage_cases,
    )


def completion_text(result: dict[str, Any]) -> str | None:
    choices = (result.get("completion") or {}).get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    text = choices[0].get("text")
    return text if isinstance(text, str) else None


def completion_text_from_payload(completion: dict[str, Any]) -> str | None:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    text = choices[0].get("text")
    return text if isinstance(text, str) else None


def finish_reason(result: dict[str, Any]) -> str | None:
    choices = (result.get("completion") or {}).get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None


def finish_reason_from_payload(completion: dict[str, Any]) -> str | None:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None


def usage_value(result: dict[str, Any], key: str) -> int | None:
    usage = (result.get("completion") or {}).get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) else None


def usage_value_from_payload(completion: dict[str, Any], key: str) -> int | None:
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) else None


def summarize_completion_cases(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_cases = result.get("completion_cases")
    if not isinstance(raw_cases, list):
        return []
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        completion = raw_case.get("completion")
        if not isinstance(completion, dict):
            completion = {}
        cases.append(
            {
                "name": raw_case.get("name"),
                "prompt": raw_case.get("prompt"),
                "max_tokens": raw_case.get("max_tokens"),
                "completion_seconds": raw_case.get("completion_seconds"),
                "completion_text": completion_text_from_payload(completion),
                "finish_reason": finish_reason_from_payload(completion),
                "completion_tokens": usage_value_from_payload(
                    completion,
                    "completion_tokens",
                ),
                "total_tokens": usage_value_from_payload(completion, "total_tokens"),
            }
        )
    return cases


def sum_case_usage(result: dict[str, Any], key: str) -> int | None:
    cases = summarize_completion_cases(result)
    values = [case.get(key) for case in cases]
    if not values or not all(isinstance(value, int) for value in values):
        return usage_value(result, key)
    return sum(values)


def token_throughput(result: dict[str, Any]) -> float | None:
    generated_tokens = sum_case_usage(result, "completion_tokens")
    latency = result.get("completion_seconds")
    if not isinstance(generated_tokens, int) or not isinstance(latency, (int, float)):
        return None
    if latency <= 0:
        return None
    return round(generated_tokens / latency, 3)


def summarize_gpu_read_contract(result: dict[str, Any]) -> dict[str, Any]:
    read_report = result.get("kv_read_offset_table_report") or {}
    write_report = result.get("kv_write_replace_candidate_report") or {}
    backed_report = result.get("backed_allocator") or {}
    backed_pool_stats = backed_report.get("pool_stats") or {}
    return {
        "read_path": read_report.get("read_path"),
        "replacement_backend": read_report.get("replacement_backend"),
        "gpu_kernel_calls": read_report.get("gpu_kernel_calls"),
        "stream_aware_kernel_calls": read_report.get("stream_aware_kernel_calls"),
        "reference_read_bytes": read_report.get("reference_read_bytes"),
        "replacement_calls": read_report.get("replacement_calls"),
        "offset_table_builds": read_report.get("offset_table_builds"),
        "offset_table_cache_hits": read_report.get("offset_table_cache_hits"),
        "offset_table_cache_rebuilds": read_report.get(
            "offset_table_cache_rebuilds"
        ),
        "read_min_entries_total_blocks_calls": read_report.get(
            "min_entries_total_blocks_calls"
        ),
        "read_block_table_validation_enabled": read_report.get(
            "block_table_validation_enabled"
        ),
        "read_fast_current_context_launch": read_report.get(
            "fast_current_context_launch"
        ),
        "read_gpu_kernel_precompile_requested": read_report.get(
            "gpu_kernel_precompile_requested"
        ),
        "read_gpu_kernel_precompile_succeeded": read_report.get(
            "gpu_kernel_precompile_succeeded"
        ),
        "read_gpu_kernel_precompile_calls": read_report.get(
            "gpu_kernel_precompile_calls"
        ),
        "read_gpu_kernel_precompile_elapsed_ms": read_report.get(
            "gpu_kernel_precompile_elapsed_ms"
        ),
        "read_report_on_update": read_report.get("report_on_update"),
        "read_report_write_count": read_report.get("report_write_count"),
        "read_host_profile": read_report.get("host_profile"),
        "native_write_skipped_calls": write_report.get("native_skipped_calls"),
        "write_verification_enabled": write_report.get(
            "write_verification_enabled"
        ),
        "write_verify_rows_per_call": write_report.get("verify_rows_per_call"),
        "write_report_on_update": write_report.get("report_on_update"),
        "write_report_write_count": write_report.get("report_write_count"),
        "write_host_profile": write_report.get("host_profile"),
        "kcmm_write_verified_rows": write_report.get("verified_rows"),
        "write_device_slot_enabled": write_report.get("device_slot_write_enabled"),
        "write_device_slot_active": write_report.get("device_slot_write_active"),
        "write_device_slot_calls": write_report.get("device_slot_write_calls"),
        "write_host_slot_calls": write_report.get("host_slot_write_calls"),
        "write_device_slot_status_checks": write_report.get(
            "device_slot_status_checks"
        ),
        "write_device_slot_status_error_count": write_report.get(
            "device_slot_status_error_count"
        ),
        "write_device_slot_status_codes": write_report.get(
            "device_slot_status_codes"
        ),
        "write_device_slot_kernel_precompile_requested": write_report.get(
            "device_slot_kernel_precompile_requested"
        ),
        "write_device_slot_kernel_precompile_succeeded": write_report.get(
            "device_slot_kernel_precompile_succeeded"
        ),
        "write_device_slot_kernel_precompile_calls": write_report.get(
            "device_slot_kernel_precompile_calls"
        ),
        "write_device_slot_kernel_precompile_elapsed_ms": write_report.get(
            "device_slot_kernel_precompile_elapsed_ms"
        ),
        "write_device_slot_offset_table_cache_hits": write_report.get(
            "device_slot_offset_table_cache_hits"
        ),
        "write_device_slot_offset_table_cache_rebuilds": write_report.get(
            "device_slot_offset_table_cache_rebuilds"
        ),
        "write_device_slot_valid_table_cache_hits": write_report.get(
            "device_slot_valid_table_cache_hits"
        ),
        "write_device_slot_valid_table_cache_rebuilds": write_report.get(
            "device_slot_valid_table_cache_rebuilds"
        ),
        "stream_aware_write_calls": write_report.get("stream_aware_write_calls"),
        "write_pool_shape_cached": write_report.get("pool_shape_cached"),
        "write_pool_shape_refreshes": write_report.get("pool_shape_refreshes"),
        "write_pool_block_size": write_report.get("pool_block_size"),
        "write_pool_block_bytes": write_report.get("pool_block_bytes"),
        "write_pool_step_elements": write_report.get("pool_step_elements"),
        "write_pool_num_layers": write_report.get("pool_num_layers"),
        "write_known_slot_blocks": write_report.get("known_slot_blocks"),
        "write_slot_block_ensure_cache_hits": write_report.get(
            "slot_block_ensure_cache_hits"
        ),
        "write_slot_block_ensure_cache_misses": write_report.get(
            "slot_block_ensure_cache_misses"
        ),
        "force_non_default_stream": read_report.get("force_non_default_stream"),
        "read_forced_non_default_stream_calls": read_report.get(
            "forced_non_default_stream_calls"
        ),
        "read_last_stream_ptr": read_report.get("last_stream_ptr"),
        "read_last_original_stream_ptr": read_report.get("last_original_stream_ptr"),
        "read_last_default_stream_ptr": read_report.get("last_default_stream_ptr"),
        "gpu_kernel_profile": read_report.get("gpu_kernel_profile"),
        "write_forced_non_default_stream_calls": write_report.get(
            "forced_non_default_stream_calls"
        ),
        "write_last_stream_ptr": write_report.get("last_stream_ptr"),
        "write_last_original_stream_ptr": write_report.get("last_original_stream_ptr"),
        "write_last_default_stream_ptr": write_report.get("last_default_stream_ptr"),
        "max_read_batch_seen": read_report.get("max_batch_seen"),
        "max_write_batch_seen": write_report.get("max_batch_seen"),
        "write_stream_synchronize_for_verification_calls": write_report.get(
            "stream_synchronize_for_verification_calls"
        ),
        "storage_of_record": write_report.get("storage_of_record"),
        "blocks_in_use_after_shutdown": backed_pool_stats.get("blocks_in_use"),
    }


def summarize_success(mode_name: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "success": True,
        "mode": mode_name,
        "server_mode": result.get("mode"),
        "startup_seconds": result.get("startup_seconds"),
        "request_latency_seconds": result.get("completion_seconds"),
        "tokens_per_second": token_throughput(result),
        "completion_text": completion_text(result),
        "finish_reason": finish_reason(result),
        "completion_tokens": sum_case_usage(result, "completion_tokens"),
        "total_tokens": sum_case_usage(result, "total_tokens"),
        "completion_cases": summarize_completion_cases(result),
        "gpu_memory": result.get("gpu_memory"),
        "generated_model": result.get("generated_model"),
        "instrument_kv_reads": result.get("instrument_kv_reads"),
        "kv_write_verify": result.get("kv_write_verify"),
        "kv_write_device_slots": result.get("kv_write_device_slots"),
        "kv_read_fast_current_context_launch": result.get(
            "kv_read_fast_current_context_launch"
        ),
        "kv_read_precompile_gpu_kernel": result.get(
            "kv_read_precompile_gpu_kernel"
        ),
        "tracker_report_on_update": result.get("tracker_report_on_update"),
        "tracker_host_profile": result.get("tracker_host_profile"),
        "log_path": result.get("log_path"),
    }
    if mode_name == "kcmm_gpu_read":
        summary["kcmm_gpu_read_contract"] = summarize_gpu_read_contract(result)
    return summary


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
        ("backed_report_path", smoke_config.backed_report_path),
        ("kv_write_mirror_report_path", smoke_config.kv_write_mirror_report_path),
        (
            "kv_read_offset_table_report_path",
            smoke_config.kv_read_offset_table_report_path,
        ),
        ("kv_read_trace_path", smoke_config.kv_read_trace_path),
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


def add_correctness_failures(modes: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    stock = modes.get("stock", {})
    gpu_read = modes.get("kcmm_gpu_read", {})
    if not stock.get("success"):
        failures.append(
            {
                "mode": "stock",
                "reason": "stock_failed",
                "detail": stock.get("error", "stock vLLM did not complete"),
            }
        )
        return failures
    if not gpu_read.get("success"):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "kcmm_gpu_read_failed_after_stock_passed",
                "detail": gpu_read.get(
                    "error",
                    "KCMM GPU read-kernel mode did not complete",
                ),
            }
        )
        return failures

    for key in ("completion_text", "finish_reason", "completion_tokens", "total_tokens"):
        if stock.get(key) != gpu_read.get(key):
            failures.append(
                {
                    "mode": "kcmm_gpu_read",
                    "reason": f"{key}_mismatch",
                    "stock_value": stock.get(key),
                    "kcmm_value": gpu_read.get(key),
                }
            )
    stock_cases = {
        case.get("name"): case
        for case in stock.get("completion_cases", [])
        if isinstance(case, dict)
    }
    gpu_cases = {
        case.get("name"): case
        for case in gpu_read.get("completion_cases", [])
        if isinstance(case, dict)
    }
    if set(stock_cases) != set(gpu_cases):
        failures.append(
            {
                "mode": "kcmm_gpu_read",
                "reason": "coverage_case_set_mismatch",
                "stock_cases": sorted(str(key) for key in stock_cases),
                "kcmm_cases": sorted(str(key) for key in gpu_cases),
            }
        )
        return failures
    comparison_keys = (
        "completion_text",
        "finish_reason",
        "completion_tokens",
        "total_tokens",
    )
    for case_name in sorted(stock_cases):
        stock_case = stock_cases[case_name]
        gpu_case = gpu_cases[case_name]
        for key in comparison_keys:
            if stock_case.get(key) != gpu_case.get(key):
                failures.append(
                    {
                        "mode": "kcmm_gpu_read",
                        "reason": f"coverage_case_{key}_mismatch",
                        "case": case_name,
                        "stock_value": stock_case.get(key),
                        "kcmm_value": gpu_case.get(key),
                    }
                )
    return failures


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def nested_number(value: Any, *keys: str) -> float | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return number(current)


def performance_comparison(modes: dict[str, Any]) -> dict[str, Any]:
    stock = modes.get("stock", {})
    gpu_read = modes.get("kcmm_gpu_read", {})
    if not stock.get("success") or not gpu_read.get("success"):
        return {"available": False, "reason": "both modes must pass first"}

    metrics = [
        (
            "startup_seconds",
            number(stock.get("startup_seconds")),
            number(gpu_read.get("startup_seconds")),
            "lower_is_better",
        ),
        (
            "request_latency_seconds",
            number(stock.get("request_latency_seconds")),
            number(gpu_read.get("request_latency_seconds")),
            "lower_is_better",
        ),
        (
            "tokens_per_second",
            number(stock.get("tokens_per_second")),
            number(gpu_read.get("tokens_per_second")),
            "higher_is_better",
        ),
        (
            "gpu_memory_peak_delta_mib",
            nested_number(stock, "gpu_memory", "peak_delta_mib"),
            nested_number(gpu_read, "gpu_memory", "peak_delta_mib"),
            "lower_is_better",
        ),
    ]
    return {
        "available": True,
        "metrics": {
            name: {
                "stock": stock_value,
                "kcmm_gpu_read": kcmm_value,
                "kcmm_to_stock_ratio": ratio(kcmm_value, stock_value),
                "direction": direction,
            }
            for name, stock_value, kcmm_value, direction in metrics
        },
    }


def add_ratio_warning(
    warnings: list[dict[str, Any]],
    *,
    metric: str,
    stock_value: float | None,
    mode_value: float | None,
    ratio_threshold: float,
    higher_is_worse: bool,
) -> None:
    if stock_value is None or mode_value is None:
        return
    if stock_value <= 0:
        return
    threshold = stock_value * ratio_threshold
    if higher_is_worse and mode_value > threshold:
        warnings.append(
            {
                "mode": "kcmm_gpu_read",
                "metric": metric,
                "stock_value": round(stock_value, 3),
                "mode_value": round(mode_value, 3),
                "threshold": round(threshold, 3),
                "classification": "performance_warning",
            }
        )
    if not higher_is_worse and mode_value < threshold:
        warnings.append(
            {
                "mode": "kcmm_gpu_read",
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
                "mode": "kcmm_gpu_read",
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
    gpu_read = modes.get("kcmm_gpu_read", {})
    if not stock.get("success") or not gpu_read.get("success"):
        return warnings

    add_ratio_warning(
        warnings,
        metric="startup_seconds",
        stock_value=number(stock.get("startup_seconds")),
        mode_value=number(gpu_read.get("startup_seconds")),
        ratio_threshold=config.latency_warning_ratio,
        higher_is_worse=True,
    )
    add_ratio_warning(
        warnings,
        metric="request_latency_seconds",
        stock_value=number(stock.get("request_latency_seconds")),
        mode_value=number(gpu_read.get("request_latency_seconds")),
        ratio_threshold=config.latency_warning_ratio,
        higher_is_worse=True,
    )
    add_ratio_warning(
        warnings,
        metric="tokens_per_second",
        stock_value=number(stock.get("tokens_per_second")),
        mode_value=number(gpu_read.get("tokens_per_second")),
        ratio_threshold=config.throughput_warning_ratio,
        higher_is_worse=False,
    )
    add_memory_warning(
        warnings,
        config=config,
        stock_delta=nested_number(stock, "gpu_memory", "peak_delta_mib"),
        mode_delta=nested_number(gpu_read, "gpu_memory", "peak_delta_mib"),
    )
    return warnings


def run_gate(config: GateConfig) -> dict[str, Any]:
    run_id = int(time.time() * 1000)
    run_dir = Path(tempfile.gettempdir()) / f"kcmm-vllm-phase-ii-c-gpu-read-ab-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    created_model_dir = not config.model_path.exists()
    model_existed = (
        tiny_model_exists(config.model_path)
        if config.generate_tiny_model
        else config.model_path.exists()
    )
    modes: dict[str, Any] = {}
    try:
        for mode_name in MODE_ORDER:
            print(f"run GPU read A/B mode: {mode_name}", flush=True)
            modes[mode_name] = run_mode(config, mode_name, run_dir)
    finally:
        if config.generate_tiny_model and created_model_dir and not config.keep_model:
            shutil.rmtree(config.model_path, ignore_errors=True)

    correctness_failures = add_correctness_failures(modes)
    performance_warnings = add_performance_warnings(config, modes)
    report = {
        "phase": "II.C",
        "gate": "stock-vs-kcmm-gpu-read-kernel-ab",
        "passed": not correctness_failures,
        "started_at_unix_ms": run_id,
        "repo_root": str(repo_root()),
        "run_dir": str(run_dir),
        "model_path": str(config.model_path),
        "model_name": config.model_name,
        "generate_tiny_model": config.generate_tiny_model,
        "model_existed_before_gate": model_existed,
        "prompt": config.prompt,
        "max_tokens": config.max_tokens,
        "coverage_cases": [
            {
                "name": case.name,
                "prompt": case.prompt,
                "max_tokens": case.max_tokens,
            }
            for case in config.coverage_cases
        ],
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
        "completion_concurrency": config.completion_concurrency,
        "kv_force_non_default_stream": config.kv_force_non_default_stream,
        "kv_read_profile": config.kv_read_profile,
        "kv_read_validate_block_tables": config.kv_read_validate_block_tables,
        "instrument_kv_reads": config.instrument_kv_reads,
        "kv_write_verify": config.kv_write_verify,
        "kv_write_device_slots": config.kv_write_device_slots,
        "kv_read_fast_current_context_launch": (
            config.kv_read_fast_current_context_launch
        ),
        "kv_read_precompile_gpu_kernel": config.kv_read_precompile_gpu_kernel,
        "tracker_report_on_update": config.tracker_report_on_update,
        "tracker_host_profile": config.tracker_host_profile,
        "mode_order": list(MODE_ORDER),
        "modes": modes,
        "correctness_failures": correctness_failures,
        "performance_comparison": performance_comparison(modes),
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
