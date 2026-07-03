"""Self-terminating KCMM/vLLM server smoke test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = ".scratch/kcmm-vllm/tiny-opt-head64"
DEFAULT_MODEL_NAME = "tiny-opt-kcmm"
DEFAULT_KCMM_LIB_PATH = "target/debug/libbaseline_llm_os.so"
_DIRECT_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SmokeFailure(RuntimeError):
    """Raised when the smoke test cannot complete successfully."""


@dataclass(frozen=True)
class CompletionCase:
    name: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True)
class SmokeConfig:
    mode: str
    host: str
    port: int
    model_path: Path
    model_name: str
    kcmm_lib_path: Path
    timeout_seconds: float
    shutdown_timeout_seconds: float
    prompt: str
    max_tokens: int
    build_kcmm: bool
    keep_model: bool
    generate_tiny_model: bool
    print_seams: bool
    instrument_allocators: bool
    instrument_kv_writes: bool
    instrument_kv_reads: bool
    kv_read_offset_table: bool
    kv_read_replace_candidate: bool
    kv_read_gpu_kernel_candidate: bool
    kv_read_profile: bool
    kv_read_validate_block_tables: bool
    kv_read_fast_current_context_launch: bool
    kv_read_precompile_gpu_kernel: bool
    tracker_report_on_update: bool
    tracker_host_profile: bool
    kv_write_mirror: bool
    kv_write_replace_candidate: bool
    kv_write_verify: bool
    kv_write_device_slots: bool
    kv_force_non_default_stream: bool
    runtime_derived_pool: bool
    shadow_allocations: bool
    backed_allocations: bool
    allocator_trace_path: Path
    kv_write_trace_path: Path
    kv_read_trace_path: Path
    kv_read_offset_table_report_path: Path
    kv_write_mirror_report_path: Path
    shadow_report_path: Path
    backed_report_path: Path
    require_allocator_seams: bool
    require_kv_write_seams: bool
    require_kv_read_seams: bool
    log_path: Path
    max_model_len: int = 64
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 64
    gpu_memory_utilization: float = 0.25
    tensor_parallel_size: int = 1
    completion_concurrency: int = 1
    completion_cases: tuple[CompletionCase, ...] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root() / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("kcmm", "stock"),
        default="kcmm",
        help="kcmm runs the observer launcher; stock passes --kcmm-skip-observer.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--completion-concurrency",
        type=int,
        default=1,
        help="Number of completion requests to keep in flight within one smoke run.",
    )
    parser.add_argument(
        "--build-kcmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the KCMM shared library when running in kcmm mode.",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Keep the generated tiny model after the smoke run.",
    )
    parser.add_argument(
        "--generate-tiny-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate the local tiny OPT model when --model-path is missing. "
            "Disable this for externally supplied real model directories."
        ),
    )
    parser.add_argument(
        "--print-seams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the KCMM launcher to print vLLM seam inspection output.",
    )
    parser.add_argument(
        "--instrument-allocators",
        action="store_true",
        help="Enable observer-only vLLM V2 allocator seam instrumentation.",
    )
    parser.add_argument(
        "--runtime-derived-pool",
        action="store_true",
        help="Size the KCMM pool from vLLM runtime cache/model configuration.",
    )
    parser.add_argument(
        "--instrument-kv-writes",
        action="store_true",
        help="Enable observer-only reshape_and_cache KV write instrumentation.",
    )
    parser.add_argument(
        "--instrument-kv-reads",
        action="store_true",
        help="Enable observer-only paged_attention KV read instrumentation.",
    )
    parser.add_argument(
        "--kv-read-offset-table",
        action="store_true",
        help=(
            "Build the Phase II.C A2 KCMM block_id->offset table at "
            "paged_attention read seams without replacing the native kernel."
        ),
    )
    parser.add_argument(
        "--kv-read-replace-candidate",
        action="store_true",
        help=(
            "Skip native paged_attention and fill output with a KCMM-backed "
            "reference attention implementation."
        ),
    )
    parser.add_argument(
        "--kv-read-gpu-kernel-candidate",
        action="store_true",
        help=(
            "Skip native paged_attention and fill output with KCMM's GPU "
            "paged-attention kernel."
        ),
    )
    parser.add_argument(
        "--kv-read-profile",
        action="store_true",
        help=(
            "Record per-call CUDA event timing for the KCMM GPU read kernel. "
            "Requires --kv-read-gpu-kernel-candidate."
        ),
    )
    parser.add_argument(
        "--tracker-report-on-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write KCMM tracker reports after every observed seam call. "
            "Disable for performance-clean gates that only need final reports."
        ),
    )
    parser.add_argument(
        "--kv-read-validate-block-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate sampled paged-attention block_tables on the host. "
            "Disable for performance-clean gates after correctness coverage passes."
        ),
    )
    parser.add_argument(
        "--kv-read-fast-current-context-launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the KCMM read launch ABI that assumes PyTorch/vLLM already "
            "made the correct CUDA context current."
        ),
    )
    parser.add_argument(
        "--kv-read-precompile-gpu-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Precompile/load the KCMM paged-attention read kernel before "
            "serving the measured request."
        ),
    )
    parser.add_argument(
        "--tracker-host-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect section-level host timings in KCMM tracker final reports.",
    )
    parser.add_argument(
        "--kv-write-mirror",
        action="store_true",
        help="Mirror reshape_and_cache writes into KCMM after native vLLM writes.",
    )
    parser.add_argument(
        "--kv-write-replace-candidate",
        action="store_true",
        help=(
            "Skip native reshape_and_cache writes and write only to KCMM. "
            "This validates the Phase II.B write candidate only; Phase II.C "
            "read replacement is still required for correctness."
        ),
    )
    parser.add_argument(
        "--kv-write-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable bounded D2H verification of KCMM KV writes. Disable this "
            "for performance-clean gates after correctness has been proven."
        ),
    )
    parser.add_argument(
        "--kv-write-device-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the device-resident vLLM slot_mapping tensor for KCMM KV "
            "writes. Requires --no-kv-write-verify."
        ),
    )
    parser.add_argument(
        "--kv-force-non-default-stream",
        action="store_true",
        help=(
            "Route KCMM KV write/read replacement launches through a dedicated "
            "non-default CUDA stream with explicit stream waits."
        ),
    )
    parser.add_argument(
        "--shadow-allocations",
        action="store_true",
        help="Mirror vLLM GPU block allocations into KCMM shadow allocator.",
    )
    parser.add_argument(
        "--backed-allocations",
        action="store_true",
        help="Let KCMM choose vLLM GPU block IDs behind an opt-in flag.",
    )
    parser.add_argument(
        "--allocator-trace-path",
        default=None,
        help="Allocator instrumentation JSONL trace path. Defaults to a /tmp file.",
    )
    parser.add_argument(
        "--require-allocator-seams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when instrumentation does not observe required allocator seams.",
    )
    parser.add_argument(
        "--kv-write-trace-path",
        default=None,
        help="KV write instrumentation JSONL trace path. Defaults to a /tmp file.",
    )
    parser.add_argument(
        "--kv-read-trace-path",
        default=None,
        help="KV read instrumentation JSONL trace path. Defaults to a /tmp file.",
    )
    parser.add_argument(
        "--kv-read-offset-table-report-path",
        default=None,
        help="KCMM KV read offset-table JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument(
        "--kv-write-mirror-report-path",
        default=None,
        help="KCMM KV write mirror JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument(
        "--require-kv-write-seams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when KV write instrumentation does not observe reshape_and_cache.",
    )
    parser.add_argument(
        "--require-kv-read-seams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when KV read instrumentation does not observe paged_attention.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Combined vLLM stdout/stderr log path. Defaults to a /tmp file.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> SmokeConfig:
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
    log_path = (
        Path(args.log_path)
        if args.log_path
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-smoke-{int(time.time() * 1000)}.log"
    )
    allocator_trace_path = (
        Path(args.allocator_trace_path)
        if args.allocator_trace_path
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-allocator-trace-{int(time.time() * 1000)}.jsonl"
    )
    kv_write_trace_path = (
        Path(args.kv_write_trace_path)
        if args.kv_write_trace_path
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-kv-write-trace-{int(time.time() * 1000)}.jsonl"
    )
    kv_read_trace_path = (
        Path(args.kv_read_trace_path)
        if args.kv_read_trace_path
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-kv-read-trace-{int(time.time() * 1000)}.jsonl"
    )
    kv_read_offset_table_report_path = (
        Path(args.kv_read_offset_table_report_path)
        if args.kv_read_offset_table_report_path
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-kv-read-offset-table-{int(time.time() * 1000)}.json"
    )
    kv_write_mirror_report_path = (
        Path(args.kv_write_mirror_report_path)
        if args.kv_write_mirror_report_path
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-kv-write-mirror-{int(time.time() * 1000)}.json"
    )
    shadow_report_path = (
        Path(tempfile.gettempdir())
        / f"kcmm-vllm-shadow-report-{int(time.time() * 1000)}.json"
    )
    backed_report_path = (
        Path(tempfile.gettempdir())
        / f"kcmm-vllm-backed-report-{int(time.time() * 1000)}.json"
    )
    return SmokeConfig(
        mode=args.mode,
        host=args.host,
        port=args.port,
        model_path=resolve_repo_path(args.model_path),
        model_name=args.model_name,
        kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
        timeout_seconds=args.timeout_seconds,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        build_kcmm=args.build_kcmm,
        keep_model=args.keep_model,
        generate_tiny_model=args.generate_tiny_model,
        print_seams=args.print_seams,
        instrument_allocators=args.instrument_allocators,
        instrument_kv_writes=args.instrument_kv_writes,
        instrument_kv_reads=args.instrument_kv_reads,
        kv_read_offset_table=args.kv_read_offset_table,
        kv_read_replace_candidate=args.kv_read_replace_candidate,
        kv_read_gpu_kernel_candidate=args.kv_read_gpu_kernel_candidate,
        kv_read_profile=args.kv_read_profile,
        kv_read_validate_block_tables=args.kv_read_validate_block_tables,
        kv_read_fast_current_context_launch=(
            args.kv_read_fast_current_context_launch
        ),
        kv_read_precompile_gpu_kernel=args.kv_read_precompile_gpu_kernel,
        tracker_report_on_update=args.tracker_report_on_update,
        tracker_host_profile=args.tracker_host_profile,
        kv_write_mirror=args.kv_write_mirror,
        kv_write_replace_candidate=args.kv_write_replace_candidate,
        kv_write_verify=args.kv_write_verify,
        kv_write_device_slots=args.kv_write_device_slots,
        kv_force_non_default_stream=args.kv_force_non_default_stream,
        runtime_derived_pool=(
            args.runtime_derived_pool
            or args.shadow_allocations
            or args.backed_allocations
            or args.kv_write_mirror
            or args.kv_write_replace_candidate
            or args.kv_read_offset_table
            or args.kv_read_replace_candidate
            or args.kv_read_gpu_kernel_candidate
            or args.kv_read_profile
            or args.kv_force_non_default_stream
        ),
        shadow_allocations=args.shadow_allocations,
        backed_allocations=args.backed_allocations,
        allocator_trace_path=allocator_trace_path,
        kv_write_trace_path=kv_write_trace_path,
        kv_read_trace_path=kv_read_trace_path,
        kv_read_offset_table_report_path=kv_read_offset_table_report_path,
        kv_write_mirror_report_path=kv_write_mirror_report_path,
        shadow_report_path=shadow_report_path,
        backed_report_path=backed_report_path,
        require_allocator_seams=args.require_allocator_seams,
        require_kv_write_seams=args.require_kv_write_seams,
        require_kv_read_seams=args.require_kv_read_seams,
        log_path=log_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        completion_concurrency=args.completion_concurrency,
    )


def tail_file(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port_closed(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not port_is_open(host, port):
            return True
        time.sleep(0.25)
    return not port_is_open(host, port)


def live_process_group_members(pgid: int) -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat=,cmd="],
        check=False,
        capture_output=True,
        text=True,
    )
    members: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        pid_text, pgid_text, stat = parts[:3]
        cmd = parts[3] if len(parts) == 4 else ""
        try:
            member_pgid = int(pgid_text)
        except ValueError:
            continue
        if member_pgid == pgid and "Z" not in stat:
            members.append(f"{pid_text} {pgid_text} {stat} {cmd}".rstrip())
    return members


def wait_process_exit(process: subprocess.Popen[None], timeout_seconds: float) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=max(timeout_seconds, 0.1))
        return True
    except subprocess.TimeoutExpired:
        return process.poll() is not None


def gpu_memory_used_mib() -> list[int] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(int(text.split()[0]))
        except (IndexError, ValueError):
            return None
    return values or None


class GpuMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.before: list[int] | None = None
        self.after: list[int] | None = None
        self.samples: list[list[int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.before = gpu_memory_used_mib()
        if self.before is None:
            return
        self.samples.append(self.before)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        self.after = gpu_memory_used_mib()
        if self.after is not None:
            self.samples.append(self.after)

        if not self.samples:
            return {
                "available": False,
                "reason": "nvidia-smi unavailable or unreadable",
            }

        totals = [sum(sample) for sample in self.samples]
        peak_index = max(range(len(self.samples)), key=lambda index: totals[index])
        before_total = sum(self.before) if self.before is not None else None
        after_total = sum(self.after) if self.after is not None else None
        peak_total = totals[peak_index]
        return {
            "available": True,
            "unit": "MiB",
            "sample_count": len(self.samples),
            "before_per_gpu_mib": self.before,
            "after_per_gpu_mib": self.after,
            "peak_per_gpu_mib": self.samples[peak_index],
            "before_total_mib": before_total,
            "after_total_mib": after_total,
            "peak_total_mib": peak_total,
            "peak_delta_mib": (
                peak_total - before_total if before_total is not None else None
            ),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            sample = gpu_memory_used_mib()
            if sample is not None:
                self.samples.append(sample)


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 2.0,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _DIRECT_HTTP_OPENER.open(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"body": body}
        return exc.code, parsed


def run_checked(command: list[str], description: str) -> None:
    print(f"{description}: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=repo_root(), check=True)
    except subprocess.CalledProcessError as exc:
        raise SmokeFailure(f"{description} failed with exit code {exc.returncode}") from exc


def ensure_kcmm_library(config: SmokeConfig) -> None:
    if config.mode != "kcmm":
        return
    if config.build_kcmm or not config.kcmm_lib_path.exists():
        cargo = shutil.which("cargo")
        if cargo is None:
            raise SmokeFailure("cargo not found; cannot build KCMM shared library")
        run_checked([cargo, "build", "--features", "kcmm"], "build KCMM")
    if not config.kcmm_lib_path.exists():
        raise SmokeFailure(f"KCMM shared library not found: {config.kcmm_lib_path}")


def ensure_tiny_model(model_path: Path) -> bool:
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    if all((model_path / name).exists() for name in required):
        return False

    script = repo_root() / "scripts" / "kcmm" / "create_tiny_opt_model.py"
    run_checked(
        [sys.executable, str(script), "--output", str(model_path)],
        "create tiny OPT model",
    )
    return True


def ensure_model(config: SmokeConfig) -> bool:
    if config.generate_tiny_model:
        return ensure_tiny_model(config.model_path)
    if not config.model_path.exists():
        raise SmokeFailure(
            "model path does not exist and tiny model generation is disabled: "
            f"{config.model_path}"
        )
    return False


def vllm_command(config: SmokeConfig) -> list[str]:
    command = [sys.executable, "-m", "scripts.kcmm"]
    if config.mode == "stock":
        command.append("--kcmm-skip-observer")
    else:
        command.extend(["--kcmm-lib-path", str(config.kcmm_lib_path)])
        if config.runtime_derived_pool:
            command.extend(["--kcmm-pool-mode", "runtime"])
        if config.shadow_allocations:
            command.extend(
                [
                    "--kcmm-shadow-allocations",
                    "--kcmm-shadow-report-path",
                    str(config.shadow_report_path),
                ]
            )
        if config.backed_allocations:
            command.extend(
                [
                    "--kcmm-backed-allocations",
                    "--kcmm-backed-report-path",
                    str(config.backed_report_path),
                ]
            )
        if config.kv_write_mirror or config.kv_write_replace_candidate:
            flag = (
                "--kcmm-kv-write-replace-candidate"
                if config.kv_write_replace_candidate
                else "--kcmm-kv-write-mirror"
            )
            command.extend(
                [
                    flag,
                    "--kcmm-kv-write-mirror-report-path",
                    str(config.kv_write_mirror_report_path),
                ]
            )
            if not config.kv_write_verify:
                command.append("--no-kcmm-kv-write-verify")
            if config.kv_write_device_slots:
                command.append("--kcmm-kv-write-device-slots")
        if config.kv_force_non_default_stream:
            command.append("--kcmm-kv-force-non-default-stream")
        if (
            config.kv_read_offset_table
            or config.kv_read_replace_candidate
            or config.kv_read_gpu_kernel_candidate
        ):
            if config.kv_read_gpu_kernel_candidate:
                mode_flag = "--kcmm-kv-read-gpu-kernel-candidate"
            elif config.kv_read_replace_candidate:
                mode_flag = "--kcmm-kv-read-replace-candidate"
            else:
                mode_flag = "--kcmm-kv-read-offset-table"
            command.extend(
                [
                    mode_flag,
                    "--kcmm-kv-read-offset-table-report-path",
                    str(config.kv_read_offset_table_report_path),
                ]
            )
            if config.kv_read_profile:
                command.append("--kcmm-kv-read-profile")
            if not config.kv_read_validate_block_tables:
                command.append("--no-kcmm-kv-read-validate-block-tables")
            if config.kv_read_fast_current_context_launch:
                command.append("--kcmm-kv-read-fast-current-context-launch")
            if config.kv_read_precompile_gpu_kernel:
                command.append("--kcmm-kv-read-precompile-gpu-kernel")
        if (
            not config.tracker_report_on_update
            and (
                config.kv_write_mirror
                or config.kv_write_replace_candidate
                or config.kv_read_offset_table
                or config.kv_read_replace_candidate
                or config.kv_read_gpu_kernel_candidate
            )
        ):
            command.append("--no-kcmm-tracker-report-on-update")
        if (
            config.tracker_host_profile
            and (
                config.kv_write_mirror
                or config.kv_write_replace_candidate
                or config.kv_read_offset_table
                or config.kv_read_replace_candidate
                or config.kv_read_gpu_kernel_candidate
            )
        ):
            command.append("--kcmm-tracker-host-profile")
    if config.instrument_allocators:
        command.extend(
            [
                "--kcmm-instrument-allocators",
                "--kcmm-allocator-trace-path",
                str(config.allocator_trace_path),
            ]
        )
        if config.require_allocator_seams:
            command.append("--kcmm-require-allocator-seams")
    if config.instrument_kv_writes:
        command.extend(
            [
                "--kcmm-instrument-kv-writes",
                "--kcmm-kv-write-trace-path",
                str(config.kv_write_trace_path),
            ]
        )
        if config.require_kv_write_seams:
            command.append("--kcmm-require-kv-write-seams")
    if config.instrument_kv_reads:
        command.extend(
            [
                "--kcmm-instrument-kv-reads",
                "--kcmm-kv-read-trace-path",
                str(config.kv_read_trace_path),
            ]
        )
        if config.require_kv_read_seams:
            command.append("--kcmm-require-kv-read-seams")
    if config.print_seams:
        command.append("--kcmm-print-seams")
    command.extend(
        [
            "serve",
            str(config.model_path),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--dtype",
            "float16",
            "--max-model-len",
            str(config.max_model_len),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--max-num-seqs",
            str(config.max_num_seqs),
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--enforce-eager",
            "--max-seq-len-to-capture",
            str(config.max_model_len),
            "--guided-decoding-backend",
            "lm-format-enforcer",
            "--disable-log-requests",
            "--served-model-name",
            config.model_name,
            "--use-v2-block-manager",
        ]
    )
    if (
        config.instrument_allocators
        or config.instrument_kv_writes
        or config.instrument_kv_reads
        or config.kv_read_offset_table
        or config.kv_read_replace_candidate
        or config.kv_read_gpu_kernel_candidate
        or config.kv_write_mirror
        or config.kv_write_replace_candidate
        or config.runtime_derived_pool
        or config.shadow_allocations
        or config.backed_allocations
    ):
        # Keep vLLM's engine in this Python process so launcher monkey-patches
        # apply to the block manager and allocator objects being exercised.
        command.append("--disable-frontend-multiprocessing")
    return command


def read_allocator_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"allocator trace was not written: {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    summary = next(
        (event for event in reversed(events) if event.get("event") == "summary"),
        None,
    )
    if summary is None:
        raise SmokeFailure(f"allocator trace has no summary event: {path}")
    missing = summary.get("missing_required_groups") or {}
    if missing:
        raise SmokeFailure(
            "allocator instrumentation did not observe required seams: "
            + json.dumps(missing, sort_keys=True)
        )
    return {
        "path": str(path),
        "event_count": len(events),
        "counts": summary.get("counts", {}),
        "missing_required_groups": missing,
    }


def read_kv_write_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"KV write trace was not written: {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    summary = next(
        (event for event in reversed(events) if event.get("event") == "summary"),
        None,
    )
    if summary is None:
        raise SmokeFailure(f"KV write trace has no summary event: {path}")
    missing = summary.get("missing_required_groups") or {}
    if missing:
        raise SmokeFailure(
            "KV write instrumentation did not observe required seams: "
            + json.dumps(missing, sort_keys=True)
        )
    write_events = [event for event in events if event.get("event") == "kv_write_call"]
    if not write_events:
        raise SmokeFailure(f"KV write trace has no write events: {path}")
    invalid_contracts: list[dict[str, Any]] = []
    for event in write_events:
        contract = event.get("args", {}).get("slot_mapping_contract", {})
        if not contract.get("valid", False):
            invalid_contracts.append(
                {
                    "seq": event.get("seq"),
                    "key": event.get("key"),
                    "contract": contract,
                }
            )
    if invalid_contracts:
        raise SmokeFailure(
            "KV write slot_mapping contract validation failed: "
            + json.dumps(invalid_contracts, sort_keys=True)
        )
    return {
        "path": str(path),
        "event_count": len(events),
        "write_event_count": len(write_events),
        "counts": summary.get("counts", {}),
        "missing_required_groups": missing,
        "first_write": write_events[0],
        "first_slot_mapping_contract": write_events[0]
        .get("args", {})
        .get("slot_mapping_contract", {}),
    }


def read_kv_read_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"KV read trace was not written: {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    summary = next(
        (event for event in reversed(events) if event.get("event") == "summary"),
        None,
    )
    if summary is None:
        raise SmokeFailure(f"KV read trace has no summary event: {path}")
    missing = summary.get("missing_required_groups") or {}
    if missing:
        raise SmokeFailure(
            "KV read instrumentation did not observe required seams: "
            + json.dumps(missing, sort_keys=True)
        )
    read_events = [event for event in events if event.get("event") == "kv_read_call"]
    if not read_events:
        raise SmokeFailure(f"KV read trace has no read events: {path}")
    invalid_contracts: list[dict[str, Any]] = []
    for event in read_events:
        contract = event.get("args", {}).get("block_tables_contract", {})
        if not contract.get("valid", False):
            invalid_contracts.append(
                {
                    "seq": event.get("seq"),
                    "key": event.get("key"),
                    "contract": contract,
                }
            )
    if invalid_contracts:
        raise SmokeFailure(
            "KV read block_tables contract validation failed: "
            + json.dumps(invalid_contracts, sort_keys=True)
        )
    return {
        "path": str(path),
        "event_count": len(events),
        "read_event_count": len(read_events),
        "counts": summary.get("counts", {}),
        "missing_required_groups": missing,
        "first_read": read_events[0],
        "first_block_tables_contract": read_events[0]
        .get("args", {})
        .get("block_tables_contract", {}),
    }


def read_kv_write_mirror_report(
    path: Path,
    *,
    expect_verification: bool = True,
    expect_device_slots: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"KCMM KV write mirror report was not written: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("error_count", 0):
        raise SmokeFailure(
            "KCMM KV write report recorded errors: "
            + json.dumps(report, sort_keys=True)
        )
    if not report.get("pool_attached", False):
        raise SmokeFailure(
            "KCMM KV write mirror never attached to a pool: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("mirror_calls", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV write report did not record any KCMM write calls: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("mirrored_rows", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV write report did not write any rows: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("stream_aware_write_calls", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV write report used no stream-aware writes: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("device_slot_status_error_count", 0) != 0:
        raise SmokeFailure(
            "KCMM KV write device-slot report recorded status errors: "
            + json.dumps(report, sort_keys=True)
        )
    if expect_device_slots:
        if report.get("device_slot_write_enabled") is not True:
            raise SmokeFailure(
                "KCMM KV write report did not enable device-slot writes: "
                + json.dumps(report, sort_keys=True)
            )
        if report.get("device_slot_write_calls", 0) <= 0:
            raise SmokeFailure(
                "KCMM KV write report used no device-slot writes: "
                + json.dumps(report, sort_keys=True)
            )
        if report.get("host_slot_write_calls", 0) != 0:
            raise SmokeFailure(
                "KCMM KV write report used host-slot writes despite "
                "device-slot mode: "
                + json.dumps(report, sort_keys=True)
            )
    verification_enabled = bool(report.get("write_verification_enabled", True))
    if expect_verification and not verification_enabled:
        raise SmokeFailure(
            "KCMM KV write report unexpectedly disabled verification: "
            + json.dumps(report, sort_keys=True)
        )
    if expect_verification and report.get("verified_rows", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV write report did not verify any rows: "
            + json.dumps(report, sort_keys=True)
        )
    if not expect_verification and verification_enabled:
        raise SmokeFailure(
            "KCMM KV write report left verification enabled: "
            + json.dumps(report, sort_keys=True)
        )
    if not expect_verification and report.get("verified_rows", 0) != 0:
        raise SmokeFailure(
            "KCMM KV write report verified rows despite disabled verification: "
            + json.dumps(report, sort_keys=True)
        )
    if not expect_verification and report.get("verification_bytes", 0) != 0:
        raise SmokeFailure(
            "KCMM KV write report copied verification bytes despite disabled "
            "verification: "
            + json.dumps(report, sort_keys=True)
        )
    if (
        not expect_verification
        and report.get("stream_synchronize_for_verification_calls", 0) != 0
    ):
        raise SmokeFailure(
            "KCMM KV write report synchronized for verification despite disabled "
            "verification: "
            + json.dumps(report, sort_keys=True)
        )
    return {"path": str(path), **report}


def require_non_default_stream_report(
    report: dict[str, Any],
    *,
    name: str,
) -> None:
    if not report.get("force_non_default_stream", False):
        raise SmokeFailure(
            f"{name} report did not enable forced non-default stream mode: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("forced_non_default_stream_calls", 0) <= 0:
        raise SmokeFailure(
            f"{name} report recorded no forced non-default stream calls: "
            + json.dumps(report, sort_keys=True)
        )
    stream_ptr = report.get("last_stream_ptr")
    default_stream_ptr = report.get("last_default_stream_ptr")
    if not isinstance(stream_ptr, int) or stream_ptr == 0:
        raise SmokeFailure(
            f"{name} report did not record a non-zero stream pointer: "
            + json.dumps(report, sort_keys=True)
        )
    if isinstance(default_stream_ptr, int) and stream_ptr == default_stream_ptr:
        raise SmokeFailure(
            f"{name} report used the default stream despite forced mode: "
            + json.dumps(report, sort_keys=True)
        )


def read_kv_read_offset_table_report(
    path: Path,
    *,
    expect_replacement: bool,
    expect_gpu_kernel: bool = False,
    expect_profile: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(
            f"KCMM KV read offset-table report was not written: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("error_count", 0):
        raise SmokeFailure(
            "KCMM KV read offset-table report recorded errors: "
            + json.dumps(report, sort_keys=True)
        )
    if not report.get("pool_attached", False):
        raise SmokeFailure(
            "KCMM KV read offset-table planner never attached to a pool: "
            + json.dumps(report, sort_keys=True)
        )
    if bool(report.get("kernel_replaced", False)) != expect_replacement:
        raise SmokeFailure(
            "KCMM KV read report had unexpected kernel replacement state: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("read_calls", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV read offset-table planner saw no read calls: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("planned_calls", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV read offset-table planner built no read plans: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("offset_table_builds", 0) <= 0:
        raise SmokeFailure(
            "KCMM KV read offset-table planner built no offset tables: "
            + json.dumps(report, sort_keys=True)
        )
    if expect_replacement:
        if report.get("replacement_calls", 0) <= 0:
            raise SmokeFailure(
                "KCMM KV read replacement candidate replaced no calls: "
                + json.dumps(report, sort_keys=True)
            )
        if expect_gpu_kernel:
            if report.get("replacement_backend") != "gpu_kernel":
                raise SmokeFailure(
                    "KCMM KV read report did not use the GPU kernel backend: "
                    + json.dumps(report, sort_keys=True)
                )
            if report.get("read_path") != "kcmm_paged_attn_decode_f16":
                raise SmokeFailure(
                    "KCMM KV read report did not use the GPU kernel read path: "
                    + json.dumps(report, sort_keys=True)
                )
            if report.get("gpu_kernel_calls", 0) <= 0:
                raise SmokeFailure(
                    "KCMM KV read GPU kernel candidate launched no kernels: "
                    + json.dumps(report, sort_keys=True)
                )
            if report.get("stream_aware_kernel_calls", 0) <= 0:
                raise SmokeFailure(
                    "KCMM KV read GPU kernel candidate used no stream-aware launches: "
                    + json.dumps(report, sort_keys=True)
                )
            if report.get("reference_read_bytes", 0) != 0:
                raise SmokeFailure(
                    "KCMM KV read GPU kernel candidate used CPU-staged reads: "
                    + json.dumps(report, sort_keys=True)
                )
            profile = report.get("gpu_kernel_profile")
            if expect_profile:
                if not isinstance(profile, dict):
                    raise SmokeFailure(
                        "KCMM KV read profile summary was not written: "
                        + json.dumps(report, sort_keys=True)
                    )
                if not profile.get("enabled", False):
                    raise SmokeFailure(
                        "KCMM KV read profile summary was not enabled: "
                        + json.dumps(report, sort_keys=True)
                    )
                if profile.get("count", 0) <= 0:
                    raise SmokeFailure(
                        "KCMM KV read profile recorded no samples: "
                        + json.dumps(report, sort_keys=True)
                    )
                if not isinstance(profile.get("avg_ms"), (int, float)):
                    raise SmokeFailure(
                        "KCMM KV read profile recorded no average duration: "
                        + json.dumps(report, sort_keys=True)
                    )
        elif report.get("reference_read_bytes", 0) <= 0:
            raise SmokeFailure(
                "KCMM KV read replacement candidate read no KCMM bytes: "
                + json.dumps(report, sort_keys=True)
            )
    if not report.get("recent_calls"):
        raise SmokeFailure(
            "KCMM KV read offset-table planner recorded no recent calls: "
            + json.dumps(report, sort_keys=True)
        )
    return {"path": str(path), **report}


def read_shadow_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"shadow allocator report was not written: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("error_count", 0):
        raise SmokeFailure(
            "shadow allocator reported errors: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("outstanding_mappings", 0):
        raise SmokeFailure(
            "shadow allocator leaked mappings: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("native_gpu_allocations", 0) <= 0:
        raise SmokeFailure(
            "shadow allocator did not observe any GPU allocations: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("kcmm_allocations") != report.get("kcmm_frees"):
        raise SmokeFailure(
            "shadow allocator KCMM allocation/free count mismatch: "
            + json.dumps(report, sort_keys=True)
        )
    return {"path": str(path), **report}


def read_backed_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"KCMM-backed allocator report was not written: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("stop_condition"):
        raise SmokeFailure(
            "KCMM-backed allocator stopped before completion: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("error_count", 0):
        raise SmokeFailure(
            "KCMM-backed allocator reported errors: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("outstanding_mappings", 0):
        raise SmokeFailure(
            "KCMM-backed allocator leaked mappings: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("native_gpu_allocations", 0) <= 0:
        raise SmokeFailure(
            "KCMM-backed allocator did not observe GPU allocations: "
            + json.dumps(report, sort_keys=True)
        )
    if report.get("kcmm_allocations") != report.get("kcmm_frees"):
        raise SmokeFailure(
            "KCMM-backed allocator KCMM allocation/free count mismatch: "
            + json.dumps(report, sort_keys=True)
        )
    pool_stats = report.get("pool_stats") or {}
    if pool_stats.get("blocks_in_use", 0) != 0:
        raise SmokeFailure(
            "KCMM-backed allocator left KCMM blocks in use: "
            + json.dumps(report, sort_keys=True)
        )
    return {"path": str(path), **report}


def start_server(config: SmokeConfig) -> subprocess.Popen[None]:
    if port_is_open(config.host, config.port):
        raise SmokeFailure(f"port already listening: {config.host}:{config.port}")
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = config.log_path.open("w", encoding="utf-8")
    command = vllm_command(config)
    env = os.environ.copy()
    no_proxy_entries = [config.host, "127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = env.get(key, "")
        existing_entries = [entry for entry in existing.split(",") if entry]
        merged = [*existing_entries]
        for entry in no_proxy_entries:
            if entry not in merged:
                merged.append(entry)
        env[key] = ",".join(merged)
    print(f"start vLLM: {' '.join(command)}", flush=True)
    print(f"log: {config.log_path}", flush=True)
    return subprocess.Popen(
        command,
        cwd=repo_root(),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def terminate_server(
    process: subprocess.Popen[None],
    config: SmokeConfig,
) -> None:
    if process.poll() is None:
        deadlines = [
            (signal.SIGINT, config.shutdown_timeout_seconds * 0.30),
            (signal.SIGTERM, config.shutdown_timeout_seconds * 0.15),
            (signal.SIGKILL, config.shutdown_timeout_seconds * 0.55),
        ]
        for sig, timeout in deadlines:
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            if wait_process_exit(process, max(timeout, 1.0)):
                break

    if not wait_for_port_closed(
        config.host,
        config.port,
        timeout_seconds=config.shutdown_timeout_seconds,
    ):
        raise SmokeFailure(f"port still listening after shutdown: {config.port}")

    # vLLM can close the HTTP port before Python atexit hooks and CUDA teardown
    # have finished. Give the process a bounded cleanup window before treating
    # the process group as leaked.
    wait_process_exit(
        process,
        max(2.0, min(5.0, config.shutdown_timeout_seconds * 0.25)),
    )
    live_members = live_process_group_members(process.pid)
    if process.poll() is None and live_members:
        raise SmokeFailure(
            "vLLM process group still has live members after shutdown: "
            + "; ".join(live_members)
        )


def wait_for_ready(
    process: subprocess.Popen[None],
    config: SmokeConfig,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.timeout_seconds
    last_error = ""
    url = f"{config.base_url}/v1/models"
    while time.monotonic() < deadline:
        rc = process.poll()
        if rc is not None:
            raise SmokeFailure(
                f"vLLM exited before readiness with code {rc}\n"
                f"last log lines:\n{tail_file(config.log_path)}"
            )
        try:
            status, payload = http_json("GET", url, timeout_seconds=1.0)
            if status == 200 and payload.get("object") == "list":
                return payload
            last_error = f"GET /v1/models returned {status}: {payload}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        time.sleep(0.5)

    raise SmokeFailure(
        f"vLLM did not become ready within {config.timeout_seconds}s; "
        f"last error: {last_error}\nlast log lines:\n{tail_file(config.log_path)}"
    )


def run_completion(
    config: SmokeConfig,
    *,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    status, payload = http_json(
        "POST",
        f"{config.base_url}/v1/completions",
        payload={
            "model": config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout_seconds=config.timeout_seconds,
    )
    if status != 200:
        raise SmokeFailure(
            f"POST /v1/completions returned {status}: {payload}\n"
            f"last log lines:\n{tail_file(config.log_path)}"
        )
    choices = payload.get("choices")
    if not choices:
        raise SmokeFailure(f"completion response has no choices: {payload}")
    return payload


def completion_case_sequence(config: SmokeConfig) -> tuple[CompletionCase, ...]:
    if config.completion_cases:
        return config.completion_cases
    return (
        CompletionCase(
            name="default",
            prompt=config.prompt,
            max_tokens=config.max_tokens,
        ),
    )


def run_completion_case(config: SmokeConfig, case: CompletionCase) -> dict[str, Any]:
    case_started_at = time.monotonic()
    completion = run_completion(
        config,
        prompt=case.prompt,
        max_tokens=case.max_tokens,
    )
    case_completed_at = time.monotonic()
    return {
        "name": case.name,
        "prompt": case.prompt,
        "max_tokens": case.max_tokens,
        "completion_seconds": round(case_completed_at - case_started_at, 3),
        "completion": completion,
    }


def run_completion_cases(
    config: SmokeConfig,
    cases: tuple[CompletionCase, ...],
) -> list[dict[str, Any]]:
    if config.completion_concurrency <= 1 or len(cases) <= 1:
        return [run_completion_case(config, case) for case in cases]

    max_workers = min(config.completion_concurrency, len(cases))
    results: list[dict[str, Any] | None] = [None] * len(cases)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_completion_case, config, case): index
            for index, case in enumerate(cases)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return [result for result in results if result is not None]


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    if config.shadow_allocations and config.mode != "kcmm":
        raise SmokeFailure("--shadow-allocations requires --mode kcmm")
    if config.backed_allocations and config.mode != "kcmm":
        raise SmokeFailure("--backed-allocations requires --mode kcmm")
    if config.runtime_derived_pool and config.mode != "kcmm":
        raise SmokeFailure("--runtime-derived-pool requires --mode kcmm")
    if config.backed_allocations and config.shadow_allocations:
        raise SmokeFailure("--backed-allocations cannot be combined with --shadow-allocations")
    if config.kv_write_mirror and not config.backed_allocations:
        raise SmokeFailure("--kv-write-mirror requires --backed-allocations")
    if config.kv_write_replace_candidate and config.kv_write_mirror:
        raise SmokeFailure(
            "--kv-write-replace-candidate cannot be combined with --kv-write-mirror"
        )
    if config.kv_write_replace_candidate and not config.backed_allocations:
        raise SmokeFailure("--kv-write-replace-candidate requires --backed-allocations")
    if config.kv_write_device_slots and not (
        config.kv_write_mirror or config.kv_write_replace_candidate
    ):
        raise SmokeFailure(
            "--kv-write-device-slots requires --kv-write-mirror or "
            "--kv-write-replace-candidate"
        )
    if config.kv_write_device_slots and config.kv_write_verify:
        raise SmokeFailure(
            "--kv-write-device-slots requires --no-kv-write-verify"
        )
    if config.kv_read_offset_table and not config.backed_allocations:
        raise SmokeFailure("--kv-read-offset-table requires --backed-allocations")
    if config.kv_read_replace_candidate and config.kv_read_offset_table:
        raise SmokeFailure(
            "--kv-read-replace-candidate cannot be combined with --kv-read-offset-table"
        )
    if config.kv_read_gpu_kernel_candidate and (
        config.kv_read_offset_table or config.kv_read_replace_candidate
    ):
        raise SmokeFailure(
            "--kv-read-gpu-kernel-candidate cannot be combined with "
            "--kv-read-offset-table or --kv-read-replace-candidate"
        )
    if config.kv_read_replace_candidate and not config.backed_allocations:
        raise SmokeFailure("--kv-read-replace-candidate requires --backed-allocations")
    if config.kv_read_gpu_kernel_candidate and not config.backed_allocations:
        raise SmokeFailure("--kv-read-gpu-kernel-candidate requires --backed-allocations")
    if config.kv_read_replace_candidate and not (
        config.kv_write_mirror or config.kv_write_replace_candidate
    ):
        raise SmokeFailure(
            "--kv-read-replace-candidate requires --kv-write-mirror or "
            "--kv-write-replace-candidate"
        )
    if config.kv_read_gpu_kernel_candidate and not (
        config.kv_write_mirror or config.kv_write_replace_candidate
    ):
        raise SmokeFailure(
            "--kv-read-gpu-kernel-candidate requires --kv-write-mirror or "
            "--kv-write-replace-candidate"
        )
    if config.kv_read_profile and not config.kv_read_gpu_kernel_candidate:
        raise SmokeFailure("--kv-read-profile requires --kv-read-gpu-kernel-candidate")
    if config.kv_force_non_default_stream and not (
        config.kv_write_mirror
        or config.kv_write_replace_candidate
        or config.kv_read_gpu_kernel_candidate
    ):
        raise SmokeFailure(
            "--kv-force-non-default-stream requires a KCMM KV write or GPU read path"
        )
    ensure_kcmm_library(config)
    generated_model = ensure_model(config)
    process: subprocess.Popen[None] | None = None
    gpu_monitor = GpuMemoryMonitor()
    gpu_monitor.start()
    gpu_memory: dict[str, Any] | None = None
    started_at = time.monotonic()
    result: dict[str, Any] | None = None
    try:
        process = start_server(config)
        models = wait_for_ready(process, config)
        ready_at = time.monotonic()
        request_started_at = time.monotonic()
        completion_cases = run_completion_cases(
            config,
            completion_case_sequence(config),
        )
        request_completed_at = time.monotonic()
        completion_case_seconds_total = sum(
            float(case["completion_seconds"]) for case in completion_cases
        )
        completion = completion_cases[0]["completion"]
        result = {
            "mode": config.mode,
            "base_url": config.base_url,
            "model_path": str(config.model_path),
            "model_name": config.model_name,
            "log_path": str(config.log_path),
            "startup_seconds": round(ready_at - started_at, 3),
            "completion_seconds": round(request_completed_at - request_started_at, 3),
            "completion_case_seconds_total": round(completion_case_seconds_total, 3),
            "completion_concurrency": config.completion_concurrency,
            "models": models,
            "completion": completion,
            "completion_cases": completion_cases,
            "generated_model": generated_model,
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tensor_parallel_size,
            "generate_tiny_model": config.generate_tiny_model,
            "runtime_derived_pool": config.runtime_derived_pool,
            "instrument_kv_writes": config.instrument_kv_writes,
            "instrument_kv_reads": config.instrument_kv_reads,
            "kv_read_offset_table": config.kv_read_offset_table,
            "kv_read_replace_candidate": config.kv_read_replace_candidate,
            "kv_read_gpu_kernel_candidate": config.kv_read_gpu_kernel_candidate,
            "kv_read_profile": config.kv_read_profile,
            "kv_read_validate_block_tables": config.kv_read_validate_block_tables,
            "kv_read_fast_current_context_launch": (
                config.kv_read_fast_current_context_launch
            ),
            "kv_read_precompile_gpu_kernel": config.kv_read_precompile_gpu_kernel,
            "tracker_report_on_update": config.tracker_report_on_update,
            "tracker_host_profile": config.tracker_host_profile,
            "kv_write_mirror": config.kv_write_mirror,
            "kv_write_replace_candidate": config.kv_write_replace_candidate,
            "kv_write_verify": config.kv_write_verify,
            "kv_write_device_slots": config.kv_write_device_slots,
            "kv_force_non_default_stream": config.kv_force_non_default_stream,
            "shadow_allocations": config.shadow_allocations,
            "backed_allocations": config.backed_allocations,
        }
    finally:
        try:
            if process is not None:
                terminate_server(process, config)
        finally:
            gpu_memory = gpu_monitor.stop()
            if generated_model and not config.keep_model:
                shutil.rmtree(config.model_path, ignore_errors=True)

    if result is None:
        raise SmokeFailure("smoke run exited without a result")
    if gpu_memory is not None:
        result["gpu_memory"] = gpu_memory
    if config.instrument_allocators:
        result["allocator_trace"] = read_allocator_trace(config.allocator_trace_path)
    if config.instrument_kv_writes:
        result["kv_write_trace"] = read_kv_write_trace(config.kv_write_trace_path)
    if config.instrument_kv_reads:
        result["kv_read_trace"] = read_kv_read_trace(config.kv_read_trace_path)
    if config.kv_write_mirror or config.kv_write_replace_candidate:
        report = read_kv_write_mirror_report(
            config.kv_write_mirror_report_path,
            expect_verification=config.kv_write_verify,
            expect_device_slots=config.kv_write_device_slots,
        )
        if config.kv_force_non_default_stream:
            require_non_default_stream_report(report, name="KCMM KV write")
        if config.kv_write_replace_candidate:
            result["kv_write_replace_candidate_report"] = report
        else:
            result["kv_write_mirror"] = report
    if (
        config.kv_read_offset_table
        or config.kv_read_replace_candidate
        or config.kv_read_gpu_kernel_candidate
    ):
        result["kv_read_offset_table_report"] = read_kv_read_offset_table_report(
            config.kv_read_offset_table_report_path,
            expect_replacement=(
                config.kv_read_replace_candidate
                or config.kv_read_gpu_kernel_candidate
            ),
            expect_gpu_kernel=config.kv_read_gpu_kernel_candidate,
            expect_profile=config.kv_read_profile,
        )
        if config.kv_force_non_default_stream and config.kv_read_gpu_kernel_candidate:
            require_non_default_stream_report(
                result["kv_read_offset_table_report"],
                name="KCMM KV read",
            )
    if config.shadow_allocations:
        result["shadow_allocator"] = read_shadow_report(config.shadow_report_path)
    if config.backed_allocations:
        result["backed_allocator"] = read_backed_report(config.backed_report_path)
    return result


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    try:
        result = run_smoke(config)
    except SmokeFailure as exc:
        print(f"KCMM vLLM smoke failed: {exc}", file=sys.stderr)
        if config.backed_allocations and config.backed_report_path.exists():
            print(
                f"\nKCMM-backed allocator report ({config.backed_report_path}):",
                file=sys.stderr,
            )
            print(config.backed_report_path.read_text(encoding="utf-8"), file=sys.stderr)
        if (
            (config.kv_write_mirror or config.kv_write_replace_candidate)
            and config.kv_write_mirror_report_path.exists()
        ):
            print(
                f"\nKCMM KV write mirror report ({config.kv_write_mirror_report_path}):",
                file=sys.stderr,
            )
            print(
                config.kv_write_mirror_report_path.read_text(encoding="utf-8"),
                file=sys.stderr,
            )
        if (
            (
                config.kv_read_offset_table
                or config.kv_read_replace_candidate
                or config.kv_read_gpu_kernel_candidate
            )
            and config.kv_read_offset_table_report_path.exists()
        ):
            print(
                "\nKCMM KV read offset-table report "
                f"({config.kv_read_offset_table_report_path}):",
                file=sys.stderr,
            )
            print(
                config.kv_read_offset_table_report_path.read_text(encoding="utf-8"),
                file=sys.stderr,
            )
        if config.log_path.exists():
            print(f"\nLog tail ({config.log_path}):", file=sys.stderr)
            print(tail_file(config.log_path), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
