"""Phase II.C GPU read-kernel A/B gate across tiny OPT shape variants."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.kcmm.vllm_gpu_read_ab_gate import (
    DEFAULT_COVERAGE_CASES,
    GateConfig,
    parse_coverage_case,
    run_gate,
)
from scripts.kcmm.vllm_smoke import (
    CompletionCase,
    DEFAULT_KCMM_LIB_PATH,
    DEFAULT_MODEL_NAME,
    repo_root,
    resolve_repo_path,
)


DEFAULT_VARIANTS = (
    "head64_layers2:128:2:2:256",
    "head80_layers2:160:2:2:320",
    "head96_layers2:192:2:2:384",
    "head128_layers2:256:2:2:512",
    "head192_layers2:384:2:2:768",
    "head256_layers2:512:2:2:1024",
)
SUPPORTED_HEAD_DIMS = (64, 80, 96, 112, 120, 128, 192, 256)
DEFAULT_SHAPE_COVERAGE_CASES = (
    DEFAULT_COVERAGE_CASES[0],
    DEFAULT_COVERAGE_CASES[1],
    CompletionCase(
        name="long_context",
        prompt=DEFAULT_COVERAGE_CASES[2].prompt,
        max_tokens=1,
    ),
)


@dataclass(frozen=True)
class ShapeVariant:
    name: str
    hidden_size: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    seed: int = 0
    max_position_embeddings: int = 8192

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def model_path(self) -> Path:
        return repo_root() / ".scratch" / "kcmm-vllm" / "shape-gate" / self.name


@dataclass(frozen=True)
class ShapeGateConfig:
    host: str
    port: int
    model_name: str
    kcmm_lib_path: Path
    timeout_seconds: float
    shutdown_timeout_seconds: float
    build_kcmm: bool
    keep_model: bool
    print_seams: bool
    output_path: Path
    variants: tuple[ShapeVariant, ...]
    coverage_cases: tuple[Any, ...]
    latency_warning_ratio: float
    throughput_warning_ratio: float
    memory_warning_ratio: float
    memory_warning_min_delta_mib: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        metavar="NAME:HIDDEN_SIZE:NUM_HEADS:NUM_LAYERS:FFN_DIM",
        help=(
            "Tiny OPT shape variant to run. May be repeated. Defaults to "
            f"{', '.join(DEFAULT_VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--coverage-case",
        action="append",
        default=None,
        metavar="NAME:MAX_TOKENS:PROMPT",
        help=(
            "Completion case to compare for every shape variant. May be repeated. "
            "Defaults to short multi-token cases plus a single-token "
            "long-context case that still forces multi-block decode reads."
        ),
    )
    parser.add_argument(
        "--build-kcmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the KCMM shared library before each KCMM mode if needed.",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Keep generated tiny model variants after the gate run.",
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
        help="Shape gate JSON report path. Defaults to a /tmp file.",
    )
    parser.add_argument("--latency-warning-ratio", type=float, default=2.0)
    parser.add_argument("--throughput-warning-ratio", type=float, default=0.5)
    parser.add_argument("--memory-warning-ratio", type=float, default=1.5)
    parser.add_argument("--memory-warning-min-delta-mib", type=int, default=256)
    return parser


def parse_variant(value: str) -> ShapeVariant:
    parts = value.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "variants must use NAME:HIDDEN_SIZE:NUM_HEADS:NUM_LAYERS:FFN_DIM"
        )
    name, hidden_text, heads_text, layers_text, ffn_text = parts
    if not name:
        raise argparse.ArgumentTypeError("variant name cannot be empty")
    try:
        hidden_size = int(hidden_text)
        num_heads = int(heads_text)
        num_layers = int(layers_text)
        ffn_dim = int(ffn_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid variant integer field: {value}") from exc
    if min(hidden_size, num_heads, num_layers, ffn_dim) <= 0:
        raise argparse.ArgumentTypeError("variant numeric fields must be positive")
    if hidden_size % num_heads != 0:
        raise argparse.ArgumentTypeError("variant hidden size must divide by heads")
    head_dim = hidden_size // num_heads
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise argparse.ArgumentTypeError(
            "variant head_dim must be one of "
            f"{SUPPORTED_HEAD_DIMS} for this CUDA 11.8 vLLM/XFormers stack "
            "and the current GPU read kernel"
        )
    return ShapeVariant(
        name=name,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
    )


def parse_variants(values: list[str] | None) -> tuple[ShapeVariant, ...]:
    variants = tuple(parse_variant(value) for value in (values or list(DEFAULT_VARIANTS)))
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate variant names: {names}")
    return variants


def parse_config(argv: list[str] | None = None) -> ShapeGateConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        variants = parse_variants(args.variant)
        coverage_cases = (
            tuple(parse_coverage_case(value) for value in args.coverage_case)
            if args.coverage_case
            else DEFAULT_SHAPE_COVERAGE_CASES
        )
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    timestamp_ms = int(time.time() * 1000)
    output_path = (
        Path(args.output)
        if args.output
        else Path(tempfile.gettempdir())
        / f"kcmm-vllm-phase-ii-c-gpu-read-shape-gate-{timestamp_ms}.json"
    )
    return ShapeGateConfig(
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
        timeout_seconds=args.timeout_seconds,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
        build_kcmm=args.build_kcmm,
        keep_model=args.keep_model,
        print_seams=args.print_seams,
        output_path=output_path,
        variants=variants,
        coverage_cases=coverage_cases,
        latency_warning_ratio=args.latency_warning_ratio,
        throughput_warning_ratio=args.throughput_warning_ratio,
        memory_warning_ratio=args.memory_warning_ratio,
        memory_warning_min_delta_mib=args.memory_warning_min_delta_mib,
    )


def tiny_model_matches(model_path: Path, variant: ShapeVariant) -> bool:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = {
        "hidden_size": variant.hidden_size,
        "num_attention_heads": variant.num_heads,
        "num_hidden_layers": variant.num_layers,
        "ffn_dim": variant.ffn_dim,
    }
    return all(config.get(key) == value for key, value in expected.items())


def create_tiny_model(variant: ShapeVariant) -> None:
    script = repo_root() / "scripts" / "kcmm" / "create_tiny_opt_model.py"
    command = [
        sys.executable,
        str(script),
        "--output",
        str(variant.model_path),
        "--hidden-size",
        str(variant.hidden_size),
        "--num-heads",
        str(variant.num_heads),
        "--num-layers",
        str(variant.num_layers),
        "--ffn-dim",
        str(variant.ffn_dim),
        "--max-position-embeddings",
        str(variant.max_position_embeddings),
        "--seed",
        str(variant.seed),
    ]
    print(f"create shape variant {variant.name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root(), check=True)


def ensure_variant_model(variant: ShapeVariant) -> bool:
    if tiny_model_matches(variant.model_path, variant):
        return False
    if variant.model_path.exists():
        shutil.rmtree(variant.model_path)
    create_tiny_model(variant)
    return True


def gate_config_for_variant(
    config: ShapeGateConfig,
    variant: ShapeVariant,
    output_path: Path,
) -> GateConfig:
    return GateConfig(
        host=config.host,
        port=config.port,
        model_path=variant.model_path,
        model_name=config.model_name,
        kcmm_lib_path=config.kcmm_lib_path,
        timeout_seconds=config.timeout_seconds,
        shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        generate_tiny_model=True,
        prompt="Hello",
        max_tokens=4,
        coverage_cases=config.coverage_cases,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        tensor_parallel_size=1,
        completion_concurrency=1,
        kv_force_non_default_stream=False,
        kv_read_profile=False,
        kv_read_validate_block_tables=True,
        instrument_kv_reads=True,
        kv_write_verify=True,
        tracker_report_on_update=True,
        tracker_host_profile=False,
        build_kcmm=config.build_kcmm,
        keep_model=True,
        print_seams=config.print_seams,
        output_path=output_path,
        latency_warning_ratio=config.latency_warning_ratio,
        throughput_warning_ratio=config.throughput_warning_ratio,
        memory_warning_ratio=config.memory_warning_ratio,
        memory_warning_min_delta_mib=config.memory_warning_min_delta_mib,
    )


def run_shape_gate(config: ShapeGateConfig) -> dict[str, Any]:
    started_at_ms = int(time.time() * 1000)
    report_dir = config.output_path.parent / f"{config.output_path.stem}-reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    variant_reports: dict[str, Any] = {}
    generated_paths: list[Path] = []
    try:
        for variant in config.variants:
            generated_model = ensure_variant_model(variant)
            if generated_model:
                generated_paths.append(variant.model_path)
            output_path = report_dir / f"{variant.name}.json"
            print(f"run shape variant: {variant.name}", flush=True)
            variant_report = run_gate(
                gate_config_for_variant(config, variant, output_path)
            )
            output_path.write_text(
                json.dumps(variant_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            variant_reports[variant.name] = {
                "variant": asdict(variant),
                "model_path": str(variant.model_path),
                "output_path": str(output_path),
                "report": variant_report,
            }
    finally:
        if not config.keep_model:
            for path in generated_paths:
                shutil.rmtree(path, ignore_errors=True)

    correctness_failures = [
        {
            "variant": name,
            "failures": entry["report"].get("correctness_failures", []),
        }
        for name, entry in variant_reports.items()
        if entry["report"].get("correctness_failures")
    ]
    failed_variants = [
        name
        for name, entry in variant_reports.items()
        if not entry["report"].get("passed", False)
    ]
    performance_warnings = [
        {
            "variant": name,
            "warnings": entry["report"].get("performance_warnings", []),
        }
        for name, entry in variant_reports.items()
        if entry["report"].get("performance_warnings")
    ]
    return {
        "phase": "II.C",
        "gate": "stock-vs-kcmm-gpu-read-shape-gate",
        "passed": not failed_variants,
        "started_at_unix_ms": started_at_ms,
        "repo_root": str(repo_root()),
        "variant_order": [variant.name for variant in config.variants],
        "coverage_cases": [
            {
                "name": case.name,
                "prompt": case.prompt,
                "max_tokens": case.max_tokens,
            }
            for case in config.coverage_cases
        ],
        "variants": variant_reports,
        "failed_variants": failed_variants,
        "correctness_failures": correctness_failures,
        "performance_warnings": performance_warnings,
        "output_path": str(config.output_path),
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    report = run_shape_gate(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
