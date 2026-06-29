"""Configuration helpers for the KCMM vLLM observer launcher."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, replace

from .bindings import KcmmConfig


EVICTION_POLICY_CODES = {
    "lru": 0,
    "lfu": 1,
    "fifo": 2,
}

POOL_MODES = ("fixed", "runtime")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


@dataclass(frozen=True)
class VllmRuntimeSizing:
    """vLLM runtime cache/model sizing values used to shape a KCMM pool."""

    vllm_version: str
    block_size: int
    num_gpu_blocks: int
    num_cpu_blocks: int
    effective_num_gpu_blocks: int
    num_layers: int
    kv_heads: int
    head_dim: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    cache_dtype: str
    model_dtype: str
    use_v2_block_manager: bool
    enforce_eager: bool
    enable_prefix_caching: bool

    def validate(self) -> None:
        positive_fields = (
            "block_size",
            "num_gpu_blocks",
            "effective_num_gpu_blocks",
            "num_layers",
            "kv_heads",
            "head_dim",
            "max_model_len",
            "max_num_seqs",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        )
        for field in positive_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"vLLM runtime sizing field {field} must be positive")
        if not self.use_v2_block_manager:
            raise ValueError("Phase II.A runtime pool sizing requires --use-v2-block-manager")
        if not self.enforce_eager:
            raise ValueError("Phase II.A runtime pool sizing requires --enforce-eager")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObserverConfig:
    """KCMM launcher settings for Phase I.C.

    The default model shape is intentionally tiny. Phase I.C only proves that
    the Python process can create a KCMM CUDA pool beside vLLM and sample its
    observer metrics; it does not size the pool for the served model yet.
    """

    library_path: str | None = None
    device_ordinal: int = 0
    block_size: int = 16
    max_blocks: int = 64
    num_layers: int = 1
    kv_heads: int = 1
    head_dim: int = 64
    max_batch: int = 1
    max_seq_len: int = 16
    cpu_cache_path: str = "/dev/shm/kcmm_vllm_observer"
    enable_tiering: bool = False
    eviction_policy: str = "lru"
    prefetch_window: int = 4
    max_batch_blocks: int = 64
    low_watermark_threshold: float = 0.2
    background_evict_interval_ms: int = 100
    attention_sink_blocks: int = 1
    recent_window_blocks: int = 4
    probe_blocks: int = 1
    pool_mode: str = "fixed"
    observer_only: bool = False
    skip_observer: bool = False
    destroy_before_vllm: bool = False
    print_seams: bool = False
    instrument_allocators: bool = False
    allocator_trace_path: str | None = None
    require_allocator_seams: bool = False
    instrument_kv_writes: bool = False
    kv_write_trace_path: str | None = None
    require_kv_write_seams: bool = False
    kv_write_mirror: bool = False
    kv_write_replace_candidate: bool = False
    kv_write_mirror_report_path: str | None = None
    kv_write_verify: bool = True
    instrument_kv_reads: bool = False
    kv_read_trace_path: str | None = None
    require_kv_read_seams: bool = False
    kv_read_offset_table: bool = False
    kv_read_replace_candidate: bool = False
    kv_read_gpu_kernel_candidate: bool = False
    kv_read_profile: bool = False
    kv_read_offset_table_report_path: str | None = None
    kv_force_non_default_stream: bool = False
    shadow_allocations: bool = False
    shadow_report_path: str | None = None
    backed_allocations: bool = False
    backed_report_path: str | None = None

    @classmethod
    def from_env(cls) -> "ObserverConfig":
        return cls(
            library_path=os.environ.get("KCMM_LIB_PATH") or None,
            device_ordinal=_env_int("KCMM_DEVICE_ORDINAL", 0),
            block_size=_env_int("KCMM_BLOCK_SIZE", 16),
            max_blocks=_env_int("KCMM_MAX_BLOCKS", 64),
            num_layers=_env_int("KCMM_NUM_LAYERS", 1),
            kv_heads=_env_int("KCMM_KV_HEADS", 1),
            head_dim=_env_int("KCMM_HEAD_DIM", 64),
            max_batch=_env_int("KCMM_MAX_BATCH", 1),
            max_seq_len=_env_int("KCMM_MAX_SEQ_LEN", 16),
            cpu_cache_path=os.environ.get(
                "KCMM_CPU_CACHE_PATH", "/dev/shm/kcmm_vllm_observer"
            ),
            enable_tiering=_env_bool("KCMM_ENABLE_TIERING", False),
            eviction_policy=os.environ.get("KCMM_EVICTION_POLICY", "lru"),
            prefetch_window=_env_int("KCMM_PREFETCH_WINDOW", 4),
            max_batch_blocks=_env_int("KCMM_MAX_BATCH_BLOCKS", 64),
            low_watermark_threshold=_env_float("KCMM_LOW_WATERMARK_THRESHOLD", 0.2),
            background_evict_interval_ms=_env_int(
                "KCMM_BACKGROUND_EVICT_INTERVAL_MS", 100
            ),
            attention_sink_blocks=_env_int("KCMM_ATTENTION_SINK_BLOCKS", 1),
            recent_window_blocks=_env_int("KCMM_RECENT_WINDOW_BLOCKS", 4),
            probe_blocks=_env_int("KCMM_PROBE_BLOCKS", 1),
            pool_mode=os.environ.get("KCMM_POOL_MODE", "fixed"),
            observer_only=_env_bool("KCMM_OBSERVER_ONLY", False),
            skip_observer=_env_bool("KCMM_SKIP_OBSERVER", False),
            destroy_before_vllm=_env_bool("KCMM_DESTROY_BEFORE_VLLM", False),
            print_seams=_env_bool("KCMM_PRINT_SEAMS", False),
            instrument_allocators=_env_bool("KCMM_INSTRUMENT_ALLOCATORS", False),
            allocator_trace_path=os.environ.get("KCMM_ALLOCATOR_TRACE_PATH") or None,
            require_allocator_seams=_env_bool("KCMM_REQUIRE_ALLOCATOR_SEAMS", False),
            instrument_kv_writes=_env_bool("KCMM_INSTRUMENT_KV_WRITES", False),
            kv_write_trace_path=os.environ.get("KCMM_KV_WRITE_TRACE_PATH") or None,
            require_kv_write_seams=_env_bool("KCMM_REQUIRE_KV_WRITE_SEAMS", False),
            kv_write_mirror=_env_bool("KCMM_KV_WRITE_MIRROR", False),
            kv_write_replace_candidate=_env_bool(
                "KCMM_KV_WRITE_REPLACE_CANDIDATE", False
            ),
            kv_write_mirror_report_path=(
                os.environ.get("KCMM_KV_WRITE_MIRROR_REPORT_PATH") or None
            ),
            kv_write_verify=_env_bool("KCMM_KV_WRITE_VERIFY", True),
            instrument_kv_reads=_env_bool("KCMM_INSTRUMENT_KV_READS", False),
            kv_read_trace_path=os.environ.get("KCMM_KV_READ_TRACE_PATH") or None,
            require_kv_read_seams=_env_bool("KCMM_REQUIRE_KV_READ_SEAMS", False),
            kv_read_offset_table=_env_bool("KCMM_KV_READ_OFFSET_TABLE", False),
            kv_read_replace_candidate=_env_bool(
                "KCMM_KV_READ_REPLACE_CANDIDATE", False
            ),
            kv_read_gpu_kernel_candidate=_env_bool(
                "KCMM_KV_READ_GPU_KERNEL_CANDIDATE", False
            ),
            kv_read_profile=_env_bool("KCMM_KV_READ_PROFILE", False),
            kv_read_offset_table_report_path=(
                os.environ.get("KCMM_KV_READ_OFFSET_TABLE_REPORT_PATH") or None
            ),
            kv_force_non_default_stream=_env_bool(
                "KCMM_KV_FORCE_NON_DEFAULT_STREAM", False
            ),
            shadow_allocations=_env_bool("KCMM_SHADOW_ALLOCATIONS", False),
            shadow_report_path=os.environ.get("KCMM_SHADOW_REPORT_PATH") or None,
            backed_allocations=_env_bool("KCMM_BACKED_ALLOCATIONS", False),
            backed_report_path=os.environ.get("KCMM_BACKED_REPORT_PATH") or None,
        )

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "ObserverConfig":
        base = cls.from_env()
        values = {}
        for field in base.__dataclass_fields__:
            arg_name = (
                f"kcmm_{field}" if field != "library_path" else "kcmm_lib_path"
            )
            value = getattr(namespace, arg_name, None)
            values[field] = getattr(base, field) if value is None else value
        return cls(**values)

    def to_c_config(self) -> KcmmConfig:
        if self.eviction_policy not in EVICTION_POLICY_CODES:
            raise ValueError(f"unsupported eviction policy: {self.eviction_policy}")

        path = self.cpu_cache_path.encode("utf-8")
        if len(path) >= 256:
            raise ValueError("cpu_cache_path must fit in 255 bytes")

        cfg = KcmmConfig()
        cfg.block_size = self.block_size
        cfg.max_blocks = self.max_blocks
        cfg.cpu_cache_path = path
        cfg.tiering = 1 if self.enable_tiering else 0
        cfg.eviction_policy = EVICTION_POLICY_CODES[self.eviction_policy]
        cfg.prefetch_window = self.prefetch_window
        cfg.max_batch_blocks = self.max_batch_blocks
        cfg.device_ordinal = self.device_ordinal
        cfg.num_layers = self.num_layers
        cfg.kv_heads = self.kv_heads
        cfg.head_dim = self.head_dim
        cfg.max_batch = self.max_batch
        cfg.max_seq_len = self.max_seq_len
        cfg.low_watermark_threshold = self.low_watermark_threshold
        cfg.background_evict_interval_ms = self.background_evict_interval_ms
        cfg.attention_sink_blocks = self.attention_sink_blocks
        cfg.recent_window_blocks = self.recent_window_blocks
        return cfg

    def validate(self) -> None:
        if self.pool_mode not in POOL_MODES:
            raise ValueError(
                f"unsupported KCMM pool mode: {self.pool_mode}; "
                f"expected one of {', '.join(POOL_MODES)}"
            )
        if self.enable_tiering and self.pool_mode == "runtime":
            raise ValueError("Phase II.A runtime-derived KCMM pool requires tiering disabled")
        if self.shadow_allocations and self.pool_mode != "runtime":
            raise ValueError("KCMM shadow allocation mode requires --kcmm-pool-mode runtime")
        if self.shadow_allocations and self.skip_observer:
            raise ValueError("KCMM shadow allocation mode requires the KCMM observer pool")
        if self.backed_allocations and self.pool_mode != "runtime":
            raise ValueError("KCMM-backed allocation mode requires --kcmm-pool-mode runtime")
        if self.backed_allocations and self.skip_observer:
            raise ValueError("KCMM-backed allocation mode requires the KCMM observer pool")
        if self.backed_allocations and self.shadow_allocations:
            raise ValueError("KCMM-backed allocation mode cannot be combined with shadow mode")
        if self.kv_write_mirror and self.pool_mode != "runtime":
            raise ValueError("KCMM KV write mirror requires --kcmm-pool-mode runtime")
        if self.kv_write_mirror and self.skip_observer:
            raise ValueError("KCMM KV write mirror requires the KCMM observer pool")
        if self.kv_write_mirror and not self.backed_allocations:
            raise ValueError("KCMM KV write mirror requires --kcmm-backed-allocations")
        if self.kv_write_replace_candidate and self.kv_write_mirror:
            raise ValueError(
                "KCMM KV write replacement candidate cannot be combined with mirror mode"
            )
        if self.kv_write_replace_candidate and self.pool_mode != "runtime":
            raise ValueError(
                "KCMM KV write replacement candidate requires --kcmm-pool-mode runtime"
            )
        if self.kv_write_replace_candidate and self.skip_observer:
            raise ValueError(
                "KCMM KV write replacement candidate requires the KCMM observer pool"
            )
        if self.kv_write_replace_candidate and not self.backed_allocations:
            raise ValueError(
                "KCMM KV write replacement candidate requires --kcmm-backed-allocations"
            )
        if self.kv_read_offset_table and self.pool_mode != "runtime":
            raise ValueError(
                "KCMM KV read offset-table planning requires --kcmm-pool-mode runtime"
            )
        if self.kv_read_offset_table and self.skip_observer:
            raise ValueError(
                "KCMM KV read offset-table planning requires the KCMM observer pool"
            )
        if self.kv_read_offset_table and not self.backed_allocations:
            raise ValueError(
                "KCMM KV read offset-table planning requires --kcmm-backed-allocations"
            )
        if self.kv_read_replace_candidate and self.kv_read_offset_table:
            raise ValueError(
                "KCMM KV read replacement candidate cannot be combined with "
                "offset-table planning mode"
            )
        if self.kv_read_gpu_kernel_candidate and (
            self.kv_read_offset_table or self.kv_read_replace_candidate
        ):
            raise ValueError(
                "KCMM KV read GPU kernel candidate cannot be combined with "
                "offset-table planning or reference replacement mode"
            )
        if self.kv_read_replace_candidate and self.pool_mode != "runtime":
            raise ValueError(
                "KCMM KV read replacement candidate requires --kcmm-pool-mode runtime"
            )
        if self.kv_read_gpu_kernel_candidate and self.pool_mode != "runtime":
            raise ValueError(
                "KCMM KV read GPU kernel candidate requires --kcmm-pool-mode runtime"
            )
        if self.kv_read_replace_candidate and self.skip_observer:
            raise ValueError(
                "KCMM KV read replacement candidate requires the KCMM observer pool"
            )
        if self.kv_read_gpu_kernel_candidate and self.skip_observer:
            raise ValueError(
                "KCMM KV read GPU kernel candidate requires the KCMM observer pool"
            )
        if self.kv_read_replace_candidate and not self.backed_allocations:
            raise ValueError(
                "KCMM KV read replacement candidate requires --kcmm-backed-allocations"
            )
        if self.kv_read_gpu_kernel_candidate and not self.backed_allocations:
            raise ValueError(
                "KCMM KV read GPU kernel candidate requires --kcmm-backed-allocations"
            )
        if self.kv_read_replace_candidate and not (
            self.kv_write_mirror or self.kv_write_replace_candidate
        ):
            raise ValueError(
                "KCMM KV read replacement candidate requires KCMM KV writes via "
                "--kcmm-kv-write-mirror or --kcmm-kv-write-replace-candidate"
            )
        if self.kv_read_gpu_kernel_candidate and not (
            self.kv_write_mirror or self.kv_write_replace_candidate
        ):
            raise ValueError(
                "KCMM KV read GPU kernel candidate requires KCMM KV writes via "
                "--kcmm-kv-write-mirror or --kcmm-kv-write-replace-candidate"
            )
        if self.kv_read_profile and not self.kv_read_gpu_kernel_candidate:
            raise ValueError(
                "KCMM KV read profiling requires "
                "--kcmm-kv-read-gpu-kernel-candidate"
            )
        if self.kv_force_non_default_stream and not (
            self.kv_write_mirror
            or self.kv_write_replace_candidate
            or self.kv_read_gpu_kernel_candidate
        ):
            raise ValueError(
                "KCMM forced non-default stream mode requires a stream-aware "
                "KV write or GPU read path"
            )

    def with_runtime_sizing(self, sizing: VllmRuntimeSizing) -> "ObserverConfig":
        sizing.validate()
        return replace(
            self,
            block_size=sizing.block_size,
            max_blocks=sizing.effective_num_gpu_blocks,
            num_layers=sizing.num_layers,
            kv_heads=sizing.kv_heads,
            head_dim=sizing.head_dim,
            max_batch=sizing.max_num_seqs,
            max_seq_len=sizing.max_model_len,
            enable_tiering=False,
        )

    def pool_shape_dict(self) -> dict[str, int | float | bool | str | None]:
        return {
            "pool_mode": self.pool_mode,
            "device_ordinal": self.device_ordinal,
            "block_size": self.block_size,
            "max_blocks": self.max_blocks,
            "num_layers": self.num_layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "max_batch": self.max_batch,
            "max_seq_len": self.max_seq_len,
            "max_batch_blocks": self.max_batch_blocks,
            "probe_blocks": self.probe_blocks,
            "enable_tiering": self.enable_tiering,
            "cpu_cache_path": self.cpu_cache_path,
        }


def add_kcmm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--kcmm-help", action="store_true", default=None)
    parser.add_argument("--kcmm-lib-path", default=None)
    parser.add_argument("--kcmm-device-ordinal", type=int, default=None)
    parser.add_argument("--kcmm-block-size", type=int, default=None)
    parser.add_argument("--kcmm-max-blocks", type=int, default=None)
    parser.add_argument("--kcmm-num-layers", type=int, default=None)
    parser.add_argument("--kcmm-kv-heads", type=int, default=None)
    parser.add_argument("--kcmm-head-dim", type=int, default=None)
    parser.add_argument("--kcmm-max-batch", type=int, default=None)
    parser.add_argument("--kcmm-max-seq-len", type=int, default=None)
    parser.add_argument("--kcmm-cpu-cache-path", default=None)
    parser.add_argument("--kcmm-enable-tiering", action="store_true", default=None)
    parser.add_argument(
        "--kcmm-disable-tiering",
        dest="kcmm_enable_tiering",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--kcmm-eviction-policy",
        choices=sorted(EVICTION_POLICY_CODES),
        default=None,
    )
    parser.add_argument("--kcmm-prefetch-window", type=int, default=None)
    parser.add_argument("--kcmm-max-batch-blocks", type=int, default=None)
    parser.add_argument("--kcmm-low-watermark-threshold", type=float, default=None)
    parser.add_argument("--kcmm-background-evict-interval-ms", type=int, default=None)
    parser.add_argument("--kcmm-attention-sink-blocks", type=int, default=None)
    parser.add_argument("--kcmm-recent-window-blocks", type=int, default=None)
    parser.add_argument("--kcmm-probe-blocks", type=int, default=None)
    parser.add_argument("--kcmm-pool-mode", choices=POOL_MODES, default=None)
    parser.add_argument("--kcmm-observer-only", action="store_true", default=None)
    parser.add_argument("--kcmm-skip-observer", action="store_true", default=None)
    parser.add_argument("--kcmm-destroy-before-vllm", action="store_true", default=None)
    parser.add_argument("--kcmm-print-seams", action="store_true", default=None)
    parser.add_argument(
        "--kcmm-instrument-allocators",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-allocator-trace-path", default=None)
    parser.add_argument(
        "--kcmm-require-allocator-seams",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-instrument-kv-writes",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-kv-write-trace-path", default=None)
    parser.add_argument(
        "--kcmm-require-kv-write-seams",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-kv-write-mirror",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-kv-write-replace-candidate",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-kv-write-mirror-report-path", default=None)
    parser.add_argument(
        "--kcmm-kv-write-verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable bounded D2H verification of KCMM KV write rows. "
            "Disable with --no-kcmm-kv-write-verify for performance-clean gates."
        ),
    )
    parser.add_argument(
        "--kcmm-instrument-kv-reads",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-kv-read-trace-path", default=None)
    parser.add_argument(
        "--kcmm-require-kv-read-seams",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-kv-read-offset-table",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-kv-read-replace-candidate",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-kv-read-gpu-kernel-candidate",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-kv-read-profile",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-kv-read-offset-table-report-path", default=None)
    parser.add_argument(
        "--kcmm-kv-force-non-default-stream",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--kcmm-shadow-allocations",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-shadow-report-path", default=None)
    parser.add_argument(
        "--kcmm-backed-allocations",
        action="store_true",
        default=None,
    )
    parser.add_argument("--kcmm-backed-report-path", default=None)
    return parser
