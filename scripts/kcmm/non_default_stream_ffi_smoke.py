"""Phase II.C non-default CUDA stream smoke for KCMM stream-aware FFI."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bindings import KcmmError, KcmmLibrary
from .config import ObserverConfig
from .vllm_smoke import DEFAULT_KCMM_LIB_PATH, repo_root, resolve_repo_path


class NonDefaultStreamSmokeFailure(RuntimeError):
    """Raised when the non-default stream FFI smoke cannot complete."""


@dataclass(frozen=True)
class NonDefaultStreamSmokeConfig:
    kcmm_lib_path: Path
    build_kcmm: bool
    device_ordinal: int
    block_size: int
    max_blocks: int
    num_layers: int
    kv_heads: int
    num_q_heads: int
    head_dim: int
    max_batch: int
    max_seq_len: int
    cpu_cache_path: str
    output_path: Path | None


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
        self._check(self.lib.cuInit(0), "cuInit")

    @staticmethod
    def _check(rc: int, operation: str) -> None:
        if rc != 0:
            raise NonDefaultStreamSmokeFailure(
                f"{operation} failed with CUDA rc={rc}"
            )

    def memcpy_dtoh(self, src_device_ptr: int, byte_count: int) -> bytes:
        buffer = (ctypes.c_ubyte * byte_count)()
        rc = self.lib.cuMemcpyDtoH_v2(
            ctypes.c_void_p(ctypes.addressof(buffer)),
            ctypes.c_uint64(src_device_ptr),
            ctypes.c_size_t(byte_count),
        )
        self._check(rc, "cuMemcpyDtoH_v2")
        return bytes(buffer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kcmm-lib-path", default=DEFAULT_KCMM_LIB_PATH)
    parser.add_argument(
        "--build-kcmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the KCMM shared library if requested or missing.",
    )
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument(
        "--cpu-cache-path",
        default="/dev/shm/kcmm_non_default_stream_smoke",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser


def parse_config(argv: list[str] | None = None) -> NonDefaultStreamSmokeConfig:
    args = build_parser().parse_args(argv)
    return NonDefaultStreamSmokeConfig(
        kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
        build_kcmm=args.build_kcmm,
        device_ordinal=args.device_ordinal,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        num_layers=args.num_layers,
        kv_heads=args.kv_heads,
        num_q_heads=args.num_q_heads,
        head_dim=args.head_dim,
        max_batch=args.max_batch,
        max_seq_len=args.max_seq_len,
        cpu_cache_path=args.cpu_cache_path,
        output_path=Path(args.output) if args.output else None,
    )


def run_checked(command: list[str], description: str) -> None:
    print(f"{description}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=repo_root(), check=False)
    if result.returncode != 0:
        raise NonDefaultStreamSmokeFailure(
            f"{description} failed with exit code {result.returncode}"
        )


def ensure_kcmm_library(config: NonDefaultStreamSmokeConfig) -> None:
    if config.build_kcmm or not config.kcmm_lib_path.exists():
        run_checked(["cargo", "build", "--features", "kcmm"], "build KCMM")
    if not config.kcmm_lib_path.exists():
        raise NonDefaultStreamSmokeFailure(
            f"KCMM shared library not found: {config.kcmm_lib_path}"
        )


def tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def first_mismatch(actual: bytes, expected: bytes) -> int:
    for index, (left, right) in enumerate(zip(actual, expected, strict=False)):
        if left != right:
            return index
    return min(len(actual), len(expected))


def assert_bytes_equal(name: str, actual: bytes, expected: bytes) -> None:
    if actual == expected:
        return
    mismatch_at = first_mismatch(actual, expected)
    raise NonDefaultStreamSmokeFailure(
        f"{name} byte mismatch at offset {mismatch_at}: "
        f"actual_len={len(actual)} expected_len={len(expected)}"
    )


def observer_config(config: NonDefaultStreamSmokeConfig) -> ObserverConfig:
    return ObserverConfig(
        library_path=str(config.kcmm_lib_path),
        device_ordinal=config.device_ordinal,
        block_size=config.block_size,
        max_blocks=config.max_blocks,
        num_layers=config.num_layers,
        kv_heads=config.kv_heads,
        head_dim=config.head_dim,
        max_batch=config.max_batch,
        max_seq_len=config.max_seq_len,
        cpu_cache_path=config.cpu_cache_path,
        enable_tiering=False,
    )


def expected_decode_output(v_rows: Any, *, num_q_heads: int, kv_heads: int) -> Any:
    import torch

    v_heads = v_rows[0].view(kv_heads, -1)
    per_query_head = [
        v_heads[qh * kv_heads // num_q_heads] for qh in range(num_q_heads)
    ]
    return torch.stack(per_query_head, dim=0).view(1, num_q_heads, -1)


def validate_config(config: NonDefaultStreamSmokeConfig) -> None:
    if config.block_size <= 0:
        raise NonDefaultStreamSmokeFailure("block_size must be positive")
    if config.max_blocks <= 0:
        raise NonDefaultStreamSmokeFailure("max_blocks must be positive")
    if config.num_layers <= 0:
        raise NonDefaultStreamSmokeFailure("num_layers must be positive")
    if config.kv_heads <= 0:
        raise NonDefaultStreamSmokeFailure("kv_heads must be positive")
    if config.num_q_heads <= 0:
        raise NonDefaultStreamSmokeFailure("num_q_heads must be positive")
    if config.head_dim <= 0 or config.head_dim > 256:
        raise NonDefaultStreamSmokeFailure("head_dim must be in [1, 256]")
    if config.max_batch <= 0:
        raise NonDefaultStreamSmokeFailure("max_batch must be positive")
    if config.max_seq_len <= 0:
        raise NonDefaultStreamSmokeFailure("max_seq_len must be positive")


def run_smoke(config: NonDefaultStreamSmokeConfig) -> dict[str, Any]:
    validate_config(config)
    ensure_kcmm_library(config)

    import torch

    if not torch.cuda.is_available():
        raise NonDefaultStreamSmokeFailure("PyTorch reports CUDA unavailable")
    if config.device_ordinal >= torch.cuda.device_count():
        raise NonDefaultStreamSmokeFailure(
            f"device ordinal {config.device_ordinal} outside available range"
        )

    torch.cuda.set_device(config.device_ordinal)
    device = torch.device("cuda", config.device_ordinal)
    torch.empty(1, device=device)
    torch.cuda.synchronize(config.device_ordinal)

    library = KcmmLibrary(config.kcmm_lib_path)
    pool = library.create_pool(observer_config(config).to_c_config())
    blocks: list[int] = []
    seq_idx: int | None = None
    registered_seq_idx: int | None = None
    started_at = time.monotonic()
    try:
        blocks = pool.alloc_blocks(1)
        seq_idx = pool.register_sequence(blocks)
        registered_seq_idx = seq_idx
        block_id = blocks[0]
        if pool.block_table(seq_idx, 1) != [block_id]:
            raise NonDefaultStreamSmokeFailure(
                "registered block table did not round-trip"
            )

        with torch.cuda.device(config.device_ordinal):
            stream = torch.cuda.Stream()
        default_stream = torch.cuda.default_stream(config.device_ordinal)
        current_stream_before = torch.cuda.current_stream(config.device_ordinal)
        stream_ptr = int(stream.cuda_stream)
        default_stream_ptr = int(default_stream.cuda_stream)
        current_stream_ptr_before = int(current_stream_before.cuda_stream)
        if stream_ptr == 0:
            raise NonDefaultStreamSmokeFailure(
                "torch.cuda.Stream() returned legacy default stream handle 0"
            )
        if stream_ptr == default_stream_ptr:
            raise NonDefaultStreamSmokeFailure(
                "torch.cuda.Stream() matched the default stream handle"
            )

        step_elements = config.kv_heads * config.head_dim
        byte_count = step_elements * 2
        slot = block_id * config.block_size
        offsets_f16 = pool.all_block_offsets_f16(min_entries=block_id + 1)
        if block_id >= len(offsets_f16):
            raise NonDefaultStreamSmokeFailure(
                f"missing offset table entry for block {block_id}"
            )

        with torch.cuda.stream(stream):
            k_src = (
                torch.arange(step_elements, dtype=torch.float16, device=device) + 100
            ).reshape(1, step_elements)
            v_src = (
                torch.arange(step_elements, dtype=torch.float16, device=device) + 1000
            ).reshape(1, step_elements)
            query = torch.zeros(
                (1, config.num_q_heads, config.head_dim),
                dtype=torch.float16,
                device=device,
            )
            out = torch.full(
                (1, config.num_q_heads, config.head_dim),
                -1,
                dtype=torch.float16,
                device=device,
            )
            block_tables = torch.tensor(
                [[block_id]],
                dtype=torch.int32,
                device=device,
            )
            seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
            block_offsets = torch.tensor(
                offsets_f16,
                dtype=torch.int64,
                device=device,
            )

            pool.append_kv_slots(
                layer_idx=0,
                slot_mapping=[slot],
                k_src_ptr=int(k_src.data_ptr()),
                v_src_ptr=int(v_src.data_ptr()),
                stream_ptr=stream_ptr,
            )
            pool.paged_attn_decode_f16(
                layer_idx=0,
                query_ptr=int(query.data_ptr()),
                out_ptr=int(out.data_ptr()),
                block_tables_ptr=int(block_tables.data_ptr()),
                seq_lens_ptr=int(seq_lens.data_ptr()),
                block_offsets_f16_ptr=int(block_offsets.data_ptr()),
                batch=1,
                num_q_heads=config.num_q_heads,
                kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                block_size=config.block_size,
                max_blocks_per_seq=1,
                scale=1.0,
                stream_ptr=stream_ptr,
            )

        current_stream_ptr_after_context = int(
            torch.cuda.current_stream(config.device_ordinal).cuda_stream
        )
        stream.synchronize()

        driver = CudaDriver()
        block_offset = pool.block_va_offset(block_id)
        va_k = pool.va_k(0)
        va_v = pool.va_v(0)
        k_addr = va_k + block_offset
        v_addr = va_v + block_offset
        expected_k = tensor_bytes(k_src[0])
        expected_v = tensor_bytes(v_src[0])
        actual_k = driver.memcpy_dtoh(k_addr, byte_count)
        actual_v = driver.memcpy_dtoh(v_addr, byte_count)
        assert_bytes_equal("k[slot=0]", actual_k, expected_k)
        assert_bytes_equal("v[slot=0]", actual_v, expected_v)

        expected_out = expected_decode_output(
            v_src,
            num_q_heads=config.num_q_heads,
            kv_heads=config.kv_heads,
        ).cpu()
        actual_out = out.detach().cpu()
        if not torch.equal(actual_out, expected_out):
            max_abs_diff = (actual_out - expected_out).abs().max().item()
            raise NonDefaultStreamSmokeFailure(
                "decode output mismatch after non-default-stream write/read: "
                f"max_abs_diff={max_abs_diff}"
            )

        stats_before_unregister = pool.stats()
        pool.unregister_sequence(seq_idx)
        seq_idx = None
        pool.synchronize()
        stats_after_unregister = pool.stats()
        if stats_after_unregister.get("blocks_in_use") != 0:
            raise NonDefaultStreamSmokeFailure(
                "KCMM blocks still in use after unregister: "
                + json.dumps(stats_after_unregister, sort_keys=True)
            )

        return {
            "phase": "II.C",
            "gate": "kcmm-non-default-stream-ffi",
            "passed": True,
            "library_path": str(config.kcmm_lib_path),
            "device_ordinal": config.device_ordinal,
            "device_name": torch.cuda.get_device_name(config.device_ordinal),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "shape": {
                "block_size": config.block_size,
                "max_blocks": config.max_blocks,
                "num_layers": config.num_layers,
                "kv_heads": config.kv_heads,
                "num_q_heads": config.num_q_heads,
                "head_dim": config.head_dim,
                "max_batch": config.max_batch,
                "max_seq_len": config.max_seq_len,
                "step_elements": step_elements,
            },
            "stream": {
                "stream_ptr": stream_ptr,
                "default_stream_ptr": default_stream_ptr,
                "current_stream_ptr_before": current_stream_ptr_before,
                "current_stream_ptr_after_context": current_stream_ptr_after_context,
                "non_default": stream_ptr != default_stream_ptr,
                "non_zero": stream_ptr != 0,
            },
            "direct_slot_write": {
                "stream_aware": True,
                "stream_ptr": stream_ptr,
                "slot_formula": "slot = block_id * block_size + offset_in_block",
                "slot": slot,
                "block": block_id,
                "offset_in_block": 0,
                "byte_count_per_kv_row": byte_count,
                "k_addr": k_addr,
                "v_addr": v_addr,
                "verified_rows": 1,
            },
            "gpu_read": {
                "stream_aware": True,
                "stream_ptr": stream_ptr,
                "read_path": "kcmm_paged_attn_decode_f16_on_stream",
                "batch": 1,
                "seq_len": 1,
                "block_tables": [[block_id]],
                "seq_lens": [1],
                "expected_output": expected_out.flatten().tolist(),
                "actual_output": actual_out.flatten().tolist(),
            },
            "synchronization": {
                "preflight_device_synchronize": True,
                "verification_stream_synchronize": True,
                "post_cleanup_pool_synchronize": True,
                "device_synchronize_after_stream_work": False,
            },
            "allocated_blocks": blocks,
            "registered_seq_idx": registered_seq_idx,
            "stats_before_unregister": stats_before_unregister,
            "stats_after_unregister": stats_after_unregister,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }
    finally:
        try:
            if seq_idx is not None:
                pool.unregister_sequence(seq_idx)
            elif blocks:
                stats = pool.stats()
                if stats.get("blocks_in_use"):
                    pool.free_blocks(blocks)
            pool.synchronize()
        finally:
            pool.destroy()


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    try:
        report = run_smoke(config)
    except (KcmmError, NonDefaultStreamSmokeFailure) as exc:
        print(f"KCMM non-default stream FFI smoke failed: {exc}", file=sys.stderr)
        return 1
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
