"""Phase II.B preflight smoke for KCMM KV write FFI."""

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


class KvWriteSmokeFailure(RuntimeError):
    """Raised when the KCMM KV write FFI smoke cannot complete."""


@dataclass(frozen=True)
class KvWriteSmokeConfig:
    kcmm_lib_path: Path
    build_kcmm: bool
    device_ordinal: int
    block_size: int
    max_blocks: int
    num_layers: int
    kv_heads: int
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
            raise KvWriteSmokeFailure(f"{operation} failed with CUDA rc={rc}")

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
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--cpu-cache-path", default="/dev/shm/kcmm_kv_write_smoke")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> KvWriteSmokeConfig:
    args = build_parser().parse_args(argv)
    return KvWriteSmokeConfig(
        kcmm_lib_path=resolve_repo_path(args.kcmm_lib_path),
        build_kcmm=args.build_kcmm,
        device_ordinal=args.device_ordinal,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        num_layers=args.num_layers,
        kv_heads=args.kv_heads,
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
        raise KvWriteSmokeFailure(
            f"{description} failed with exit code {result.returncode}"
        )


def ensure_kcmm_library(config: KvWriteSmokeConfig) -> None:
    if config.build_kcmm or not config.kcmm_lib_path.exists():
        run_checked(["cargo", "build", "--features", "kcmm"], "build KCMM")
    if not config.kcmm_lib_path.exists():
        raise KvWriteSmokeFailure(f"KCMM shared library not found: {config.kcmm_lib_path}")


def tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def assert_bytes_equal(name: str, actual: bytes, expected: bytes) -> None:
    if actual == expected:
        return
    mismatch_at = next(
        (
            index
            for index, (left, right) in enumerate(zip(actual, expected, strict=False))
            if left != right
        ),
        min(len(actual), len(expected)),
    )
    raise KvWriteSmokeFailure(
        f"{name} byte mismatch at offset {mismatch_at}: "
        f"actual_len={len(actual)} expected_len={len(expected)}"
    )


def observer_config(config: KvWriteSmokeConfig) -> ObserverConfig:
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


def run_smoke(config: KvWriteSmokeConfig) -> dict[str, Any]:
    ensure_kcmm_library(config)

    import torch

    if not torch.cuda.is_available():
        raise KvWriteSmokeFailure("PyTorch reports CUDA unavailable")
    if config.device_ordinal >= torch.cuda.device_count():
        raise KvWriteSmokeFailure(
            f"device ordinal {config.device_ordinal} outside available range"
        )

    torch.cuda.set_device(config.device_ordinal)
    torch.empty(1, device=f"cuda:{config.device_ordinal}")
    torch.cuda.synchronize(config.device_ordinal)

    library = KcmmLibrary(config.kcmm_lib_path)
    pool = library.create_pool(observer_config(config).to_c_config())
    blocks: list[int] = []
    seq_idx: int | None = None
    registered_seq_idx: int | None = None
    started_at = time.monotonic()
    try:
        blocks = pool.alloc_blocks(2)
        seq_idx = pool.register_sequence(blocks)
        registered_seq_idx = seq_idx
        if pool.block_table(seq_idx, 4) != blocks:
            raise KvWriteSmokeFailure("registered block table did not round-trip")
        offsets = pool.block_table_va_offsets(seq_idx, 4)
        if len(offsets) != len(blocks):
            raise KvWriteSmokeFailure("registered block VA offsets did not round-trip")

        batch = 2
        step = config.kv_heads * config.head_dim
        byte_count = step * 2
        device = f"cuda:{config.device_ordinal}"
        k_src = torch.arange(batch * step, dtype=torch.float16, device=device).reshape(
            batch,
            step,
        )
        v_src = (torch.arange(batch * step, dtype=torch.float16, device=device) + 1000).reshape(
            batch,
            step,
        )
        positions = [0, config.block_size + 1]
        pool.append_kv_step(
            layer_idx=0,
            seq_indices=[seq_idx, seq_idx],
            positions=positions,
            k_src_ptr=int(k_src.data_ptr()),
            v_src_ptr=int(v_src.data_ptr()),
        )
        pool.synchronize()
        torch.cuda.synchronize(config.device_ordinal)

        driver = CudaDriver()
        va_k = pool.va_k(0)
        va_v = pool.va_v(0)
        comparisons: list[dict[str, Any]] = []
        for row, position in enumerate(positions):
            block = blocks[position // config.block_size]
            offset_in_block = position % config.block_size
            block_offset = pool.block_va_offset(block)
            token_offset_bytes = offset_in_block * step * 2
            k_addr = va_k + block_offset + token_offset_bytes
            v_addr = va_v + block_offset + token_offset_bytes
            expected_k = tensor_bytes(k_src[row])
            expected_v = tensor_bytes(v_src[row])
            actual_k = driver.memcpy_dtoh(k_addr, byte_count)
            actual_v = driver.memcpy_dtoh(v_addr, byte_count)
            assert_bytes_equal(f"k[row={row}]", actual_k, expected_k)
            assert_bytes_equal(f"v[row={row}]", actual_v, expected_v)
            comparisons.append(
                {
                    "row": row,
                    "seq_idx": seq_idx,
                    "position": position,
                    "block": block,
                    "offset_in_block": offset_in_block,
                    "byte_count": byte_count,
                    "k_addr": k_addr,
                    "v_addr": v_addr,
                }
            )

        pool.unregister_sequence(seq_idx)
        seq_idx = None
        pool.synchronize()
        stats_after_unregister = pool.stats()
        if stats_after_unregister.get("blocks_in_use") != 0:
            raise KvWriteSmokeFailure(
                "KCMM blocks still in use after unregister: "
                + json.dumps(stats_after_unregister, sort_keys=True)
            )

        return {
            "phase": "II.B",
            "gate": "kcmm-kv-write-ffi",
            "passed": True,
            "library_path": str(config.kcmm_lib_path),
            "device_ordinal": config.device_ordinal,
            "shape": {
                "block_size": config.block_size,
                "max_blocks": config.max_blocks,
                "num_layers": config.num_layers,
                "kv_heads": config.kv_heads,
                "head_dim": config.head_dim,
                "max_batch": config.max_batch,
                "max_seq_len": config.max_seq_len,
                "step_elements": step,
            },
            "allocated_blocks": blocks,
            "registered_seq_idx": registered_seq_idx,
            "comparisons": comparisons,
            "stats_after_unregister": stats_after_unregister,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }
    finally:
        try:
            if seq_idx is not None:
                pool.unregister_sequence(seq_idx)
            elif blocks:
                # unregister_sequence owns normal cleanup after registration.
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
    except (KcmmError, KvWriteSmokeFailure) as exc:
        print(f"KCMM KV write FFI smoke failed: {exc}", file=sys.stderr)
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
