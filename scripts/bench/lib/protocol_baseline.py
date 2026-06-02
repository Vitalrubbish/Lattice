"""protocol_baseline.py — TCP JSON-lines client for the baseline inference server.

Extracted from scripts/bench_baseline_comprehensive.py.
"""

import asyncio
import json
import random
import socket
import sys
import time
from typing import Dict, Optional


def send_infer_baseline(host: str, port: int, prompt_len: int,
                        max_new_tokens: int, eos_token_id: int = 1_000_000,
                        timeout: float = 300.0) -> dict:
    """Send an inference request to the baseline TCP server.

    Uses the baseline's line-delimited JSON TCP protocol.
    Sets eos_token_id to a high value to effectively disable EOS (ignore_eos),
    matching the vLLM benchmark's ignore_eos=True for fair capacity comparison.

    Returns:
        dict with keys: prompt_tokens, completion_tokens, total_ms, success, [error]
    """
    req = {
        "id": random.randint(0, 2**63),
        "prompt_tokens": [1] * max(1, prompt_len),
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
    }
    body = json.dumps(req) + "\n"

    t0 = time.time()
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.sendall(body.encode())

        # Read response (line-delimited JSON)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        sock.close()

        line = buf.split(b"\n")[0].decode()
        resp = json.loads(line)
        elapsed_ms = (time.time() - t0) * 1000

        generated_tokens = resp.get("generated_tokens", [])
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(generated_tokens),
            "total_ms": elapsed_ms,
            "success": True,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


async def send_infer_baseline_async(host: str, port: int, prompt_len: int,
                                    max_new_tokens: int,
                                    eos_token_id: int = 1_000_000,
                                    timeout: float = 300.0) -> dict:
    """Async version of send_infer_baseline using asyncio streams.

    Uses asyncio TCP connection instead of blocking socket I/O, enabling true
    concurrency without thread-pool bottlenecks.

    See send_infer_baseline for full parameter documentation.
    """
    req = {
        "id": random.randint(0, 2**63),
        "prompt_tokens": [1] * max(1, prompt_len),
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
    }
    body = json.dumps(req) + "\n"

    t0 = time.time()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.write(body.encode())
        await writer.drain()

        # Read response (line-delimited JSON)
        buf = await asyncio.wait_for(reader.readuntil(b"\n"), timeout=timeout)
        writer.close()
        await writer.wait_closed()

        line = buf.decode().rstrip("\n")
        resp = json.loads(line)
        elapsed_ms = (time.time() - t0) * 1000

        generated_tokens = resp.get("generated_tokens", [])
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(generated_tokens),
            "total_ms": elapsed_ms,
            "success": True,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


def query_baseline_stats(host: str, port: int, timeout: float = 5.0) -> Optional[dict]:
    """Query the baseline server for live UFS stats.

    Returns:
        dict with keys: sample_count, internal_frag_rate, block_utilization,
        physical_memory_efficiency, runtime_frag_index, active_sequences,
        blocks_in_use, total_blocks_allocated, total_tokens, etc.
        Returns None on failure.
    """
    req = json.dumps({"type": "stats"}) + "\n"
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.sendall(req.encode())
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        sock.close()
        line = buf.split(b"\n")[0].decode()
        return json.loads(line)
    except Exception as e:
        print(f"   Stats query failed: {e}", file=sys.stderr)
        return None


def wait_for_baseline_server(host: str, port: int, timeout_s: int = 300) -> bool:
    """Wait for the baseline server to accept TCP connections.

    Returns True if server is reachable within timeout_s seconds.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            return True
        except Exception:
            pass
        time.sleep(0.5)
    return False
