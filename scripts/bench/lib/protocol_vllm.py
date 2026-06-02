"""protocol_vllm.py — HTTP client for vLLM OpenAI-compatible API.

Extracted from scripts/bench_vllm_comprehensive.py and scripts/bench_vllm.py.

Fixes applied during extraction:
  - Added import re (was missing, caused NameError in log parsing)
  - Added send_completion_vllm() with prompt_token_ids support for fair comparison
"""

import asyncio
import http.client
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional


def send_completion_vllm(port: int, model: str, max_tokens: int,
                         prompt_text: str = "",
                         prompt_token_ids: Optional[List[int]] = None,
                         ignore_eos: bool = False,
                         timeout: int = 300) -> dict:
    """Send one completion request to vLLM; return timings + token counts.

    When ignore_eos=True, the model is instructed not to stop on EOS,
    guaranteeing max_tokens tokens are generated for fair capacity benchmarks.

    For fair comparison with baseline (which sends exact token IDs), use
    prompt_token_ids to pass exact token counts. When provided, prompt_text is
    ignored (OpenAI API uses token_ids over text when both are given).

    Args:
        port: vLLM server port
        model: model name
        max_tokens: max tokens to generate
        prompt_text: text prompt (ignored if prompt_token_ids is provided)
        prompt_token_ids: exact token IDs (produces identical KV cache usage to baseline)
        ignore_eos: if True, ignore EOS token and generate all max_tokens
        timeout: request timeout in seconds

    Returns:
        dict with keys: prompt_tokens, completion_tokens, total_ms, success, [error]
    """
    req_body: dict = {
        "model": model,
        "max_tokens": max_tokens,
    }

    # For fair comparison: use exact token IDs when provided
    if prompt_token_ids is not None:
        req_body["prompt_token_ids"] = prompt_token_ids
    elif prompt_text:
        req_body["prompt"] = prompt_text
    else:
        req_body["prompt"] = "Hello"  # fallback

    if ignore_eos:
        req_body["ignore_eos"] = True

    body = json.dumps(req_body)

    t0 = time.time()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("POST", "/v1/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        elapsed_ms = (time.time() - t0) * 1000

        usage = data.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_ms": elapsed_ms,
            "success": resp.status == 200,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        prompt_len = len(prompt_token_ids) if prompt_token_ids else len(prompt_text.split())
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }



async def send_completion_vllm_async(port: int, model: str, max_tokens: int,
                                     prompt_text: str = "",
                                     prompt_token_ids: Optional[List[int]] = None,
                                     ignore_eos: bool = False,
                                     timeout: int = 300,
                                     session = None) -> dict:
    """Async version of send_completion_vllm using aiohttp.

    Accepts an optional aiohttp.ClientSession to reuse connections across
    concurrent requests. When session is None, a new session is created per call
    (simpler but less efficient for high concurrency).

    See send_completion_vllm for full parameter documentation.
    """
    import aiohttp

    req_body: dict = {
        "model": model,
        "max_tokens": max_tokens,
    }

    if prompt_token_ids is not None:
        req_body["prompt_token_ids"] = prompt_token_ids
    elif prompt_text:
        req_body["prompt"] = prompt_text
    else:
        req_body["prompt"] = "Hello"

    if ignore_eos:
        req_body["ignore_eos"] = True

    url = f"http://127.0.0.1:{port}/v1/completions"
    t0 = time.time()

    try:
        if session is not None:
            async with session.post(url, json=req_body,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                data = await resp.json()
                elapsed_ms = (time.time() - t0) * 1000
                usage = data.get("usage", {})
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_ms": elapsed_ms,
                    "success": resp.status == 200,
                }
        else:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as s:
                async with s.post(url, json=req_body) as resp:
                    data = await resp.json()
                    elapsed_ms = (time.time() - t0) * 1000
                    usage = data.get("usage", {})
                    return {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_ms": elapsed_ms,
                        "success": resp.status == 200,
                    }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        prompt_len = len(prompt_token_ids) if prompt_token_ids else len(prompt_text.split())
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_ms": elapsed_ms,
            "success": False,
            "error": str(e),
        }


def wait_for_vllm_server(port: int, timeout_s: int = 300) -> bool:
    """Wait for vLLM server to accept HTTP connections.

    Returns True if server is reachable within timeout_s seconds.
    """
    for _ in range(timeout_s * 2):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def warmup_vllm(port: int, model: str):
    """Send a warmup request to trigger JIT compilation in vLLM."""
    try:
        send_completion_vllm(port, model, max_tokens=4,
                             prompt_text="Hello " * 8, timeout=120)
        print("   Warmup complete.", file=sys.stderr)
    except Exception as e:
        print(f"   Warmup warning: {e}", file=sys.stderr)


def get_vllm_num_gpu_blocks(port: Optional[int] = None,
                            server_log_path: Optional[str] = None,
                            block_size: int = 16) -> int:
    """Determine vLLM's pre-allocated block pool size.

    Tries in order:
      1. Parse server log for 'GPU KV cache size: N tokens'
      2. Query /metrics for vllm:num_gpu_blocks
      3. Return 0 if both fail (caller should fall back to estimation)

    Args:
        port: vLLM server port (for /metrics query)
        server_log_path: path to vLLM server log file
        block_size: block size in tokens (default 16)

    Returns:
        num_gpu_blocks, or 0 if unable to determine
    """
    # Method 1: parse server log
    if server_log_path:
        for attempt in range(5):
            try:
                if not os.path.exists(server_log_path):
                    if attempt < 4:
                        time.sleep(1.0)
                        continue
                    break
                with open(server_log_path, 'r', errors='replace') as f:
                    for line in f:
                        m = re.search(r'GPU KV cache size:\s*([0-9,]+)\s+tokens', line)
                        if m:
                            tokens = int(m.group(1).replace(',', ''))
                            return tokens // block_size
            except Exception as e:
                if attempt == 4:
                    print(f"   (log parse: {e})", file=sys.stderr)
            if attempt < 4:
                time.sleep(1.0)

    # Method 2: /metrics endpoint
    if port:
        try:
            import urllib.request
            url = f"http://127.0.0.1:{port}/metrics"
            resp = urllib.request.urlopen(url, timeout=5)
            body = resp.read().decode()
            m = re.search(r'vllm:num_gpu_blocks\S*\s+(\d+)', body)
            if m:
                return int(m.group(1))
        except Exception:
            pass

    return 0


def get_gpu_memory() -> dict:
    """Get GPU memory via nvidia-smi.

    Returns dict with total_mib and used_mib, or empty dict on failure.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5
        ).decode().strip()
        total, used = out.split(",")
        return {"total_mib": float(total.strip()), "used_mib": float(used.strip())}
    except Exception:
        return {}
