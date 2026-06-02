"""workload.py — Shared prompt distributions, model parameters, and workload configs."""

# ── Prompt-length distribution (145 samples, sonnet-derived) ──
# Used by throughput and fragmentation benchmarks.
# Mirrors examples/bench_throughput.rs:29-40
SONNET_PROMPT_LENS = [
    8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13,
    39, 39, 40, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42,
    42, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45,
    46, 46, 46, 46, 46, 47, 47, 48, 48, 50, 72, 72, 73, 73, 73,
    74, 74, 75, 75, 76, 76, 77, 77, 77, 78, 79, 80, 80, 80, 80,
    106, 122, 126, 128, 135, 145, 146, 152, 152, 152, 153, 155, 155, 156, 157,
    160, 162, 170, 239, 251, 263, 273, 288, 289, 289,
]

# ── Short prompts for max_concurrency tests ──
# Maximises admitted sequences by using minimal prompt length.
SHORT_PROMPT_LENS = [8, 16, 32]

# ── TinyLlama-1.1B model parameters ──
# Must match the model being served.
# Mirrors src/cache/unified_frag.rs and bench_vllm_comprehensive.py:61-68
TINYLLAMA_PARAMS = {
    "kv_heads": 4,
    "head_dim": 64,
    "num_layers": 22,
    "block_size": 16,
}
# sizeof(f16) = 2 bytes
# block_bytes = kv_heads × head_dim × block_size × sizeof(f16)
TINYLLAMA_PARAMS["block_bytes"] = (
    TINYLLAMA_PARAMS["kv_heads"]
    * TINYLLAMA_PARAMS["head_dim"]
    * TINYLLAMA_PARAMS["block_size"]
    * 2
)

# ── Default workload parameters ──
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_NUM_REQUESTS = 100
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 300  # seconds per request
DEFAULT_TIME_BUDGET = 600  # seconds for max_concurrency ramp
DEFAULT_EOS_TOKEN_ID = 1_000_000  # effectively disables EOS
