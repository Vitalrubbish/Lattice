"""Create a local tiny OPT model for KCMM/vLLM smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM


DEFAULT_BASE_TOKENIZER = "hf-internal-testing/tiny-random-OPTForCausalLM"
DEFAULT_OUTPUT = ".scratch/kcmm-vllm/tiny-opt-head64"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--base-tokenizer", default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--max-position-embeddings", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.hidden_size % args.num_heads != 0:
        raise ValueError("hidden-size must be divisible by num-heads")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer,
        cache_dir=args.cache_dir,
    )
    config = OPTConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        word_embed_proj_dim=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_position_embeddings=args.max_position_embeddings,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="relu",
        do_layer_norm_before=True,
        layerdrop=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    model = OPTForCausalLM(config)
    model.eval()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    print(f"output={output.resolve()}")
    print(f"params={sum(param.numel() for param in model.parameters())}")
    print(f"head_dim={config.hidden_size // config.num_attention_heads}")
    print(f"max_position_embeddings={config.max_position_embeddings}")
    print(f"seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
