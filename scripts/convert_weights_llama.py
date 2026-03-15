#!/usr/bin/env python3
"""Convert HuggingFace Llama-family weights to Espresso BLOBFILE layout.

Requires:
    pip install torch transformers

Usage:
    ./scripts/convert_weights_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output /tmp/tinyllama
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def make_blob_header(data_size: int) -> bytes:
    header = bytearray(128)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into("<I", header, 72, data_size)
    struct.pack_into("<I", header, 80, 128)
    return bytes(header)


def write_blob(tensor: torch.Tensor, path: Path) -> None:
    payload = tensor.detach().cpu().float().to(torch.float16).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(make_blob_header(len(payload)))
        handle.write(payload)


def expand_gqa_projection(weight: torch.Tensor, n_head: int, n_kv_head: int, head_dim: int) -> torch.Tensor:
    if n_kv_head == n_head:
        return weight
    if n_head % n_kv_head != 0:
        raise ValueError(f"Cannot expand GQA weights: n_head={n_head} n_kv_head={n_kv_head}")

    repeats = n_head // n_kv_head
    in_dim = weight.shape[1]
    expanded = weight.reshape(n_kv_head, head_dim, in_dim).repeat_interleave(repeats, dim=0)
    return expanded.reshape(n_head * head_dim, in_dim).contiguous()


def write_causal_masks(output_dir: Path, max_seq: int) -> None:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    size = 1
    while size <= max_seq:
        mask = torch.full((size, size), 0.0, dtype=torch.float16)
        mask = torch.triu(mask.fill_(-1e4), diagonal=1) + torch.tril(torch.zeros_like(mask))
        payload = mask.numpy().tobytes()
        with (mask_dir / f"causal_{size}.bin").open("wb") as handle:
            handle.write(make_blob_header(len(payload)))
            handle.write(payload)
        size *= 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    state = model.state_dict()
    config = model.config

    hidden_dim = getattr(config, "intermediate_size")
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    n_kv_head = getattr(config, "num_key_value_heads", config.num_attention_heads)

    metadata = {
        "name": getattr(config, "_name_or_path", args.model).split("/")[-1],
        "nLayer": config.num_hidden_layers,
        "nHead": config.num_attention_heads,
        "nKVHead": n_kv_head,
        "dModel": config.hidden_size,
        "headDim": head_dim,
        "hiddenDim": hidden_dim,
        "vocab": config.vocab_size,
        "maxSeq": config.max_position_embeddings,
        "normEps": config.rms_norm_eps,
        "architecture": "llama",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    write_blob(state["model.embed_tokens.weight"], output_dir / "embeddings" / "token.bin")
    write_blob(state["model.norm.weight"], output_dir / "final_norm.bin")
    lm_head = state["lm_head.weight"] if "lm_head.weight" in state else state["model.embed_tokens.weight"]
    write_blob(lm_head, output_dir / "lm_head.bin")

    for layer in range(config.num_hidden_layers):
        layer_dir = output_dir / "layers" / str(layer)
        prefix = f"model.layers.{layer}"

        write_blob(state[f"{prefix}.input_layernorm.weight"], layer_dir / "rms_att.bin")
        write_blob(state[f"{prefix}.post_attention_layernorm.weight"], layer_dir / "rms_ffn.bin")
        write_blob(state[f"{prefix}.self_attn.q_proj.weight"], layer_dir / "wq.bin")

        wk = expand_gqa_projection(
            state[f"{prefix}.self_attn.k_proj.weight"],
            n_head=config.num_attention_heads,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )
        wv = expand_gqa_projection(
            state[f"{prefix}.self_attn.v_proj.weight"],
            n_head=config.num_attention_heads,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

        write_blob(wk, layer_dir / "wk.bin")
        write_blob(wv, layer_dir / "wv.bin")
        write_blob(state[f"{prefix}.self_attn.o_proj.weight"], layer_dir / "wo.bin")
        write_blob(state[f"{prefix}.mlp.gate_proj.weight"], layer_dir / "w1.bin")
        write_blob(state[f"{prefix}.mlp.down_proj.weight"], layer_dir / "w2.bin")
        write_blob(state[f"{prefix}.mlp.up_proj.weight"], layer_dir / "w3.bin")

    write_causal_masks(output_dir, config.max_position_embeddings)


if __name__ == "__main__":
    main()
