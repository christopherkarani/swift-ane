#!/usr/bin/env python3
"""Load Espresso BLOBFILE Llama-family weights into a Torch/HF-compatible shape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BLOBFILE_HEADER_BYTES = 128


@dataclass(frozen=True)
class EspressoLlamaMetadata:
    name: str
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    head_dim: int
    hidden_dim: int
    vocab: int
    max_seq: int
    norm_eps: float
    rope_theta: float
    eos_token: int | None


def load_espresso_metadata(weights_dir: Path) -> EspressoLlamaMetadata:
    payload = json.loads((weights_dir / "metadata.json").read_text(encoding="utf-8"))
    return EspressoLlamaMetadata(
        name=payload["name"],
        n_layer=int(payload["nLayer"]),
        n_head=int(payload["nHead"]),
        n_kv_head=int(payload.get("nKVHead", payload["nHead"])),
        d_model=int(payload["dModel"]),
        head_dim=int(payload["headDim"]),
        hidden_dim=int(payload["hiddenDim"]),
        vocab=int(payload["vocab"]),
        max_seq=int(payload["maxSeq"]),
        norm_eps=float(payload["normEps"]),
        rope_theta=float(payload.get("ropeTheta", 10_000.0)),
        eos_token=(int(payload["eosToken"]) if "eosToken" in payload else None),
    )


def read_blobfile_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    with path.open("rb") as handle:
        handle.seek(BLOBFILE_HEADER_BYTES)
        array = np.fromfile(handle, dtype=np.float16, count=count)
        trailing = handle.read(1)
    if array.size != count:
        raise ValueError(f"truncated BLOBFILE payload at {path}")
    if trailing:
        raise ValueError(f"unexpected trailing payload at {path}")
    return array.reshape(shape)


def load_espresso_llama_state_dict(
    weights_dir: Path,
    metadata: EspressoLlamaMetadata | None = None,
) -> dict[str, np.ndarray]:
    weights_dir = weights_dir.expanduser().resolve()
    metadata = metadata or load_espresso_metadata(weights_dir)

    state_dict: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": read_blobfile_array(
            weights_dir / "embeddings" / "token.bin",
            (metadata.vocab, metadata.d_model),
        ),
        "model.norm.weight": read_blobfile_array(
            weights_dir / "final_norm.bin",
            (metadata.d_model,),
        ),
        "lm_head.weight": read_blobfile_array(
            weights_dir / "lm_head.bin",
            (metadata.vocab, metadata.d_model),
        ),
    }

    for layer_index in range(metadata.n_layer):
        layer_dir = weights_dir / "layers" / str(layer_index)
        prefix = f"model.layers.{layer_index}"
        state_dict[f"{prefix}.input_layernorm.weight"] = read_blobfile_array(
            layer_dir / "rms_att.bin",
            (metadata.d_model,),
        )
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = read_blobfile_array(
            layer_dir / "rms_ffn.bin",
            (metadata.d_model,),
        )
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = read_blobfile_array(
            layer_dir / "wq.bin",
            (metadata.d_model, metadata.d_model),
        )
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = read_blobfile_array(
            layer_dir / "wk.bin",
            (metadata.d_model, metadata.d_model),
        )
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = read_blobfile_array(
            layer_dir / "wv.bin",
            (metadata.d_model, metadata.d_model),
        )
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = read_blobfile_array(
            layer_dir / "wo.bin",
            (metadata.d_model, metadata.d_model),
        )
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = read_blobfile_array(
            layer_dir / "w1.bin",
            (metadata.hidden_dim, metadata.d_model),
        )
        state_dict[f"{prefix}.mlp.down_proj.weight"] = read_blobfile_array(
            layer_dir / "w2.bin",
            (metadata.d_model, metadata.hidden_dim),
        )
        state_dict[f"{prefix}.mlp.up_proj.weight"] = read_blobfile_array(
            layer_dir / "w3.bin",
            (metadata.hidden_dim, metadata.d_model),
        )

    return state_dict


def llama_config_kwargs_from_metadata(metadata: EspressoLlamaMetadata) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "hidden_size": metadata.d_model,
        "intermediate_size": metadata.hidden_dim,
        "num_hidden_layers": metadata.n_layer,
        "num_attention_heads": metadata.n_head,
        "num_key_value_heads": metadata.n_kv_head,
        "vocab_size": metadata.vocab,
        "max_position_embeddings": metadata.max_seq,
        "rms_norm_eps": metadata.norm_eps,
        "rope_theta": metadata.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
    }
    if metadata.eos_token is not None:
        kwargs["eos_token_id"] = metadata.eos_token
    return kwargs


def load_espresso_llama_for_causal_lm(weights_dir: Path, torch_dtype=None):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    metadata = load_espresso_metadata(weights_dir)
    model = LlamaForCausalLM(LlamaConfig(**llama_config_kwargs_from_metadata(metadata)))
    state_dict = load_espresso_llama_state_dict(weights_dir, metadata)
    torch_state_dict = {
        name: torch.from_numpy(array.astype(np.float32, copy=False)).to(dtype=torch_dtype or torch.float32)
        for name, array in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(torch_state_dict, strict=False)
    if missing:
        raise ValueError(f"Missing Llama tensors for Espresso weights: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected Espresso tensors for Llama model: {unexpected}")
    model.eval()
    return metadata, model
