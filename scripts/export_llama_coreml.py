#!/usr/bin/env python3
"""Export a Hugging Face Llama-family Core ML model.

By default the exporter writes the existing stateless fixed-sequence trunk that
returns hidden states before the final RMSNorm / LM head.

With `--stateful`, the exporter writes a single-token decode-step model whose
KV caches live in Core ML `MLState`. The resulting package accepts:

- `input_ids`: shape `[1, 1]`, dtype `int32`
- `cache_position`: shape `[1]`, dtype `int32`

and returns:

- `hidden_states`: shape `[1, 1, hidden]`, dtype `float16`

The caller is responsible for incrementing `cache_position` on each step while
reusing the same `MLState` instance across predictions.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import coremltools as ct
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from espresso_llama_weights import load_espresso_llama_for_causal_lm


def configure_eager_llama(hf_model: AutoModelForCausalLM) -> None:
    hf_model.model.config._attn_implementation = "eager"
    for layer in hf_model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"


class LlamaTrunkBeforeNorm(torch.nn.Module):
    def __init__(self, hf_model: AutoModelForCausalLM, seq_len: int):
        super().__init__()
        self.model = hf_model.model
        configure_eager_llama(hf_model)
        dtype = self.model.embed_tokens.weight.dtype
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        mask = torch.full(
            (1, 1, seq_len, seq_len),
            torch.finfo(dtype).min,
            dtype=dtype,
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("position_ids", position_ids, persistent=False)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.long()
        hidden_states = self.model.embed_tokens(input_ids)
        position_ids = self.position_ids.to(hidden_states.device)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        mask = self.causal_mask.to(hidden_states.device, dtype=hidden_states.dtype)

        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )[0]

        return hidden_states


class StatefulLlamaDecodeStepBeforeNorm(torch.nn.Module):
    def __init__(self, hf_model: AutoModelForCausalLM, max_cache_len: int):
        super().__init__()
        configure_eager_llama(hf_model)
        self.model = hf_model.model
        self.layers = self.model.layers
        self.embed_tokens = self.model.embed_tokens
        self.rotary_emb = self.model.rotary_emb
        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.max_cache_len = max_cache_len
        self.dtype = self.embed_tokens.weight.dtype

        self.register_buffer(
            "mask_positions",
            torch.arange(max_cache_len, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "mask_fill",
            torch.tensor(torch.finfo(self.dtype).min, dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "layer_positions",
            torch.arange(len(self.layers), dtype=torch.long),
            persistent=False,
        )

        cache_shape = (len(self.layers), self.n_kv_heads, max_cache_len, self.head_dim)
        self.register_buffer("key_cache", torch.zeros(cache_shape, dtype=self.dtype))
        self.register_buffer("value_cache", torch.zeros(cache_shape, dtype=self.dtype))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    def _update_cache(
        self,
        idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_slot = (self.layer_positions == idx).to(self.dtype).view(len(self.layers), 1, 1, 1)
        position_slot = (self.mask_positions.unsqueeze(0) == cache_position.reshape(-1, 1)).to(self.dtype)
        position_slot = position_slot.unsqueeze(1).unsqueeze(-1)
        write_slot = layer_slot * position_slot
        keep = 1.0 - write_slot
        key_update = k.squeeze(0).unsqueeze(0).to(self.dtype)
        value_update = v.squeeze(0).unsqueeze(0).to(self.dtype)
        next_key_cache = (self.key_cache * keep) + (key_update * write_slot)
        next_value_cache = (self.value_cache * keep) + (value_update * write_slot)

        # Whole-buffer state updates convert cleanly to Core ML state ops.
        self.key_cache.mul_(0)
        self.key_cache.add_(next_key_cache)
        self.value_cache.mul_(0)
        self.value_cache.add_(next_value_cache)
        return next_key_cache[idx : idx + 1], next_value_cache[idx : idx + 1]

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.long()
        cache_position = cache_position.long()
        attention_mask = self.mask_positions.unsqueeze(0) > cache_position.reshape(-1, 1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).to(self.mask_fill.dtype) * self.mask_fill
        position_ids = cache_position.unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q = attn.q_proj(hidden_states).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rotary(q, k, cos, sin)

            full_k, full_v = self._update_cache(idx, k, v, cache_position)
            full_k = self._repeat_kv(full_k)
            full_v = self._repeat_kv(full_v)

            attn_weights = torch.matmul(q, full_k.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, full_v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(1, 1, -1)
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model id or local path")
    source.add_argument(
        "--espresso-weights",
        help="Espresso Llama weights directory containing metadata.json and BLOBFILE tensors",
    )
    parser.add_argument("--seq-len", required=True, type=int, help="Fixed sequence length or stateful cache length")
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument(
        "--minimum-target",
        default="macOS15",
        choices=["macOS15"],
        help="Minimum deployment target for the converted model",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        help="Export a single-token decode-step model with KV caches stored in MLState.",
    )
    return parser.parse_args()


def minimum_target(name: str):
    if name != "macOS15":
        raise ValueError(f"Unsupported minimum target: {name}")
    return ct.target.macOS15


def build_state_specs(module: torch.nn.Module) -> list[ct.StateType]:
    states: list[ct.StateType] = []
    for name, buffer in module.named_buffers():
        if not (name.startswith("key_cache") or name.startswith("value_cache")):
            continue
        if buffer.dtype != torch.float16:
            raise ValueError(f"State buffer {name} must be float16, got {buffer.dtype}")
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(buffer.shape), dtype=np.float16),
                name=name,
            )
        )
    if not states:
        raise ValueError("Stateful export produced no KV state buffers.")
    return states


def main() -> None:
    args = parse_args()
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be > 0")

    if args.espresso_weights:
        _, model = load_espresso_llama_for_causal_lm(
            pathlib.Path(args.espresso_weights).expanduser(),
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
        model.eval()

    if args.stateful:
        export_module = StatefulLlamaDecodeStepBeforeNorm(model, args.seq_len)
        example_inputs = (
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
        )
        input_specs = [
            ct.TensorType(name="input_ids", shape=example_inputs[0].shape, dtype=example_inputs[0].numpy().dtype),
            ct.TensorType(name="cache_position", shape=example_inputs[1].shape, dtype=example_inputs[1].numpy().dtype),
        ]
        outputs = [ct.TensorType(name="hidden_states", dtype=np.float16)]
        with torch.no_grad():
            traced = torch.jit.trace(export_module.eval(), example_inputs, strict=False)
        states = build_state_specs(traced)
        mlmodel = ct.convert(
            traced,
            inputs=input_specs,
            outputs=outputs,
            states=states,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=minimum_target(args.minimum_target),
        )
    else:
        trunk = LlamaTrunkBeforeNorm(model, args.seq_len)
        trunk.eval()
        example = torch.zeros((1, args.seq_len), dtype=torch.int32)

        with torch.no_grad():
            traced = torch.jit.trace(trunk, example, strict=False)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=example.shape, dtype=example.numpy().dtype)],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=minimum_target(args.minimum_target),
        )

    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(output_path)


if __name__ == "__main__":
    main()
