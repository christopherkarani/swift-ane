#!/usr/bin/env python3
"""Generate a single-layer Core ML transformer baseline for EspressoBench."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


DIM = 768
HIDDEN = 2048
SEQ_LEN = 256
HEADS = 12
HEAD_DIM = DIM // HEADS
WEIGHT_SCALE = 0.02
RNG = np.random.default_rng(1234)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/models/transformer_layer.mlpackage",
        help="Destination .mlpackage path",
    )
    return parser.parse_args()


def random_weight(shape: tuple[int, ...]) -> np.ndarray:
    return (RNG.standard_normal(shape).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)


def const(name: str, value: np.ndarray):
    return mb.const(val=value, name=name)


def rms_norm(x, prefix: str):
    weight = const(f"{prefix}_weight", np.ones((1, DIM, 1, 1), dtype=np.float16))
    squared = mb.mul(x=x, y=x, name=f"{prefix}_squared")
    mean = mb.reduce_mean(x=squared, axes=[1], keep_dims=True, name=f"{prefix}_mean")
    epsilon = const(f"{prefix}_epsilon", np.array([1e-5], dtype=np.float16))
    inv_rms = mb.rsqrt(x=mb.add(x=mean, y=epsilon, name=f"{prefix}_mean_eps"), name=f"{prefix}_inv_rms")
    normalized = mb.mul(x=x, y=inv_rms, name=f"{prefix}_normalized")
    return mb.mul(x=normalized, y=weight, name=f"{prefix}_output")


def transformer_layer(x):
    x_norm = rms_norm(x, "attn_rms")

    wq = const("wq", random_weight((DIM, DIM, 1, 1)))
    wk = const("wk", random_weight((DIM, DIM, 1, 1)))
    wv = const("wv", random_weight((DIM, DIM, 1, 1)))
    wo = const("wo", random_weight((DIM, DIM, 1, 1)))
    w1 = const("w1", random_weight((HIDDEN, DIM, 1, 1)))
    w3 = const("w3", random_weight((HIDDEN, DIM, 1, 1)))
    w2 = const("w2", random_weight((DIM, HIDDEN, 1, 1)))

    q = mb.conv(x=x_norm, weight=wq, name="q_proj")
    k = mb.conv(x=x_norm, weight=wk, name="k_proj")
    v = mb.conv(x=x_norm, weight=wv, name="v_proj")

    q_heads = mb.reshape(x=q, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="q_heads")
    k_heads = mb.reshape(x=k, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="k_heads")
    v_heads = mb.reshape(x=v, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="v_heads")

    q_tokens = mb.transpose(x=q_heads, perm=[0, 1, 3, 2], name="q_tokens")
    v_tokens = mb.transpose(x=v_heads, perm=[0, 1, 3, 2], name="v_tokens")

    scores = mb.matmul(x=q_tokens, y=k_heads, name="attn_scores_raw")
    scale = const("attn_scale", np.array([1.0 / np.sqrt(HEAD_DIM)], dtype=np.float16))
    scaled_scores = mb.mul(x=scores, y=scale, name="attn_scores")

    mask = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e4, dtype=np.float16), k=1)
    causal_mask = const("causal_mask", mask.reshape((1, 1, SEQ_LEN, SEQ_LEN)))
    masked_scores = mb.add(x=scaled_scores, y=causal_mask, name="attn_scores_masked")
    attn_weights = mb.softmax(x=masked_scores, axis=-1, name="attn_weights")

    attended = mb.matmul(x=attn_weights, y=v_tokens, name="attn_weighted_values")
    attended = mb.transpose(x=attended, perm=[0, 1, 3, 2], name="attn_heads_to_channels")
    attended = mb.reshape(x=attended, shape=[1, DIM, 1, SEQ_LEN], name="attn_concat")

    projected = mb.conv(x=attended, weight=wo, name="attn_output_projection")
    residual_attn = mb.add(x=x, y=projected, name="attn_residual")

    ffn_norm = rms_norm(residual_attn, "ffn_rms")
    hidden_w1 = mb.conv(x=ffn_norm, weight=w1, name="ffn_w1")
    hidden_w3 = mb.conv(x=ffn_norm, weight=w3, name="ffn_w3")
    gate = mb.sigmoid(x=hidden_w1, name="ffn_gate_sigmoid")
    silu = mb.mul(x=hidden_w1, y=gate, name="ffn_silu")
    swiglu = mb.mul(x=silu, y=hidden_w3, name="ffn_swiglu")
    ffn_output = mb.conv(x=swiglu, weight=w2, name="ffn_w2")

    return mb.add(x=residual_attn, y=ffn_output, name="ffn_residual")


@mb.program(
    input_specs=[mb.TensorSpec(shape=(1, DIM, 1, SEQ_LEN), dtype=types.fp16)],
)
def transformer_program(x):
    return transformer_layer(x)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = ct.convert(
        transformer_program,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(str(output_path))
    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()
