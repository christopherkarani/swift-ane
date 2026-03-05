#!/usr/bin/env python3
"""Generate a Core ML model matching the Espresso transformer architecture.

Dimensions (from ModelConfig):
  dim=768, hidden=2048, seqLen=256, heads=12, headDim=64

Architecture (single layer):
  1. RMSNorm -> QKV Projection -> SDPA -> Output Projection + Residual
  2. RMSNorm -> SwiGLU FFN (W1, W3, SiLU gate, W2) + Residual

Usage:
  pip install coremltools numpy
  python3 scripts/generate_coreml_model.py
  # Output: benchmarks/models/transformer_layer.mlpackage
"""

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import os
import pathlib

# Match ModelConfig exactly
DIM = 768
HIDDEN = 2048
SEQ_LEN = 256
HEADS = 12
HEAD_DIM = DIM // HEADS


def build_transformer_layer():
    """Build a single transformer layer using coremltools MIL builder."""

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, DIM, 1, SEQ_LEN), dtype=types.fp16),
        ]
    )
    def transformer(x):
        # --- Attention Block ---
        # RMSNorm (simplified: normalize then scale)
        rms_att_weight = mb.const(
            val=np.ones((1, DIM, 1, 1), dtype=np.float16),
            name="rms_att_weight"
        )
        x_sq = mb.mul(x=x, y=x, name="x_sq")
        x_mean = mb.reduce_mean(x=x_sq, axes=[1], keep_dims=True, name="x_mean")
        eps = mb.const(val=np.float16(1e-5), name="eps")
        x_mean_eps = mb.add(x=x_mean, y=eps, name="x_mean_eps")
        x_rsqrt = mb.rsqrt(x=x_mean_eps, name="x_rsqrt")
        x_norm = mb.mul(x=x, y=x_rsqrt, name="x_norm_pre")
        x_norm = mb.mul(x=x_norm, y=rms_att_weight, name="x_norm")

        # QKV projections (as conv2d with 1x1 kernels -- ANE-native)
        wq = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wq")
        wk = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wk")
        wv = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wv")

        q = mb.conv(x=x_norm, weight=wq, name="q_proj")
        k = mb.conv(x=x_norm, weight=wk, name="k_proj")
        v = mb.conv(x=x_norm, weight=wv, name="v_proj")

        # Reshape for multi-head: [1, DIM, 1, SEQ] -> [1, HEADS, HEAD_DIM, SEQ]
        q_r = mb.reshape(x=q, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="q_reshape")
        k_r = mb.reshape(x=k, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="k_reshape")
        v_r = mb.reshape(x=v, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="v_reshape")

        # Transpose Q,V to standard layout: [1, HEADS, SEQ, HEAD_DIM]
        q_t = mb.transpose(x=q_r, perm=[0, 1, 3, 2], name="q_transpose")
        v_t = mb.transpose(x=v_r, perm=[0, 1, 3, 2], name="v_transpose")
        # K stays as [1, HEADS, HEAD_DIM, SEQ] — acts as K^T in matmul

        # Attention scores: Q_t @ K_r = (SEQ, HEAD_DIM) @ (HEAD_DIM, SEQ) = (SEQ, SEQ)
        scale = mb.const(val=np.float16(1.0 / np.sqrt(HEAD_DIM)), name="scale")
        scores = mb.matmul(x=q_t, y=k_r, name="attn_scores_raw")
        scores = mb.mul(x=scores, y=scale, name="attn_scores")

        # Causal mask: [1, 1, SEQ, SEQ] broadcasts over heads
        mask_np = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e4, dtype=np.float16), k=1)
        mask = mb.const(val=mask_np.reshape(1, 1, SEQ_LEN, SEQ_LEN), name="causal_mask")
        scores = mb.add(x=scores, y=mask, name="attn_scores_masked")

        # Softmax
        attn_weights = mb.softmax(x=scores, axis=-1, name="attn_weights")

        # Attention output: weights @ V_t = (SEQ, SEQ) @ (SEQ, HEAD_DIM) = (SEQ, HEAD_DIM)
        attn_out = mb.matmul(x=attn_weights, y=v_t, name="attn_out_heads")

        # Transpose back: [1, HEADS, SEQ, HEAD_DIM] -> [1, HEADS, HEAD_DIM, SEQ]
        attn_out = mb.transpose(x=attn_out, perm=[0, 1, 3, 2], name="attn_out_transpose")
        # Reshape: [1, HEADS, HEAD_DIM, SEQ] -> [1, DIM, 1, SEQ]
        attn_out = mb.reshape(x=attn_out, shape=[1, DIM, 1, SEQ_LEN], name="attn_out_concat")

        # Output projection
        wo = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wo")
        o_out = mb.conv(x=attn_out, weight=wo, name="o_proj")

        # Residual
        x2 = mb.add(x=x, y=o_out, name="residual_attn")

        # --- FFN Block ---
        # RMSNorm
        rms_ffn_weight = mb.const(
            val=np.ones((1, DIM, 1, 1), dtype=np.float16),
            name="rms_ffn_weight"
        )
        x2_sq = mb.mul(x=x2, y=x2, name="x2_sq")
        x2_mean = mb.reduce_mean(x=x2_sq, axes=[1], keep_dims=True, name="x2_mean")
        x2_mean_eps = mb.add(x=x2_mean, y=eps, name="x2_mean_eps")
        x2_rsqrt = mb.rsqrt(x=x2_mean_eps, name="x2_rsqrt")
        x2_norm = mb.mul(x=x2, y=x2_rsqrt, name="x2_norm_pre")
        x2_norm = mb.mul(x=x2_norm, y=rms_ffn_weight, name="x2_norm")

        # SwiGLU FFN
        w1 = mb.const(val=np.random.randn(HIDDEN, DIM, 1, 1).astype(np.float16) * 0.02, name="w1")
        w3 = mb.const(val=np.random.randn(HIDDEN, DIM, 1, 1).astype(np.float16) * 0.02, name="w3")
        w2 = mb.const(val=np.random.randn(DIM, HIDDEN, 1, 1).astype(np.float16) * 0.02, name="w2")

        h1 = mb.conv(x=x2_norm, weight=w1, name="ffn_w1")
        h3 = mb.conv(x=x2_norm, weight=w3, name="ffn_w3")

        # SiLU gate
        silu = mb.sigmoid(x=h1, name="sigmoid_h1")
        silu = mb.mul(x=h1, y=silu, name="silu_h1")
        gate = mb.mul(x=silu, y=h3, name="gate_out")

        ffn_out = mb.conv(x=gate, weight=w2, name="ffn_w2")

        # Residual
        output = mb.add(x=x2, y=ffn_out, name="residual_ffn")

        return output

    return transformer


def main():
    print(f"Generating Core ML transformer layer model...")
    print(f"  dim={DIM}, hidden={HIDDEN}, seq_len={SEQ_LEN}, heads={HEADS}")

    prog = build_transformer_layer()

    # Convert to Core ML model
    model = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "benchmarks", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transformer_layer.mlpackage")

    model.save(output_path)
    total_size = sum(
        f.stat().st_size for f in pathlib.Path(output_path).rglob('*') if f.is_file()
    )
    print(f"  Saved to: {output_path}")
    print(f"  Model size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
