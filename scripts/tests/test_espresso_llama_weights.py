import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import espresso_llama_weights as script


def make_blobfile(path: Path, values: np.ndarray) -> None:
    payload = values.astype(np.float16).tobytes()
    header = bytearray(script.BLOBFILE_HEADER_BYTES)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into("<I", header, 72, len(payload))
    struct.pack_into("<I", header, 80, script.BLOBFILE_HEADER_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + payload)


def write_fixture(weights_dir: Path) -> None:
    metadata = {
        "name": "stories110m",
        "nLayer": 2,
        "nHead": 2,
        "nKVHead": 2,
        "dModel": 4,
        "headDim": 2,
        "hiddenDim": 6,
        "vocab": 8,
        "maxSeq": 16,
        "normEps": 1e-5,
        "architecture": "llama",
    }
    (weights_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    make_blobfile(weights_dir / "embeddings" / "token.bin", np.arange(32, dtype=np.float32).reshape(8, 4))
    make_blobfile(weights_dir / "final_norm.bin", np.arange(4, dtype=np.float32))
    make_blobfile(weights_dir / "lm_head.bin", np.arange(32, dtype=np.float32).reshape(8, 4))

    for layer_index in range(2):
        layer_dir = weights_dir / "layers" / str(layer_index)
        offset = layer_index * 100
        make_blobfile(layer_dir / "rms_att.bin", np.arange(offset, offset + 4, dtype=np.float32))
        make_blobfile(layer_dir / "rms_ffn.bin", np.arange(offset + 4, offset + 8, dtype=np.float32))
        make_blobfile(layer_dir / "wq.bin", np.arange(offset + 8, offset + 24, dtype=np.float32).reshape(4, 4))
        make_blobfile(layer_dir / "wk.bin", np.arange(offset + 24, offset + 40, dtype=np.float32).reshape(4, 4))
        make_blobfile(layer_dir / "wv.bin", np.arange(offset + 40, offset + 56, dtype=np.float32).reshape(4, 4))
        make_blobfile(layer_dir / "wo.bin", np.arange(offset + 56, offset + 72, dtype=np.float32).reshape(4, 4))
        make_blobfile(layer_dir / "w1.bin", np.arange(offset + 72, offset + 96, dtype=np.float32).reshape(6, 4))
        make_blobfile(layer_dir / "w2.bin", np.arange(offset + 96, offset + 120, dtype=np.float32).reshape(4, 6))
        make_blobfile(layer_dir / "w3.bin", np.arange(offset + 120, offset + 144, dtype=np.float32).reshape(6, 4))


class EspressoLlamaWeightsTests(unittest.TestCase):
    def test_load_espresso_metadata_reads_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_dir = Path(directory)
            write_fixture(weights_dir)

            metadata = script.load_espresso_metadata(weights_dir)

        self.assertEqual(metadata.n_layer, 2)
        self.assertEqual(metadata.hidden_dim, 6)
        self.assertEqual(metadata.max_seq, 16)
        self.assertEqual(metadata.rope_theta, 10_000.0)
        self.assertIsNone(metadata.eos_token)

    def test_read_blobfile_array_reads_payload_after_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            expected = np.arange(6, dtype=np.float32).reshape(2, 3)
            make_blobfile(path, expected)

            loaded = script.read_blobfile_array(path, (2, 3))

        np.testing.assert_array_equal(loaded, expected.astype(np.float16))

    def test_load_espresso_llama_state_dict_maps_expected_tensor_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_dir = Path(directory)
            write_fixture(weights_dir)

            state_dict = script.load_espresso_llama_state_dict(weights_dir)

        self.assertEqual(state_dict["model.embed_tokens.weight"].shape, (8, 4))
        self.assertEqual(state_dict["model.layers.0.self_attn.q_proj.weight"].shape, (4, 4))
        self.assertEqual(state_dict["model.layers.1.mlp.gate_proj.weight"].shape, (6, 4))
        self.assertEqual(state_dict["model.norm.weight"].shape, (4,))
        self.assertEqual(state_dict["lm_head.weight"].shape, (8, 4))

    def test_llama_config_kwargs_from_metadata_sets_llama_specific_fields(self) -> None:
        metadata = script.EspressoLlamaMetadata(
            name="stories110m",
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            d_model=4,
            head_dim=4,
            hidden_dim=6,
            vocab=8,
            max_seq=16,
            norm_eps=1e-5,
            rope_theta=500_000.0,
            eos_token=7,
        )

        kwargs = script.llama_config_kwargs_from_metadata(metadata)

        self.assertEqual(kwargs["num_key_value_heads"], 1)
        self.assertEqual(kwargs["rope_theta"], 500_000.0)
        self.assertEqual(kwargs["eos_token_id"], 7)
        self.assertFalse(kwargs["tie_word_embeddings"])


if __name__ == "__main__":
    unittest.main()
