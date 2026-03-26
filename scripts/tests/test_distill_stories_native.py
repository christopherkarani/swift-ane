import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import distill_stories_native as script


class DistillStoriesNativeTests(unittest.TestCase):
    def test_load_config_parses_student_and_export_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "teacher": {
                            "source": "hf",
                            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        },
                        "student": {
                            "name": "stories-gqa4",
                            "n_layer": 4,
                            "n_head": 8,
                            "n_kv_head": 2,
                            "d_model": 512,
                            "head_dim": 64,
                            "hidden_dim": 1536,
                            "vocab": 32000,
                            "max_seq": 256,
                            "norm_eps": 1e-5,
                            "rope_theta": 10000.0,
                        },
                        "train": {
                            "texts": ["Hello world"],
                            "sequence_length": 32,
                            "window_stride": 8,
                            "max_samples": 1,
                            "batch_size": 1,
                            "steps": 0,
                            "learning_rate": 1e-5,
                            "device": "cpu",
                        },
                        "export": {
                            "output_dir": "/tmp/stories-gqa4",
                            "context_target_tokens": 256,
                            "optimization_recipe": "stories-gqa4-proof",
                            "quality_gate": "proof-only",
                        },
                        "initialization": {
                            "mode": "teacher_copy",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = script.DistillationConfig.load(config_path)

        self.assertEqual(config.student.n_kv_head, 2)
        self.assertEqual(config.student.to_llama_config().num_key_value_heads, 2)
        self.assertEqual(config.export.context_target_tokens, 256)
        self.assertEqual(config.export.optimization_recipe, "stories-gqa4-proof")
        self.assertEqual(config.initialization.mode, "teacher_copy")
        self.assertEqual(config.train.window_stride, 8)

    def test_initialize_student_from_teacher_supports_truncate_prefix(self) -> None:
        class TinyModel:
            def __init__(self, state):
                self._state = state

            def state_dict(self):
                return self._state

        teacher = TinyModel(
            {
                "model.embed_tokens.weight": torch.tensor([[1.0, 2.0]]),
                "model.layers.0.self_attn.q_proj.weight": torch.tensor([[3.0, 4.0]]),
                "model.layers.1.self_attn.q_proj.weight": torch.tensor([[5.0, 6.0]]),
                "model.norm.weight": torch.tensor([7.0, 8.0]),
                "lm_head.weight": torch.tensor([[9.0, 10.0]]),
            }
        )
        student_state = {
            "model.embed_tokens.weight": torch.zeros((1, 2)),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 2)),
            "model.norm.weight": torch.zeros(2),
            "lm_head.weight": torch.zeros((1, 2)),
        }
        student = TinyModel(student_state)

        mode = script.initialize_student_from_teacher(
            student_model=student,
            teacher_model=teacher,
            initialization=script.InitializationSpec(mode="teacher_truncate_prefix"),
        )

        self.assertEqual(mode, "teacher_truncate_prefix")
        self.assertTrue(torch.equal(student_state["model.embed_tokens.weight"], teacher.state_dict()["model.embed_tokens.weight"]))
        self.assertTrue(
            torch.equal(
                student_state["model.layers.0.self_attn.q_proj.weight"],
                teacher.state_dict()["model.layers.0.self_attn.q_proj.weight"],
            )
        )

    def test_build_training_examples_uses_text_inputs(self) -> None:
        class Tokenizer:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text.split()) + 1))

        examples = script.build_training_examples(
            ["one two three", "alpha beta"],
            Tokenizer(),
            sequence_length=8,
            max_samples=2,
            window_stride=4,
        )

        self.assertEqual(len(examples), 2)
        self.assertTrue(all(example.numel() >= 2 for example in examples))

    def test_build_training_examples_slides_windows_for_long_inputs(self) -> None:
        class Tokenizer:
            def encode(self, text: str) -> list[int]:
                return list(range(10))

        examples = script.build_training_examples(
            ["long input"],
            Tokenizer(),
            sequence_length=4,
            max_samples=3,
            window_stride=2,
        )

        self.assertEqual(len(examples), 3)
        self.assertEqual([example.tolist() for example in examples], [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8]])


if __name__ == "__main__":
    unittest.main()
