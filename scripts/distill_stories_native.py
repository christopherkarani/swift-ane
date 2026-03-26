#!/usr/bin/env python3
"""Config-driven Stories distillation and native Espresso export pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.convert_weights_llama import write_blob, write_causal_masks
from scripts.espresso_llama_weights import load_espresso_llama_for_causal_lm
from scripts.stories_model_identity import EspressoSentencePieceTokenizer


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class TeacherSpec:
    source: str
    model: str | None = None
    weights_dir: str | None = None
    tokenizer_dir: str | None = None

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "TeacherSpec":
        return TeacherSpec(
            source=str(payload["source"]),
            model=payload.get("model"),
            weights_dir=payload.get("weights_dir"),
            tokenizer_dir=payload.get("tokenizer_dir"),
        )


@dataclass(frozen=True)
class StudentSpec:
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

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "StudentSpec":
        return StudentSpec(
            name=str(payload["name"]),
            n_layer=int(payload["n_layer"]),
            n_head=int(payload["n_head"]),
            n_kv_head=int(payload.get("n_kv_head", payload["n_head"])),
            d_model=int(payload["d_model"]),
            head_dim=int(payload["head_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            vocab=int(payload["vocab"]),
            max_seq=int(payload["max_seq"]),
            norm_eps=float(payload["norm_eps"]),
            rope_theta=float(payload.get("rope_theta", 10_000.0)),
            eos_token=_optional_int(payload.get("eos_token")),
        )

    def to_llama_config(self) -> LlamaConfig:
        kwargs: dict[str, Any] = {
            "hidden_size": self.d_model,
            "intermediate_size": self.hidden_dim,
            "num_hidden_layers": self.n_layer,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_kv_head,
            "vocab_size": self.vocab,
            "max_position_embeddings": self.max_seq,
            "rms_norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
        }
        if self.eos_token is not None:
            kwargs["eos_token_id"] = self.eos_token
        return LlamaConfig(**kwargs)


@dataclass(frozen=True)
class InitializationSpec:
    mode: str

    @staticmethod
    def from_dict(payload: dict[str, Any] | None) -> "InitializationSpec":
        payload = payload or {}
        return InitializationSpec(mode=str(payload.get("mode", "random")))


@dataclass(frozen=True)
class TrainSpec:
    texts: list[str]
    sequence_length: int
    window_stride: int
    max_samples: int
    batch_size: int
    steps: int
    learning_rate: float
    kl_weight: float
    ce_weight: float
    temperature: float
    device: str

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "TrainSpec":
        texts = list(payload.get("texts", []))
        if text_file := payload.get("texts_file"):
            lines = Path(text_file).read_text(encoding="utf-8").splitlines()
            texts.extend(line for line in lines if line and not line.startswith("#"))
        return TrainSpec(
            texts=texts,
            sequence_length=int(payload["sequence_length"]),
            window_stride=int(payload.get("window_stride", max(1, int(payload["sequence_length"]) // 2))),
            max_samples=int(payload.get("max_samples", len(texts))),
            batch_size=int(payload.get("batch_size", 1)),
            steps=int(payload["steps"]),
            learning_rate=float(payload["learning_rate"]),
            kl_weight=float(payload.get("kl_weight", 1.0)),
            ce_weight=float(payload.get("ce_weight", 0.0)),
            temperature=float(payload.get("temperature", 1.0)),
            device=str(payload.get("device", "cpu")),
        )


@dataclass(frozen=True)
class ExportSpec:
    output_dir: str
    context_target_tokens: int
    bundle_path: str | None
    model_tier: str
    behavior_class: str
    optimization_recipe: str
    quality_gate: str
    performance_target: str | None
    teacher_model: str | None

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExportSpec":
        return ExportSpec(
            output_dir=str(payload["output_dir"]),
            context_target_tokens=int(payload["context_target_tokens"]),
            bundle_path=payload.get("bundle_path"),
            model_tier=str(payload.get("model_tier", "optimized")),
            behavior_class=str(payload.get("behavior_class", "approximate")),
            optimization_recipe=str(payload["optimization_recipe"]),
            quality_gate=str(payload["quality_gate"]),
            performance_target=payload.get("performance_target"),
            teacher_model=payload.get("teacher_model"),
        )


@dataclass(frozen=True)
class DistillationConfig:
    teacher: TeacherSpec
    student: StudentSpec
    initialization: InitializationSpec
    train: TrainSpec
    export: ExportSpec

    @staticmethod
    def load(path: Path) -> "DistillationConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DistillationConfig(
            teacher=TeacherSpec.from_dict(payload["teacher"]),
            student=StudentSpec.from_dict(payload["student"]),
            initialization=InitializationSpec.from_dict(payload.get("initialization")),
            train=TrainSpec.from_dict(payload["train"]),
            export=ExportSpec.from_dict(payload["export"]),
        )


class NativeTokenizerAdapter:
    def __init__(self, tokenizer_dir: Path) -> None:
        self._tokenizer = EspressoSentencePieceTokenizer(tokenizer_dir / "tokenizer.model")

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)


class HFTokenizerAdapter:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device mps but torch.backends.mps.is_available() is false")
    return requested


def load_teacher(teacher: TeacherSpec):
    if teacher.source == "espresso":
        if not teacher.weights_dir or not teacher.tokenizer_dir:
            raise ValueError("espresso teacher requires weights_dir and tokenizer_dir")
        _, model = load_espresso_llama_for_causal_lm(Path(teacher.weights_dir), torch_dtype=torch.float32)
        tokenizer = NativeTokenizerAdapter(Path(teacher.tokenizer_dir))
        return model, tokenizer, teacher.weights_dir
    if teacher.source == "hf":
        if not teacher.model:
            raise ValueError("hf teacher requires model")
        model = AutoModelForCausalLM.from_pretrained(teacher.model, torch_dtype=torch.float32)
        from transformers import AutoTokenizer

        tokenizer = HFTokenizerAdapter(AutoTokenizer.from_pretrained(teacher.model))
        return model, tokenizer, teacher.model
    raise ValueError(f"Unsupported teacher source: {teacher.source}")


def build_training_examples(
    texts: list[str],
    tokenizer,
    sequence_length: int,
    max_samples: int,
    window_stride: int,
) -> list[torch.Tensor]:
    examples: list[torch.Tensor] = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue
        if len(token_ids) <= sequence_length + 1:
            examples.append(torch.tensor(token_ids, dtype=torch.long))
            if len(examples) >= max_samples:
                break
            continue

        max_start = len(token_ids) - (sequence_length + 1)
        for start in range(0, max_start + 1, max(1, window_stride)):
            window = token_ids[start : start + sequence_length + 1]
            if len(window) < 2:
                continue
            examples.append(torch.tensor(window, dtype=torch.long))
            if len(examples) >= max_samples:
                break
        if len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError("No usable training examples were produced from the configured texts")
    return examples


def evaluate_student_against_teacher(
    teacher_model,
    student_model: LlamaForCausalLM,
    examples: list[torch.Tensor],
    device: str,
) -> dict[str, float]:
    teacher_model.eval()
    student_model.eval()

    total_tokens = 0
    matching_tokens = 0
    total_label_tokens = 0
    matching_labels = 0
    total_kl = 0.0
    exact_two_token_passes = 0
    exact_two_token_token0_matches = 0
    exact_two_token_future_accepts = 0

    with torch.no_grad():
        for example in examples:
            sample = example.to(device)
            input_ids = sample[:-1].unsqueeze(0)
            labels = sample[1:].unsqueeze(0)

            teacher_logits = teacher_model(input_ids=input_ids).logits
            student_logits = student_model(input_ids=input_ids).logits

            teacher_argmax = teacher_logits.argmax(dim=-1)
            student_argmax = student_logits.argmax(dim=-1)

            total_tokens += int(teacher_argmax.numel())
            matching_tokens += int((teacher_argmax == student_argmax).sum().item())
            total_label_tokens += int(labels.numel())
            matching_labels += int((student_argmax == labels).sum().item())

            total_kl += float(
                F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                ).cpu().item()
            )

            teacher_token0 = int(teacher_argmax[0, -1].item())
            student_token0 = int(student_argmax[0, -1].item())
            exact_two_token_passes += 1
            if teacher_token0 != student_token0:
                continue

            exact_two_token_token0_matches += 1
            extended_input_ids = torch.cat(
                [input_ids, torch.tensor([[teacher_token0]], dtype=torch.long, device=device)],
                dim=1,
            )
            teacher_next_logits = teacher_model(input_ids=extended_input_ids).logits
            student_next_logits = student_model(input_ids=extended_input_ids).logits
            teacher_token1 = int(teacher_next_logits.argmax(dim=-1)[0, -1].item())
            student_token1 = int(student_next_logits.argmax(dim=-1)[0, -1].item())
            if teacher_token1 == student_token1:
                exact_two_token_future_accepts += 1

    mean_kl = total_kl / max(len(examples), 1)
    return {
        "teacher_token_agreement": matching_tokens / max(total_tokens, 1),
        "label_token_accuracy": matching_labels / max(total_label_tokens, 1),
        "mean_teacher_student_kl": mean_kl,
        "exact_two_token_token0_match_rate": exact_two_token_token0_matches / max(exact_two_token_passes, 1),
        "exact_two_token_future_accept_rate": exact_two_token_future_accepts / max(exact_two_token_passes, 1),
    }


def initialize_student_from_teacher(
    student_model: LlamaForCausalLM,
    teacher_model,
    initialization: InitializationSpec,
) -> str:
    if initialization.mode == "random":
        return "random"
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    copied = 0

    if initialization.mode == "teacher_copy":
        for name, tensor in student_state.items():
            teacher_tensor = teacher_state.get(name)
            if teacher_tensor is None or teacher_tensor.shape != tensor.shape:
                raise ValueError(
                    f"teacher_copy requires matching tensor for {name}; "
                    f"student shape={tuple(tensor.shape)} teacher shape={None if teacher_tensor is None else tuple(teacher_tensor.shape)}"
                )
            tensor.copy_(teacher_tensor.to(device=tensor.device, dtype=tensor.dtype))
            copied += 1
        if copied != len(student_state):
            raise ValueError(f"teacher_copy copied {copied} tensors but student has {len(student_state)} tensors")
        return "teacher_copy"

    if initialization.mode == "teacher_truncate_prefix":
        for name, tensor in student_state.items():
            teacher_name = name
            if name.startswith("model.layers."):
                parts = name.split(".")
                layer_index = int(parts[2])
                teacher_name = ".".join(["model", "layers", str(layer_index)] + parts[3:])
            teacher_tensor = teacher_state.get(teacher_name)
            if teacher_tensor is None:
                raise ValueError(f"teacher_truncate_prefix requires tensor {teacher_name} for {name}")
            if teacher_tensor.shape != tensor.shape:
                raise ValueError(
                    f"teacher_truncate_prefix requires matching tensor shape for {name}; "
                    f"student shape={tuple(tensor.shape)} teacher shape={tuple(teacher_tensor.shape)}"
                )
            tensor.copy_(teacher_tensor.to(device=tensor.device, dtype=tensor.dtype))
            copied += 1
        if copied != len(student_state):
            raise ValueError(
                f"teacher_truncate_prefix copied {copied} tensors but student has {len(student_state)} tensors"
            )
        return "teacher_truncate_prefix"

    raise ValueError(f"Unsupported initialization mode: {initialization.mode}")


def export_student_to_espresso(model: LlamaForCausalLM, student: StudentSpec, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    metadata = {
        "name": student.name,
        "nLayer": student.n_layer,
        "nHead": student.n_head,
        "nKVHead": student.n_kv_head,
        "dModel": student.d_model,
        "headDim": student.head_dim,
        "hiddenDim": student.hidden_dim,
        "vocab": student.vocab,
        "maxSeq": student.max_seq,
        "normEps": student.norm_eps,
        "ropeTheta": student.rope_theta,
        "architecture": "llama",
    }
    if student.eos_token is not None:
        metadata["eosToken"] = student.eos_token
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    write_blob(state["model.embed_tokens.weight"], output_dir / "embeddings" / "token.bin")
    write_blob(state["model.norm.weight"], output_dir / "final_norm.bin")
    lm_head = state["lm_head.weight"] if "lm_head.weight" in state else state["model.embed_tokens.weight"]
    write_blob(lm_head, output_dir / "lm_head.bin")

    for layer_index in range(student.n_layer):
        prefix = f"model.layers.{layer_index}"
        layer_dir = output_dir / "layers" / str(layer_index)
        write_blob(state[f"{prefix}.input_layernorm.weight"], layer_dir / "rms_att.bin")
        write_blob(state[f"{prefix}.post_attention_layernorm.weight"], layer_dir / "rms_ffn.bin")
        write_blob(state[f"{prefix}.self_attn.q_proj.weight"], layer_dir / "wq.bin")
        write_blob(state[f"{prefix}.self_attn.k_proj.weight"], layer_dir / "wk.bin")
        write_blob(state[f"{prefix}.self_attn.v_proj.weight"], layer_dir / "wv.bin")
        write_blob(state[f"{prefix}.self_attn.o_proj.weight"], layer_dir / "wo.bin")
        write_blob(state[f"{prefix}.mlp.gate_proj.weight"], layer_dir / "w1.bin")
        write_blob(state[f"{prefix}.mlp.down_proj.weight"], layer_dir / "w2.bin")
        write_blob(state[f"{prefix}.mlp.up_proj.weight"], layer_dir / "w3.bin")

    write_causal_masks(output_dir, student.max_seq)
    return output_dir


def maybe_pack_bundle(export_root: Path, config: DistillationConfig, teacher_tokenizer_dir: str | None) -> None:
    if not config.export.bundle_path:
        return
    if not teacher_tokenizer_dir:
        raise ValueError("Bundle packing requires a tokenizer directory")

    espc = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "espc"
    if not espc.exists():
        raise FileNotFoundError(f"espc not found at {espc}")

    command = [
        str(espc),
        "pack-native",
        str(export_root),
        config.export.bundle_path,
        "--tokenizer-dir",
        teacher_tokenizer_dir,
        "--context-target",
        str(config.export.context_target_tokens),
        "--model-tier",
        config.export.model_tier,
        "--behavior-class",
        config.export.behavior_class,
        "--optimization-recipe",
        config.export.optimization_recipe,
        "--quality-gate",
        config.export.quality_gate,
    ]
    if config.export.teacher_model:
        command.extend(["--teacher-model", config.export.teacher_model])
    if config.export.performance_target:
        command.extend(["--performance-target", config.export.performance_target])
    subprocess.run(command, check=True)


def run_distillation(config: DistillationConfig, dry_run: bool = False) -> dict[str, Any]:
    device = resolve_device(config.train.device)
    teacher_model, tokenizer, teacher_ref = load_teacher(config.teacher)
    teacher_model.eval().to(device)
    student_model = LlamaForCausalLM(config.student.to_llama_config()).to(device)
    initialization_mode = initialize_student_from_teacher(
        student_model=student_model,
        teacher_model=teacher_model,
        initialization=config.initialization,
    )
    student_model.train()

    examples = build_training_examples(
        config.train.texts,
        tokenizer,
        sequence_length=config.train.sequence_length,
        max_samples=config.train.max_samples,
        window_stride=config.train.window_stride,
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.train.learning_rate)
    step_metrics: list[dict[str, float]] = []
    total_steps = 0
    initial_evaluation = evaluate_student_against_teacher(
        teacher_model=teacher_model,
        student_model=student_model,
        examples=examples,
        device=device,
    )
    student_model.train()

    if not dry_run:
        for step_index in range(config.train.steps):
            sample = examples[step_index % len(examples)].to(device)
            input_ids = sample[:-1].unsqueeze(0)
            labels = sample[1:].unsqueeze(0)

            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids).logits
            student_logits = student_model(input_ids=input_ids).logits

            temp = config.train.temperature
            kl = F.kl_div(
                F.log_softmax(student_logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1),
                reduction="batchmean",
            ) * (temp * temp)
            ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
            loss = (config.train.kl_weight * kl) + (config.train.ce_weight * ce)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_steps += 1
            step_metrics.append(
                {
                    "step": float(step_index + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "kl": float(kl.detach().cpu().item()),
                    "ce": float(ce.detach().cpu().item()),
                }
            )

    final_evaluation = evaluate_student_against_teacher(
        teacher_model=teacher_model,
        student_model=student_model,
        examples=examples,
        device=device,
    )

    export_root = Path(config.export.output_dir).expanduser().resolve()
    export_student_to_espresso(student_model.cpu(), config.student, export_root)
    maybe_pack_bundle(export_root, config, config.teacher.tokenizer_dir)

    report = {
        "teacher_ref": teacher_ref,
        "student_name": config.student.name,
        "student_n_kv_head": config.student.n_kv_head,
        "context_target_tokens": config.export.context_target_tokens,
        "steps_requested": config.train.steps,
        "steps_completed": total_steps,
        "device": device,
        "initialization_mode": initialization_mode,
        "behavior_class": config.export.behavior_class,
        "optimization_recipe": config.export.optimization_recipe,
        "bundle_path": config.export.bundle_path,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "metrics": step_metrics,
    }
    (export_root / "distill-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON distillation config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and export without optimization steps")
    args = parser.parse_args()

    config = DistillationConfig.load(Path(args.config))
    report = run_distillation(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
