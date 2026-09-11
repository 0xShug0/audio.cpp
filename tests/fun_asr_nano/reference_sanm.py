#!/usr/bin/env python3
"""Generate deterministic SAN-M block parity fixtures from Transformers #46180."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


TRANSFORMERS_COMMIT = "48e7f65fb274172e15aa88875d780c67c37606c7"
TORCH_VERSION = "2.11.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def projection_block(module, hidden_states):
    """Run the stem after the encoder-wide scale and positional embedding steps."""
    normalized = module.self_attn_layer_norm(hidden_states)
    value_states = module.self_attn.v_proj(normalized)
    attention_output, _ = module.self_attn(normalized)
    fsmn_output = module.feedforward_sequential_memory(value_states)
    hidden_states = attention_output + fsmn_output

    residual = hidden_states
    hidden_states = module.final_layer_norm(hidden_states)
    hidden_states = module.fc1(hidden_states)
    hidden_states = module.activation_fn(hidden_states)
    hidden_states = module.fc2(hidden_states)
    return residual + hidden_states


def main() -> None:
    args = parse_args()
    transformers_src = args.transformers_src.resolve()
    sys.path.insert(0, str(transformers_src))

    import torch
    import transformers
    from transformers import FunAsrNanoEncoderConfig
    from transformers.models.fun_asr_nano.modeling_fun_asr_nano import (
        FunAsrNanoEncoderLayer,
        FunAsrNanoEncoderStem,
    )

    transformers_module = Path(transformers.__file__).resolve()
    if transformers_src not in transformers_module.parents:
        raise RuntimeError(
            f"expected Transformers from {transformers_src}, imported {transformers_module}"
        )
    transformers_root = transformers_src.parent
    actual_commit = subprocess.check_output(
        ["git", "-C", str(transformers_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != TRANSFORMERS_COMMIT:
        raise RuntimeError(f"expected Transformers {TRANSFORMERS_COMMIT}, found {actual_commit}")
    modeling_path = transformers_src / "transformers/models/fun_asr_nano/modeling_fun_asr_nano.py"
    modeling_relative = modeling_path.relative_to(transformers_root)
    modeling_status = subprocess.check_output(
        ["git", "-C", str(transformers_root), "status", "--porcelain", "--", str(modeling_relative)],
        text=True,
    ).strip()
    if modeling_status:
        raise RuntimeError(f"Fun-ASR-Nano modeling source has uncommitted changes: {modeling_status}")
    if torch.__version__.split("+", maxsplit=1)[0] != TORCH_VERSION:
        raise RuntimeError(f"expected torch {TORCH_VERSION}, found {torch.__version__}")

    torch.set_num_threads(1)
    torch.manual_seed(20260729)
    config = FunAsrNanoEncoderConfig(
        num_mel_bins=12,
        num_stacked_frames=1,
        d_model=8,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        encoder_layers=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="relu",
        max_position_embeddings=32,
        kernel_size=3,
    )
    projection = FunAsrNanoEncoderStem(config).eval()
    residual = FunAsrNanoEncoderLayer(config).eval()

    projection_input = torch.linspace(-1.1, 1.3, steps=5 * 12, dtype=torch.float32).reshape(1, 5, 12)
    residual_input = torch.linspace(0.9, -0.7, steps=5 * 8, dtype=torch.float32).reshape(1, 5, 8)
    with torch.no_grad():
        projection_norm = projection.self_attn_layer_norm(projection_input)
        projection_output = projection_block(projection, projection_input)
        positions = projection.position_embeddings(6)[1:].to(dtype=projection_input.dtype)
        raw_stem_input = (projection_input - positions.unsqueeze(0)) / math.sqrt(config.d_model)
        actual_stem_output = projection(raw_stem_input)
        torch.testing.assert_close(actual_stem_output, projection_output, atol=2e-6, rtol=2e-6)
        residual_norm = residual.self_attn_layer_norm(residual_input)
        residual_output = residual(residual_input)

    binary_data = bytearray()

    def store_tensor(tensor) -> dict[str, object]:
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        descriptor = {
            "shape": list(array.shape),
            "offset_f32": len(binary_data) // 4,
            "count": int(array.size),
        }
        binary_data.extend(array.reshape(-1).tobytes())
        return descriptor

    def store_block(module, input_tensor, output_tensor, checkpoints) -> dict[str, object]:
        return {
            "input": store_tensor(input_tensor),
            "output": store_tensor(output_tensor),
            "checkpoints": {
                name: store_tensor(tensor)
                for name, tensor in checkpoints.items()
            },
            "weights": {
                name: store_tensor(tensor)
                for name, tensor in sorted(module.state_dict().items())
            },
        }

    blocks = {
        "projection": store_block(
            projection,
            projection_input,
            projection_output,
            {"self_attn_layer_norm": projection_norm},
        ),
        "residual": store_block(
            residual,
            residual_input,
            residual_output,
            {"self_attn_layer_norm": residual_norm},
        ),
    }
    binary_path = args.output.with_suffix(".bin")
    binary_bytes = bytes(binary_data)
    payload = {
        "schema_version": 1,
        "reference": "transformers.FunAsrNanoEncoderStem/FunAsrNanoEncoderLayer",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "modeling_sha256": hashlib.sha256(modeling_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "data_file": binary_path.name,
        "data_format": "little-endian-float32",
        "data_sha256": hashlib.sha256(binary_bytes).hexdigest(),
        "config": {
            "input_size": 12,
            "model_size": 8,
            "num_heads": 2,
            "ffn_size": 16,
            "fsmn_kernel_size": 3,
            "layer_norm_eps": 1e-5,
            "frames": 5,
        },
        "blocks": blocks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(binary_bytes)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
