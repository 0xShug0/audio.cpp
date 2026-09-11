#!/usr/bin/env python3
"""Generate deterministic full-encoder checkpoints from Transformers #46180."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


TRANSFORMERS_COMMIT = "48e7f65fb274172e15aa88875d780c67c37606c7"
MODEL_REVISION = "854d88f94205cd17d2afdb24332130d86fbe654a"
MODEL_CONFIG_SHA256 = "c7c4a30316929631ac5fabc5fb3c0dd3278dcc9809670720c5920186285d004a"
MODEL_SAFETENSORS_SHA256 = "335ca3e74917f1156690400e2c344350112950165789cf78ce3d0a367affd821"
TORCH_VERSION = "2.11.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-src", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    transformers_src = args.transformers_src.resolve()
    model_dir = args.model_dir.resolve()
    sys.path.insert(0, str(transformers_src))

    import torch
    import transformers
    from safetensors import safe_open
    from transformers import FunAsrNanoConfig
    from transformers.models.fun_asr_nano.modeling_fun_asr_nano import (
        FunAsrNanoEncoder,
    )

    imported_module = Path(transformers.__file__).resolve()
    if transformers_src not in imported_module.parents:
        raise RuntimeError(
            f"expected Transformers from {transformers_src}, imported {imported_module}"
        )
    transformers_root = transformers_src.parent
    actual_commit = subprocess.check_output(
        ["git", "-C", str(transformers_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != TRANSFORMERS_COMMIT:
        raise RuntimeError(
            f"expected Transformers {TRANSFORMERS_COMMIT}, found {actual_commit}"
        )
    modeling_path = (
        transformers_src
        / "transformers/models/fun_asr_nano/modeling_fun_asr_nano.py"
    )
    modeling_relative = modeling_path.relative_to(transformers_root)
    modeling_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(transformers_root),
            "status",
            "--porcelain",
            "--",
            str(modeling_relative),
        ],
        text=True,
    ).strip()
    if modeling_status:
        raise RuntimeError(
            f"Fun-ASR-Nano modeling source has uncommitted changes: {modeling_status}"
        )
    if torch.__version__.split("+", maxsplit=1)[0] != TORCH_VERSION:
        raise RuntimeError(f"expected torch {TORCH_VERSION}, found {torch.__version__}")

    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise RuntimeError("model directory is missing config.json or model.safetensors")
    actual_config_sha256 = sha256_file(config_path)
    actual_model_sha256 = sha256_file(model_path)
    if actual_config_sha256 != MODEL_CONFIG_SHA256:
        raise RuntimeError(
            f"expected config SHA-256 {MODEL_CONFIG_SHA256}, found {actual_config_sha256}"
        )
    if actual_model_sha256 != MODEL_SAFETENSORS_SHA256:
        raise RuntimeError(
            f"expected model SHA-256 {MODEL_SAFETENSORS_SHA256}, found {actual_model_sha256}"
        )

    config = FunAsrNanoConfig.from_pretrained(model_dir, local_files_only=True)
    encoder = FunAsrNanoEncoder(config.encoder_config).eval()
    state_dict = {}
    catalog_lines = []
    with safe_open(model_path, framework="pt", device="cpu") as source:
        for name in source.keys():
            if not name.startswith("model.audio_tower."):
                continue
            tensor_slice = source.get_slice(name)
            shape = list(tensor_slice.get_shape())
            catalog_lines.append(
                json.dumps(
                    {"name": name, "dtype": tensor_slice.get_dtype(), "shape": shape},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            target_name = name.removeprefix("model.audio_tower.").replace(
                ".fsmn.", ".feedforward_sequential_memory."
            )
            state_dict[target_name] = source.get_tensor(name).float()
    if len(state_dict) != 1194:
        raise RuntimeError(
            f"expected 1194 audio-tower tensors, found {len(state_dict)}"
        )
    encoder.load_state_dict(state_dict, strict=True, assign=True)

    torch.set_num_threads(1)
    input_features = torch.linspace(
        -1.0, 1.0, steps=3 * config.encoder_config.input_size, dtype=torch.float32
    ).reshape(1, 3, config.encoder_config.input_size)
    input_mask = torch.ones((1, 3), dtype=torch.int64)
    checkpoints = {}
    with torch.no_grad():
        hidden = encoder.stem(input_features, input_mask)
        checkpoints["stem"] = hidden
        for index, layer in enumerate(encoder.layers):
            hidden = layer(hidden, input_mask)
            if index in (0, 24, 48):
                checkpoints[f"main_layer_{index}"] = hidden
        hidden = encoder.layer_norm(hidden)
        checkpoints["main_layer_norm"] = hidden
        for index, layer in enumerate(encoder.timestamp_prediction_layers):
            hidden = layer(hidden, input_mask)
            if index in (0, 10, 19):
                checkpoints[f"timestamp_layer_{index}"] = hidden
        hidden = encoder.timestamp_prediction_layer_norm(hidden)
        checkpoints["final"] = hidden
        actual = encoder(input_features, input_features_mask=input_mask).last_hidden_state
        torch.testing.assert_close(actual, hidden, atol=0.0, rtol=0.0)

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

    binary_path = args.output.with_suffix(".bin")
    binary_bytes = bytes(binary_data)
    catalog_text = "\n".join(sorted(catalog_lines)) + "\n"
    payload = {
        "schema_version": 1,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "model_revision": MODEL_REVISION,
        "modeling_sha256": hashlib.sha256(modeling_path.read_bytes()).hexdigest(),
        "model_config_sha256": actual_config_sha256,
        "model_safetensors_sha256": actual_model_sha256,
        "weight_catalog_count": len(catalog_lines),
        "weight_catalog_sha256": hashlib.sha256(catalog_text.encode()).hexdigest(),
        "torch_version": torch.__version__,
        "data_file": binary_path.name,
        "data_format": "little-endian-float32",
        "data_sha256": hashlib.sha256(binary_bytes).hexdigest(),
        "input": store_tensor(input_features),
        "valid_frames": 3,
        "checkpoints": {
            name: store_tensor(tensor) for name, tensor in checkpoints.items()
        },
    }
    binary_bytes = bytes(binary_data)
    payload["data_sha256"] = hashlib.sha256(binary_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(binary_bytes)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
