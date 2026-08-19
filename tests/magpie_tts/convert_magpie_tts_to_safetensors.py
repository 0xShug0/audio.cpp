#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a MagpieTTS .nemo checkpoint to audio.cpp safetensors.")
    parser.add_argument("source", type=Path, help="MagpieTTS .nemo archive or extracted checkpoint directory")
    parser.add_argument("output", type=Path, help="Output model directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_if_needed(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if source.is_dir():
        return source, None
    if not source.is_file() or source.suffix != ".nemo":
        raise RuntimeError(f"expected a .nemo archive or extracted directory, got {source}")
    temp = tempfile.TemporaryDirectory(prefix="magpie_tts_nemo_")
    root = Path(temp.name)
    with tarfile.open(source, "r:*") as archive:
        archive.extractall(root, filter="data")
    return root, temp


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload in {path}")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        if torch.is_tensor(value):
            tensors[str(name)] = value.detach().cpu().contiguous()
    if not tensors:
        raise RuntimeError(f"no tensors found in {path}")
    return tensors


def detect_format(config: dict[str, Any], tensors: dict[str, torch.Tensor]) -> str:
    if "baked_context_embedding.weight" in tensors and "text_embedding.weight" in tensors:
        return "magpie_tts_safetensors"
    if any(name.startswith("audio_decoder.") for name in tensors) and any(
        name.startswith("vector_quantizer.") for name in tensors
    ):
        return "nemo_nano_codec_safetensors"
    return str(config.get("target", "nemo_safetensors")).replace(".", "_")


def normalize_config(config: dict[str, Any], tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    out = dict(config)
    out["audio_cpp_format"] = detect_format(config, tensors)
    out["audio_cpp_format_version"] = 1
    out["tensor_file"] = "model.safetensors"
    if "_baked_embedding_T" in tensors:
        out["baked_context_length"] = int(tensors["_baked_embedding_T"].item())
    if "_baked_embedding_D" in tensors:
        out["baked_context_dim"] = int(tensors["_baked_embedding_D"].item())
    if "baked_context_embedding.weight" in tensors:
        out["baked_speakers"] = int(tensors["baked_context_embedding.weight"].shape[0])
    if "text_embedding.weight" in tensors:
        out["text_vocab_size"] = int(tensors["text_embedding.weight"].shape[0])
    if "audio_embeddings.0.weight" in tensors:
        out["audio_vocab_size"] = int(tensors["audio_embeddings.0.weight"].shape[0])
    stacked_audio_channels = len(
        [name for name in tensors if name.startswith("audio_embeddings.") and name.endswith(".weight")]
    )
    if stacked_audio_channels:
        out["stacked_audio_channels"] = stacked_audio_channels
        frame_stacking_factor = int(out.get("frame_stacking_factor", 1))
        if frame_stacking_factor <= 0 or stacked_audio_channels % frame_stacking_factor != 0:
            raise RuntimeError(
                f"invalid Magpie frame_stacking_factor={frame_stacking_factor} for {stacked_audio_channels} stacked channels"
            )
        out["audio_codebooks"] = stacked_audio_channels // frame_stacking_factor
    return out


def copy_runtime_assets(root: Path, output: Path) -> None:
    for source in root.iterdir():
        if source.name in {"model_weights.ckpt", "model_config.yaml"}:
            continue
        if source.is_file():
            shutil.copy2(source, output / source.name)
        elif source.is_dir():
            destination = output / source.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{output} already exists and is not empty; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    root, temp = extract_if_needed(source)
    try:
        config_path = root / "model_config.yaml"
        weights_path = root / "model_weights.ckpt"
        if not config_path.is_file():
            raise RuntimeError(f"missing {config_path}")
        if not weights_path.is_file():
            raise RuntimeError(f"missing {weights_path}")

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise RuntimeError("model_config.yaml must contain a mapping")
        tensors = load_state_dict(weights_path)

        save_file(tensors, str(output / "model.safetensors"))
        (output / "config.json").write_text(
            json.dumps(normalize_config(config, tensors), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(config_path, output / "model_config.yaml")
        copy_runtime_assets(root, output)
    finally:
        if temp is not None:
            temp.cleanup()

    print(f"wrote {output / 'model.safetensors'}")
    print(f"wrote {output / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
