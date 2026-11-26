#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Irodori Semantic-DACVAE PyTorch checkpoint to safetensors."
    )
    parser.add_argument(
        "--input",
        default="models/Semantic-DACVAE-Japanese-32dim/weights.pth",
        help="PyTorch DACVAE checkpoint to convert.",
    )
    parser.add_argument(
        "--output",
        default="models/Semantic-DACVAE-Japanese-32dim/weights.safetensors",
        help="Safetensors file to write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def checkpoint_state_dict(payload: Any) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is missing a dict state_dict")

    metadata: dict[str, str] = {
        "source_format": "pytorch",
        "source_state_key": "state_dict",
        "tensor_count": str(len(state)),
    }
    raw_metadata = payload.get("metadata")
    if raw_metadata is not None:
        metadata["pytorch_metadata_json"] = json.dumps(
            raw_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return state, metadata


def tensor_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise RuntimeError(f"checkpoint contains a non-string tensor key: {key!r}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"checkpoint state contains a non-tensor value at key: {key}")
        tensors[key] = value.detach().cpu().contiguous()
    if not tensors:
        raise RuntimeError("refusing to write an empty safetensors file")
    return tensors


def save_checked(
    tensors: dict[str, torch.Tensor],
    output_path: Path,
    metadata: dict[str, str],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=metadata)

    reloaded = load_file(str(output_path), device="cpu")
    if set(reloaded.keys()) != set(tensors.keys()):
        raise RuntimeError(f"saved safetensors key set does not match source: {output_path}")
    for key, expected in tensors.items():
        actual = reloaded[key]
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"saved tensor shape changed for {key}: {tuple(expected.shape)} -> {tuple(actual.shape)}"
            )
        if actual.dtype != expected.dtype:
            raise RuntimeError(f"saved tensor dtype changed for {key}: {expected.dtype} -> {actual.dtype}")
        if not torch.equal(actual, expected):
            raise RuntimeError(f"saved tensor value changed for {key}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if not input_path.is_file():
        raise RuntimeError(f"input checkpoint does not exist: {input_path}")

    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    state, metadata = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    metadata["source_file"] = str(input_path)
    metadata["tensor_count"] = str(len(tensors))
    save_checked(tensors, output_path, metadata, args.overwrite)

    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    print(f"wrote {output_path}")
    print(f"tensors: {len(tensors)}")
    print(f"tensor_bytes: {total_bytes}")


if __name__ == "__main__":
    main()
