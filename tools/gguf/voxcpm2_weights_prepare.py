#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a VoxCPM2 PyTorch checkpoint to safetensors.")
    parser.add_argument(
        "--input",
        default="models/VoxCPM2/audiovae.pth",
        help="PyTorch checkpoint to convert.",
    )
    parser.add_argument(
        "--output",
        default="models/VoxCPM2/audiovae.safetensors",
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
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint state_dict must be a dict, got {type(state).__name__}")
    return state


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


def write_checked(tensors: dict[str, torch.Tensor], output_path: Path, input_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_format": "pytorch",
        "source_file": str(input_path),
        "tensor_count": str(len(tensors)),
    }
    save_file(tensors, str(output_path), metadata=metadata)

    reloaded = load_file(str(output_path), device="cpu")
    if set(reloaded.keys()) != set(tensors.keys()):
        raise RuntimeError(f"saved safetensors key set does not match source: {output_path}")
    for key, value in tensors.items():
        actual = reloaded[key]
        if actual.shape != value.shape:
            raise RuntimeError(f"saved tensor shape changed for {key}: {tuple(value.shape)} -> {tuple(actual.shape)}")
        if actual.dtype != value.dtype:
            raise RuntimeError(f"saved tensor dtype changed for {key}: {value.dtype} -> {actual.dtype}")
        if not torch.equal(actual, value):
            raise RuntimeError(f"saved tensor value changed for {key}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if not input_path.is_file():
        raise RuntimeError(f"input checkpoint does not exist: {input_path}")

    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    write_checked(tensors, output_path, input_path, args.overwrite)
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    print(f"wrote {output_path}")
    print(f"tensors: {len(tensors)}")
    print(f"tensor_bytes: {total_bytes}")


if __name__ == "__main__":
    main()
