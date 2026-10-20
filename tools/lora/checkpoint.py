"""Checkpoint containers and logical-file discovery, independent of adapter layout."""

from pathlib import Path

import torch
from safetensors.torch import load_file


TORCH_EXTENSIONS = (".pt", ".pth", ".bin", ".ckpt")
EXTENSIONS = (".safetensors", *TORCH_EXTENSIONS)


def find_checkpoint(directory, stem, required=True):
    for extension in EXTENSIONS:
        path = directory / (stem + extension)
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(f"No {stem} checkpoint in {directory}; expected {', '.join(EXTENSIONS)}")
    return None


def load_checkpoint(path):
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".safetensors":
        state = load_file(path, device="cpu")
    elif extension in TORCH_EXTENSIONS:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError(f"Unsupported checkpoint container: {path}")
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Expected a tensor state dictionary: {path}")
    return state
