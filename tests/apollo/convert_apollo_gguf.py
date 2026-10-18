#!/usr/bin/env python3
"""Package the official Apollo checkpoint without changing any tensor values."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audiocpp-gguf", type=Path, default=REPO_ROOT / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])

    with args.checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Apollo checkpoint does not match the official release SHA256")
    # The pinned official checkpoint contains OmegaConf metadata, not only tensors.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["model_name"] != "Apollo":
        raise ValueError("Expected an Apollo checkpoint")
    state = {key: value.detach().cpu().contiguous() for key, value in checkpoint["state_dict"].items()}
    config = OmegaConf.to_container(checkpoint["model_args"], resolve=True)
    config.update(model_type="apollo", checkpoint_sha256=digest, upstream="JusperLee/Apollo")
    window = torch.hann_window(int(config["sr"] * config["win"] // 1000))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="apollo-convert-") as temp:
        staging = Path(temp)
        save_file(state, staging / "model.safetensors")
        save_file({"window": window}, staging / "frontend.safetensors")
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = [
            str(args.audiocpp_gguf.resolve()),
            "--input", f"weights={staging / 'model.safetensors'}",
            "--input", f"frontend={staging / 'frontend.safetensors'}",
            "--output", str(args.output.resolve()), "--type", "orig",
            "--family", "apollo", "--model-spec", str(REPO_ROOT / "model_specs/apollo.json"),
            "--root", str(staging),
        ]
        if args.overwrite:
            command.append("--overwrite")
        logging.info("command=%s", command)
        converted = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logging.info("%s", converted.stdout.rstrip())
        converted.check_returncode()

    expected = {f"weights/{key}": value for key, value in state.items()}
    expected["frontend/window"] = window
    reader = gguf.GGUFReader(str(args.output))
    if {tensor.name for tensor in reader.tensors} != set(expected):
        raise ValueError("GGUF tensor names do not match the checkpoint and frontend")
    ranks = reader.get_field("audiocpp.tensor_ranks").contents()
    shapes = reader.get_field("audiocpp.tensor_shapes").contents()
    cursor = 0
    for tensor, rank in zip(reader.tensors, ranks, strict=True):
        original = expected[tensor.name].numpy()
        exact_shape = tuple(shapes[cursor:cursor + rank])
        cursor += rank
        physical_shape = tuple(tensor.shape) + (1,) * (4 - len(tensor.shape))
        expected_shape = original.shape[::-1] + (1,) * (4 - original.ndim)
        if exact_shape != original.shape or physical_shape != expected_shape or tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"GGUF shape/type mismatch: {tensor.name}")
        if not np.array_equal(tensor.data.reshape(-1).view(np.uint8), original.reshape(-1).view(np.uint8)):
            raise ValueError(f"GGUF byte mismatch: {tensor.name}")
    if cursor != len(shapes):
        raise ValueError("Unexpected trailing GGUF shape metadata")
    logging.info("PASS: %d checkpoint tensors and periodic Hann window byte-exact; checkpoint_sha256=%s",
                 len(state), digest)
    logging.info("output=%s bytes=%d", args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
