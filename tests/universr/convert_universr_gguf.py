#!/usr/bin/env python3
"""Package UniverSR audio or speech checkpoints in their original precision."""

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
import yaml
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audiocpp-gguf", type=Path, default=REPO_ROOT / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])

    checkpoint = args.model_dir / "pytorch_model.bin"
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    state = {key: value.detach().cpu().contiguous() for key, value in
             torch.load(checkpoint, map_location="cpu", weights_only=True).items()}
    config = yaml.safe_load((args.model_dir / "config.yaml").read_text(encoding="utf-8"))
    config.update(model_type="universr", checkpoint_sha256=digest)
    transform = config["transform"]
    window = getattr(torch.signal.windows, transform["window_fn"])(transform["n_fft"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="universr-convert-") as temp:
        staging = Path(temp)
        save_file(state, staging / "model.safetensors")
        save_file({"window": window}, staging / "frontend.safetensors")
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = [
            str(args.audiocpp_gguf.resolve()),
            "--input", f"weights={staging / 'model.safetensors'}",
            "--input", f"frontend={staging / 'frontend.safetensors'}",
            "--output", str(args.output.resolve()), "--type", "orig",
            "--family", "universr", "--model-spec", str(REPO_ROOT / "model_specs/universr.json"),
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
    logging.info("PASS: %d checkpoint tensors and reference window byte-exact; checkpoint_sha256=%s",
                 len(state), digest)
    logging.info("output=%s bytes=%d", args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
