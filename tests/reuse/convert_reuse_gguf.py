#!/usr/bin/env python3
"""Package RE-USE directly from its original safetensors checkpoint."""

import argparse
import hashlib
import logging
from pathlib import Path
import subprocess

import gguf
import numpy as np
from safetensors import safe_open


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=("f32", "f16", "bf16", "q8_0"), default="f32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audiocpp-gguf", type=Path, default=root / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])
    checkpoint = args.source / "model.safetensors"
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != "87a4a970ce9aa79d5d92e71899ab034defcf13d93e5e4393ec0dc7db6d4ec048":
        raise ValueError("Expected the official NVIDIA RE-USE checkpoint")
    command = [str(args.audiocpp_gguf.resolve()), "--input", f"weights={checkpoint.resolve()}",
               "--output", str(args.output.resolve()), "--type", args.type, "--family", "reuse",
               "--model-spec", str(root / "model_specs/reuse.json"), "--root", str(args.source.resolve())]
    if args.overwrite:
        command.append("--overwrite")
    with safe_open(checkpoint, framework="np") as source:
        for name in source.keys():
            shape = source.get_slice(name).get_shape()
            if len(shape) == 1 or name.endswith(".A_log") or name.endswith(".conv1d.weight"):
                command.extend(["--keep-type", f"weights/{name}=f32"])
    logging.info("command=%s", command)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logging.info("%s", result.stdout)
    result.check_returncode()
    reader = gguf.GGUFReader(str(args.output))
    with safe_open(checkpoint, framework="np") as source:
        if {tensor.name for tensor in reader.tensors} != {"weights/" + key for key in source.keys()}:
            raise ValueError("GGUF tensor inventory mismatch")
        ranks = reader.get_field("audiocpp.tensor_ranks").contents()
        shapes = reader.get_field("audiocpp.tensor_shapes").contents()
        cursor = 0
        for tensor, rank in zip(reader.tensors, ranks, strict=True):
            original = source.get_tensor(tensor.name.removeprefix("weights/"))
            if tuple(shapes[cursor:cursor + rank]) != original.shape:
                raise ValueError(f"GGUF shape mismatch: {tensor.name}")
            cursor += rank
            if args.type == "f32" and tensor.tensor_type != gguf.GGMLQuantizationType.F32:
                raise ValueError(f"GGUF type mismatch: {tensor.name}")
            if tensor.tensor_type == gguf.GGMLQuantizationType.F32 and not np.array_equal(
                    tensor.data.reshape(-1).view(np.uint8), original.reshape(-1).view(np.uint8)):
                raise ValueError(f"GGUF type/value mismatch: {tensor.name}")
        if cursor != len(shapes):
            raise ValueError("Trailing GGUF tensor shape metadata")
    logging.info("PASS: %d tensors with matching shapes; retained F32 tensors byte-exact; sha256=%s",
                 len(reader.tensors), digest)


if __name__ == "__main__":
    main()
