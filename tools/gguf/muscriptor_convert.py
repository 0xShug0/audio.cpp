#!/usr/bin/env python3
"""Convert an official MuScriptor Small, Medium or Large checkpoint to F32 GGUF."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from safetensors import safe_open


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory containing config.json and model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True,
                        help="Path to audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    checkpoint = source / "model.safetensors"
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "muscriptor":
        raise ValueError("Expected a MuScriptor config.json")
    if config["variant"] not in ("small", "medium", "large"):
        raise ValueError("Expected an official Small, Medium or Large variant")
    dim, heads, layers, card = (config[key] for key in
                                ("dim", "num_heads", "num_layers", "card"))
    if min(dim, heads, layers, card) <= 0 or dim % heads:
        raise ValueError("Invalid MuScriptor dimensions")
    with safe_open(checkpoint, framework="np") as tensors:
        for name in tensors.keys():
            if tensors.get_slice(name).get_dtype() != "F32":
                raise ValueError(f"Expected original F32 weights: {name}")
        for name, shape in {
            "emb.0.weight": [card + 1, dim],
            "linears.0.weight": [card, dim],
            "condition_provider.conditioners.self_wav.output_proj.weight": [dim, 512],
            **{f"transformer.layers.{layer}.self_attn.in_proj_weight": [dim * 3, dim]
               for layer in range(layers)},
        }.items():
            if tensors.get_slice(name).get_shape() != shape:
                raise ValueError(f"Checkpoint/config shape mismatch: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.converter.resolve()), "--input", str(checkpoint),
               "--output", str(args.output.resolve()), "--type", "f32",
               "--family", "muscriptor", "--model-spec", str(root / "model_specs/muscriptor.json")]
    if args.overwrite:
        command.append("--overwrite")
    # A clean sidecar root excludes download caches and macOS metadata files.
    with tempfile.TemporaryDirectory(prefix="muscriptor-gguf-") as package_root:
        shutil.copyfile(source / "config.json", Path(package_root) / "config.json")
        subprocess.run(command + ["--root", package_root], check=True)


if __name__ == "__main__":
    main()
