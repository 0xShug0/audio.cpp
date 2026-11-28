#!/usr/bin/env python3
"""Fold OpenVoice weight normalization and package the standalone converter."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=("f32", "f16", "bf16", "q8_0"), default="f32")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(args.upstream.resolve()))
    from openvoice.models import SynthesizerTrn

    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    model = SynthesizerTrn(
        0, config["data"]["filter_length"] // 2 + 1,
        n_speakers=config["data"]["n_speakers"], **config["model"]
    ).eval()
    checkpoint = torch.load(args.checkpoint_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    for module in model.modules():
        if hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    tensors = {name: value.detach().contiguous() for name, value in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = repo / "build/logs/tone_color_vc"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=staging) as temporary:
        root = Path(temporary)
        source = root / "model.safetensors"
        save_file(tensors, source)
        restored = load_file(source)
        for name, value in tensors.items():
            if not torch.equal(value, restored[name]):
                raise RuntimeError(f"Safetensors round-trip mismatch: {name}")
        shutil.copyfile(args.checkpoint_dir / "config.json", root / "config.json")
        type_options = []
        for name, tensor in tensors.items():
            if tensor.ndim == 1:
                type_options.extend(["--keep-type", f"weights/{name}=f32"])
            elif args.type == "q8_0":
                storage = "q8_0" if tensor.ndim == 2 and tensor.shape[-1] % 32 == 0 else "f16"
                type_options.extend(["--keep-type", f"weights/{name}={storage}"])
        subprocess.run([
            str(repo / "build/debug/bin/audiocpp_gguf"),
            "--input", f"weights={source}", "--root", str(root),
            "--output", str(args.output), "--type", args.type,
            "--family", "tone_color_vc",
            "--model-spec", str(repo / "model_specs/tone_color_vc.json"),
            "--overwrite",
            *type_options,
        ], check=True)
    print(f"Converted {len(tensors)} tensors with folded weight normalization")


if __name__ == "__main__":
    main()
