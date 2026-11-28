#!/usr/bin/env python3
"""Package an Irodori inference checkpoint and DACVAE as a standalone GGUF."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from safetensors import safe_open

K_QUANT_TYPES = ("q4_k", "q5_k", "q6_k")
K_QUANT_BLOCK = 256


def k_quant_fallbacks(source) -> list[str]:
    """2-D weights that a k-quant would leave in F32 because their input width
    is not a multiple of 256 (v4 Large's 1152-wide T5Gemma 2 projections).
    They are stored as Q8_0 instead. Embeddings keep the converter's F16 rule."""
    names = []
    for name in source.keys():
        shape = source.get_slice(name).get_shape()
        if (name.endswith(".weight") and len(shape) == 2 and "embed" not in name.lower()
                and shape[-1] % K_QUANT_BLOCK != 0 and shape[-1] % 32 == 0):
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Irodori model.safetensors.")
    parser.add_argument("--codec", type=Path, required=True, help="DACVAE weights.safetensors.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=["orig", "f16", "bf16", "q8_0", *K_QUANT_TYPES], default="q8_0")
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    codec = args.codec.resolve(strict=True)
    converter = args.converter.resolve(strict=True)
    with safe_open(model, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        fallbacks = k_quant_fallbacks(source) if args.type in K_QUANT_TYPES else []
    config = json.loads(metadata["config_json"])
    sidecars = {"model_config.json": config}
    if config.get("text_encoder_type") == "pretrained":
        sidecars["text_encoder_config.json"] = json.loads(metadata["text_encoder_config_json"])
    tokenizer = model.parent / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Missing checkpoint tokenizer: {tokenizer}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="irodori_gguf_") as directory:
        command = [str(converter), "--family", "irodori_tts",
                   "--input", f"model_weights={model}",
                   "--input", f"codec_weights={codec}",
                   "--root", str(model.parent), "--type", args.type,
                   "--output", str(args.output.resolve())]
        for name, value in sidecars.items():
            path = Path(directory) / name
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            command.extend(["--sidecar", f"{path}={name}"])
        for name in fallbacks:
            command.extend(["--keep-type", f"model_weights/{name}=q8_0"])
        if args.overwrite:
            command.append("--overwrite")
        if args.log:
            command.append("--log")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
