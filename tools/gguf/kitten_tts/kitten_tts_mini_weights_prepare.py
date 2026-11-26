#!/usr/bin/env python3
"""Prepare official TTS2 Mini weights for the existing kitten_tts spec-v1 packer.

The neural state dict is retained unchanged; voice packs and frontend metadata
are embedded so the resulting GGUF needs no sidecar files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.model / "config.json").read_text())
    if source["type"] != "KITTEN2_TINY" or source["family"] != "mini":
        raise ValueError("This converter requires the KittenTTS 2 Mini checkpoint")
    if len(source["models"]) != 1:
        raise ValueError("Mini must have one multilingual neural model")
    model = next(iter(source["models"].values()))
    architecture = model["architecture"]
    if architecture["decoder"]["type"] != "istftnet" or architecture["hidden_dim"] != 512:
        raise ValueError("Unsupported Mini decoder architecture")
    tensors = load_file(args.model / model["safetensors"])
    config = {k: v for k, v in architecture.items() if k not in ("albert", "decoder")}
    config.update(variant="mini_2", plbert=architecture["albert"], istftnet=architecture["decoder"],
                  vocab=json.loads((args.model / source["vocab"]).read_text()),
                  voice_aliases={}, voice_languages={}, default_voices={})
    for language, entry in source["languages"].items():
        with np.load(args.model / entry["voices"]) as pack:
            for voice in source["voices"][language]:
                key = voice["key"]
                identifier = f"{language}.{key}"
                style = np.ascontiguousarray(pack[key], dtype=np.float32)
                if style.ndim != 2 or style.shape[1] != 256:
                    raise ValueError(f"Invalid voice shape: {identifier}: {style.shape}")
                tensors[f"voices.{identifier}"] = style
                config["voice_languages"][identifier] = language
                config["default_voices"].setdefault(language, identifier)
                # Unqualified duplicated names resolve to the first language
                # advertised by upstream; language-qualified IDs are unambiguous.
                for alias in (voice["name"], key):
                    config["voice_aliases"].setdefault(alias, identifier)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ggml").mkdir(exist_ok=True)
    output = args.output_dir / "ggml/kitten_tts.safetensors"
    if output.exists():
        raise FileExistsError(output)
    save_file(tensors, output)
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(tensors)} tensors, {len(config['voice_languages'])} voices: {args.output_dir}")


if __name__ == "__main__":
    main()
