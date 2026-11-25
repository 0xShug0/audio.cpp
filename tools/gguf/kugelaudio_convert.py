#!/usr/bin/env python3
"""Package KugelAudio with preset voices and an optional reference encoder."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from safetensors.torch import save_file


def conversion_spec(repo, include_encoder):
    spec = json.loads((repo / "model_specs/kugelaudio.json").read_text(encoding="utf-8"))
    if not include_encoder:
        # The published legacy packages contain only preset features. Do not
        # advertise a raw-audio clone path from a preset-only embedded contract.
        spec["tasks"] = [task for task in spec["tasks"] if task != "clone"]
        spec["capabilities"].pop("clone", None)
        spec["ui"]["tags"] = [tag for tag in spec["ui"]["tags"] if tag != "Clone"]
        spec["description"] = "KugelAudio TTS with bundled preset voices and incremental output."
    return spec


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", choices=("native", "bf16", "f16", "q8_0", "q4_k"), default="native")
    parser.add_argument("--include-reference-encoder", action="store_true",
                        help="retain the original acoustic encoder for reference-audio voice cloning")
    parser.add_argument("--converter", type=Path, default=repo / "build/debug/bin/audiocpp_gguf")
    args = parser.parse_args()
    source = args.model_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = repo / "build/logs/kugelaudio"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert-", dir=staging_root) as temporary:
        staging = Path(temporary)
        shutil.copyfile(source / "config.json", staging / "config.json")
        tokenizer = staging / "text_tokenizer"
        tokenizer.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(args.tokenizer_dir / name, tokenizer / name)
        index = json.loads((source / "model.safetensors.index.json").read_text())
        excluded = ("model.semantic_tokenizer.", "model.semantic_connector.")
        if not args.include_reference_encoder:
            excluded += ("model.acoustic_tokenizer.encoder.",)
        weights = {name: shard for name, shard in index["weight_map"].items()
                   if not name.startswith(excluded)}
        if args.include_reference_encoder and not any(name.startswith("model.acoustic_tokenizer.encoder.") for name in weights):
            raise ValueError("the source checkpoint omits acoustic encoder weights")
        for shard in set(weights.values()):
            (staging / shard).symlink_to(source / shard)
        voices = {}
        for name in ("default", "clear", "english_female", "english_male"):
            voice = torch.load(source / "voices" / f"{name}.pt", map_location="cpu", weights_only=True)
            voices[f"voices.{name}.acoustic_mean"] = voice["acoustic_mean"].contiguous()
            voices[f"voices.{name}.acoustic_std"] = torch.as_tensor(voice["acoustic_std"]).reshape(1).contiguous()
        save_file(voices, staging / "voices.safetensors")
        weights.update({name: "voices.safetensors" for name in voices})
        (staging / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
        model_spec = staging / "model_spec.json"
        model_spec.write_text(json.dumps(conversion_spec(repo, args.include_reference_encoder), indent=2), encoding="utf-8")
        suffix = "bf16" if args.type == "native" else args.type
        if args.include_reference_encoder:
            suffix += "-clone"
        output = args.output_dir / f"kugelaudio-0-open-{suffix}.gguf"
        subprocess.run([
            str(args.converter), "--input", str(staging / "model.safetensors.index.json"),
            "--root", str(staging), "--output", str(output), "--type", args.type,
            "--family", "kugelaudio", "--model-spec", str(model_spec),
            "--keep-type", "model.acoustic_tokenizer.encoder.*=orig",
        ], check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        subprocess.run([str(args.converter), "--inspect", str(output)], check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


if __name__ == "__main__":
    main()
