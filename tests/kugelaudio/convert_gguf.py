#!/usr/bin/env python3
"""Package original KugelAudio weights, tokenizer and four preset voices as GGUF."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from safetensors.torch import save_file


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", choices=("native", "bf16", "f16", "q8_0", "q4_k"), default="native")
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
        # The open generator consumes preset acoustic features, never either encoder.
        excluded = ("model.acoustic_tokenizer.encoder.", "model.semantic_tokenizer.", "model.semantic_connector.")
        weights = {name: shard for name, shard in index["weight_map"].items()
                   if not name.startswith(excluded)}
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
        suffix = "bf16" if args.type == "native" else args.type
        output = args.output_dir / f"kugelaudio-0-open-{suffix}.gguf"
        subprocess.run([
            str(args.converter), "--input", str(staging / "model.safetensors.index.json"),
            "--root", str(staging), "--output", str(output), "--type", args.type,
            "--family", "kugelaudio", "--model-spec", str(repo / "model_specs/kugelaudio.json"),
        ], check=True)
        subprocess.run([str(args.converter), "--inspect", str(output)], check=True)


if __name__ == "__main__":
    main()
