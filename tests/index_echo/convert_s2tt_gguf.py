#!/usr/bin/env python3
"""Pack an Index-Echo-S2TT checkpoint into a GGUF without unused vision weights."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file


REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "build/debug/bin/audiocpp_gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", default="orig", choices=("orig", "f16", "bf16", "q8_0", "q4_k"))
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    llm = source / "llm"
    config = json.loads((llm / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["text_config"]["hidden_size"] if "text_config" in config else config["hidden_size"])
    size = "2b" if hidden == 2048 else "9b" if hidden == 4096 else None
    if size is None:
        raise ValueError(f"unsupported Index-Echo decoder hidden size: {hidden}")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="index-echo-s2tt-", dir=output) as temporary:
        staging = Path(temporary)
        if "text_config" in config:
            index = json.loads((llm / "model.safetensors.index.json").read_text(encoding="utf-8"))
            weights = {
                name: shard for name, shard in index["weight_map"].items()
                if name.startswith("model.language_model.")
            }
            if not weights:
                raise ValueError("Index-Echo nested decoder weights are missing")
            decoder_source = staging / "model.safetensors.index.json"
            decoder_source.write_text(
                json.dumps({"weight_map": weights}, sort_keys=True), encoding="utf-8"
            )
            for shard in set(weights.values()):
                (staging / shard).symlink_to(llm / shard)
        else:
            decoder_source = llm / "model.safetensors"
            if not decoder_source.is_file():
                raise ValueError("Index-Echo standalone decoder weights are missing")

        audio_weights = load_file(source / "audio_tower.safetensors")
        normalized_audio = staging / "audio_tower.safetensors"
        save_file(
            {"thinker.audio_tower." + name: tensor for name, tensor in audio_weights.items()},
            normalized_audio,
        )
        del audio_weights

        destination = output / f"index-echo-s2tt-{size}-{args.type}.gguf"
        temporary_output = staging / destination.name
        command = [
            str(CONVERTER),
            "--input", f"stlm={decoder_source}",
            "--input", f"audio={normalized_audio}",
            "--input", f"connector={source / 'connector.safetensors'}",
            "--output", str(temporary_output),
            "--type", args.type,
            "--family", "index_echo",
            "--model-spec", str(REPO / "model_specs/index_echo.json"),
            "--sidecar", f"{llm / 'config.json'}=llm/config.json",
            "--sidecar", f"{llm / 'tokenizer.json'}=llm/tokenizer.json",
            "--sidecar", f"{llm / 'tokenizer_config.json'}=llm/tokenizer_config.json",
            "--sidecar", f"{source / 'audio_config.json'}=audio_config.json",
            "--overwrite",
        ]
        if size == "2b" and args.type == "q4_k":
            # The 2B linear-attention QKV projections trigger GGML Q4_K's
            # nearest_int assertion; preserve those projections at Q8_0.
            for name in sorted(weights):
                if name.endswith("linear_attn.in_proj_qkv.weight"):
                    command.extend(["--keep-type", f"stlm/{name}=q8_0"])
        subprocess.run(command, check=True)
        subprocess.run([str(CONVERTER), "--inspect", str(temporary_output)], check=True)
        os.replace(temporary_output, destination)


if __name__ == "__main__":
    main()
