#!/usr/bin/env python3
"""Build a self-contained Maya1 GGUF from the original checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "build/debug/bin/audiocpp_gguf"
SPEC = REPO / "model_specs/maya1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    codec_dir = args.codec_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(REPO / "build/debug"),
            "--target",
            "audiocpp_gguf",
            "-j",
            str(os.cpu_count() or 8),
        ],
        check=True,
    )

    with tempfile.TemporaryDirectory(prefix="maya1-convert-") as temporary:
        staging = Path(temporary)
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            shutil.copyfile(model_dir / name, staging / name)

        model_index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        weight_map = dict(model_index["weight_map"])
        for shard in set(weight_map.values()):
            (staging / shard).symlink_to(model_dir / shard)

        codec_state = torch.load(
            codec_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        codec_state = {
            "codec." + name: tensor.contiguous()
            for name, tensor in codec_state.items()
            if name.startswith(("quantizer.quantizers.", "decoder."))
        }
        codec_shard = "codec.safetensors"
        save_file(codec_state, staging / codec_shard)
        weight_map.update({name: codec_shard for name in codec_state})
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, sort_keys=True),
            encoding="utf-8",
        )

        for storage_type, filename in (
            ("native", "maya1-orig.gguf"),
            ("q8_0", "maya1-q8_0.gguf"),
        ):
            descriptor, temporary_output = tempfile.mkstemp(
                prefix="maya1-", suffix=".gguf", dir=output_dir
            )
            os.close(descriptor)
            temporary_output = Path(temporary_output)
            try:
                subprocess.run(
                    [
                        str(CONVERTER),
                        "--input",
                        str(staging / "model.safetensors.index.json"),
                        "--root",
                        str(staging),
                        "--output",
                        str(temporary_output),
                        "--type",
                        storage_type,
                        "--family",
                        "maya1",
                        "--model-spec",
                        str(SPEC),
                        "--overwrite",
                    ],
                    check=True,
                )
                output = output_dir / filename
                os.replace(temporary_output, output)
                subprocess.run(
                    [str(CONVERTER), "--inspect", str(output)], check=True
                )
                print(output)
            finally:
                temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
