#!/usr/bin/env python3
"""Build AuK component GGUFs from upstream safetensors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "build/debug/bin/audiocpp_gguf"


def main() -> None:
    variants = (
        ("qwen-bf16", "qwen2.5-omni-3b-bf16.gguf", "bf16", 923),
        ("qwen-q8", "qwen2.5-omni-3b-q8_0.gguf", "q8_0", 923),
        ("base-f32", "auk-base-f32.gguf", "f32", 420),
        ("base-f16", "auk-base-f16.gguf", "f16", 420),
        ("base-q8", "auk-base-q8_0.gguf", "q8_0", 420),
        ("flash-f32", "auk-flash-f32.gguf", "f32", 420),
        ("flash-f16", "auk-flash-f16.gguf", "f16", 420),
        ("flash-q8", "auk-flash-q8_0.gguf", "q8_0", 420),
        ("vae-f32", "auk-vae-f32.gguf", "f32", 1137),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="AuK Base checkpoint directory")
    parser.add_argument("--flash-dir", type=Path, required=True, help="AuK-Flash checkpoint directory")
    parser.add_argument("--qwen-dir", type=Path, required=True, help="Qwen2.5-Omni-3B checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="destination for component GGUFs and sidecars")
    parser.add_argument("--only", choices=[item[0] for item in variants], action="append")
    args = parser.parse_args()
    selected = set(args.only or [item[0] for item in variants])
    base_dir = args.base_dir.resolve()
    flash_dir = args.flash_dir.resolve()
    qwen_dir = args.qwen_dir.resolve()
    output_dir = args.output_dir.resolve()
    subprocess.run(
        ["cmake", "--build", str(REPO / "build/debug"), "--target", "audiocpp_gguf", "-j", "8"],
        check=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    (output_dir / "tokenizer").mkdir(exist_ok=True)
    shutil.copyfile(base_dir / "config.yaml", output_dir / "config/auk-base.yaml")
    shutil.copyfile(flash_dir / "config.yaml", output_dir / "config/auk-flash.yaml")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(qwen_dir / name, output_dir / "tokenizer" / name)

    with tempfile.TemporaryDirectory(prefix="auk-qwen-index-") as workspace:
        shards = Path(workspace)
        index = json.loads((qwen_dir / "model.safetensors.index.json").read_text())
        weight_map = {
            name: shard for name, shard in index["weight_map"].items()
            if name.startswith(("thinker.model.", "thinker.audio_tower."))
        }
        if len(weight_map) != 923:
            raise RuntimeError(f"expected 923 AuK Qwen tensors, found {len(weight_map)}")
        (shards / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, sort_keys=True))
        for name in set(weight_map.values()):
            (shards / name).symlink_to(qwen_dir / name)

        sources = {
            "qwen": shards / "model.safetensors.index.json",
            "base": base_dir / "auk_base.safetensors",
            "flash": flash_dir / "auk_flash.safetensors",
            "vae": base_dir / "vae.safetensors",
        }
        for variant, filename, weight_type, tensor_count in variants:
            if variant not in selected:
                continue
            source = sources[variant.split("-")[0]]
            fd, temporary = tempfile.mkstemp(prefix=variant + "-", suffix=".gguf", dir=output_dir)
            os.close(fd)
            temporary = Path(temporary)
            try:
                subprocess.run([
                    str(CONVERTER), "--input", str(source), "--root", str(output_dir),
                    "--output", str(temporary), "--type", weight_type,
                    "--family", "auk", "--model-spec", str(REPO / "model_specs/auk.json"),
                    "--no-sidecars", "--allow-missing-model-spec", "--overwrite",
                ], check=True)
                inspection = subprocess.run(
                    [str(CONVERTER), "--inspect", str(temporary)],
                    check=True, capture_output=True, text=True,
                ).stdout
                if f"tensors={tensor_count}" not in inspection.splitlines() or \
                   "embedded_sidecar_count=0" not in inspection.splitlines():
                    raise RuntimeError(f"AuK component inspection failed:\n{inspection}")
                output = output_dir / filename
                os.replace(temporary, output)
                print(f"Rebuilt {output}\n{inspection}", flush=True)
            finally:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
