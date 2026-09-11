#!/usr/bin/env python3
"""Convert moonshine-ai/moonshine streaming safetensors checkpoints to audio.cpp GGUF packages."""

import argparse
import json
import shutil
import subprocess
import struct
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "model_specs" / "moonshine_stt.json"
SCALAR_TENSORS = {"model.encoder.embedder.comp.log_k"}


def rank1_scalar_safetensors(source: Path, destination: Path) -> None:
    """Copy safetensors while exposing known rank-0 scalars as length-1 tensors."""
    with source.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        payload = f.read()

    changed = False
    for name in SCALAR_TENSORS:
        entry = header.get(name)
        if entry is not None and entry.get("shape") == []:
            entry["shape"] = [1]
            changed = True

    if not changed:
        shutil.copyfile(source, destination)
        return

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(encoded) % 8)) % 8
    destination.write_bytes(struct.pack("<Q", len(encoded) + padding) + encoded + (b" " * padding) + payload)


def convert(converter: Path, checkpoint: Path, output: Path, quant_type: str, overwrite: bool) -> None:
    ckpt = checkpoint / "model.safetensors"
    if not ckpt.exists():
        candidates = sorted(checkpoint.glob("*.safetensors"))
        if not candidates:
            raise SystemExit(f"No .safetensors checkpoint found in {checkpoint}")
        ckpt = candidates[-1]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moonshine-stt-convert-") as tmp_dir:
        converted_input = Path(tmp_dir) / ckpt.name
        rank1_scalar_safetensors(ckpt, converted_input)
        command = [
            str(converter),
            "--input",
            str(converted_input),
            "--root",
            str(checkpoint),
            "--family",
            "moonshine_stt",
            "--model-spec",
            str(SPEC),
            "--type",
            quant_type,
            "--output",
            str(output),
        ]
        if overwrite:
            command.append("--overwrite")
        print("+", " ".join(command))
        subprocess.run(command, check=True)

    for name in ["config.json", "tokenizer.json"]:
        source = checkpoint / name
        if source.exists():
            shutil.copyfile(source, output.parent / name)
            print(f"copied {name} -> {output.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="HF checkpoint directory")
    parser.add_argument("--converter", type=Path, required=True, help="path to audiocpp_gguf")
    parser.add_argument("--output", type=Path, required=True, help="output .gguf path")
    parser.add_argument(
        "--type",
        default="q8_0",
        choices=["orig", "f32", "f16", "bf16", "q8_0", "q4_k", "q5_k", "q6_k"],
        help="GGUF tensor storage type",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite output")
    args = parser.parse_args()
    convert(args.converter, args.checkpoint, args.output, args.type, args.overwrite)


if __name__ == "__main__":
    main()
