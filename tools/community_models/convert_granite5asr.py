#!/usr/bin/env python3
"""Convert IBM Granite Speech 5.0 470M TurboCTC safetensors checkpoints to audio.cpp GGUF packages.

Produces a self-contained GGUF with weights from model.safetensors and copies
tokenizer.json and config.json alongside it, giving a complete model directory
that audiocpp_cli / audiocpp_server load with --family granite5asr.

Examples:
  python tools/community_models/convert_granite5asr.py \
      --checkpoint granite5asr \
      --converter build/windows-cpu-release/bin/audiocpp_gguf.exe \
      --type q8_0 \
      --output models/granite5asr-gguf/granite5asr-q8_0.gguf
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "model_specs" / "granite5asr.json"


def convert(
    converter: Path,
    checkpoint: Path,
    output: Path,
    quant_type: str,
    overwrite: bool,
) -> None:
    ckpt = checkpoint / "model.safetensors"
    if not ckpt.exists():
        candidates = sorted(checkpoint.glob("*.safetensors"))
        if not candidates:
            raise SystemExit(f"No .safetensors checkpoint found in {checkpoint}")
        ckpt = candidates[-1]

    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(converter),
        "--input",
        str(ckpt),
        "--root",
        str(checkpoint),
        "--family",
        "granite5asr",
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

    for asset_name in ["tokenizer.json", "config.json", "preprocessor_config.json"]:
        src_asset = checkpoint / asset_name
        if src_asset.exists():
            dst_asset = output.parent / asset_name
            shutil.copyfile(src_asset, dst_asset)
            print(f"copied {asset_name} -> {output.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "granite5asr",
        help="checkpoint directory containing model.safetensors and tokenizer.json",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        required=True,
        help="path to the audiocpp_gguf binary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gguf-out/granite-speech-5.0-470m-turboctc-q8_0.gguf"),
        help="output .gguf file path",
    )
    parser.add_argument(
        "--type",
        default="q8_0",
        choices=["orig", "f32", "f16", "bf16", "q8_0", "q4_k", "q5_k", "q6_k"],
        help="quantization type for GGUF tensors",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing output file",
    )

    args = parser.parse_args()
    convert(
        converter=args.converter,
        checkpoint=args.checkpoint,
        output=args.output,
        quant_type=args.type,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
