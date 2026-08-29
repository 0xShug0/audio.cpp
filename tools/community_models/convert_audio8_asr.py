#!/usr/bin/env python3
"""Convert Audio8/Audio8-ASR-0.1B safetensors checkpoints to audio.cpp GGUF packages.

Produces a self-contained GGUF with weights from model.safetensors and embeds
the config/tokenizer sidecars, giving a complete model directory that
audiocpp_cli / audiocpp_server load with --family audio8_asr.

The checkpoint is CC-BY-NC-4.0: run this against a checkpoint you downloaded
yourself from https://huggingface.co/Audio8/Audio8-ASR-0.1B and keep the
result local. Do not redistribute the converted GGUF.

Examples:
  python tools/community_models/convert_audio8_asr.py \
      --checkpoint models/Audio8-ASR-0.1B-hf \
      --converter build/debug/bin/audiocpp_gguf \
      --type q8_0 \
      --output models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf
"""

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "model_specs" / "audio8_asr.json"


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
        "audio8_asr",
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

    for asset_name in ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
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
        default=REPO_ROOT / "models" / "Audio8-ASR-0.1B-hf",
        help="checkpoint directory containing model.safetensors, configs, and tokenizer files",
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
        default=Path("models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf"),
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
