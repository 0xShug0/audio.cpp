#!/usr/bin/env python3
"""Convert Audio8 TTS safetensors checkpoints to audio.cpp GGUF packages.

Produces one self-contained GGUF per precision with two tensor namespaces:
  model_weights.*  — DualAR slow/fast transformer weights (model.safetensors)
  codec_weights.*  — arktts neural codec weights (converted codec.safetensors)

and embeds config.json / tokenizer_config.json / tokenizer.json plus the
audio8_tts package spec from the repository, giving a standalone model file
that audiocpp_cli / audiocpp_server load with --family audio8_tts.

The tool deliberately does not download files or read PyTorch checkpoints.
Point --model-dir at a snapshot that already contains the HF weights plus the
codec.safetensors artifact produced by convert_audio8_tts_codec.py.

Examples:
  # 16-bit reference package (AR stays BF16, codec downcasts F32 -> BF16)
  python3 tools/community_models/convert_audio8_tts.py \
      --model-dir /models/Audio8-TTS-Preview-0.6b \
      --converter build/bin/audiocpp_gguf --type bf16

  # default Q8_0 package like fish_audio ships
  python3 tools/community_models/convert_audio8_tts.py \
      --model-dir /models/Audio8-TTS-Preview-0.6b \
      --converter build/bin/audiocpp_gguf --type q8_0

Codec conv stacks are sensitive to quantization. If a Q8_0 package drifts,
reconvert with mixed storage by keeping the codec namespace at 16 bit:
append "--keep-type", "codec_weights*=bf16" to a Q8_0 conversion.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "model_specs" / "audio8_tts.json"
FAMILY = "audio8_tts"


def convert(converter: Path, model_dir: Path, codec: Path, output: Path,
            quant_type: str, overwrite: bool) -> None:
    command = [
        str(converter),
        "--input", f"model_weights={model_dir / 'model.safetensors'}",
        "--input", f"codec_weights={codec}",
        "--root", str(model_dir),
        "--family", FAMILY,
        "--model-spec", str(SPEC),
        "--type", quant_type,
        "--output", str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="HF snapshot directory with config.json, tokenizer files, "
                             "model.safetensors, and codec.safetensors")
    parser.add_argument("--codec-safetensors", type=Path,
                        help="codec.safetensors override "
                             "(default: <model-dir>/codec.safetensors)")
    parser.add_argument("--converter", type=Path, required=True,
                        help="path to the audiocpp_gguf binary")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("gguf-out") / "audio8-tts",
                        help="directory for the produced GGUF files")
    parser.add_argument("--name", default="audio8-tts-preview-0.6b",
                        help="GGUF base name")
    parser.add_argument("--type", default="bf16",
                        choices=["orig", "f16", "bf16", "q8_0", "q2_k", "q3_k",
                                 "q4_k", "q5_k", "q6_k"],
                        help="GGUF storage type (default bf16)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    converter = args.converter.resolve()
    if not converter.is_file():
        raise SystemExit(f"converter not found: {converter} (build target audiocpp_gguf)")
    if not SPEC.is_file():
        raise SystemExit(f"package spec not found: {SPEC}")
    model_dir = args.model_dir.resolve()
    if not (model_dir / "model.safetensors").is_file():
        raise SystemExit(f"model.safetensors not found in {model_dir}")
    codec = (args.codec_safetensors or model_dir / "codec.safetensors").resolve()
    if not codec.is_file():
        raise SystemExit(f"codec safetensors not found: {codec} "
                         "(run convert_audio8_tts_codec.py first)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.name}-{args.type}.gguf"
    try:
        convert(converter, model_dir, codec, output, args.type, args.overwrite)
    except subprocess.CalledProcessError:
        sys.exit(1)

    print(f"\nDone. Load it with:\n"
          f"  audiocpp_cli --task tts --family {FAMILY} \\\n"
          f"      --model {output} --text \"...\" --out out.wav")


if __name__ == "__main__":
    main()
