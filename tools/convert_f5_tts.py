#!/usr/bin/env python3
"""Convert Habibi/F5-TTS safetensors checkpoints to audio.cpp GGUF packages.

Produces one self-contained GGUF per checkpoint with two tensor namespaces:
  transformer.*  — the DiT flow-matching transformer (raw EMA torch names)
  vocos.*        — the Vocos mel vocoder

and copies vocab.txt alongside it, giving a complete model directory that
audiocpp_cli / audiocpp_server load with --family f5_tts (aliases: habibi,
habibi_tts). A standalone vocoder GGUF can also be produced for use with the
original safetensors checkpoints.

Examples:
  # unified checkpoint (default package)
  python3 tools/convert_f5_tts.py \
      --checkpoint /models/Habibi-TTS/Unified \
      --vocos /models/vocos-mel-24khz/vocos.safetensors \
      --converter build/bin/audiocpp_gguf --name habibi-unified

  # every specialized dialect checkpoint under a Habibi-TTS root
  python3 tools/convert_f5_tts.py --checkpoint-root /models/Habibi-TTS \
      --vocos /models/vocos-mel-24khz/vocos.safetensors \
      --converter build/bin/audiocpp_gguf

  # standalone vocoder package
  python3 tools/convert_f5_tts.py --vocos-only \
      --vocos /models/vocos-mel-24khz/vocos.safetensors \
      --converter build/bin/audiocpp_gguf
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = REPO_ROOT / "model_specs" / "f5_tts.json"


def convert(converter: Path, checkpoint: Path, vocos: Path, output: Path,
            quant_type: str, overwrite: bool) -> None:
    ckpt = _find_checkpoint(checkpoint)
    command = [
        str(converter),
        "--input", f"transformer={ckpt}",
        "--input", f"vocos={vocos}",
        "--root", str(checkpoint),
        "--family", "f5_tts",
        "--model-spec", str(SPEC),
        "--type", quant_type,
        "--output", str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    print("+", " ".join(command))
    subprocess.run(command, check=True)
    shutil.copyfile(checkpoint / "vocab.txt", output.parent / "vocab.txt")
    print(f"copied vocab.txt -> {output.parent}")


def convert_vocos(converter: Path, vocos: Path, output: Path,
                  quant_type: str, overwrite: bool) -> None:
    command = [
        str(converter),
        "--input", f"vocos={vocos}",
        "--root", str(vocos.parent),
        "--type", quant_type,
        "--allow-missing-model-spec",
        "--no-sidecars",
        "--output", str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def _find_checkpoint(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.safetensors"))
    if not candidates:
        raise SystemExit(f"no .safetensors checkpoint in {directory}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        help="one checkpoint directory (model_*.safetensors + vocab.txt)")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="Habibi-TTS root: converts Unified/ and every Specialized/* dir")
    parser.add_argument("--vocos", type=Path, required=True,
                        help="vocos.safetensors (bundled into each checkpoint GGUF)")
    parser.add_argument("--vocos-only", action="store_true",
                        help="only convert the standalone vocoder package")
    parser.add_argument("--converter", type=Path, required=True,
                        help="path to the audiocpp_gguf binary")
    parser.add_argument("--output-dir", type=Path, default=Path("gguf-out"))
    parser.add_argument("--type", default="orig",
                        choices=["orig", "f16", "bf16", "q8_0", "q2_k", "q3_k",
                                 "q4_k", "q5_k", "q6_k"],
                        help="GGUF storage type (default orig = keep f32)")
    parser.add_argument("--name", help="GGUF base name for --checkpoint (default: dir name, lowercased)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    converter = args.converter.resolve()
    if not converter.is_file():
        raise SystemExit(f"converter not found: {converter} (build target audiocpp_gguf)")
    if not args.vocos.is_file():
        raise SystemExit(f"vocos checkpoint not found: {args.vocos}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vocos_only:
        convert_vocos(converter, args.vocos.resolve(),
                      args.output_dir / f"vocos-mel-24khz-{args.type}.gguf",
                      args.type, args.overwrite)
        return

    jobs: list[tuple[Path, str]] = []
    if args.checkpoint:
        name = args.name or args.checkpoint.name.lower()
        jobs.append((args.checkpoint, name))
    elif args.checkpoint_root:
        root = args.checkpoint_root
        if (root / "Unified").is_dir():
            jobs.append((root / "Unified", "habibi-unified"))
        for d in sorted((root / "Specialized").glob("*")):
            if d.is_dir():
                jobs.append((d, f"habibi-{d.name.lower()}"))
    else:
        raise SystemExit("pass --checkpoint or --checkpoint-root (or --vocos-only)")

    for checkpoint, name in jobs:
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        convert(converter, checkpoint, args.vocos.resolve(),
                out_dir / f"{name}-{args.type}.gguf", args.type, args.overwrite)

    # standalone vocoder package alongside the checkpoints
    convert_vocos(converter, args.vocos.resolve(),
                  args.output_dir / f"vocos-mel-24khz-{args.type}.gguf",
                  args.type, args.overwrite)
    print("\nDone. Upload the .gguf files + each package's vocab.txt to the hosting repo.")


if __name__ == "__main__":
    main()
