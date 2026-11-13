#!/usr/bin/env python3
"""Add original acoustic encoder weights to a preset-only KugelAudio GGUF.

Preserves existing quantized tensor types/values and embedded tokenizer/configs.
The resulting package embeds the current clone-aware model spec. The encoder
input must come from the same KugelAudio checkpoint as the main model.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--converter", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.model.resolve(), args.encoder.resolve()):
        parser.error("choose a new output path; source files are preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.converter.resolve()), "--input", str(args.model.resolve()),
        "--input", "reference_encoder=" + str(args.encoder.resolve()), "--type", "orig", "--family", "kugelaudio",
        "--model-spec", str(repo / "model_specs/kugelaudio.json"), "--output", str(args.output.resolve())],
        check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


if __name__ == "__main__":
    main()
