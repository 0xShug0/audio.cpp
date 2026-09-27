#!/usr/bin/env python3
"""Convert an official Hugging Face GigaAM ASR snapshot to a self-contained GGUF."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file


VARIANTS = (
    "v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt",
    "multilingual_ctc", "multilingual_large_ctc",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Hugging Face model snapshot")
    parser.add_argument("output", type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--spec", type=Path, default=Path("model_specs/gigaam_asr.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--type", choices=["orig", "f16"], default="orig",
                        help="Original precision, or F16 weights with F32 frontend and normalization")
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())["cfg"]["model"]["cfg"]
    variant = config["model_name"]
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported ASR variant: {variant}")
    original = torch.load(args.source / "pytorch_model.bin", map_location="cpu", weights_only=True)
    state = {name.removeprefix("model."): value for name, value in original.items()}
    if not any(name.startswith("head.") for name in state):
        raise ValueError("Expected a fine-tuned ASR checkpoint, not an SSL encoder")
    uses_sentencepiece = bool(config["decoding"].get("model_path"))
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.staging_dir / "model.safetensors")
    config["decoding"]["model_path"] = "tokenizer.model" if uses_sentencepiece else None
    (args.staging_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if uses_sentencepiece:
        shutil.copyfile(args.source / "tokenizer.model", args.staging_dir / "tokenizer.model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.converter.resolve()),
        "--input", str(args.staging_dir / "model.safetensors"),
        "--output", str(args.output), "--type", args.type,
        "--family", "gigaam_asr", "--model-spec", str(args.spec),
        "--root", str(args.staging_dir),
        "--sidecar", f"{args.staging_dir / 'config.json'}=config.json",
    ]
    if args.type == "f16":
        command += ["--keep-type", "preprocessor.*=f32"]
        for name, tensor in tensors.items():
            if tensor.ndim == 1:
                command += ["--keep-type", f"{name}*=f32"]
    if uses_sentencepiece:
        command += ["--sidecar", f"{args.staging_dir / 'tokenizer.model'}=tokenizer.model"]
    if args.overwrite:
        command += ["--overwrite"]
    subprocess.run(command, check=True)
    print(f"Converted {variant}: {len(tensors)} tensors, storage={args.type}")


if __name__ == "__main__":
    main()
