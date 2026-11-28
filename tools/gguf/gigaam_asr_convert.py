#!/usr/bin/env python3
"""Convert an official Hugging Face GigaAM ASR snapshot (v3, Multilingual or v4) to a self-contained GGUF."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


VARIANTS = (
    "v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt",
    "multilingual_ctc", "multilingual_large_ctc",
)


def convert_v4(args, cfg):
    """GigaAM-v4 dual-head Zapformer snapshot (ai-sage/GigaAM-v4, rnnt revision).

    Stages tensors without running model code: drops the "model." prefix and the RoPE tables, stores scalars as [1],
    adds "<stack>.proj.weight_t" (the transposed OrthogonalLinear used by the stack bypass) and embeds the config and
    both SentencePiece tokenizers.
    """
    import numpy as np
    from safetensors.numpy import load_file, save_file

    if not cfg.get("encoder", {}).get("_target_", "").endswith("ZapformerEncoder") or "heads" not in cfg:
        raise ValueError("Expected a GigaAM-v4 dual-head Zapformer ASR snapshot")
    for name, head in cfg["heads"].items():
        if not head["_target_"].endswith("RNNTHead"):
            raise ValueError(f"Only RNN-T heads are supported for now, got {head['_target_']} for {name}")

    state = load_file(str(args.source / "model.safetensors"))
    tensors = {}
    for name, value in state.items():
        name = name.removeprefix("model.")
        if ".rope_bank." in name:
            continue
        if value.ndim == 0:
            value = value.reshape(1)
        tensors[name] = np.ascontiguousarray(value)
    for name in list(tensors):
        if name.endswith(".proj.weight") and name.startswith("encoder.encoder.encoders."):
            tensors[name + "_t"] = np.ascontiguousarray(tensors[name].T)
    if args.type == "f32":
        tensors = {k: v.astype(np.float32) for k, v in tensors.items()}

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.staging_dir / "model.safetensors"))
    tokenizers = []
    for name, decoding in cfg["decodings"].items():
        file = Path(decoding["model_path"]).name
        decoding["model_path"] = file
        tokenizers.append(file)
        shutil.copyfile(args.source / file, args.staging_dir / file)
    (args.staging_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.converter.resolve()),
        "--input", str(args.staging_dir / "model.safetensors"),
        "--output", str(args.output), "--type", "orig" if args.type == "f32" else args.type,
        "--family", "gigaam_asr", "--model-spec", str(args.spec),
        "--root", str(args.staging_dir),
        "--sidecar", f"{args.staging_dir / 'config.json'}=config.json",
    ]
    if args.type in ("f16", "q8_0"):
        command += ["--keep-type", "preprocessor.*=f32"]
        for name, tensor in tensors.items():
            if tensor.ndim == 1:
                command += ["--keep-type", f"{name}*=f32"]
    if args.type == "q8_0":
        # Quantize only the encoder matrices; the frontend convolutions, depthwise kernels and heads stay as stored.
        command += ["--keep-type", "encoder.encoder_embed.conv*=orig", "--keep-type", "*depthwise_conv*=orig",
                    "--keep-type", "heads.*=orig"]
    for file in tokenizers:
        command += ["--sidecar", f"{args.staging_dir / file}={file}"]
    if args.overwrite:
        command += ["--overwrite"]
    if args.command_out:
        args.command_out.write_text("\n".join(command) + "\n")
        print(f"Staged GigaAM-v4 {cfg.get('model_name')}: {len(tensors)} tensors; command in {args.command_out}")
        return
    subprocess.run(command, check=True)
    print(f"Converted GigaAM-v4 {cfg.get('model_name')}: {len(tensors)} tensors, storage={args.type}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Hugging Face model snapshot")
    parser.add_argument("output", type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--spec", type=Path, default=Path("model_specs/gigaam_asr.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--type", choices=["orig", "f16", "f32", "q8_0"], default="orig",
                        help="Original precision, F16 weights with F32 frontend and normalization, "
                             "F32 (v4 only) or Q8_0 encoder matrices (v4 only)")
    parser.add_argument("--command-out", type=Path,
                        help="v4 only: stage, then write the audiocpp_gguf command (one argument per line) here")
    args = parser.parse_args()

    raw_config = json.loads((args.source / "config.json").read_text())["cfg"]
    if raw_config.get("encoder", {}).get("_target_", "").endswith("ZapformerEncoder"):
        convert_v4(args, raw_config)
        return
    if args.type not in ("orig", "f16"):
        raise ValueError("--type f32 and q8_0 are only supported for GigaAM-v4 checkpoints")
    config = raw_config["model"]["cfg"]
    variant = config["model_name"]
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported ASR variant: {variant}")
    import torch
    from safetensors.torch import save_file

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
