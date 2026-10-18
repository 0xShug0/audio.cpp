#!/usr/bin/env python3
"""Convert an official ESPnet OWSM-CTC v4 checkpoint to GGUF."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official Hugging Face snapshot root")
    parser.add_argument("output", type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--spec", type=Path, default=Path("model_specs/owsm_ctc.json"))
    parser.add_argument("--type", choices=("orig", "q8_0", "q4_k"), default="orig")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configs = list(args.source.glob("exp/s2t_train_*/config.yaml"))
    checkpoints = list(args.source.glob("exp/s2t_train_*/valid.total_count.ave_5best*.pth"))
    if len(configs) != 1 or len(checkpoints) != 1:
        raise ValueError("Expected one official OWSM-CTC v4 config and one ave-5 checkpoint")
    config = yaml.safe_load(configs[0].read_text())
    encoder = config["encoder_conf"]
    if config["encoder"] != "e_branchformer_ctc" or (encoder["output_size"], encoder["num_blocks"]) != (1024, 27):
        raise ValueError("Expected the OWSM-CTC v4 1B encoder")
    variant = "medium"
    if encoder["linear_units"] != encoder["output_size"] * 4:
        raise ValueError("OWSM-CTC v4 requires a 4x feed-forward width")

    state = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if name != "frontend.logmel.melmat"
    }
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.staging_dir / "model.safetensors")
    tokenizer = args.source / config["bpemodel"]
    shutil.copyfile(tokenizer, args.staging_dir / "bpe.model")
    runtime_config = {
        "variant": variant,
        "hidden_size": encoder["output_size"],
        "num_heads": encoder["attention_heads"],
        "encoder_layers": encoder["num_blocks"],
        "intermediate_size": encoder["linear_units"],
        "vocabulary_size": len(config["token_list"]),
        "max_audio_samples": 480000,
        "frontend_frames": 3001,
        "encoder_frames": 374,
    }
    prompt = config["promptencoder_conf"]
    runtime_config.update({
        "prompt_hidden_size": prompt["output_size"],
        "prompt_num_heads": prompt["attention_heads"],
        "prompt_layers": prompt["num_blocks"],
        "prompt_intermediate_size": prompt["linear_units"],
        "interctc_layers": encoder["interctc_layer_idx"],
        "cross_attention_layers": [
            i for i, enabled in enumerate(encoder["use_cross_attention"]) if enabled
        ],
    })
    (args.staging_dir / "config.json").write_text(json.dumps(runtime_config, indent=2) + "\n")

    command = [
        str(args.converter.resolve()),
        "--input", str((args.staging_dir / "model.safetensors").resolve()),
        "--root", str(args.staging_dir.resolve()),
        "--output", str(args.output.resolve()),
        "--type", args.type,
        "--family", "owsm_ctc",
        "--model-spec", str(args.spec.resolve()),
        "--sidecar", f"{(args.staging_dir / 'config.json').resolve()}=config.json",
        "--sidecar", f"{(args.staging_dir / 'bpe.model').resolve()}=bpe.model",
    ]
    if args.type != "orig":
        for name, tensor in tensors.items():
            if tensor.ndim != 2:
                command.extend(("--keep-type", f"{name}=f32"))
    if args.overwrite:
        command.append("--overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    print(f"Converted OWSM-CTC v4 {variant}: {len(tensors)} tensors, storage={args.type}")


if __name__ == "__main__":
    main()
