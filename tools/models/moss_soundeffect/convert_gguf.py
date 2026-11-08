#!/usr/bin/env python3
"""Package an official MOSS-SoundEffect v2 checkpoint for audio.cpp."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BLOCK_NAMES = {
    **{f"attn{index}.{hf}.{suffix}": f"{native}.{name}.{suffix}"
       for index, native in [(1, "self_attn"), (2, "cross_attn")]
       for hf, name in [("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "o")]
       for suffix in ["weight", "bias"]},
    **{f"attn{index}.norm_{name}.weight": f"{native}.norm_{name}.weight"
       for index, native in [(1, "self_attn"), (2, "cross_attn")] for name in ["q", "k"]},
    **{f"ffn.net.{hf}.{suffix}": f"ffn.{native}.{suffix}"
       for hf, native in [("0.proj", "0"), ("2", "2")] for suffix in ["weight", "bias"]},
    "norm2.weight": "norm3.weight", "norm2.bias": "norm3.bias",
    "scale_shift_table": "modulation",
}
GLOBAL_NAMES = {
    **{f"condition_embedder.{hf}.{suffix}": f"{native}.{suffix}"
       for hf, native in [("text_embedder.linear_1", "text_embedding.0"),
                          ("text_embedder.linear_2", "text_embedding.2"),
                          ("time_embedder.linear_1", "time_embedding.0"),
                          ("time_embedder.linear_2", "time_embedding.2"),
                          ("time_proj", "time_projection.1")]
       for suffix in ["weight", "bias"]},
    "scale_shift_table": "head.modulation", "proj_out.weight": "head.head.weight",
    "proj_out.bias": "head.head.bias", "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", choices=["f32", "f16", "bf16", "q8_0"], default="bf16")
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.model_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = Path(__file__).resolve().parents[3] / "model_specs/moss_soundeffect.json"
    with tempfile.TemporaryDirectory(prefix="moss-sfx-", dir=args.output_dir) as tmp:
        stage = Path(tmp)
        original = load_file(source / "transformer/diffusion_pytorch_model.safetensors")
        dit = {}
        for name, value in original.items():
            if name.startswith("blocks."):
                _, index, suffix = name.split(".", 2)
                target = f"blocks.{index}.{BLOCK_NAMES[suffix]}"
            else:
                target = GLOBAL_NAMES[name]
            if target in dit:
                raise ValueError(f"duplicate DiT tensor: {target}")
            dit[target] = value
        f32_dit = [name for name in dit if name.endswith(("modulation", ".bias")) or "norm" in name
                   or name.startswith(("time_embedding.", "time_projection."))]
        save_file(dit, stage / "dit.safetensors")
        del original, dit
        # The official DAC archive is a state dictionary and constructor metadata.
        archive = torch.load(source / "vae/vae_128d_48k.pth", map_location="cpu", weights_only=False)
        vae_config = archive["metadata"]["kwargs"]
        if not vae_config["continuous"]:
            raise ValueError("MOSS-SoundEffect v2 requires the continuous DAC VAE")
        original = archive["state_dict"]
        vae = {}
        for name, value in original.items():
            if not name.startswith(("decoder.", "post_quant_conv.")):
                continue
            if name.endswith(".weight_g"):
                continue
            if name.endswith(".weight_v"):
                prefix = name.removesuffix("weight_v")
                value = torch._weight_norm(value, original[prefix + "weight_g"], 0)
                name = prefix + "weight"
            vae[name] = value.contiguous()
        save_file(vae, stage / "vae.safetensors")
        del archive, original, vae
        index = json.loads((source / "model_index.json").read_text())
        config = {
            "sample_rate": index["sample_rate"], "max_duration_sec": index["max_inference_seconds"],
            "transformer": json.loads((source / "transformer/config.json").read_text()),
            "text_encoder": json.loads((source / "text_encoder/config.json").read_text()),
            "vae": vae_config,
        }
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]:
            shutil.copyfile(source / "tokenizer" / name, stage / name)
        output = args.output_dir / f"moss-soundeffect-v2-{args.type}.gguf"
        command = [str(args.converter.resolve()), "--input", f"text_encoder={source / 'text_encoder/model.safetensors.index.json'}",
                   "--input", f"dit={stage / 'dit.safetensors'}", "--input", f"vae={stage / 'vae.safetensors'}",
                   "--root", str(stage), "--output", str(output.resolve()), "--type", args.type,
                   "--model-spec", str(spec), "--keep-type", "vae/*=f32",
                   "--exclude-prefix", "text_encoder/lm_head.", "--log"]
        for name in f32_dit:
            command.extend(["--keep-type", f"dit/{name}=f32"])
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
