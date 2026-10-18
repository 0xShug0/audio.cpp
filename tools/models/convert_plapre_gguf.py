#!/usr/bin/env python3
"""Package Plapre's AR model, Kanade codec, WavLM and HiFT in one GGUF."""

import argparse
import importlib.resources
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import torch
import torchaudio
import yaml
from safetensors.torch import save_file


def wavlm_weights(state, layers):
    result = {}
    for name, tensor in state.items():
        if name.startswith("feature_extractor.conv_layers."):
            name = name.replace(".conv.weight", ".0.weight").replace(".layer_norm.", ".2.")
        elif name.startswith("encoder.feature_projection."):
            name = name.replace("encoder.feature_projection.layer_norm", "layer_norm")
            name = name.replace("encoder.feature_projection.projection", "post_extract_proj")
        elif name.startswith("encoder.transformer.pos_conv_embed.conv."):
            name = name.replace("encoder.transformer.pos_conv_embed.conv", "encoder.pos_conv.0")
            name = name.replace(".parametrizations.weight.original0", ".weight_g")
            name = name.replace(".parametrizations.weight.original1", ".weight_v")
        elif name.startswith("encoder.transformer.layer_norm."):
            name = name.replace("encoder.transformer.layer_norm", "encoder.layer_norm")
        else:
            match = re.fullmatch(r"encoder\.transformer\.layers\.(\d+)\.(.+)", name)
            if match is None:
                raise ValueError(f"Unrecognized WavLM tensor: {name}")
            layer, suffix = match.groups()
            if int(layer) >= layers:
                continue
            prefix = f"encoder.layers.{layer}."
            if suffix in ("attention.attention.in_proj_weight", "attention.attention.in_proj_bias"):
                parameter = "weight" if suffix.endswith("weight") else "bias"
                for kind, value in zip(("q", "k", "v"), tensor.chunk(3, dim=0), strict=True):
                    result[prefix + f"self_attn.{kind}_proj.{parameter}"] = value.contiguous()
                continue
            mapping = {
                "attention.gru_rel_pos_const": "self_attn.grep_a",
                "attention.rel_attn_embed.weight": "self_attn.relative_attention_bias.weight",
                "attention.attention.out_proj.weight": "self_attn.out_proj.weight",
                "attention.attention.out_proj.bias": "self_attn.out_proj.bias",
                "attention.gru_rel_pos_linear.weight": "self_attn.grep_linear.weight",
                "attention.gru_rel_pos_linear.bias": "self_attn.grep_linear.bias",
                "layer_norm.weight": "self_attn_layer_norm.weight",
                "layer_norm.bias": "self_attn_layer_norm.bias",
                "feed_forward.intermediate_dense.weight": "fc1.weight",
                "feed_forward.intermediate_dense.bias": "fc1.bias",
                "feed_forward.output_dense.weight": "fc2.weight",
                "feed_forward.output_dense.bias": "fc2.bias",
                "final_layer_norm.weight": "final_layer_norm.weight",
                "final_layer_norm.bias": "final_layer_norm.bias",
            }
            name = prefix + mapping[suffix]
        if name in result:
            raise ValueError(f"Duplicate WavLM tensor: {name}")
        result[name] = tensor.contiguous()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kanade", type=Path, required=True)
    parser.add_argument("--hift", type=Path, required=True)
    parser.add_argument("--speakers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    codec = args.kanade.resolve()
    output = args.output.resolve()
    spec = Path(__file__).resolve().parents[2] / "model_specs/plapre.json"
    speakers = args.speakers or Path(str(importlib.resources.files("plapre") / "speakers.json"))
    config = yaml.safe_load((codec / "config.yaml").read_text())["model"]["init_args"]
    ssl = config["ssl_feature_extractor"]["init_args"]
    if ssl["model_name"] != "wavlm_base_plus":
        raise ValueError("This converter supports Kanade's WavLM Base+ frontend")
    wavlm = wavlm_weights(torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().state_dict(), ssl["output_layer"])
    hift = torch.load(args.hift, map_location="cpu", weights_only=True)
    hift = {key.removeprefix("generator."): value.contiguous() for key, value in hift.items()}
    speaker_projection = torch.load(source / "speaker_proj.pt", map_location="cpu", weights_only=True)
    speaker_projection = {key: value.contiguous() for key, value in speaker_projection.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".plapre-convert-", dir=output.parent) as temporary:
        root = Path(temporary)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(source / name, root / name)
        shutil.copyfile(speakers, root / "speakers.json")
        (root / "kanade_config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(wavlm, root / "wavlm.safetensors")
        save_file(hift, root / "hift.safetensors")
        save_file(speaker_projection, root / "speaker.safetensors")
        command = [str(args.converter.resolve()),
                   "--input", str(source / "model.safetensors"),
                   "--input", f"speaker_proj={root / 'speaker.safetensors'}",
                   "--input", f"kanade={codec / 'model.safetensors'}",
                   "--input", f"wavlm={root / 'wavlm.safetensors'}",
                   "--input", f"hift={root / 'hift.safetensors'}",
                   "--root", str(root), "--model-spec", str(spec),
                   "--type", "orig", "--output", str(output)]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
