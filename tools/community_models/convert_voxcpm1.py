#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file


ATTN_MAP = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
}

MLP_MAP = {
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OpenBMB VoxCPM-0.5B weights to an audio.cpp-native GGUF."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/media/leo/Share/audio.cpp/models/VoxCPM-0.5b"),
        help="Directory containing OpenBMB config/tokenizer files, pytorch_model.bin, and audiovae.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/media/leo/Share/models/audio.cpp-gguf/VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf"),
        help="Output GGUF path.",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("build/debug/bin/audiocpp_gguf"),
        help="Path to the audiocpp_gguf converter.",
    )
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=Path("model_specs/voxcpm1.json"),
        help="VoxCPM1 model spec JSON.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("build/voxcpm1_native_gguf"),
        help="Directory for staged safetensors and sidecars.",
    )
    parser.add_argument("--type", default="q8_0", help="Main model conversion type.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output GGUF.")
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def unwrap_state_dict(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} checkpoint must be a dict")
    if "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"{label} state_dict must be a dict")
    return value


def map_layer(prefix, out_prefix, key):
    match = re.fullmatch(rf"{re.escape(prefix)}\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight", key)
    if match:
        return f"{out_prefix}.blk.{match.group(1)}.{ATTN_MAP[match.group(2)]}.weight"

    match = re.fullmatch(rf"{re.escape(prefix)}\.layers\.(\d+)\.input_layernorm\.weight", key)
    if match:
        return f"{out_prefix}.blk.{match.group(1)}.attn_norm.weight"

    match = re.fullmatch(rf"{re.escape(prefix)}\.layers\.(\d+)\.post_attention_layernorm\.weight", key)
    if match:
        return f"{out_prefix}.blk.{match.group(1)}.ffn_norm.weight"

    match = re.fullmatch(rf"{re.escape(prefix)}\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight", key)
    if match:
        return f"{out_prefix}.blk.{match.group(1)}.{MLP_MAP[match.group(2)]}.weight"

    return None


def map_main_key(key):
    direct = {
        "base_lm.embed_tokens.weight": "token_embd.weight",
        "base_lm.norm.weight": "output_norm.weight",
        "residual_lm.norm.weight": "residual_lm.output_norm.weight",
        "feat_encoder.in_proj.weight": "locenc.in_proj.weight",
        "feat_encoder.in_proj.bias": "locenc.in_proj.bias",
        "feat_encoder.special_token": "locenc.special_token",
        "feat_encoder.encoder.norm.weight": "locenc.output_norm.weight",
        "feat_decoder.estimator.in_proj.weight": "locdit.in_proj.weight",
        "feat_decoder.estimator.in_proj.bias": "locdit.in_proj.bias",
        "feat_decoder.estimator.cond_proj.weight": "locdit.cond_proj.weight",
        "feat_decoder.estimator.cond_proj.bias": "locdit.cond_proj.bias",
        "feat_decoder.estimator.out_proj.weight": "locdit.out_proj.weight",
        "feat_decoder.estimator.out_proj.bias": "locdit.out_proj.bias",
        "feat_decoder.estimator.time_mlp.linear_1.weight": "locdit.time_mlp.linear_1.weight",
        "feat_decoder.estimator.time_mlp.linear_1.bias": "locdit.time_mlp.linear_1.bias",
        "feat_decoder.estimator.time_mlp.linear_2.weight": "locdit.time_mlp.linear_2.weight",
        "feat_decoder.estimator.time_mlp.linear_2.bias": "locdit.time_mlp.linear_2.bias",
        "feat_decoder.estimator.delta_time_mlp.linear_1.weight": "locdit.delta_time_mlp.linear_1.weight",
        "feat_decoder.estimator.delta_time_mlp.linear_1.bias": "locdit.delta_time_mlp.linear_1.bias",
        "feat_decoder.estimator.delta_time_mlp.linear_2.weight": "locdit.delta_time_mlp.linear_2.weight",
        "feat_decoder.estimator.delta_time_mlp.linear_2.bias": "locdit.delta_time_mlp.linear_2.bias",
        "feat_decoder.estimator.decoder.norm.weight": "locdit.output_norm.weight",
        "fsq_layer.in_proj.weight": "fsq.in_proj.weight",
        "fsq_layer.in_proj.bias": "fsq.in_proj.bias",
        "fsq_layer.out_proj.weight": "fsq.out_proj.weight",
        "fsq_layer.out_proj.bias": "fsq.out_proj.bias",
        "enc_to_lm_proj.weight": "proj.enc_to_lm.weight",
        "enc_to_lm_proj.bias": "proj.enc_to_lm.bias",
        "lm_to_dit_proj.weight": "proj.lm_to_dit.weight",
        "lm_to_dit_proj.bias": "proj.lm_to_dit.bias",
        "res_to_dit_proj.weight": "proj.res_to_dit.weight",
        "res_to_dit_proj.bias": "proj.res_to_dit.bias",
        "stop_proj.weight": "stop.stop_proj.weight",
        "stop_proj.bias": "stop.stop_proj.bias",
        "stop_head.weight": "stop.stop_head.weight",
    }
    if key in direct:
        return direct[key]

    mapped = map_layer("base_lm", "", key)
    if mapped is not None:
        return mapped.removeprefix(".")

    mapped = map_layer("residual_lm", "residual_lm", key)
    if mapped is not None:
        return mapped

    mapped = map_layer("feat_encoder.encoder", "locenc", key)
    if mapped is not None:
        return mapped

    mapped = map_layer("feat_decoder.estimator.decoder", "locdit", key)
    if mapped is not None:
        return mapped

    raise KeyError(f"unmapped VoxCPM1 tensor: {key}")


def main_tensor_for_gguf(name, tensor):
    tensor = tensor.detach().cpu().contiguous()
    if name == "stop.stop_head.weight":
        return tensor.float()
    if tensor.ndim != 2:
        return tensor.float()
    return tensor


def load_main_tensors(path):
    state = unwrap_state_dict(torch.load(path, map_location="cpu"), "main")
    out = {}
    for key, tensor in state.items():
        mapped = map_main_key(key)
        if mapped in out:
            raise RuntimeError(f"duplicate mapped tensor: {mapped}")
        out[mapped] = main_tensor_for_gguf(mapped, tensor)
    return out


def convert_weight_norm(state):
    out = {}
    weight_norm_bases = set()
    for key in state:
        if key.endswith(".weight_g") or key.endswith(".weight_v"):
            weight_norm_bases.add(key.rsplit(".", 1)[0])

    for key, tensor in state.items():
        if key.endswith(".weight_g") or key.endswith(".weight_v"):
            continue
        out[key] = tensor

    for base in sorted(weight_norm_bases):
        weight_g_name = f"{base}.weight_g"
        weight_v_name = f"{base}.weight_v"
        if weight_g_name not in state or weight_v_name not in state:
            raise KeyError(f"missing weight-norm pair for {base}")
        weight_g = state[weight_g_name]
        weight_v = state[weight_v_name]
        norm_dims = [
            index
            for index in range(weight_v.dim())
            if weight_g.shape[index] == 1 and weight_v.shape[index] > 1
        ]
        norm = weight_v.norm(dim=tuple(norm_dims), keepdim=True) if norm_dims else weight_v.norm(keepdim=True)
        out[f"{base}.weight"] = weight_g * weight_v / (norm + 1.0e-12)

    return out


def load_audio_vae_tensors(path, config):
    state = unwrap_state_dict(torch.load(path, map_location="cpu"), "AudioVAE")
    state = convert_weight_norm(state)
    out = {}
    for key, tensor in state.items():
        mapped = f"audio_vae.{key}"
        if mapped in out:
            raise RuntimeError(f"duplicate mapped tensor: {mapped}")
        out[mapped] = tensor.detach().cpu().contiguous()

    boundaries = config["audio_vae_config"].get("sr_bin_boundaries", [])
    buckets = len(boundaries) + 1
    decoder_dim = int(config["audio_vae_config"]["decoder_dim"])
    decoder_rates = config["audio_vae_config"]["decoder_rates"]
    for index, _rate in enumerate(decoder_rates, start=2):
        channels = decoder_dim // (1 << (index - 2))
        prefix = f"audio_vae.decoder.sr_cond_model.{index}"
        out[f"{prefix}.scale_embed.weight"] = torch.ones((buckets, channels), dtype=torch.float32)
        out[f"{prefix}.bias_embed.weight"] = torch.zeros((buckets, channels), dtype=torch.float32)
    return out


def audio_vae_f16_weight_names(tensors):
    names = []
    for name, tensor in tensors.items():
        if not name.startswith("audio_vae."):
            continue
        if name.startswith("audio_vae.decoder.sr_cond_model."):
            continue
        if name.endswith(".weight") and tensor.ndim >= 2:
            names.append(name)
    return sorted(set(names))


def voxcpm1_config_for_runtime(config):
    out = dict(config)
    if "audio_vae_config" not in out:
        out["audio_vae_config"] = {
            "encoder_dim": 128,
            "encoder_rates": [2, 5, 8, 8],
            "latent_dim": 64,
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 5, 2],
            "sr_bin_boundaries": [],
            "sample_rate": 16000,
            "out_sample_rate": 16000,
        }
    return out


def stage_sidecars(source, staging, config):
    with open(staging / "config.json", "w", encoding="utf-8") as handle:
        json.dump(voxcpm1_config_for_runtime(config), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy2(require_file(source / name), staging / name)


def run_converter(args, staged_safetensors, staging, vae_f16_weights):
    command = [
        str(args.converter),
        "--input", str(staged_safetensors),
        "--output", str(args.output),
        "--type", args.type,
        "--keep-type", "token_embd.weight=q8_0",
        "--keep-type", "stop.stop_head.weight=f32",
        "--keep-type", "audio_vae.decoder.sr_cond_model.*=f32",
        "--root", str(staging),
        "--family", "voxcpm1",
        "--model-spec", str(args.model_spec),
    ]
    for name in vae_f16_weights:
        command.extend(["--keep-type", f"{name}=f16"])
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    source = args.source
    require_file(source / "pytorch_model.bin")
    require_file(source / "audiovae.pth")
    require_file(source / "config.json")
    require_file(args.converter)
    require_file(args.model_spec)

    with open(source / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    staging = args.work_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tensors = load_main_tensors(source / "pytorch_model.bin")
    runtime_config = voxcpm1_config_for_runtime(config)
    tensors.update(load_audio_vae_tensors(source / "audiovae.pth", runtime_config))
    vae_f16_weights = audio_vae_f16_weight_names(tensors)
    staged_safetensors = staging / "voxcpm1-native.safetensors"
    save_file(tensors, staged_safetensors)
    stage_sidecars(source, staging, config)

    run_converter(args, staged_safetensors, staging, vae_f16_weights)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
