#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "reference" / "Mel-Band-Roformer-Vocal-Model" / "configs" / "config_vocals_mel_band_roformer.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Mel-Band RoFormer reference checkpoint for audio.cpp C++ runtime.")
    parser.add_argument("--ckpt", default="models/melbandroformer/MelBandRoformer.ckpt")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", default="models/mel-roformer-mlx")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid RoFormer config payload: {path}")
    return payload


def export_config_json(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    inference = dict(config["inference"])
    return {
        "model_type": "mel_band_roformer",
        "sample_rate": int(model["sample_rate"]),
        "chunk_size": int(inference["chunk_size"]),
        "num_overlap": int(inference["num_overlap"]),
        "dim": int(model["dim"]),
        "depth": int(model["depth"]),
        "num_bands": int(model["num_bands"]),
        "num_stems": int(model["num_stems"]),
        "time_transformer_depth": int(model["time_transformer_depth"]),
        "freq_transformer_depth": int(model["freq_transformer_depth"]),
        "dim_head": int(model["dim_head"]),
        "heads": int(model["heads"]),
        "n_fft": int(model["stft_n_fft"]),
        "hop_length": int(model["stft_hop_length"]),
        "win_length": int(model["stft_win_length"]),
        "stft_normalized": bool(model["stft_normalized"]),
        "mask_estimator_depth": int(model["mask_estimator_depth"]),
        "mlp_expansion_factor": 4,
        "has_final_norm": False,
    }


def split_qkv_weights(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".rotary_embed.freqs"):
            continue
        if key.endswith(".to_qkv.weight"):
            prefix = key[:-len(".to_qkv.weight")]
            q, k, v = tensor.chunk(3, dim=0)
            converted[prefix + ".to_q.weight"] = q.contiguous()
            converted[prefix + ".to_k.weight"] = k.contiguous()
            converted[prefix + ".to_v.weight"] = v.contiguous()
            continue
        converted[key] = tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor
    return converted


def main() -> int:
    args = parse_args()
    ckpt_path = REPO_ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    config_path = REPO_ROOT / args.config_path if not Path(args.config_path).is_absolute() else Path(args.config_path)
    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected checkpoint payload type: {type(state)}")
    config = load_config(config_path)
    converted = split_qkv_weights(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(
        json.dumps(export_config_json(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"converted_ckpt={ckpt_path}")
    print(f"output_dir={output_dir}")
    print(f"tensor_count={len(converted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
