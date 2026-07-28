#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an audio-separator BS-RoFormer checkpoint for audio.cpp."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid BS-RoFormer config payload: {path}")
    return payload


def export_config_json(config: dict[str, Any]) -> dict[str, Any]:
    audio = dict(config["audio"])
    model = dict(config["model"])
    inference = dict(config["inference"])
    return {
        "model_type": "bs_roformer",
        "sample_rate": int(audio["sample_rate"]),
        "stereo": bool(model["stereo"]),
        "chunk_size": int(audio["chunk_size"]),
        "batch_size": int(inference.get("batch_size", 1)),
        "num_overlap": int(inference["num_overlap"]),
        "normalize": False,
        "dim": int(model["dim"]),
        "depth": int(model["depth"]),
        "num_stems": int(model["num_stems"]),
        "time_transformer_depth": int(model["time_transformer_depth"]),
        "freq_transformer_depth": int(model["freq_transformer_depth"]),
        "linear_transformer_depth": int(model.get("linear_transformer_depth", 0)),
        "freqs_per_bands": [int(value) for value in model["freqs_per_bands"]],
        "dim_head": int(model["dim_head"]),
        "heads": int(model["heads"]),
        "n_fft": int(model["stft_n_fft"]),
        "hop_length": int(model["stft_hop_length"]),
        "win_length": int(model["stft_win_length"]),
        "stft_normalized": bool(model["stft_normalized"]),
        "mask_estimator_depth": int(model["mask_estimator_depth"]),
        "mlp_expansion_factor": int(model.get("mlp_expansion_factor", 4)),
        "skip_connection": bool(model.get("skip_connection", False)),
    }


def extract_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested:
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"unexpected checkpoint payload type: {type(payload)}")

    converted: dict[str, torch.Tensor] = {}
    for name, tensor in payload.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if name.endswith(".rotary_embed.freqs"):
            continue
        converted[name] = tensor.detach().cpu().contiguous()
    if not converted:
        raise RuntimeError("checkpoint contains no tensors")
    return converted


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    config_path = Path(args.config_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    payload = torch.load(
        ckpt_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    state = extract_state(payload)
    config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(
        json.dumps(export_config_json(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"converted_ckpt={ckpt_path}")
    print(f"output_dir={output_dir}")
    print(f"tensor_count={len(state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
