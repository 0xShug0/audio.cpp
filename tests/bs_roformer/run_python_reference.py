#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import soundfile as sf
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the audio-separator BS-RoFormer Python reference."
    )
    parser.add_argument("--audio-separator-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Use the reference's explicit matmul/softmax attention path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    separator_root = Path(args.audio_separator_root).resolve()
    sys.path.insert(0, str(separator_root))

    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer

    config = yaml.load(
        Path(args.config_path).read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    model_config = dict(config["model"])
    if args.no_flash_attn:
        model_config["flash_attn"] = False
    model = BSRoformer(**model_config)
    state = torch.load(
        Path(args.ckpt),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    expected_rate = int(config["audio"]["sample_rate"])
    if sample_rate != expected_rate:
        raise RuntimeError(f"sample-rate mismatch: expected {expected_rate}, got {sample_rate}")
    expected_channels = 2 if bool(config["model"]["stereo"]) else 1
    if audio.shape[1] != expected_channels:
        raise RuntimeError(
            f"channel mismatch: expected {expected_channels}, got {audio.shape[1]}"
        )

    tensor = torch.from_numpy(audio.T.copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        separated = model(tensor)[0].detach().cpu().numpy().T

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, separated, sample_rate, subtype="FLOAT")
    print(f"audio_out={output_path}")
    print(f"frames={separated.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
