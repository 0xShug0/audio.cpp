#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def import_yue(reference_root: Path) -> None:
    sys.path.insert(0, str(reference_root / "src"))


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().float().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump realistic YuE2 VAE Python reference tensors.")
    parser.add_argument("--reference-root", type=Path, default=Path("reference/YuE"))
    parser.add_argument("--model", type=Path, default=Path("/home/leo/Desktop/YuE2/YuE2-Vae"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=32)
    parser.add_argument("--audio-frames", type=int, default=61440)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

    import_yue(args.reference_root.resolve())
    from yue2.modeling_vae import YuE2VAE

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    model = YuE2VAE.from_pretrained(args.model, device="cpu", decoder_only=False).eval()

    latents = torch.randn(1, model.config.latent_dim, args.latent_frames, dtype=torch.float32) * 0.35
    decoded = model.decoder(latents)
    audio = torch.randn(1, model.config.audio_channels, args.audio_frames, dtype=torch.float32) * 0.08
    encoded = model.encoder(audio)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_f32(args.out_dir / "decode_input.f32", latents)
    write_f32(args.out_dir / "decode_ref.f32", decoded)
    write_f32(args.out_dir / "encode_input.f32", audio)
    write_f32(args.out_dir / "encode_ref.f32", encoded)
    metadata = {
        "seed": args.seed,
        "latent_frames": args.latent_frames,
        "audio_frames": args.audio_frames,
        "decode_shape": list(decoded.shape),
        "encode_shape": list(encoded.shape),
        "sample_rate": model.config.sample_rate,
        "downsampling_ratio": model.config.downsampling_ratio,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
