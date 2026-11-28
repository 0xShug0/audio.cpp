#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "models" / "Ace-Step1.5" / "acestep-v15-turbo" / "silence_latent.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ACE-Step silence_latent.pt into a safetensors file."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to silence_latent.pt",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output safetensors path. Defaults to the input path with .safetensors suffix.",
    )
    parser.add_argument(
        "--key",
        default="silence_latent",
        help="Tensor key to write into the safetensors file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise RuntimeError(f"silence latent input not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".safetensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tensor = torch.load(str(input_path), map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"expected torch.Tensor in {input_path}, got {type(tensor)}")

    save_file({args.key: tensor.detach().cpu().contiguous()}, str(output_path))
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"key={args.key}")
    print(f"shape={tuple(tensor.shape)}")
    print(f"dtype={tensor.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
