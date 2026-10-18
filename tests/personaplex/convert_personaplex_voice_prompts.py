#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_prompt(path: Path, output_dir: Path) -> Path:
    prompt = torch.load(path, map_location="cpu")
    if not isinstance(prompt, dict):
        raise TypeError(f"{path} is not a tensor dictionary")
    tensors = {}
    for name, value in prompt.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{path}:{name} is not a tensor")
        tensors[name] = value.contiguous()
    output = output_dir / f"{path.stem}.safetensors"
    save_file(tensors, output, metadata={"format": "personaplex_voice_prompt", "source": path.name})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PersonaPlex voice prompt .pt files to safetensors.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [convert_prompt(path, args.output_dir) for path in sorted(args.input_dir.glob("*.pt"))]
    if not outputs:
        raise RuntimeError(f"no .pt voice prompts found in {args.input_dir}")
    print(f"converted {len(outputs)} PersonaPlex voice prompts to {args.output_dir}")


if __name__ == "__main__":
    main()
