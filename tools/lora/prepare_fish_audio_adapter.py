#!/usr/bin/env python3
"""Export Fish Speech Lightning LoRA factors as runtime-ready safetensors."""

import argparse
import collections
from pathlib import Path
import typing

import torch
from safetensors.torch import save_file

from checkpoint import load_checkpoint


def prepare(input_path: Path, output_path: Path, alpha: float) -> None:
    # Lightning stores scheduler settings as OmegaConf objects, not just tensors.
    from omegaconf import ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    allowed = [
        dict, list, int, collections.defaultdict, typing.Any,
        ListConfig, ContainerMetadata, Metadata, AnyNode,
    ]
    with torch.serialization.safe_globals(allowed):
        state = load_checkpoint(input_path)
    if (input_path.parent / "adapter_config.json").is_file() or any(name.endswith(".alpha") for name in state):
        # Already-prepared adapters retain their own scaling metadata.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, output_path, metadata={"format": "pt", "family": "fish_audio"})
        return
    tensors = {}
    targets = set()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not name.startswith("model."):
            raise ValueError(f"Unexpected Fish adapter entry: {name}")
        name = name.removeprefix("model.")
        if not name.endswith((".lora_A", ".lora_B")):
            raise ValueError(f"Expected LoRA factor, got {name}")
        target, factor = name.rsplit(".", 1)
        if value.ndim != 2:
            raise ValueError(f"Expected matrix factor: {name}")
        if target in {"embeddings", "codebook_embeddings", "fast_embeddings"}:
            # loralib.Embedding adds (B @ A).T to its [vocabulary, width] weight.
            factor = "lora_B" if factor == "lora_A" else "lora_A"
            value = value.T
        tensors[f"{target}.{factor}"] = value.contiguous()
        targets.add(target)
    if not targets:
        raise ValueError("Checkpoint contains no LoRA targets")
    for target in targets:
        a = tensors[f"{target}.lora_A"]
        b = tensors[f"{target}.lora_B"]
        if a.shape[0] != b.shape[1] or a.shape[0] <= 0:
            raise ValueError(f"Inconsistent LoRA rank: {target}")
        tensors[f"{target}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path, metadata={"format": "pt", "family": "fish_audio"})
    print(f"adapter={output_path}")
    print(f"targets={len(targets)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Fish Speech Lightning .ckpt")
    parser.add_argument("--output", type=Path, required=True, help="Output adapter .safetensors")
    parser.add_argument("--alpha", type=float, required=True, help="Training LoRA alpha, not alpha/r")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not torch.isfinite(torch.tensor(args.alpha)):
        parser.error("--alpha must be finite")
    prepare(args.input, args.output, args.alpha)


if __name__ == "__main__":
    main()
