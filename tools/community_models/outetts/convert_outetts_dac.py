#!/usr/bin/env python3
"""Convert OuteTTS 1.0's official IBM DAC checkpoint to safetensors.

The official OuteTTS runtime loads a trusted PyTorch checkpoint. audio.cpp uses
the resulting plain tensor source for both native safetensors loading and GGUF
packing; no Python or pickle loader is needed at inference time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dac
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="weights_24khz_1.5kbps_v1.0.pth")
    parser.add_argument("output", type=Path, help="output model.safetensors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {args.output}")

    model = dac.DAC.load(str(args.input)).cpu().eval()
    if model.sample_rate != 24000 or model.hop_length != 320:
        raise RuntimeError(
            f"expected OuteTTS DAC at 24 kHz with hop 320, got "
            f"{model.sample_rate} Hz / hop {model.hop_length}"
        )
    if model.n_codebooks != 2 or model.codebook_size != 1024:
        raise RuntimeError(
            f"expected two 1024-entry codebooks, got "
            f"{model.n_codebooks} x {model.codebook_size}"
        )

    tensors = {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "format": "pt",
            "source": "ibm-research/DAC.speech.v1.0",
            "checkpoint": args.input.name,
        },
    )
    print(f"wrote {args.output} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
