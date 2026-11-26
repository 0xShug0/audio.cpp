#!/usr/bin/env python3
"""Prepare the bundled Silero VAD weights from the official snakers4/silero-vad v6.2 checkpoint.

Writes assets/framework/models/silero_vad/silero_vad_16k.safetensors: the 16 kHz sub-model of the
official TorchScript checkpoint, with its tensors renamed to the keys src/models/silero_vad loads
(stft_conv, conv1-4, lstm_cell, final_conv) and stored unchanged as float32.

Source: https://github.com/snakers4/silero-vad (MIT), tag v6.2, src/silero_vad/data/silero_vad.jit.
The output is checked against the pinned sha256 below.

Do not use silero-vad's tinygrad export (src/silero_vad/data/silero_vad_16k.safetensors) instead:
its conv and LSTM weights match no official release, and its outputs differ from the TorchScript
and ONNX models (snakers4/silero-vad#768).

Requires torch, numpy and safetensors:
    uv run --no-project --with torch --with numpy --with safetensors python \\
        tools/gguf/silero_vad_weights_prepare.py \\
        --output assets/framework/models/silero_vad/silero_vad_16k.safetensors
Use --jit <path> to convert a checkpoint downloaded beforehand instead of fetching it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

JIT_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.jit"
OUTPUT_SHA256 = "D7FB67A5B4DE0414A0178270EE81439B9E3067C883C304AA758C247D50FFD79D"

# audio.cpp key -> key in the checkpoint's 16 kHz sub-model, with the expected shape.
TENSORS = {
    "stft_conv.weight": ("_model.stft.forward_basis_buffer", (258, 1, 256)),
    "conv1.weight": ("_model.encoder.0.reparam_conv.weight", (128, 129, 3)),
    "conv1.bias": ("_model.encoder.0.reparam_conv.bias", (128,)),
    "conv2.weight": ("_model.encoder.1.reparam_conv.weight", (64, 128, 3)),
    "conv2.bias": ("_model.encoder.1.reparam_conv.bias", (64,)),
    "conv3.weight": ("_model.encoder.2.reparam_conv.weight", (64, 64, 3)),
    "conv3.bias": ("_model.encoder.2.reparam_conv.bias", (64,)),
    "conv4.weight": ("_model.encoder.3.reparam_conv.weight", (128, 64, 3)),
    "conv4.bias": ("_model.encoder.3.reparam_conv.bias", (128,)),
    "lstm_cell.weight_ih": ("_model.decoder.rnn.weight_ih", (512, 128)),
    "lstm_cell.weight_hh": ("_model.decoder.rnn.weight_hh", (512, 128)),
    "lstm_cell.bias_ih": ("_model.decoder.rnn.bias_ih", (512,)),
    "lstm_cell.bias_hh": ("_model.decoder.rnn.bias_hh", (512,)),
    "final_conv.weight": ("_model.decoder.decoder.2.weight", (1, 128, 1)),
    "final_conv.bias": ("_model.decoder.decoder.2.bias", (1,)),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def convert(jit_path: Path, output: Path) -> None:
    import safetensors.numpy
    import torch

    state = torch.jit.load(str(jit_path), map_location="cpu").state_dict()
    tensors = {}
    for name, (source, shape) in TENSORS.items():
        if source not in state:
            raise SystemExit(f"checkpoint has no tensor {source}; is it silero-vad v6.2?")
        array = state[source].detach().cpu().numpy().astype("float32")
        if array.shape != shape:
            raise SystemExit(f"{source} has shape {array.shape}, expected {shape}")
        tensors[name] = array
    output.parent.mkdir(parents=True, exist_ok=True)
    safetensors.numpy.save_file(tensors, str(output))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the bundled Silero VAD v6.2 weights.")
    parser.add_argument("--output", type=Path, required=True, help="safetensors file to write")
    parser.add_argument("--jit", type=Path, help="local silero_vad.jit (v6.2); fetched from GitHub when omitted")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        jit = args.jit
        if jit is None:
            jit = Path(tmp) / "silero_vad.jit"
            with urllib.request.urlopen(JIT_URL, timeout=120) as response:
                jit.write_bytes(response.read())
        convert(jit, args.output)

    actual = sha256(args.output)
    if actual != OUTPUT_SHA256:
        print(f"error: {args.output} has sha256 {actual}, expected {OUTPUT_SHA256}", file=sys.stderr)
        return 1
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, sha256 {actual})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
