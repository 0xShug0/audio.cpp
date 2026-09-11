#!/usr/bin/env python3
"""Dump NeMo Parakeet-TDT reference activations for numerical parity testing.

Loads the real `nvidia/parakeet-tdt-0.6b-v3` NeMo checkpoint, runs it on a
given audio clip, and saves a set of intermediate activations as .npy files
plus a summary.json of their shapes/means/stds. compare_parity.py then
compares these against the equivalent C++ dump produced by
dump_cpp_reference.cpp.

Captured tensors:
  mel_features       preprocessor output, [1, n_mels, T] (before the encoder)
  pre_encode         encoder.pre_encode output == the FastConformer
                      subsampling stack's linear projection, BEFORE the
                      xscale (sqrt(d_model)) multiply. [1, T', d_model]
  pos_emb            the raw sinusoidal relative positional encoding fed
                      into every layer's linear_pos projection (shared
                      across layers), [1, 2*T'-1, d_model]
  layer_{i}          full output of encoder.layers[i] for every layer,
                      [1, T', d_model]
  enc_out            final encoder output (after all layers, before the
                      joint's encoder projection), [1, d_model, T']

Setup (NeMo + its dependency chain isn't a lightweight install):
  python3 -m venv /tmp/parakeet_parity_venv
  /tmp/parakeet_parity_venv/bin/pip install torch nemo_toolkit[asr] librosa pysqlite3-binary numpy

  # If your system Python lacks a usable sqlite3 module (common in minimal
  # containers), NeMo's dependency chain pulls it in transitively via IPython
  # history; work around it by installing pysqlite3-binary and shimming
  # sys.modules['sqlite3'] before importing nemo, exactly as this script does
  # below.

Usage:
  /tmp/parakeet_parity_venv/bin/python3 dump_nemo_reference.py \
      --audio ../assets/2086-149220-0033.wav \
      --output-dir /tmp/parakeet_nemo_dump
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import sqlite3  # noqa: F401
except ImportError:
    sys.modules["sqlite3"] = __import__("pysqlite3")

import librosa
import numpy as np
import torch


def load_model(model_name: str):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
    model = model.cpu()
    model.eval()
    return model


def dump(model_name: str, audio_path: str, output_dir: str) -> dict:
    model = load_model(model_name)
    audio, _sr = librosa.load(audio_path, sr=16000, mono=True)

    captured: dict[str, np.ndarray] = {}

    def hook(name):
        def fn(_module, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            captured[name] = tensor.detach().cpu().numpy()

        return fn

    model.encoder.pre_encode.register_forward_hook(hook("pre_encode"))
    for i, layer in enumerate(model.encoder.layers):
        layer.register_forward_hook(hook(f"layer_{i}"))
    model.encoder.register_forward_hook(hook("enc_out"))

    # pos_enc.forward returns (scaled_x, pos_emb); wrap it to capture pos_emb
    # without disturbing the actual forward pass.
    orig_pos_enc_forward = model.encoder.pos_enc.forward

    def pos_enc_hook(x, cache_len=0):
        scaled_x, pos_emb = orig_pos_enc_forward(x, cache_len)
        captured["pos_emb"] = pos_emb.detach().cpu().numpy()
        return scaled_x, pos_emb

    model.encoder.pos_enc.forward = pos_enc_hook

    with torch.no_grad():
        mel, mel_len = model.preprocessor(
            input_signal=torch.tensor(audio).unsqueeze(0), length=torch.tensor([len(audio)])
        )
        captured["mel_features"] = mel.detach().cpu().numpy()
        model.encoder(audio_signal=mel, length=mel_len)

    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    for name, tensor in captured.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), tensor)
        summary[name] = {
            "shape": list(tensor.shape),
            "mean": float(np.mean(tensor)),
            "std": float(np.std(tensor)),
        }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, help="Path to a mono 16kHz WAV/FLAC test clip")
    parser.add_argument("--output-dir", required=True, help="Directory to write .npy dumps + summary.json into")
    parser.add_argument(
        "--model-name",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="HuggingFace/NGC model name passed to ASRModel.from_pretrained",
    )
    args = parser.parse_args()

    summary = dump(args.model_name, args.audio, args.output_dir)
    for name, stats in summary.items():
        print(f"{name}: shape={stats['shape']} mean={stats['mean']:.6f} std={stats['std']:.6f}")
    print(f"\nSaved {len(summary)} tensors to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
