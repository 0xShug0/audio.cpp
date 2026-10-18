#!/usr/bin/env python3
"""Run the official Smart Turn v3 ONNX inference path on a 16 kHz mono WAV."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("expected 16 kHz mono audio")
    audio = audio[-128000:]
    if len(audio) < 128000:
        audio = np.pad(audio, (128000 - len(audio), 0))
    features = WhisperFeatureExtractor(chunk_length=8)(
        audio, sampling_rate=16000, return_tensors="np", padding="max_length",
        max_length=128000, truncation=True, do_normalize=True,
    ).input_features.astype(np.float32)
    probability = float(ort.InferenceSession(args.onnx).run(
        None, {"input_features": features}
    )[0][0].item())
    print(f"turn_complete={int(probability > 0.5)}")
    print(f"turn_probability={probability:.9f}")


if __name__ == "__main__":
    main()
