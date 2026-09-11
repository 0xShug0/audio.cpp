#!/usr/bin/env python3
"""Generate deterministic Fun-ASR-Nano frontend parity fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
TRANSFORMERS_COMMIT = "48e7f65fb274172e15aa88875d780c67c37606c7"
TORCHAUDIO_VERSION = "2.11.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-src", type=Path, required=True)
    parser.add_argument("--sample-wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_sample(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz sample, got {sample_rate}")
    return audio.mean(axis=1, dtype=np.float32)


def synthetic_inputs(sample_wav: Path) -> dict[str, np.ndarray]:
    samples = np.arange(SAMPLE_RATE, dtype=np.float32)
    impulse = np.zeros(SAMPLE_RATE, dtype=np.float32)
    impulse[SAMPLE_RATE // 2] = 1.0
    return {
        "silence": np.zeros(SAMPLE_RATE, dtype=np.float32),
        "impulse": impulse,
        "sine_440hz": (0.5 * np.sin(2.0 * math.pi * 440.0 * samples / SAMPLE_RATE)).astype(np.float32),
        "sample_16k": load_sample(sample_wav),
    }


def summarize(features: np.ndarray, waveform: np.ndarray) -> dict[str, object]:
    flat = features.reshape(-1)
    values = flat.astype(np.float32)
    return {
        "samples": len(waveform),
        "frames": int(features.shape[0]),
        "width": int(features.shape[1]),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat, dtype=np.float64)),
        "l2": float(np.linalg.norm(flat.astype(np.float64))),
        "first32": values[:32].tolist(),
        "last32": values[-32:].tolist(),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.transformers_src.resolve()))

    import torch
    import torchaudio
    import transformers
    from transformers import FunAsrNanoFeatureExtractor

    transformers_module = Path(transformers.__file__).resolve()
    transformers_src = args.transformers_src.resolve()
    if transformers_src not in transformers_module.parents:
        raise RuntimeError(
            f"expected Transformers from {transformers_src}, imported {transformers_module}"
        )

    transformers_root = transformers_src.parent
    actual_commit = subprocess.check_output(
        ["git", "-C", str(transformers_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != TRANSFORMERS_COMMIT:
        raise RuntimeError(f"expected Transformers {TRANSFORMERS_COMMIT}, found {actual_commit}")
    feature_extractor_path = (
        transformers_src / "transformers/models/fun_asr_nano/feature_extraction_fun_asr_nano.py"
    )
    feature_extractor_relative = feature_extractor_path.relative_to(transformers_root)
    feature_status = subprocess.check_output(
        ["git", "-C", str(transformers_root), "status", "--porcelain", "--", str(feature_extractor_relative)],
        text=True,
    ).strip()
    if feature_status:
        raise RuntimeError(f"Fun-ASR-Nano feature extractor has uncommitted changes: {feature_status}")
    actual_torchaudio_version = torchaudio.__version__
    if actual_torchaudio_version.split("+", maxsplit=1)[0] != TORCHAUDIO_VERSION:
        raise RuntimeError(
            f"expected torchaudio {TORCHAUDIO_VERSION}, found {actual_torchaudio_version}"
        )

    torch.set_num_threads(1)
    extractor = FunAsrNanoFeatureExtractor()
    fixtures: dict[str, object] = {}
    binary_data = bytearray()

    def store_f32(values: np.ndarray) -> dict[str, int]:
        array = np.asarray(values, dtype="<f4").reshape(-1)
        descriptor = {"offset_f32": len(binary_data) // 4, "count": int(array.size)}
        binary_data.extend(array.tobytes())
        return descriptor

    for name, waveform in synthetic_inputs(args.sample_wav).items():
        output = extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        frames = int(output["feature_lengths"][0])
        features = output["input_features"][0, :frames].detach().cpu().numpy()
        fixture = summarize(features, waveform)
        if name != "sample_16k":
            fixture["waveform"] = store_f32(waveform)
        fixture["features"] = store_f32(features)
        fixtures[name] = fixture

    binary_path = args.output.with_suffix(".bin")
    binary_bytes = bytes(binary_data)

    payload = {
        "schema_version": 2,
        "reference": "transformers.FunAsrNanoFeatureExtractor",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "feature_extractor_sha256": hashlib.sha256(feature_extractor_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "torchaudio_version": actual_torchaudio_version,
        "sample_wav": args.sample_wav.name,
        "sample_wav_sha256": hashlib.sha256(args.sample_wav.read_bytes()).hexdigest(),
        "data_file": binary_path.name,
        "data_format": "little-endian-float32",
        "data_sha256": hashlib.sha256(binary_bytes).hexdigest(),
        "sample_rate": SAMPLE_RATE,
        "feature_size": 80,
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "lfr_m": 7,
        "lfr_n": 6,
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(binary_bytes)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(fixtures, indent=2))


if __name__ == "__main__":
    main()
