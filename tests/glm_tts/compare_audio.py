#!/usr/bin/env python3
"""Compare GLM-TTS C++ WAVs with official Python reference WAVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    return 1.0 if denominator == 0.0 else float(
        np.dot(lhs, rhs) / denominator
    )


def compare(cpp_path: Path, python_path: Path) -> dict:
    cpp, cpp_rate = mono(cpp_path)
    python, python_rate = mono(python_path)
    if cpp_rate != python_rate:
        raise ValueError(
            f"sample-rate mismatch: {cpp_rate} != {python_rate}"
        )
    common = min(cpp.size, python.size)
    wav_cosine = cosine(
        cpp[:common].astype(np.float64),
        python[:common].astype(np.float64),
    )
    mel_args = {
        "sr": cpp_rate,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "power": 2.0,
    }
    cpp_mel = np.log(
        np.maximum(
            librosa.feature.melspectrogram(y=cpp, **mel_args),
            1.0e-10,
        )
    )
    python_mel = np.log(
        np.maximum(
            librosa.feature.melspectrogram(y=python, **mel_args),
            1.0e-10,
        )
    )
    mel_frames = min(cpp_mel.shape[1], python_mel.shape[1])
    log_mel_cosine = cosine(
        cpp_mel[:, :mel_frames].reshape(-1).astype(np.float64),
        python_mel[:, :mel_frames].reshape(-1).astype(np.float64),
    )
    return {
        "cpp_path": str(cpp_path),
        "python_path": str(python_path),
        "sample_rate": cpp_rate,
        "cpp_frames": int(cpp.size),
        "python_frames": int(python.size),
        "common_frames": int(common),
        "wav_cosine": wav_cosine,
        "log_mel_cosine": log_mel_cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--cpp-suffix", default="_1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    requests = json.loads(
        args.request_file.read_text(encoding="utf-8")
    )["requests"]
    results = []
    for request in requests:
        name = request["name"]
        result = compare(
            args.cpp_dir / f"{name}{args.cpp_suffix}.wav",
            args.python_dir / f"{name}.wav",
        )
        result["name"] = name
        results.append(result)
        print(
            f"{name}: wav_cosine={result['wav_cosine']:.9f} "
            f"log_mel_cosine={result['log_mel_cosine']:.9f} "
            f"frames={result['cpp_frames']}/{result['python_frames']}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"requests": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
