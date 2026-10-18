#!/usr/bin/env python3
"""Compare matched upstream/native MiraTTS WAV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.mean(audio, axis=1, dtype=np.float32), sample_rate


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 1.0 if denominator == 0.0 else float(np.dot(left, right) / denominator)


def compare(cpp_path: Path, python_path: Path) -> dict[str, object]:
    cpp, cpp_rate = mono(cpp_path)
    python, python_rate = mono(python_path)
    if cpp_rate != python_rate:
        raise RuntimeError(
            f"sample-rate mismatch for {cpp_path.name}: {cpp_rate} != {python_rate}"
        )
    common = min(cpp.size, python.size)
    cpp_common = cpp[:common].astype(np.float64, copy=False)
    python_common = python[:common].astype(np.float64, copy=False)
    wav_cosine = cosine(cpp_common, python_common)
    mel_kwargs = {
        "sr": cpp_rate,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 2048,
        "n_mels": 128,
        "power": 2.0,
    }
    cpp_mel = np.log(
        np.maximum(librosa.feature.melspectrogram(y=cpp, **mel_kwargs), 1.0e-10)
    )
    python_mel = np.log(
        np.maximum(librosa.feature.melspectrogram(y=python, **mel_kwargs), 1.0e-10)
    )
    mel_frames = min(cpp_mel.shape[1], python_mel.shape[1])
    log_mel_cosine = cosine(
        cpp_mel[:, :mel_frames].reshape(-1).astype(np.float64, copy=False),
        python_mel[:, :mel_frames].reshape(-1).astype(np.float64, copy=False),
    )
    return {
        "cpp_audio": str(cpp_path),
        "python_audio": str(python_path),
        "sample_rate": cpp_rate,
        "cpp_frames": int(cpp.size),
        "python_frames": int(python.size),
        "exact_frame_count": cpp.size == python.size,
        "common_frames": int(common),
        "waveform_cosine": wav_cosine,
        "log_mel_cosine": log_mel_cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-summary", type=Path, required=True)
    parser.add_argument("--python-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wav-cosine-min", type=float, default=0.95)
    parser.add_argument("--log-mel-cosine-min", type=float, default=0.95)
    parser.add_argument("--require-exact-frames", action="store_true")
    args = parser.parse_args()

    cpp = json.loads(args.cpp_summary.read_text(encoding="utf-8"))
    python = json.loads(args.python_summary.read_text(encoding="utf-8"))
    cpp_by_name = {item["name"]: item for item in cpp["results"]}
    python_by_name = {item["name"]: item for item in python["results"]}
    names = sorted(set(cpp_by_name) & set(python_by_name))
    if not names:
        raise RuntimeError("the summaries contain no matching request names")

    comparisons = []
    failed = []
    for name in names:
        item = compare(
            Path(cpp_by_name[name]["audio_out"]),
            Path(python_by_name[name]["audio_out"]),
        )
        item["name"] = name
        item["passed"] = (
            item["waveform_cosine"] >= args.wav_cosine_min
            and item["log_mel_cosine"] >= args.log_mel_cosine_min
            and (not args.require_exact_frames or item["exact_frame_count"])
        )
        comparisons.append(item)
        if not item["passed"]:
            failed.append(name)
        print(json.dumps(item))

    report = {
        "thresholds": {
            "waveform_cosine": args.wav_cosine_min,
            "log_mel_cosine": args.log_mel_cosine_min,
            "require_exact_frames": args.require_exact_frames,
        },
        "passed": not failed,
        "failed": failed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"MiraTTS parity failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
