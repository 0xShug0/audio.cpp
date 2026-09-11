#!/usr/bin/env python3
"""Compare Higgs warmbench runs request by request.

Each result directory is expected to contain timing.log and audio/audio_N.wav,
as emitted by higgs_audio_tts_warm_bench. The report includes exact frame counts,
wall time, RTF, speedup, waveform cosine, and log-mel cosine per request.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected PCM16 WAV")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return sample_rate, samples / 32768.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    count = min(a.size, b.size)
    if count == 0:
        return math.nan
    a = a[:count].astype(np.float64, copy=False)
    b = b[:count].astype(np.float64, copy=False)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0.0 else math.nan


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def log_mel(samples: np.ndarray, sample_rate: int, n_fft: int = 1024, hop: int = 256, bands: int = 80) -> np.ndarray:
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    frame_count = 1 + (samples.size - n_fft) // hop
    shape = (frame_count, n_fft)
    strides = (samples.strides[0] * hop, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2

    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), bands + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sample_rate).astype(np.int64)
    bins = np.clip(bins, 0, spectrum.shape[1] - 1)
    filters = np.zeros((bands, spectrum.shape[1]), dtype=np.float64)
    for band in range(bands):
        left, center, right = bins[band : band + 3]
        if center > left:
            filters[band, left:center] = np.arange(center - left) / (center - left)
        if right > center:
            filters[band, center:right] = np.arange(right - center, 0, -1) / (right - center)
    return np.log(np.maximum(spectrum @ filters.T, 1.0e-10)).astype(np.float32)


def timings(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for line in (path / "timing.log").read_text(encoding="utf-8").splitlines():
        prefix = "higgs_audio_tts.cpp.request_"
        if not line.startswith(prefix) or ".wall_ms=" not in line:
            continue
        index_text, value = line[len(prefix) :].split(".wall_ms=", 1)
        result[int(index_text)] = float(value)
    return result


def audio_files(path: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for wav_path in (path / "audio").glob("audio_*.wav"):
        result[int(wav_path.stem.removeprefix("audio_"))] = wav_path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-same-frames", action="store_true")
    parser.add_argument("--min-wav-cosine", type=float)
    parser.add_argument("--min-logmel-cosine", type=float)
    args = parser.parse_args()

    baseline_times = timings(args.baseline)
    candidate_times = timings(args.candidate)
    baseline_audio = audio_files(args.baseline)
    candidate_audio = audio_files(args.candidate)
    indices = sorted(set(baseline_times) & set(candidate_times) & set(baseline_audio) & set(candidate_audio))
    if not indices:
        raise ValueError("no matching warmbench requests were found")

    failed = False
    requests = []
    for index in indices:
        baseline_rate, baseline_samples = read_wav(baseline_audio[index])
        candidate_rate, candidate_samples = read_wav(candidate_audio[index])
        if baseline_rate != candidate_rate:
            raise ValueError(f"request {index}: sample rates differ")
        same_frames = baseline_samples.size == candidate_samples.size
        wav_cosine = cosine(baseline_samples, candidate_samples)
        baseline_mel = log_mel(baseline_samples, baseline_rate)
        candidate_mel = log_mel(candidate_samples, candidate_rate)
        mel_frames = min(baseline_mel.shape[0], candidate_mel.shape[0])
        logmel_cosine = cosine(baseline_mel[:mel_frames].reshape(-1), candidate_mel[:mel_frames].reshape(-1))
        baseline_duration_sec = baseline_samples.size / baseline_rate
        candidate_duration_sec = candidate_samples.size / candidate_rate
        baseline_ms = baseline_times[index]
        candidate_ms = candidate_times[index]
        baseline_rtf = baseline_ms / 1000.0 / baseline_duration_sec
        candidate_rtf = candidate_ms / 1000.0 / candidate_duration_sec
        request = {
            "request_index": index,
            "baseline_frames": int(baseline_samples.size),
            "candidate_frames": int(candidate_samples.size),
            "same_frames": same_frames,
            "baseline_wall_ms": baseline_ms,
            "candidate_wall_ms": candidate_ms,
            "wall_speedup": baseline_ms / candidate_ms,
            "baseline_rtf": baseline_rtf,
            "candidate_rtf": candidate_rtf,
            "rtf_speedup": baseline_rtf / candidate_rtf,
            "wav_cosine": wav_cosine,
            "logmel_cosine": logmel_cosine,
        }
        requests.append(request)
        failed = failed or (args.require_same_frames and not same_frames)
        failed = failed or (args.min_wav_cosine is not None and wav_cosine < args.min_wav_cosine)
        failed = failed or (args.min_logmel_cosine is not None and logmel_cosine < args.min_logmel_cosine)
        print(
            f"request={index} frames={baseline_samples.size}/{candidate_samples.size} "
            f"wall_ms={baseline_ms:.3f}/{candidate_ms:.3f} "
            f"rtf={baseline_rtf:.4f}/{candidate_rtf:.4f} speedup={baseline_rtf / candidate_rtf:.3f}x "
            f"wav_cos={wav_cosine:.8f} mel_cos={logmel_cosine:.8f}"
        )

    payload = {"baseline": str(args.baseline), "candidate": str(args.candidate), "requests": requests}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
