"""Compare Kokoro GGUF output with matching extracted-F32 reference WAVs."""
import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample, resample_poly, stft
from scipy.spatial.distance import cdist


def load(path: Path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16: {path}")
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    return rate, audio


def si_sdr(reference, estimate):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / max(np.dot(reference, reference), 1e-20)
    target = scale * reference
    noise = estimate - target
    return 10 * np.log10(max(np.dot(target, target), 1e-20) / max(np.dot(noise, noise), 1e-20))


def log_spectral_distance(reference, estimate, rate):
    _, _, ref_spec = stft(reference, rate, nperseg=1024, noverlap=768, boundary=None)
    _, _, est_spec = stft(estimate, rate, nperseg=1024, noverlap=768, boundary=None)
    ref_db = 20 * np.log10(np.maximum(np.abs(ref_spec), 1e-5))
    est_db = 20 * np.log10(np.maximum(np.abs(est_spec), 1e-5))
    return float(np.mean(np.sqrt(np.mean((ref_db - est_db) ** 2, axis=0))))


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def log_mel(audio, rate, bands=40):
    _, _, spectrum = stft(audio, rate, nperseg=1024, noverlap=512, boundary=None)
    power = np.abs(spectrum) ** 2
    edges = mel_to_hz(np.linspace(hz_to_mel(40), hz_to_mel(rate / 2), bands + 2))
    bins = np.minimum((edges / (rate / 2) * (power.shape[0] - 1)).astype(int), power.shape[0] - 1)
    filters = np.zeros((bands, power.shape[0]))
    for band in range(bands):
        left, center, right = bins[band:band + 3]
        if center > left:
            filters[band, left:center] = np.arange(center - left) / (center - left)
        if right > center:
            filters[band, center:right] = np.arange(right - center, 0, -1) / (right - center)
    return np.log(np.maximum(filters @ power, 1e-10)).T


def dtw_log_mel_cosine(reference, estimate, rate):
    ref = log_mel(reference, rate)
    est = log_mel(estimate, rate)
    # Per-frame cosine distance emphasizes spectral shape over loudness.
    ref -= ref.mean(axis=1, keepdims=True)
    est -= est.mean(axis=1, keepdims=True)
    costs = cdist(ref, est, metric="cosine").astype(np.float32)
    rows, cols = costs.shape
    accumulated = np.empty_like(costs)
    steps = np.empty((rows, cols), dtype=np.uint8)
    accumulated[0, 0] = costs[0, 0]
    accumulated[1:, 0] = np.cumsum(costs[1:, 0]) + accumulated[0, 0]
    accumulated[0, 1:] = np.cumsum(costs[0, 1:]) + accumulated[0, 0]
    steps[1:, 0] = 1
    steps[0, 1:] = 2
    for row in range(1, rows):
        previous = accumulated[row - 1]
        for col in range(1, cols):
            options = (previous[col - 1], previous[col], accumulated[row, col - 1])
            step = int(np.argmin(options))
            accumulated[row, col] = costs[row, col] + options[step]
            steps[row, col] = step
    row, col = rows - 1, cols - 1
    path_cost = 0.0
    length = 0
    while row or col:
        path_cost += float(costs[row, col])
        length += 1
        step = steps[row, col]
        if step == 0:
            row -= 1; col -= 1
        elif step == 1:
            row -= 1
        else:
            col -= 1
    path_cost += float(costs[0, 0])
    return 1.0 - path_cost / (length + 1)


def compare(reference_path, candidate_path):
    rate, reference = load(reference_path)
    candidate_rate, candidate = load(candidate_path)
    if candidate_rate != rate:
        raise ValueError("sample-rate mismatch")
    original_candidate_size = len(candidate)
    # Duration prediction changes by a few frames under quantization. Normalize the
    # global duration before signal metrics so they measure acoustic similarity.
    candidate = resample(candidate, len(reference))
    reference_16k = resample_poly(reference, 2, 3)
    candidate_16k = resample_poly(candidate, 2, 3)
    eps = 1e-20
    return {
        "reference_seconds": len(reference) / rate,
        "candidate_seconds": original_candidate_size / rate,
        "duration_delta_percent": 100 * (original_candidate_size - len(reference)) / len(reference),
        "rms_delta_db": 20 * math.log10(max(np.sqrt(np.mean(candidate * candidate)), eps) /
                                         max(np.sqrt(np.mean(reference * reference)), eps)),
        "correlation": float(np.corrcoef(reference, candidate)[0, 1]),
        "si_sdr_db": float(si_sdr(reference, candidate)),
        "log_spectral_distance_db": log_spectral_distance(reference, candidate, rate),
        "dtw_log_mel_cosine": dtw_log_mel_cosine(reference, candidate, rate),
        "stoi": float(stoi(reference_16k, candidate_16k, 16000, extended=False)),
        "pesq_wb": float(pesq(16000, reference_16k, candidate_16k, "wb")),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
records = []
for precision in ("bf16-gguf", "q8-gguf"):
    for index in range(7):
        record = {"precision": precision, "request": index}
        record.update(compare(args.root / "f32-directory" / f"request_{index}.wav",
                              args.root / precision / f"request_{index}.wav"))
        records.append(record)

(args.root / "quality.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
for precision in ("bf16-gguf", "q8-gguf"):
    subset = [r for r in records if r["precision"] == precision]
    print(precision)
    for key in ("duration_delta_percent", "rms_delta_db", "correlation", "si_sdr_db",
                "log_spectral_distance_db", "dtw_log_mel_cosine", "stoi", "pesq_wb"):
        values = np.array([r[key] for r in subset])
        print(f"  {key}: mean={values.mean():.6f}, min={values.min():.6f}, max={values.max():.6f}")
