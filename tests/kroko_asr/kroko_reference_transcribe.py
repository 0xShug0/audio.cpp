#!/usr/bin/env python3
"""Run the original sherpa-onnx Kroko path and emit reproducible JSON."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("reference input must be mono PCM16 WAV")
        samples = np.frombuffer(
            source.readframes(source.getnframes()), dtype=np.int16
        ).astype(np.float32)
        return samples / 32768.0, source.getframerate()


def unpack_free_package(source: Path, destination: Path) -> Path:
    data = source.read_bytes()
    if len(data) < 4:
        raise ValueError("truncated Kroko package")
    header_size = struct.unpack_from("<I", data, 0)[0]
    offset = 4
    header = json.loads(data[offset : offset + header_size].decode("utf-8"))
    if not header.get("free", False):
        raise ValueError("reference runner accepts only free Kroko packages")
    offset += header_size
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        (destination / name).write_bytes(data[offset : offset + length])
        offset += length
    if offset != len(data):
        raise ValueError("unexpected trailing Kroko package bytes")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--decoding-method",
        choices=("greedy_search", "modified_beam_search"),
        default="greedy_search",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    samples, sample_rate = read_wave(args.audio)
    with tempfile.TemporaryDirectory(prefix="kroko-reference-") as temporary:
        model = (
            unpack_free_package(args.model, Path(temporary))
            if args.model.is_file()
            else args.model
        )
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model / "tokens.txt"),
            encoder=str(model / "encoder.onnx"),
            decoder=str(model / "decoder.onnx"),
            joiner=str(model / "joiner.onnx"),
            num_threads=args.threads,
            sample_rate=16000,
            feature_dim=80,
            low_freq=20.0,
            high_freq=-400.0,
            dither=0.0,
            normalize_samples=True,
            snip_edges=False,
            decoding_method=args.decoding_method,
            max_active_paths=4,
            provider="cpu",
        )
        stream = recognizer.create_stream()
        start = time.perf_counter()
        stream.accept_waveform(sample_rate, samples)
        stream.accept_waveform(
            sample_rate, np.zeros(round(0.66 * sample_rate), dtype=np.float32)
        )
        stream.input_finished()
        partials: list[str] = []
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            partials.append(recognizer.get_result(stream))
        result = recognizer.get_result_all(stream)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    report = {
        "audio": str(args.audio),
        "duration_seconds": len(samples) / sample_rate,
        "decoding_method": args.decoding_method,
        "text": result.text.strip(),
        "tokens": result.tokens,
        "timestamps_seconds": result.timestamps,
        "partials": partials,
        "elapsed_ms": elapsed_ms,
        "rtf": elapsed_ms / 1000.0 / (len(samples) / sample_rate),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
