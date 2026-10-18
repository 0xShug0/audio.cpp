#!/usr/bin/env python3
"""Run an official Piper ONNX voice with the matching eSpeak frontend."""

from __future__ import annotations

import argparse
import json
import subprocess
import unicodedata
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--espeak", type=Path, required=True)
    parser.add_argument("--espeak-data-parent", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.8)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    phonemes = subprocess.check_output(
        [
            str(args.espeak),
            f"--path={args.espeak_data_parent}",
            f"-v{config['espeak']['voice']}",
            "-q",
            "--ipa=3",
            args.text,
        ],
        text=True,
    ).strip()
    phonemes = unicodedata.normalize("NFD", phonemes).replace("\u200d", "").replace("\u0361", "")
    ids = [1, 0]
    mapping = config["phoneme_id_map"]
    for symbol in phonemes:
        if symbol in mapping:
            ids.extend((mapping[symbol][0], 0))
    ids.append(2)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    tokens = np.asarray([ids], dtype=np.int64)
    audio = session.run(
        None,
        {
            "input": tokens,
            "input_lengths": np.asarray([tokens.shape[1]], dtype=np.int64),
            "scales": np.asarray(
                [args.noise_scale, args.length_scale, args.noise_w], dtype=np.float32
            ),
        },
    )[0].reshape(-1)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(config["audio"]["sample_rate"])
        output.writeframes(pcm.tobytes())
    print(json.dumps({"phonemes": phonemes, "token_ids": ids, "samples": len(audio)}))


if __name__ == "__main__":
    main()
