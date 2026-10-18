#!/usr/bin/env python3
"""Capture official streaming inputs and stage timing for ARM parity probes."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from conversion import write_model
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_repo.resolve()))
    from openwakeword.model import Model

    sample_rate, audio = wavfile.read(args.audio)
    if sample_rate != 16000 or audio.dtype != np.int16 or audio.ndim != 1:
        raise ValueError("reference requires mono PCM16 at 16 kHz")
    args.output.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in ("mel", "embedding", "classifier")}
    timings = {name: [] for name in records}
    np.random.seed(0)
    model = Model(
        wakeword_models=[str(args.models / "alexa_v0.1.onnx")],
        melspec_model_path=str(args.models / "melspectrogram.onnx"),
        embedding_model_path=str(args.models / "embedding_model.onnx"),
        inference_framework="onnx", ncpu=args.threads,
    )

    initial_features = np.array(model.preprocessor.feature_buffer, copy=True)

    def capture(name, predict):
        def run(value):
            start = time.perf_counter_ns()
            output = predict(value)
            timings[name].append((time.perf_counter_ns() - start) / 1e6)
            array = np.asarray(output[0] if isinstance(output, list) else output, dtype=np.float32)
            records[name].append((np.array(value, copy=True), np.array(array, copy=True)))
            return output
        return run

    model.preprocessor.melspec_model_predict = capture("mel", model.preprocessor.melspec_model_predict)
    model.preprocessor.embedding_model_predict = capture("embedding", model.preprocessor.embedding_model_predict)
    for key, predict in model.model_prediction_function.items():
        model.model_prediction_function[key] = capture("classifier", predict)
    scores = []
    for offset in range(0, len(audio), 1280):
        frame = audio[offset:offset + 1280]
        if len(frame) < 1280:
            frame = np.pad(frame, (0, 1280 - len(frame)))
        scores.append(model.predict(frame))

    report = {"feature_threads": args.threads, "classifier_threads": 1, "audio_seconds": len(audio) / sample_rate,
              "frames": len(scores), "maximum_score": float(max(max(s.values()) for s in scores))}
    for name, values in timings.items():
        warm = values[1:]
        report[name] = {"calls": len(values), "warm_median_ms": statistics.median(warm),
                        "warm_total_ms": sum(warm)}
    print(json.dumps(report, indent=2))
    (args.output / f"reference-t{args.threads}.json").write_text(json.dumps(report, indent=2) + "\n")

    records["probabilities"] = [(audio.astype(np.float32) / 32768.0,
                                 np.array([next(iter(frame.values())) for frame in scores], dtype=np.float32))]
    write_model(args.models, args.output / "open-wake-word-reference-f32.gguf", initial_features, records)



if __name__ == "__main__":
    main()
