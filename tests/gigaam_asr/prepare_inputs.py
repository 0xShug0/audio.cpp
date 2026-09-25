#!/usr/bin/env python3
"""Prepare reproducible speech crops and a longform CLI path case."""

import argparse
import json
import random
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--long-repeat", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.source), "rb") as source:
        params = source.getparams()
        if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != 16000:
            raise ValueError("Source must be mono 16 kHz PCM16")
        samples = source.readframes(params.nframes)
    rng = random.Random(args.seed)
    records = []
    for index, seconds in enumerate([2.7, 5.3, 8.1, 12.6, 17.4]):
        count = round(seconds * params.framerate)
        start = rng.randrange(params.nframes - count + 1)
        records.append((f"random{index + 1}", start, count))
    if args.long_repeat < 1:
        raise ValueError("--long-repeat must be positive")
    long_samples = samples * args.long_repeat
    long_frames = params.nframes * args.long_repeat
    long_path = args.source
    if args.long_repeat > 1:
        long_path = args.output / "longform.wav"
        with wave.open(str(long_path), "wb") as target:
            target.setparams(params)
            target.writeframes(long_samples)
    for index, start in enumerate(range(0, long_frames, 25 * params.framerate)):
        records.append((f"long-part{index + 1:02d}", start, min(25 * params.framerate, long_frames - start)))
    manifest = []
    for name, start, count in records:
        path = args.output / (name + ".wav")
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            data = long_samples if name.startswith("long-part") else samples
            target.writeframes(data[start * 2:(start + count) * 2])
        manifest.append({"id": name, "audio": str(path), "start_sample": start, "samples": count})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    requests = [{"id": item["id"], "audio": item["audio"]} for item in manifest]
    requests.append({"id": "longform", "audio": str(long_path),
                     "options": {"audio_chunk_mode": "fixed"}})
    requests.append({"id": "longform_auto", "audio": str(long_path)})
    requests.append({"id": "longform_auto_repeat", "audio": str(long_path)})
    requests.append({"id": "repeat_after_long", "audio": manifest[0]["audio"]})
    case = {"id": "gigaam_random_longform", "family": "gigaam_asr", "task": "asr",
            "model": "GigaAM-ASR-GGUF/gigaam-v3-ctc-f16.gguf", "outputs": ["text"],
            "coverage": "Random speech crops, fixed-boundary longform, and short-after-long reuse.",
            "requests": requests}
    (args.output / "path_cases.json").write_text(json.dumps({"cases": [case]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
