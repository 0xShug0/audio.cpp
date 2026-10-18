#!/usr/bin/env python3
"""Run the original NeMo Canary model on the same fixed chunks as the native ASR path."""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=40)
    parser.add_argument("--language", default="en")
    parser.add_argument("--target-language", default=None)
    parser.add_argument("--pnc", choices=("yes", "no"), default="yes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    signal, sample_rate = sf.read(args.audio, dtype="float32")
    if sample_rate != 16000 or signal.ndim != 1:
        raise ValueError("Parity input must be mono 16 kHz")
    if not 0.02 <= args.chunk_seconds <= 40 or len(signal) < 320:
        raise ValueError("Canary requires at least 20 ms of audio and chunks between 20 ms and 40 s")
    width = int(args.chunk_seconds * sample_rate)
    boundaries = list(range(0, len(signal), width)) + [len(signal)]
    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < 320:
        boundaries[-2] = boundaries[-1] - 320
        if boundaries[-2] - boundaries[-3] < 320:
            raise ValueError("Chunk duration is too short to retain a 20 ms final chunk")
    model = EncDecMultiTaskModel.restore_from(str(args.model), map_location="cpu").eval().to(args.device)
    model.cfg.decoding.beam.beam_size = 1
    model.change_decoding_strategy(model.cfg.decoding)
    chunks = []
    with torch.inference_mode():
        for start, end in zip(boundaries, boundaries[1:]):
            samples = signal[start:end]
            if args.device == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            result = model.transcribe([samples], batch_size=1, verbose=False,
                source_lang=args.language, target_lang=args.target_language or args.language,
                pnc=args.pnc, timestamps=False)[0]
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            item = {"start_sample": start, "samples": len(samples), "text": result.text,
                    "seconds": seconds, "tokens": result.y_sequence.tolist()}
            chunks.append(item)
            print(json.dumps(item), flush=True)
    args.output.write_text(json.dumps({"chunks": chunks,
        "text": " ".join(item["text"].strip() for item in chunks),
        "audio_seconds": len(signal) / sample_rate,
        "seconds": sum(item["seconds"] for item in chunks)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
