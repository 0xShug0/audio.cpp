#!/usr/bin/env python3
"""Generate original Transformers Bark 3.0 Pro references from local WAV files.

Use the pinned source snapshot described in docs/models/underdog_bark3.md.
This deliberately does not import or use audio.cpp's prepared weights.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, ParakeetForTDT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-audio", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1:
        parser.error("iterations and threads must be positive")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("the publisher's CPU fallback uses float32")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    if args.device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support BF16")
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.source, local_files_only=True)
    model = ParakeetForTDT.from_pretrained(args.source, dtype=dtype, local_files_only=True).to(args.device).eval()
    def read_wave(path):
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000 or wave.ndim != 1 or len(wave) == 0:
            raise ValueError("reference fixtures must be nonempty mono 16 kHz WAVs")
        return wave
    def inputs_for(wave):
        # Same once-per-recording prefix as runtime/transcribe_bark_pro.py.
        padded = np.concatenate([np.zeros(4000, np.float32), wave])
        return processor(padded, sampling_rate=16000, return_tensors="pt").to(args.device, dtype)
    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()
    if args.warmup_audio:
        with torch.inference_mode():
            model.generate(**inputs_for(read_wave(args.warmup_audio)))
        synchronize()
    records = []
    for path in args.audio:
        wave = read_wave(path)  # File I/O and model loading are outside the timer.
        runs = []
        for _ in range(args.iterations):
            synchronize()
            started = time.perf_counter()
            inputs = inputs_for(wave)
            with torch.inference_mode():
                generated = model.generate(**inputs)
            synchronize()
            ids = generated.sequences if hasattr(generated, "sequences") else generated
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            runs.append(dict(seconds=time.perf_counter() - started, text=text))
        records.append(dict(name=path.stem, path=str(path.resolve()), audio_seconds=len(wave) / 16000,
                            device=args.device, dtype=args.dtype, text=text, ids=ids.cpu().tolist(), runs=runs))
        print(path.name, repr(text), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
