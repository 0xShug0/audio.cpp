#!/usr/bin/env python3
"""Run pinned Plus's unmodified official loader against local mono 16 kHz WAVs.

Download bark_loader.py from the pinned Plus revision as well as the data/base
sidecars in the model guide. This never imports audio.cpp's converted weights.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/community_models"))
from convert_underdog_bark3_plus import BASE_SHA256, PLUS_PROFILE, sha256

LOADER_SHA256 = "e86b7f130bec1a8fa14dd315cfce95785b890dbd30000f6ceb2597fdd4312b02"


def main(profile=PLUS_PROFILE, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-audio", type=Path)
    parser.add_argument("--check-official-transcribe", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.iterations <= 0:
        parser.error("threads and iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    for name, expected in {**BASE_SHA256, "bark_loader.py": LOADER_SHA256, "model.bark": profile.source_sha256}.items():
        if sha256(args.source / name) != expected:
            raise ValueError("source differs from pinned snapshot: " + name)
    torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("official_bark_plus", args.source / "bark_loader.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    model = official.load_model(args.source / "model.bark", device=args.device, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(args.source, local_files_only=True)

    def read(path):
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000 or wave.ndim != 1 or not len(wave):
            raise ValueError("reference inputs must be nonempty mono 16 kHz WAVs")
        return wave

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def generate(wave):
        padded = np.concatenate([np.zeros(4000, np.float32), wave])
        inputs = processor([padded], sampling_rate=16000, padding=True, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            result = model.generate(**inputs)
        ids = result.sequences
        return ids, processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    if args.warmup_audio:
        generate(read(args.warmup_audio))
        sync()
    records = []
    for index, path in enumerate(args.audio):
        wave = read(path)
        runs = []
        for _ in range(args.iterations):
            sync()
            start = time.perf_counter()
            ids, text = generate(wave)
            sync()
            runs.append(dict(seconds=time.perf_counter() - start, text=text))
        if args.check_official_transcribe and index == 0:
            actual = official.transcribe(model, wave, 16000)
            if actual != text:
                raise AssertionError("pinned local processor differs from official transcribe()")
        records.append(dict(name=path.stem, path=str(path.resolve()), audio_seconds=len(wave) / 16000,
                            device=args.device, dtype="float32", text=text, ids=ids.cpu().tolist(), runs=runs))
        print(path.name, repr(text), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
