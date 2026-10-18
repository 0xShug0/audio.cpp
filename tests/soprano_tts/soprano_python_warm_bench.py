#!/usr/bin/env python3
"""Soprano TTS Python warm bench — runs the official Python reference and collects timing.

Usage:
    python3 tests/soprano_tts/soprano_python_warm_bench.py --model models/Soprano-1.1-80M --out-dir build/logs/warmbench/soprano_tts_py
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_wav_duration(path: str) -> float:
    with open(path, "rb") as f:
        data = f.read()
    sr = struct.unpack("<I", data[24:28])[0]
    pos = data.find(b"data")
    if pos < 0:
        return 0.0
    data_size = struct.unpack("<I", data[pos - 4 : pos])[0]
    fmt = struct.unpack("<H", data[20:22])[0]
    bytes_per_sample = 4 if fmt == 3 else 2
    return data_size / bytes_per_sample / sr if sr > 0 else 0.0

def load_cases(path: str) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    current_section = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
            elif current_section:
                catalog.setdefault(current_section, []).append(line)
    return catalog

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Soprano-1.1-80M")
    ap.add_argument("--cases", default="tests/soprano_tts/soprano_warm_bench_cases.txt")
    ap.add_argument("--out-dir", default="build/logs/warmbench/soprano_tts_py")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model_path = os.path.join(REPO_ROOT, args.model)
    cases_path = os.path.join(REPO_ROOT, args.cases)
    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from soprano import SopranoTTS

    print("=== Soprano Python WarmBench ===")
    print(f"Model: {model_path}")
    print(f"Output: {out_dir}")

    # Load model
    t0 = time.time()
    model = SopranoTTS(backend="auto", device="cpu", model_path=model_path)
    load_time = time.time() - t0
    print(f"Load: {load_time:.2f}s")

    # Load cases
    catalog = load_cases(cases_path)

    # Warmup
    print("\nWarmup...")
    model.infer("At sunrise the studio monitors clicked on, and the first calibration phrase rolled across the room with steady timing.")
    print("Warmup complete\n")

    # Run cases
    results = []
    for section, texts in catalog.items():
        print(f"Case: {section} ({len(texts)} texts)")
        for i, text in enumerate(texts):
            case_name = f"{section}_{i}"
            out_path = os.path.join(out_dir, f"{case_name}.wav")

            t0 = time.time()
            model.infer(text, out_path)
            infer_time = time.time() - t0

            audio_dur = parse_wav_duration(out_path)
            rtf = infer_time / audio_dur if audio_dur > 0 else 0

            results.append({
                "name": case_name,
                "infer_time_s": round(infer_time, 3),
                "audio_duration_s": round(audio_dur, 3),
                "rtf": round(rtf, 4),
            })

            print(f"  {case_name}: {infer_time*1000:.0f} ms infer, {audio_dur:.3f} s audio, RTF={rtf:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Name':<20} {'Infer (s)':<12} {'Audio (s)':<12} {'RTF':<10}")
    print("-" * 54)
    for r in results:
        print(f"{r['name']:<20} {r['infer_time_s']:<12.3f} {r['audio_duration_s']:<12.3f} {r['rtf']:<10.4f}")

    if args.json:
        print(json.dumps({"load_time_s": round(load_time, 3), "results": results}, indent=2))


if __name__ == "__main__":
    main()
