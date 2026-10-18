#!/usr/bin/env python3
"""Run the official Transformers Cohere ASR reference, without changing dependencies."""

import argparse
import json
import logging
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--log", action="store_true", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(args.backend).eval()
    records = []
    with torch.inference_mode():
        for case in json.loads(args.cases.read_text()):
            waveform, sr = sf.read(case["audio"], dtype="float32")
            if waveform.ndim != 1 or sr != 16000:
                raise ValueError("Reference fixtures must be mono 16 kHz")
            durations = []
            processor.feature_extractor.max_audio_clip_s = case.get("audio_chunk_duration_sec", 35)
            processor.feature_extractor.overlap_chunk_second = min(
                5, processor.feature_extractor.max_audio_clip_s / 2
            )
            source = waveform
            if case.get("audio_chunk_mode") == "fixed":
                chunk_samples = int(processor.feature_extractor.max_audio_clip_s * sr)
                source = [waveform[i:i + chunk_samples] for i in range(0, len(waveform), chunk_samples)]
            for repeat in range(args.repeats + 1):
                if args.backend == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                inputs = processor(
                    source.copy(), sampling_rate=sr, return_tensors="pt",
                    language=case.get("language", "en"), punctuation=case.get("pnc", True),
                )
                chunk_index = inputs.pop("audio_chunk_index", None)
                inputs = inputs.to(args.backend, dtype=model.dtype)
                outputs = model.generate(**inputs, max_new_tokens=case.get("max_tokens", 256), do_sample=False)
                text = processor.decode(
                    outputs, skip_special_tokens=True, audio_chunk_index=chunk_index,
                    language=case.get("language", "en"),
                )
                if case.get("audio_chunk_mode") == "fixed":
                    separator = "" if case.get("language", "en") in ("ja", "zh") else " "
                    text = [separator.join(piece.strip() for piece in text if piece.strip())]
                if args.backend == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    durations.append(elapsed)
                logging.info("case=%s repeat=%d elapsed=%.6f text=%s", case["id"], repeat, elapsed, text)
            records.append({
                "id": case["id"], "text": text, "tokens": outputs.tolist(),
                "seconds": durations, "audio_seconds": len(waveform) / sr,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.backend == "cuda" else None,
                "backend": args.backend, "dtype": args.dtype, "torch": torch.__version__,
            })
            args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
