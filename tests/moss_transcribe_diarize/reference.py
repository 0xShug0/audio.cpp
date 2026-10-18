#!/usr/bin/env python3
"""Run the pinned upstream implementation without changing its generation path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])
    sys.path.insert(0, str(args.upstream.resolve()))
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    logging.info("torch=%s device=%s dtype=%s TF32=false", torch.__version__,
                 torch.cuda.get_device_name(device), args.dtype)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True,
        dtype=dtype, attn_implementation="sdpa").to(device).eval()
    processor = AutoProcessor.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True)
    torch.cuda.synchronize()
    logging.info("load_seconds=%.6f weight_bytes=%d resident_allocated_bytes=%d",
                 time.perf_counter() - started,
                 sum(p.numel() * p.element_size() for p in model.parameters()),
                 torch.cuda.memory_allocated())

    results = []
    for audio in args.audio:
        info = sf.info(audio)
        with audio.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        messages = build_transcription_messages(audio)
        trials = []
        for trial in range(args.repeats + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = generate_transcription(
                model, processor, messages, max_new_tokens=args.max_tokens,
                do_sample=False, device=device, dtype=dtype)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if result["generated_tokens"] >= args.max_tokens:
                raise RuntimeError(f"Generation hit max-tokens; increase it before using this as a baseline: {audio}")
            result.update(seconds=elapsed, rtf=elapsed / info.duration,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            result["segments"] = [
                dict(start=s.start, end=s.end, speaker=s.speaker, text=s.text)
                for s in parse_transcript(result["text"])]
            logging.info("audio=%s trial=%d result=%s", audio, trial, json.dumps(result, ensure_ascii=False))
            trials.append(result)
        if any(t["text"] != trials[0]["text"] for t in trials):
            raise RuntimeError(f"Upstream greedy output changed across repeated runs: {audio}")
        results.append(dict(audio=str(audio.resolve()), sha256=digest,
                            duration=info.duration, dtype=args.dtype,
                            max_tokens=args.max_tokens, trials=trials,
                            median_seconds=statistics.median(t["seconds"] for t in trials[1:])))
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
