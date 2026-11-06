#!/usr/bin/env python3
"""Run CLI request-sequence cases through the unmodified official MuLaCover pipeline."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.log)], force=True)
    sys.path.insert(0, str(args.upstream.resolve() / "src"))
    from mulacover import MuLaCoverGenPipeline

    torch.set_num_threads(args.threads)
    # Match the official CLI's component dtypes. Keep models resident so the
    # measured requests use one reusable session, as in the native CLI.
    pipeline = MuLaCoverGenPipeline.from_pretrained(str(args.model), device=torch.device("cuda"),
        dtype={"mulacover": getattr(torch, args.dtype), "codec": torch.float32,
               "qwen": torch.float32, "transcriptor": torch.float32}, lazy_load=False)
    cases = json.loads(args.requests.read_text())
    cases = cases["requests"] if isinstance(cases, dict) else cases
    results = []
    with torch.inference_mode():
        for case in cases:
            options = case.get("options", {})
            inputs = {"lyrics": options.get("lyrics", case.get("text", "")), "tags": options.get("tags", "")}
            if "audio" in case:
                inputs.update(ref_audio=str((args.requests.parent / case["audio"]).resolve()),
                              bpm=options.get("bpm"))
            else:
                inputs.update(melody_midi=options["melody_file"], chord_midi=options["chord_file"],
                              drum_midi=options.get("drum_file"))
            torch.manual_seed(int(options.get("seed", 1234)))
            torch.cuda.synchronize()
            start = time.perf_counter()
            prepared = pipeline.preprocess(inputs, cfg_scale=float(options.get("guidance_scale", 1.5)),
                max_audio_length_ms=int(float(options.get("duration_sec", 300)) * 1000))
            torch.cuda.synchronize()
            ar_start = time.perf_counter()
            generated = pipeline._forward(prepared,
                max_audio_length_ms=int(float(options.get("duration_sec", 300)) * 1000),
                temperature=float(options.get("temperature", 1)), topk=int(options.get("top_k", 250)),
                cfg_scale=float(options.get("guidance_scale", 1.5)), disable_progress=True)
            torch.cuda.synchronize()
            codec_start = time.perf_counter()
            waveform = pipeline.codec.detokenize(generated["frames"].to("cuda"), duration=29.76,
                num_steps=int(options.get("num_inference_steps", 10)),
                guidance_scale=float(options.get("codec_guidance_scale", 1.25)), disable_progress=True)
            waveform = waveform.detach().float().cpu()
            torch.cuda.synchronize()
            end = time.perf_counter()
            sample_rate = pipeline.codec.sample_rate
            seconds = waveform.shape[-1] / sample_rate
            row = {"id": case["id"], "prepare_ms": (ar_start-start)*1000,
                   "ar_ms": (codec_start-ar_start)*1000, "codec_ms": (end-codec_start)*1000,
                   "wall_ms": (end-start)*1000, "audio_sec": seconds, "rtf": (end-start)/seconds,
                   "frames": generated["frames"].shape[-1]}
            results.append(row)
            sf.write(args.output / (case["id"] + ".wav"), waveform.numpy().T, sample_rate, subtype="FLOAT")
            (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
            logging.info("case=%s session.wall_ms=%.3f mulacover.prepare_ms=%.3f mulacover.ar_ms=%.3f "
                         "mulacover.codec_ms=%.3f audio.duration_sec=%.3f session.rtf=%.6f",
                         row["id"], row["wall_ms"], row["prepare_ms"], row["ar_ms"],
                         row["codec_ms"], row["audio_sec"], row["rtf"])
            del prepared, generated, waveform


if __name__ == "__main__":
    main()
