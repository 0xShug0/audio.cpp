#!/usr/bin/env python3
"""Run the official RE-USE model for waveform parity and performance checks."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import statistics
import sys
import time

import soundfile as sf
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None,
                        help="Override TF32 for controlled parity only; ordinary runs retain PyTorch defaults.")
    parser.add_argument("--audio-chunk-duration-sec", type=float, default=0)
    parser.add_argument("--audio-chunk-overlap-sec", type=float, default=1)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])
    root = args.reference_root.resolve()
    sys.path.insert(0, str(root))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    from utils.util import pad_or_trim_to_match

    torch.set_num_threads(args.threads)
    if args.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    cfg = json.loads((root / "config.json").read_text())
    model = SEMamba(cfg)
    model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    model = model.cuda().eval()
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    if not 8000 <= sr <= 48000:
        raise ValueError("Reference input must be 8-48 kHz")
    waveform = torch.from_numpy(audio.T.copy()).cuda()
    stft = cfg["stft_cfg"]
    sizes = [int(stft[key] * sr // stft["sampling_rate"]) for key in ("n_fft", "hop_size", "win_size")]
    n_fft, hop, win = [value + value % 2 for value in sizes]
    compression = cfg["model_cfg"]["compress_factor"]
    chunk_samples = round(args.audio_chunk_duration_sec * sr)
    overlap = round(args.audio_chunk_overlap_sec * sr)
    if args.audio_chunk_duration_sec < 0 or overlap < 0 or (chunk_samples and (chunk_samples <= n_fft // 2 or overlap >= chunk_samples)):
        raise ValueError("Invalid chunk duration or overlap")
    starts = [0]
    if chunk_samples and chunk_samples < waveform.shape[1]:
        stride = chunk_samples - overlap
        count = (waveform.shape[1] - chunk_samples + stride - 1) // stride + 1
        starts = [min(i * stride, waveform.shape[1] - chunk_samples) for i in range(count)]
    else:
        chunk_samples = waveform.shape[1]
    window = torch.hann_window(chunk_samples).cuda().clamp_min(1e-7)
    timings = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for repeat in range(args.repeats + 1):
            begin = time.perf_counter()
            frontend_ms = network_ms = synthesis_ms = 0
            output = torch.zeros_like(waveform)
            if len(starts) > 1:
                counter = torch.zeros_like(waveform)
            for channel in range(waveform.shape[0]):
                for start in starts:
                    chunk = waveform[channel:channel + 1, start:start + chunk_samples]
                    torch.cuda.synchronize()
                    chunk_begin = time.perf_counter()
                    mag, phase, _ = mag_phase_stft(chunk, n_fft, hop, win, compression, addeps=False)
                    torch.cuda.synchronize()
                    frontend_end = time.perf_counter()
                    amp, phase, _ = model(mag, phase)
                    torch.cuda.synchronize()
                    network_end = time.perf_counter()
                    zeros = (torch.expm1(torch.relu(amp)) == 0).sum(1) / amp.shape[1]
                    amp[:, :, (zeros > 0.5)[0]] = 0
                    enhanced = mag_phase_istft(amp, phase, n_fft, hop, win, compression)
                    enhanced = pad_or_trim_to_match(chunk, enhanced, pad_value=1e-8)
                    if len(starts) > 1:
                        output[channel:channel + 1, start:start + chunk_samples] += enhanced * window
                        counter[channel:channel + 1, start:start + chunk_samples] += window
                    else:
                        output[channel:channel + 1] = enhanced
                    torch.cuda.synchronize()
                    frontend_ms += (frontend_end - chunk_begin) * 1000
                    network_ms += (network_end - frontend_end) * 1000
                    synthesis_ms += (time.perf_counter() - network_end) * 1000
            if len(starts) > 1:
                output /= counter
            torch.cuda.synchronize()
            end = time.perf_counter()
            row = dict(repeat=repeat, frontend_ms=frontend_ms, network_ms=network_ms,
                       synthesis_ms=synthesis_ms, total_ms=(end-begin)*1000)
            timings.append(row)
            logging.info("%s", json.dumps(row))
    sf.write(args.out, output.cpu().numpy().T, sr, subtype="FLOAT")
    with (root / "model.safetensors").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    median = statistics.median(row["total_ms"] for row in timings[1:])
    result = dict(checkpoint_sha256=digest, torch=torch.__version__, dtype="float32",
                  tf32=args.tf32, device=torch.cuda.get_device_name(), threads=args.threads,
                  input=str(args.audio.resolve()), sample_rate=sr, samples=len(audio), channels=audio.shape[1],
                  audio_chunk_duration_sec=args.audio_chunk_duration_sec,
                  audio_chunk_overlap_sec=args.audio_chunk_overlap_sec, chunks=len(starts),
                  output=str(args.out.resolve()), timings=timings, warm_median_ms=median,
                  rtf=median/1000/(len(audio)/sr),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    logging.info("%s", json.dumps(result))


if __name__ == "__main__":
    main()
