#!/usr/bin/env python3
"""Run Sidon's released TorchScript pipeline and capture component boundaries."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import save_file
from transformers import SeamlessM4TFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parity:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    encoder = torch.jit.load(str(args.model_dir / "feature_extractor_cuda.pt"), map_location="cuda").eval()
    decoder = torch.jit.load(str(args.model_dir / "decoder_cuda.pt"), map_location="cuda").eval()
    # facebook/w2v-bert-2.0/preprocessor_config.json differs from the class default.
    frontend = SeamlessM4TFeatureExtractor(padding_value=1)
    original, sample_rate = sf.read(args.audio, dtype="float64", always_2d=True)
    peak = np.abs(original).max()
    if peak == 0:
        raise ValueError("The official peak normalization is undefined for silent input")
    waveform = torch.from_numpy((0.9 * (original / peak)).astype(np.float32)).mean(dim=1).view(1, -1)
    original_frames = original.shape[0]
    target_samples = int(48000 / sample_rate * original_frames)
    measurements = []
    with torch.inference_mode():
        for run in range(args.warmup + args.repeat):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            wav = torchaudio.functional.highpass_biquad(waveform, sample_rate, 50)
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
            wav = torch.nn.functional.pad(wav, (0, 24000))
            restored = []
            cache = None
            stages = {"frontend_ms": 0.0, "encoder_ms": 0.0, "decoder_ms": 0.0}
            fixtures = {}
            for index, chunk in enumerate(wav.view(-1).split(16000 * 96)):
                stage = time.perf_counter()
                padded = torch.nn.functional.pad(chunk, (160, 160))
                features = frontend(padded, sampling_rate=16000, return_tensors="pt")["input_features"].cuda()
                torch.cuda.synchronize()
                stages["frontend_ms"] += (time.perf_counter() - stage) * 1000
                stage = time.perf_counter()
                hidden = encoder(features)["last_hidden_state"]
                torch.cuda.synchronize()
                stages["encoder_ms"] += (time.perf_counter() - stage) * 1000
                if cache is not None:
                    hidden = torch.cat([cache, hidden], dim=1)
                stage = time.perf_counter()
                audio = decoder(hidden.transpose(1, 2)).view(-1)[:-960]
                torch.cuda.synchronize()
                stages["decoder_ms"] += (time.perf_counter() - stage) * 1000
                restored.append(audio)
                cache = hidden[:, -1:]
                if args.parity:
                    fixtures[f"chunk{index}.padded"] = padded.contiguous()
                    fixtures[f"chunk{index}.features"] = features.cpu().contiguous()
                    fixtures[f"chunk{index}.hidden"] = hidden.cpu().contiguous()
                    fixtures[f"chunk{index}.audio"] = audio.cpu().contiguous()
            output = torch.cat(restored)[:target_samples].cpu().numpy()
            elapsed = (time.perf_counter() - start) * 1000
            row = {"warmup": run < args.warmup, "session.wall_ms": elapsed,
                   "audio_seconds": original_frames / sample_rate,
                   "rtf": elapsed / 1000 / (original_frames / sample_rate),
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                   **stages}
            print(json.dumps(row), flush=True)
            measurements.append(row)
    sf.write(args.output_dir / "audio.wav", output, 48000, subtype="FLOAT")
    if args.parity:
        save_file(fixtures, args.output_dir / "components.safetensors")
    (args.output_dir / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    main()
