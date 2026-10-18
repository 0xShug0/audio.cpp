#!/usr/bin/env python3
"""Run the official OpenVoice converter without its separate watermark model."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time

import librosa
import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--voice-ref", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    from openvoice.models import SynthesizerTrn
    from openvoice.mel_processing import spectrogram_torch

    if args.parity:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    data = config["data"]
    model = SynthesizerTrn(
        0, data["filter_length"] // 2 + 1,
        n_speakers=data["n_speakers"], **config["model"]
    ).to(args.device).eval()
    checkpoint = torch.load(args.checkpoint_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def load(path):
        audio, _ = librosa.load(path, sr=data["sampling_rate"])
        return torch.from_numpy(audio).unsqueeze(0).to(args.device)

    source, reference = load(args.audio), load(args.voice_ref)
    sf.write(args.output_dir / "source.wav", source[0].cpu().numpy(), data["sampling_rate"], subtype="FLOAT")
    sf.write(args.output_dir / "reference.wav", reference[0].cpu().numpy(), data["sampling_rate"], subtype="FLOAT")

    def spectrum(audio):
        return spectrogram_torch(
            audio, data["filter_length"], data["sampling_rate"],
            data["hop_length"], data["win_length"], center=False)

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    timings = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU] + (
            [torch.profiler.ProfilerActivity.CUDA] if args.device.startswith("cuda") else []),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=max(1, args.runs - 2)),
        record_shapes=True,
    ) if args.profile else nullcontext()
    with torch.inference_mode(), profiler as profile:
        for _ in range(args.runs):
            torch.manual_seed(args.seed)
            sync()
            start = time.perf_counter()
            spec = spectrum(source)
            reference_spec = spectrum(reference)
            source_embedding = model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
            target_embedding = model.ref_enc(reference_spec.transpose(1, 2)).unsqueeze(-1)
            lengths = torch.tensor([spec.shape[-1]], device=args.device)
            audio = model.voice_conversion(
                spec, lengths, source_embedding, target_embedding, tau=args.tau
            )[0][0, 0].cpu().numpy()
            sync()
            timings.append((time.perf_counter() - start) * 1000)
            if profile is not None:
                profile.step()
    if args.profile:
        (args.output_dir / "profile.txt").write_text(profile.key_averages(group_by_input_shape=True).table(
            sort_by="self_device_time_total", row_limit=40))
    sf.write(args.output_dir / "audio.wav", audio, data["sampling_rate"], subtype="FLOAT")
    duration = source.shape[-1] / data["sampling_rate"]
    median = statistics.median(timings[1:]) if len(timings) > 1 else timings[0]
    report = {
        "mode": "profile" if args.profile else ("parity" if args.parity else "performance"),
        "watermark": False, "seed": args.seed, "tau": args.tau,
        "input_duration_seconds": duration,
        "session.wall_ms": timings,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
    }
    if not args.parity and not args.profile and len(timings) > 1:
        report["warm_median_ms"] = median
        report["warm_rtf"] = median / (duration * 1000)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
