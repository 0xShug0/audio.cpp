#!/usr/bin/env python3
"""Generate the Voice Isolator parity fixtures from the Python package.

The fixtures pin inworld.voice_isolation 1.0.0 on one fixed test signal: the
public isolate and stream outputs plus the intermediate tensors the C++ port
must reproduce. Weights are read from --checkpoint-dir, a snapshot of
inworld/voice-isolator; the fixtures go to --output-dir, outside the repository.
See README.md beside this script.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from inworld.voice_isolation import VoiceIsolator
from safetensors.torch import load_file, save_file


SAMPLE_RATE = 48000
OFFLINE_WEIGHTS = "denoiser.safetensors"
REALTIME_WEIGHTS = "denoiser_realtime.safetensors"
SIGNAL_SAMPLES = 28800
LONG_SAMPLES = 960000
LONG_TAIL_SAMPLES = 24000
LONG_DECIMATION = 100
BLOCK_FRAMES = 4
HEAD_FRAMES = 2
SPECTRUM_FRAMES = 8
PUSH_SIZES = (480, 960, 1440, 333, 2048)
TOLERANCE = 1e-5
FIXTURES = (
    "offline_case0.safetensors",
    "offline_case1.safetensors",
    "realtime_case0.safetensors",
    "realtime_case1.safetensors",
    "realtime_case2.safetensors",
    "realtime_long.safetensors",
)


def signal(n: int) -> torch.Tensor:
    """Returns n samples of the fixed 48 kHz test clip as float32 mono.

    A 150 Hz and a 450 Hz tone under a 3 Hz envelope, a 3.1 kHz tone, and
    noise from a 32-bit LCG. The LCG starts at 12345 and steps
    state = (state * 1664525 + 1013904223) mod 2**32 before each sample, with
    noise = state / 2**32 * 2 - 1. The math runs in float64 and the result is
    cast to float32 last, so the C++ side can rebuild the same input.
    """
    noise = np.empty(n, dtype=np.float64)
    state = 12345
    for i in range(n):
        state = (state * 1664525 + 1013904223) % 2**32
        noise[i] = state / 2**32 * 2.0 - 1.0
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2.0 * np.pi * 3.0 * t)
    x = (
        0.3 * np.sin(2.0 * np.pi * 150.0 * t) * envelope
        + 0.15 * np.sin(2.0 * np.pi * 450.0 * t + 0.5) * envelope
        + 0.05 * np.sin(2.0 * np.pi * 3100.0 * t)
        + 0.08 * noise
    )
    return torch.from_numpy(x.astype(np.float32))


def realtime_isolator(checkpoint_dir: Path, lookahead: int) -> VoiceIsolator:
    return VoiceIsolator(
        str(checkpoint_dir / REALTIME_WEIGHTS),
        device="cpu",
        precision="fp32",
        mode="streaming",
        lookahead_frames=lookahead,
        block_frames=BLOCK_FRAMES,
        window_frames=0,
    )


def run_stream(stream, audio: torch.Tensor, push_sizes: tuple[int, ...]) -> torch.Tensor:
    """Pushes the clip in pieces cycling through push_sizes, then flushes."""
    outputs = []
    start = 0
    turn = 0
    while start < audio.shape[0]:
        size = min(push_sizes[turn % len(push_sizes)], audio.shape[0] - start)
        outputs.append(stream.push(audio[start : start + size]))
        start += size
        turn += 1
    outputs.append(stream.flush())
    return torch.cat(outputs)


def stream_spectrum(network, audio: torch.Tensor) -> torch.Tensor:
    """Spectra [1, bins, frames] as the stream analyses them.

    Frame f covers samples [f * hop - n_fft / 2, f * hop + n_fft / 2) with zeros
    outside the clip, for frames 0 through audio // hop.
    """
    n_fft, hop = network.config.n_fft, network.config.hop_length
    frames = audio.shape[0] // hop + 1
    padded = torch.nn.functional.pad(audio.float(), (n_fft // 2, n_fft))
    index = torch.arange(frames)[:, None] * hop + torch.arange(n_fft)[None, :]
    segments = padded[index]
    return torch.fft.rfft(segments * network.window.float()).T[None]


def network_heads(network, spectrum: torch.Tensor) -> dict[str, torch.Tensor]:
    """The first frames of each stage of the network, replaying its features loop."""
    causal = not network.offline
    features = network.encode(spectrum)
    film = network.film()
    heads = {
        "spectrum_head": torch.view_as_real(spectrum)[0, :, :SPECTRUM_FRAMES],
        "encoded_head": features[0, :, :HEAD_FRAMES],
    }
    for index, block in enumerate(network.blocks):
        features = block(features, causal)
        if film is not None:
            features = features * (1 + film[index, 0]) + film[index, 1]
        heads[f"block{index}_head"] = features[0, :, :HEAD_FRAMES]
    heads["raw_head"] = network.denoise_decoder(features)[0, :, :SPECTRUM_FRAMES]
    return heads


def save(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    saved = {}
    for name, value in tensors.items():
        dtype = torch.float32 if value.is_floating_point() else torch.int64
        saved[name] = value.detach().cpu().to(dtype).contiguous()
    save_file(saved, str(path), metadata={"package": "inworld.voice_isolation 1.0.0", "sample_rate": str(SAMPLE_RATE)})


@torch.inference_mode()
def generate(checkpoint_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = signal(SIGNAL_SAMPLES)

    offline = VoiceIsolator(
        str(checkpoint_dir / OFFLINE_WEIGHTS),
        device="cpu",
        precision="fp32",
        mode="offline",
    )
    network = offline.model
    spectrum = network.stft(audio[None])
    save(output_dir / "offline_case0.safetensors", {
        "input": audio,
        "output": offline.isolate(audio, SAMPLE_RATE),
        "band_edges": torch.tensor(network.edges, dtype=torch.int64),
        **network_heads(network, spectrum),
    })
    save(output_dir / "offline_case1.safetensors", {
        "input": audio,
        "output": network.enhance(audio, chunk_seconds=0.25, overlap_seconds=0.05),
        "chunk_samples": torch.tensor([round(0.25 * SAMPLE_RATE)], dtype=torch.int64),
        "overlap_samples": torch.tensor([round(0.05 * SAMPLE_RATE)], dtype=torch.int64),
    })

    realtime = realtime_isolator(checkpoint_dir, 2)
    stream = realtime.stream()
    output = run_stream(stream, audio, PUSH_SIZES)
    save(output_dir / "realtime_case0.safetensors", {
        "input": audio,
        "output": output,
        "lookahead_frames": torch.tensor([2], dtype=torch.int64),
        "latency_samples": torch.tensor([stream.latency_samples], dtype=torch.int64),
        **network_heads(realtime.model, stream_spectrum(realtime.model, audio)),
    })

    for name, lookahead in (("realtime_case1", 4), ("realtime_case2", 0)):
        stream = realtime_isolator(checkpoint_dir, lookahead).stream()
        save(output_dir / f"{name}.safetensors", {
            "input": audio,
            "output": run_stream(stream, audio, (960,)),
            "lookahead_frames": torch.tensor([lookahead], dtype=torch.int64),
            "latency_samples": torch.tensor([stream.latency_samples], dtype=torch.int64),
        })

    long_audio = signal(LONG_SAMPLES)
    long_output = run_stream(realtime_isolator(checkpoint_dir, 2).stream(), long_audio, (960,))
    save(output_dir / "realtime_long.safetensors", {
        "num_samples": torch.tensor([LONG_SAMPLES], dtype=torch.int64),
        "output_tail": long_output[-LONG_TAIL_SAMPLES:],
        "output_decimated": long_output[::LONG_DECIMATION],
    })


def compare(expected: torch.Tensor, actual: torch.Tensor) -> str | None:
    if expected.dtype != actual.dtype:
        return f"dtype {expected.dtype} != {actual.dtype}"
    if expected.shape != actual.shape:
        return f"shape {tuple(expected.shape)} != {tuple(actual.shape)}"
    if expected.is_floating_point():
        diff = float((expected.double() - actual.double()).abs().max()) if expected.numel() else 0.0
        if not diff <= TOLERANCE:
            return f"max abs diff {diff:.3g} > {TOLERANCE:g}"
    elif not torch.equal(expected, actual):
        return "integer values differ"
    return None


def check(checkpoint_dir: Path, output_dir: Path) -> int:
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        generate(checkpoint_dir, Path(scratch))
        for name in FIXTURES:
            existing = output_dir / name
            if not existing.exists():
                failures.append(f"{name}: fixture missing ({existing})")
                continue
            expected = load_file(str(existing))
            actual = load_file(str(Path(scratch) / name))
            for key in sorted(expected.keys() | actual.keys()):
                if key not in expected or key not in actual:
                    failures.append(f"{name}:{key}: present on one side only")
                    continue
                problem = compare(expected[key], actual[key])
                if problem:
                    failures.append(f"{name}:{key}: {problem}")
    if failures:
        for line in failures:
            print(f"MISMATCH {line}", file=sys.stderr)
        print(f"FAIL: {len(failures)} tensor(s) differ from the fixtures in --output-dir", file=sys.stderr)
        return 1
    print(f"OK: {len(FIXTURES)} fixtures match the files in --output-dir")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="regenerate into a temporary directory and compare with --output-dir")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.check:
        sys.exit(check(args.checkpoint_dir, args.output_dir))
    generate(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
