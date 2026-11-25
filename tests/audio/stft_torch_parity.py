#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare audio.cpp STFT against torch.stft.")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--dump-bin", type=Path, default=Path("build/debug/bin/stft_parity_dump"))
    parser.add_argument("--kind", choices=("stft",), default="stft")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--pad-mode", choices=("constant", "reflect"), default="constant")
    parser.add_argument("--preemphasis", type=float, default=0.0)
    parser.add_argument("--periodic-window", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


def run_dump(
    dump_bin: Path,
    wav: Path,
    kind: str,
    preemphasis: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    pad_mode: str,
    periodic_window: bool,
    log: bool,
) -> tuple[np.ndarray, tuple[int, ...]]:
    with tempfile.TemporaryDirectory(prefix=f"stft-{kind}-") as tmp:
        tmp_path = Path(tmp)
        f32_path = tmp_path / "spec.f32"
        shape_path = tmp_path / "shape.i64"
        cmd = [str(dump_bin), str(wav), kind, str(f32_path), str(shape_path)]
        if preemphasis != 0.0:
            cmd.extend(["--preemphasis", str(preemphasis)])
        cmd.extend(
            [
                "--n-fft",
                str(n_fft),
                "--hop-length",
                str(hop_length),
                "--win-length",
                str(win_length),
                "--pad-mode",
                pad_mode,
            ]
        )
        if periodic_window:
            cmd.append("--periodic-window")
        if log:
            cmd.append("--log")
        subprocess.run(cmd, check=True)
        shape = tuple(np.fromfile(shape_path, dtype=np.int64).tolist())
        spec = np.fromfile(f32_path, dtype=np.float32).reshape(shape)
        return spec, shape


def apply_preemphasis(audio: np.ndarray, coeff: float) -> np.ndarray:
    if coeff == 0.0 or audio.size == 0:
        return audio
    out = audio.copy()
    out[1:] = out[1:] - coeff * out[:-1]
    return out


def compare(name: str, cpp_spec: np.ndarray, torch_spec: np.ndarray, atol: float, rtol: float) -> None:
    if cpp_spec.shape != torch_spec.shape:
        raise AssertionError(f"shape mismatch: {cpp_spec.shape} != {torch_spec.shape}")
    if not np.isfinite(cpp_spec).all() or not np.isfinite(torch_spec).all():
        raise AssertionError("STFT output contains non-finite values")
    diff = cpp_spec - torch_spec
    abs_diff = np.abs(diff)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(f"{name}:")
    print(f"  shape={cpp_spec.shape}")
    print(f"  max_abs={float(abs_diff.max())}")
    print(f"  mean_abs={float(abs_diff.mean())}")
    print(f"  rmse={float(np.sqrt(np.mean(diff * diff)))}")
    print(f"  worst_idx={idx}")
    print(f"  cpp_val={float(cpp_spec[idx])}")
    print(f"  torch_val={float(torch_spec[idx])}")
    np.testing.assert_allclose(cpp_spec, torch_spec, atol=atol, rtol=rtol)


def main() -> int:
    args = parse_args()
    audio, sr = sf.read(args.wav, dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"expected 16000 Hz WAV, got {sr}")
    if audio.ndim != 1:
        raise RuntimeError("expected mono WAV")

    audio = apply_preemphasis(audio, args.preemphasis)
    x = torch.from_numpy(audio)
    window = torch.hann_window(
        args.win_length,
        periodic=args.periodic_window,
        dtype=torch.float32,
    )
    spec = torch.stft(
        x,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=window,
        center=True,
        pad_mode=args.pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    torch_spec = torch.view_as_real(spec).unsqueeze(0).numpy().astype(np.float32)

    cpp_spec, _ = run_dump(
        args.dump_bin,
        args.wav,
        args.kind,
        args.preemphasis,
        args.n_fft,
        args.hop_length,
        args.win_length,
        args.pad_mode,
        args.periodic_window,
        args.log,
    )
    compare(args.kind, cpp_spec, torch_spec, args.atol, args.rtol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
