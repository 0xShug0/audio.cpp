#!/usr/bin/env python3
"""Teacher-forced codec-decode parity: C++ vs Python reference.

The C++ target ``codec_decode_parity`` decodes a FIXED, deterministic code
matrix (codes[q][t] = (q*37 + t*5) % 1024) to a 48 kHz stereo WAV. This script
decodes the *identical* codes through the Python MOSS-Audio-Tokenizer-v2 in fp32
and reports the cosine / max-abs difference against the C++ WAV.

Unlike end-to-end audio cosine (meaningless for a free-running AR model, whose
greedy rollout diverges), this is deterministic and teacher-forced: identical
codes in, so it directly measures codec-decoder numerical parity.

Run in the ``qwen3-tts`` conda env. Example:
    # C++ side first:
    build/debug/bin/codec_decode_parity --frames 64 --out cpp.wav
    # then:
    python moss_tts_local_codec_parity.py --frames 64 --cpp-wav cpp.wav \
        --min-cosine COSINE_BOUND --max-abs-error ABS_BOUND \
        --min-cosine COSINE_BOUND --max-abs-error ABS_BOUND
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as h:
        ch, sr, sw, n = h.getnchannels(), h.getframerate(), h.getsampwidth(), h.getnframes()
        raw = h.readframes(n)
    if sw != 2 or sr != 48000 or ch != 2 or n <= 0 or len(raw) != n * ch * sw:
        raise ValueError(f"{path}: expected nonempty complete 48 kHz stereo PCM16 WAV")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return sr, audio.reshape(-1, ch)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("empty, nonfinite or mismatched codec arrays")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        raise ValueError("invalid codec signal norm")
    return float(np.dot(a, b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codec", default=CODEC_REPO, help="codec repo id or local snapshot path")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--num-quantizers", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("moss_codec_decode_py.wav"))
    parser.add_argument("--cpp-wav", type=Path, help="C++ codec_decode_parity WAV to compare against")
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--compare-only", action="store_true", help="compare existing --out and --cpp-wav without model inference")
    args = parser.parse_args()
    if args.compare_only and not args.cpp_wav:
        parser.error("--compare-only requires --cpp-wav")
    if args.frames <= 0 or not 1 <= args.num_quantizers <= 12:
        parser.error("frames must be positive and num-quantizers must be in [1, 12]")
    if args.cpp_wav and (args.min_cosine is None or not np.isfinite(args.min_cosine) or
                        not -1 <= args.min_cosine <= 1 or args.max_abs_error is None or
                        not np.isfinite(args.max_abs_error) or args.max_abs_error < 0):
        parser.error("comparison requires explicit --min-cosine and --max-abs-error")

    if not args.compare_only:
        import torch
        import torchaudio
        from transformers import AutoModel

        codec = AutoModel.from_pretrained(str(args.codec), trust_remote_code=True)
        codec = codec.to(device=args.device, dtype=torch.float32)
        if hasattr(codec, "set_compute_dtype"):
            codec.set_compute_dtype("fp32")
        codec.eval()

        nq, frames = args.num_quantizers, args.frames
        codes = torch.tensor(
            [[(q * 37 + t * 5) % 1024 for t in range(frames)] for q in range(nq)],
            dtype=torch.long,
            device=args.device,
        )  # [nq, frames], identical to the C++ target

        with torch.no_grad():
            out = codec.decode(codes, num_quantizers=nq)
        audio = getattr(out, "audio", out[0] if isinstance(out, tuple) else out)
        audio = audio.squeeze().to(torch.float32).cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() != 2 or audio.shape[0] != 2 or audio.shape[1] == 0 or not torch.isfinite(audio).all():
            raise ValueError("codec must produce finite nonempty stereo audio")
        torchaudio.save(str(args.out), audio, 48000, encoding="PCM_S", bits_per_sample=16)
        print(f"[PY] decoded codes[{nq},{frames}] -> {tuple(audio.shape)} @48000, wrote {args.out}")

    if args.cpp_wav:
        py_sr, py = read_wav(args.out)
        cpp_sr, cpp = read_wav(args.cpp_wav)
        if py_sr != cpp_sr or py.shape != cpp.shape:
            raise ValueError("codec sample rate or shape mismatch")
        c = cosine(cpp, py)
        max_abs = float(np.max(np.abs(cpp - py)))
        if not np.isfinite(c) or c < args.min_cosine or max_abs > args.max_abs_error:
            raise ValueError(f"codec parity failed: cosine={c}, max_abs={max_abs}")
        print(f"\n=== codec-decode parity (teacher-forced, fp32) ===")
        print(f"cpp={args.cpp_wav} ({cpp_sr} Hz, {cpp.shape}), py={args.out} ({py_sr} Hz, {py.shape})")
        print(f"cosine={c:.9f}  max_abs_diff={max_abs:.6e}  (16-bit WAV LSB ~= {1/32768:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
