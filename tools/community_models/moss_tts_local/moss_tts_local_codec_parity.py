#!/usr/bin/env python3
"""Teacher-forced codec-decode parity: C++ vs Python reference.

The C++ target ``codec_decode_parity`` decodes a FIXED, deterministic code
matrix (codes[q][t] = (q*37 + t*5) % 1024) to a 48 kHz stereo WAV. This script
decodes the *identical* codes through the Python MOSS-Audio-Tokenizer-v2 in fp32
and reports the cosine / max-abs difference against the C++ WAV.

Unlike end-to-end audio cosine (meaningless for a free-running AR model, whose
greedy rollout diverges), this is deterministic and teacher-forced: identical
codes in, so it directly measures codec-decoder numerical parity.

Run in the ``moss_tts_local`` conda env. Example:
    # C++ side first:
    build/windows-cuda-release/bin/codec_decode_parity.exe --frames 64 --out cpp.wav
    # then:
    python moss_tts_local_codec_parity.py --frames 64 --cpp-wav cpp.wav
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
    assert sw == 2, f"expected pcm16, got sample width {sw}"
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return sr, audio.reshape(-1, ch)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if denom == 0 else float(np.dot(a, b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codec", default=CODEC_REPO, help="codec repo id or local snapshot path")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--num-quantizers", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("moss_codec_decode_py.wav"))
    parser.add_argument("--cpp-wav", type=Path, help="C++ codec_decode_parity WAV to compare against")
    args = parser.parse_args()

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
    torchaudio.save(str(args.out), audio, 48000)
    print(f"[PY] decoded codes[{nq},{frames}] -> {tuple(audio.shape)} @48000, wrote {args.out}")

    if args.cpp_wav:
        py_sr, py = read_wav(args.out)
        cpp_sr, cpp = read_wav(args.cpp_wav)
        n = min(py.shape[0], cpp.shape[0])
        c = cosine(cpp[:n], py[:n])
        max_abs = float(np.max(np.abs(cpp[:n].reshape(-1) - py[:n].reshape(-1))))
        print(f"\n=== codec-decode parity (teacher-forced, fp32) ===")
        print(f"cpp={args.cpp_wav} ({cpp_sr} Hz, {cpp.shape}), py={args.out} ({py_sr} Hz, {py.shape})")
        print(f"cosine={c:.9f}  max_abs_diff={max_abs:.6e}  (16-bit WAV LSB ~= {1/32768:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
