#!/usr/bin/env python3
"""Generate golden R2T2 outputs from the macOS MPS baseline.

Run from the upstream Confucius4-R2T2 checkout with its MPS dependencies.
Pass explicit --model_path and --out; see README.md for examples. The default
input is audio.cpp's shared English assets/resources/sample_16k.wav.

Produces offline transcript plus the per-chunk (text, fixed_text) streaming
sequence used to verify the C++ LSP streaming port step by step.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def read_wav_16k(path):
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.asarray(wav, dtype=np.float32)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="Upstream model checkpoint path")
    ap.add_argument("--audio", type=Path, default=repo_root / "assets/resources/sample_16k.wav",
                    help="Input WAV (default: repo assets/resources/sample_16k.wav, English)")
    ap.add_argument("--language", default=None)
    ap.add_argument("--context", default="")
    ap.add_argument("--chunk_ms", type=int, default=320)
    ap.add_argument("--unfixed_chunk_num", type=int, default=2)
    ap.add_argument("--unfixed_token_num", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--dtype", default=None, help="e.g. float16 or bfloat16 (default: MPS float16)")
    ap.add_argument("--out", required=True, help="Explicit destination JSON path")
    args = ap.parse_args()

    from r2t2 import R2T2ASRModel

    audio_path = args.audio.resolve()
    audio_identity = (audio_path.relative_to(repo_root).as_posix()
                      if audio_path.is_relative_to(repo_root) else str(audio_path))
    audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    wav = read_wav_16k(audio_path)
    asr = R2T2ASRModel.from_pretrained(args.model_path, dtype=args.dtype)

    golden = {
        "audio": audio_identity,
        "audio_sha256": audio_sha256,
        "chunk_ms": args.chunk_ms,
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "language": args.language,
        "context": args.context,
    }

    offline = asr.transcribe((wav, 16000), context=args.context, language=args.language)
    golden["offline"] = offline if isinstance(offline, str) else str(offline)
    print("offline:", golden["offline"], file=sys.stderr)

    state = asr.init_streaming_state(
        context=args.context,
        language=args.language,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_ms / 1000.0,
    )
    chunk_samples = int(round(args.chunk_ms / 1000.0 * 16000))
    chunks = []
    pos = 0
    while pos < wav.shape[0]:
        seg = wav[pos:pos + chunk_samples]
        pos += seg.shape[0]
        before = state.chunk_id
        _, fixed = asr.streaming_transcribe(seg, state, args.max_new_tokens)
        if state.chunk_id == before:
            # Buffered tail shorter than one chunk: no decode happened.
            continue
        chunks.append({
            "fixed_text": fixed,
            "raw_decoded": state._raw_decoded,
            "text": state.text,
        })
    final = asr.finish_streaming_transcribe(state, args.max_new_tokens)
    golden["stream_chunks"] = chunks
    golden["stream_final_text"] = state.text
    golden["stream_finish_return"] = final
    print("stream final:", state.text, file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
