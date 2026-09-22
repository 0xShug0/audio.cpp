#!/usr/bin/env python3
"""Generate golden R2T2 outputs from the macOS MPS baseline.

Run from the Confucius4-R2T2 repo (uv environment):

    cd /Users/david/github/voice/Confucius4-R2T2
    PYTHONPATH=. uv run python /Users/david/github/voice/audio.cpp/tests/confucius4_r2t2/make_golden.py \
        --model_path checkpoints/r2t2 --audio resources/test.wav \
        --out /Users/david/github/voice/audio.cpp/tests/confucius4_r2t2/golden.json

Produces offline transcript plus the per-chunk (text, fixed_text) streaming
sequence used to verify the C++ LSP streaming port step by step.
"""

import argparse
import json
import sys

import numpy as np

from r2t2 import R2T2ASRModel
from r2t2.r2t2_asr import ASRStreamingState  # noqa: F401  (state shape doc)


def read_wav_16k(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.asarray(wav, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="checkpoints/r2t2")
    ap.add_argument("--audio", default="resources/test.wav")
    ap.add_argument("--language", default=None)
    ap.add_argument("--context", default="")
    ap.add_argument("--chunk_ms", type=int, default=320)
    ap.add_argument("--unfixed_chunk_num", type=int, default=2)
    ap.add_argument("--unfixed_token_num", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--dtype", default=None, help="e.g. float16 or bfloat16 (default: MPS float16)")
    ap.add_argument("--out", default="golden.json")
    args = ap.parse_args()

    wav = read_wav_16k(args.audio)
    asr = R2T2ASRModel.from_pretrained(args.model_path, dtype=args.dtype)

    golden = {
        "audio": args.audio,
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
