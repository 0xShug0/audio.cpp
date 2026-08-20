"""Dump a reference decode of the MOSS-Audio-Tokenizer v1 codec.

Decodes a fixed, deterministic code matrix with the checkpoint's own
MossAudioTokenizerModel and writes the fixture that
tests/moss_voicegen/codec_decode_parity.cpp compares against. The matrix follows the same
formula as the existing v2 test (tests/moss_tts_local/codec_decode_parity.cpp) so the two
generations are exercised the same way.

    python3 tools/community_models/moss_voicegen_reference_codec.py \
        --codec /path/to/MOSS-VoiceGenerator/audio_tokenizer \
        --output tests/moss_voicegen/reference/ref_codec_v1.json
"""

import argparse
import json
import pathlib

import torch
from transformers import AutoModel

CODEBOOKS = 16
FRAMES = 40
PROBE_COUNT = 64


def code_matrix() -> torch.Tensor:
    return torch.tensor(
        [[(q * 37 + t * 5) % 1024 for t in range(FRAMES)] for q in range(CODEBOOKS)],
        dtype=torch.long,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    codes = code_matrix()
    codec = AutoModel.from_pretrained(args.codec, trust_remote_code=True).eval()
    with torch.no_grad():
        out = codec.decode(codes.unsqueeze(1))          # (num_quantizers, batch, length)
    audio = (out.audio if hasattr(out, "audio") else out[0]).squeeze().float()

    step = max(1, audio.numel() // PROBE_COUNT)
    probe_index = list(range(0, audio.numel(), step))[:PROBE_COUNT]
    payload = {
        "codebooks": CODEBOOKS,
        "frames": FRAMES,
        "samples": int(audio.numel()),
        "peak": float(audio.abs().max()),
        "rms": float(audio.pow(2).mean().sqrt()),
        "probe_index": probe_index,
        "probe_values": [float(audio[i]) for i in probe_index],
    }
    pathlib.Path(args.output).write_text(json.dumps(payload))
    print(f"{payload['samples']} samples, peak={payload['peak']:.5f}, rms={payload['rms']:.5f} -> {args.output}")


if __name__ == "__main__":
    main()
