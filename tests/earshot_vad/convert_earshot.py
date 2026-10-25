#!/usr/bin/env python3
"""Convert Earshot VAD weights and frontend tables to safetensors.

Earshot (https://github.com/pykeio/earshot, MIT OR Apache-2.0) embeds its
network as ``src/weights.bin`` and its Hann window and mel filterbank as
literal tables in ``src/filters.rs``. This script copies both into one
safetensors file so the C++ port uses the exact upstream values.

Usage:
    python tests/earshot_vad/convert_earshot.py \
        --output assets/framework/models/earshot_vad/earshot_vad.safetensors

By default the sources are downloaded from the pinned upstream tag. Pass
``--source <earshot checkout>`` to use a local checkout instead. The output is
byte-for-byte deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

UPSTREAM_TAG = "1.2.2"
UPSTREAM_RAW = "https://raw.githubusercontent.com/pykeio/earshot/{tag}/{path}"
WEIGHTS_SHA256 = "27d6a6640685ac694a359cf2d55cca3b10dd70417d15fab82c10bbb2fcafdd19"

# (name, dtype, shape, offset) where offset is in f32 elements for F32 tensors
# and in f32 elements of the first i16 for I16 tensors, as in default_predictor.rs.
WEIGHT_LAYOUT = [
    ("norm_weight", "F32", [40], 0),
    ("layer1_kernel", "F32", [3, 3], 40),
    ("layer1_weight", "F32", [16], 49),
    ("layer1_bias", "F32", [16], 65),
    ("layer2_kernel", "F32", [16, 3], 81),
    ("layer2_weight", "F32", [16, 16], 129),
    ("layer2_bias", "F32", [16], 385),
    ("layer3_kernel", "F32", [16, 3], 401),
    ("layer3_weight", "F32", [16, 16], 449),
    ("rnn1_weight", "I16", [128, 80], 705),
    ("rnn2_weight", "I16", [128, 64], 5825),
    ("output_weight", "I16", [128], 9921),
]
WEIGHTS_BYTES = 39940
N_MELS = 40
WINDOW_SIZE = 768
N_BINS = 513


def read_source(source: Path | None, rel: str) -> bytes:
    if source is not None:
        return (source / rel).read_bytes()
    url = UPSTREAM_RAW.format(tag=UPSTREAM_TAG, path=rel)
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_f32_list(text: str) -> list[float]:
    return [float(item) for item in text.replace("\n", " ").split(",") if item.strip()]


def parse_filters(text: str) -> tuple[list[int], list[int], list[float], list[float]]:
    mel_match = re.search(r"MEL_COEFFS:\s*\[\(usize, &\[f32\]\); 40\]\s*=\s*\[(.*?)\];", text, re.S)
    hann_match = re.search(r"HANN_WINDOW:\s*\[f32; 768\]\s*=\s*\[(.*?)\];", text, re.S)
    if mel_match is None or hann_match is None:
        raise ValueError("filters.rs does not contain the expected MEL_COEFFS/HANN_WINDOW tables")
    starts: list[int] = []
    lengths: list[int] = []
    coeffs: list[float] = []
    for start, values in re.findall(r"\(\s*(\d+)\s*,\s*&\[(.*?)\]\s*\)", mel_match.group(1), re.S):
        band = parse_f32_list(values)
        starts.append(int(start))
        lengths.append(len(band))
        coeffs.extend(band)
    hann = parse_f32_list(hann_match.group(1))
    if len(starts) != N_MELS:
        raise ValueError(f"expected {N_MELS} mel bands, found {len(starts)}")
    if len(hann) != WINDOW_SIZE:
        raise ValueError(f"expected {WINDOW_SIZE} Hann window values, found {len(hann)}")
    if any(start + length > N_BINS for start, length in zip(starts, lengths)):
        raise ValueError("mel band exceeds the FFT bin range")
    return starts, lengths, coeffs, hann


def pack(dtype: str, values) -> bytes:
    code = {"F32": "f", "I16": "h", "I32": "i"}[dtype]
    return struct.pack(f"<{len(values)}{code}", *values)


def build_tensors(weights: bytes, filters_text: str) -> list[tuple[str, str, list[int], bytes]]:
    if len(weights) != WEIGHTS_BYTES:
        raise ValueError(f"weights.bin has {len(weights)} bytes, expected {WEIGHTS_BYTES}")
    tensors = []
    for name, dtype, shape, offset in WEIGHT_LAYOUT:
        count = 1
        for dim in shape:
            count *= dim
        start = offset * 4
        size = count * (4 if dtype == "F32" else 2)
        tensors.append((name, dtype, shape, weights[start:start + size]))
    starts, lengths, coeffs, hann = parse_filters(filters_text)
    tensors.append(("hann_window", "F32", [WINDOW_SIZE], pack("F32", hann)))
    tensors.append(("mel_start", "I32", [N_MELS], pack("I32", starts)))
    tensors.append(("mel_length", "I32", [N_MELS], pack("I32", lengths)))
    tensors.append(("mel_coeffs", "F32", [len(coeffs)], pack("F32", coeffs)))
    return tensors


def write_safetensors(path: Path, tensors, metadata: dict[str, str]) -> None:
    header: dict[str, object] = {"__metadata__": metadata}
    blobs = []
    offset = 0
    for name, dtype, shape, data in sorted(tensors, key=lambda item: item[0]):
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for blob in blobs:
            handle.write(blob)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="Local earshot checkout (default: download the pinned tag)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = read_source(args.source, "src/weights.bin")
    digest = hashlib.sha256(weights).hexdigest()
    if digest != WEIGHTS_SHA256:
        print(f"warning: weights.bin sha256 {digest} differs from the pinned {UPSTREAM_TAG} weights", file=sys.stderr)
    filters_text = read_source(args.source, "src/filters.rs").decode("utf-8")
    tensors = build_tensors(weights, filters_text)
    write_safetensors(
        args.output,
        tensors,
        {
            "format": "pt",
            "source": "https://github.com/pykeio/earshot",
            "source_version": UPSTREAM_TAG,
            "source_weights_sha256": digest,
            "license": "MIT OR Apache-2.0",
        },
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {len(tensors)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
