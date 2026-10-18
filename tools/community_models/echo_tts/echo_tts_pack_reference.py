#!/usr/bin/env python3
"""Pack an echo_ref.npz reference dump into a flat binary the C++ parity
harness can read without an npz parser.

    python3 echo_tts_pack_reference.py echo_ref.npz -o echo_ref.bin

Only the tensors the harness actually consumes are packed; the per-block
activations are 24 x 640 x 2048 and are included only with --blocks, which
takes the archive from a few MB to a few hundred.

Format, all little-endian, which is the only byte order audio.cpp targets:

    magic   8 bytes  "ECHOPAR1"
    count   int32    number of entries
    entry   int32    name length
            bytes    name, not NUL-terminated
            int32    dtype, 0 = float32, 1 = int32
            int64    element count
            data     element count * 4 bytes

Entries appear in the order written here; the reader looks them up by name, so
order is not load-bearing.
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

MAGIC = b"ECHOPAR1"
DTYPE_F32 = 0
DTYPE_I32 = 1

# The minimum needed to drive the DiT at a fixed timestep and score the result.
REQUIRED = [
    "dit.x_input",
    "dit.v_pred",
    "dit.t",
    "text.input_ids",
    "text.mask",
    "speaker.latent",
    "speaker.mask",
    "sampler.latent",
    "sampler.initial_noise",
    "config.sequence_length",
    "config.steps",
    "config.seed",
]


def pack_entry(name: str, array: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.dtype in (np.int32, np.int64):
        dtype, payload = DTYPE_I32, flat.astype("<i4")
    else:
        dtype, payload = DTYPE_F32, flat.astype("<f4")
    raw = name.encode("utf-8")
    return (
        struct.pack("<i", len(raw))
        + raw
        + struct.pack("<i", dtype)
        + struct.pack("<q", payload.size)
        + payload.tobytes()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", help="echo_ref.npz from echo_tts_reference.py")
    parser.add_argument("-o", "--output", default="echo_ref.bin")
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="also pack the 24 per-block DiT activations (large)",
    )
    args = parser.parse_args()

    data = np.load(args.npz)
    names = list(REQUIRED)
    if args.blocks:
        names += [f"dit.block.{i}" for i in range(24) if f"dit.block.{i}" in data.files]

    missing = [n for n in names if n not in data.files]
    if missing:
        print(f"missing from {args.npz}: {', '.join(missing)}", file=sys.stderr)
        return 1

    chunks = [pack_entry(n, data[n]) for n in names]
    with open(args.output, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<i", len(chunks)))
        for chunk in chunks:
            handle.write(chunk)

    total = sum(len(c) for c in chunks)
    print(f"wrote {args.output}: {len(chunks)} tensors, {total / 1e6:.1f} MB")
    for name in names:
        print(f"  {name:28s} {data[name].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
