#!/usr/bin/env python3
"""Convert a VibeASR.cpp VAE encoder GGUF into an audio.cpp GGUF package.

Upstream: https://github.com/microsoft/VibeASR.cpp

VibeASR.cpp ships its VAE encoder already quantized to its own ggml fork's
GGML_TYPE_I8_S, so there is nothing to re-quantize here. The only thing that
differs is the numeric type id: the VibeASR fork picked 36 (I2_S) and
37 (I8_S), which upstream ggml had already used for the retired
IQ4_NL_4_4 / IQ4_NL_4_8 slots. audio.cpp therefore registers the same two
types at 42 (I8_S) and 43 (I2_S).

The on-disk layout is identical either way -- an I8_S tensor is
`nelements` int8 bytes followed by a single padded F32 tensor scale, and
ggml's GGUF writer sizes every tensor with ggml_nbytes() -- so this tool
rewrites the 4-byte type field of each tensor info and copies everything else
through byte for byte. Data offsets, the data section, and the KV block are
untouched.

Examples:
  # inspect a VibeASR GGUF without writing anything
  python3 tools/community_models/convert_vibeasr_vae.py \
      --input vibeasr-vae-encoder-i8_s.gguf --list

  # produce the audio.cpp package
  python3 tools/community_models/convert_vibeasr_vae.py \
      --input vibeasr-vae-encoder-i8_s.gguf \
      --output models/vibeasr/vae_encoder-i8_s.gguf

  # confirm an already converted package needs no further remapping
  python3 tools/community_models/convert_vibeasr_vae.py \
      --input models/vibeasr/vae_encoder-i8_s.gguf --check
"""
import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# VibeASR.cpp fork id -> audio.cpp id. See external/ggml/include/ggml.h for why
# audio.cpp cannot reuse 36/37.
TYPE_REMAP = {36: 43, 37: 42}

TYPE_NAMES = {0: "f32", 1: "f16", 8: "q8_0", 42: "i8_s", 43: "i2_s"}

# GGUF metadata value type ids.
(
    KV_UINT8,
    KV_INT8,
    KV_UINT16,
    KV_INT16,
    KV_UINT32,
    KV_INT32,
    KV_FLOAT32,
    KV_BOOL,
    KV_STRING,
    KV_ARRAY,
    KV_UINT64,
    KV_INT64,
    KV_FLOAT64,
) = range(13)

KV_FIXED_SIZE = {
    KV_UINT8: 1,
    KV_INT8: 1,
    KV_UINT16: 2,
    KV_INT16: 2,
    KV_UINT32: 4,
    KV_INT32: 4,
    KV_FLOAT32: 4,
    KV_BOOL: 1,
    KV_UINT64: 8,
    KV_INT64: 8,
    KV_FLOAT64: 8,
}


class Reader:
    """Minimal forward-only GGUF header reader that tracks field offsets."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("GGUF header is truncated")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", errors="replace")

    def skip_kv_value(self, kv_type: int) -> None:
        if kv_type in KV_FIXED_SIZE:
            self.take(KV_FIXED_SIZE[kv_type])
        elif kv_type == KV_STRING:
            self.string()
        elif kv_type == KV_ARRAY:
            item_type = self.u32()
            count = self.u64()
            if item_type in KV_FIXED_SIZE:
                self.take(KV_FIXED_SIZE[item_type] * count)
            elif item_type == KV_STRING:
                for _ in range(count):
                    self.string()
            else:
                raise ValueError(f"unsupported GGUF array element type {item_type}")
        else:
            raise ValueError(f"unsupported GGUF metadata type {kv_type}")


def parse_tensor_infos(data: bytes):
    """Return (tensor_infos, alignment). Each info records where its type field lives."""
    reader = Reader(data)
    if reader.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = reader.u32()
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = reader.u64()
    n_kv = reader.u64()

    alignment = 32
    for _ in range(n_kv):
        key = reader.string()
        kv_type = reader.u32()
        if key == "general.alignment" and kv_type == KV_UINT32:
            alignment = reader.u32()
        else:
            reader.skip_kv_value(kv_type)

    infos = []
    for _ in range(n_tensors):
        name = reader.string()
        n_dims = reader.u32()
        dims = [reader.u64() for _ in range(n_dims)]
        type_offset = reader.pos
        type_id = reader.u32()
        data_offset = reader.u64()
        infos.append(
            {
                "name": name,
                "dims": dims,
                "type": type_id,
                "type_offset": type_offset,
                "data_offset": data_offset,
            }
        )
    return infos, alignment


def type_name(type_id: int) -> str:
    return TYPE_NAMES.get(type_id, f"type#{type_id}")


def list_tensors(infos) -> None:
    histogram = {}
    for info in infos:
        histogram[info["type"]] = histogram.get(info["type"], 0) + 1
        print(f"  {info['name']:<64} {type_name(info['type']):>6}  {info['dims']}")
    print(f"{len(infos)} tensors")
    for type_id in sorted(histogram):
        print(f"  {type_name(type_id):>6}: {histogram[type_id]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="VibeASR.cpp VAE encoder GGUF")
    parser.add_argument("--output", type=Path, help="audio.cpp GGUF to write")
    parser.add_argument("--list", action="store_true", help="print the tensor table and exit")
    parser.add_argument("--check", action="store_true", help="exit non-zero if any tensor still needs remapping")
    args = parser.parse_args()

    data = bytearray(args.input.read_bytes())
    infos, alignment = parse_tensor_infos(bytes(data))

    if args.list:
        list_tensors(infos)
        return 0

    stale = [info for info in infos if info["type"] in TYPE_REMAP]
    if args.check:
        if stale:
            print(f"{args.input}: {len(stale)} tensors still use VibeASR fork type ids", file=sys.stderr)
            return 1
        print(f"{args.input}: type ids are already audio.cpp native")
        return 0

    if args.output is None:
        parser.error("--output is required unless --list or --check is given")

    # Guard against a double conversion: the fork ids and the audio.cpp ids are
    # both valid ggml types, so a second pass would silently corrupt nothing but
    # would also hide a mistake in the source package.
    already = [info for info in infos if info["type"] in set(TYPE_REMAP.values())]
    if already and stale:
        raise SystemExit("input mixes VibeASR fork type ids with audio.cpp ids")
    if not stale:
        print(f"{args.input}: nothing to remap, copying through")

    for info in stale:
        struct.pack_into("<I", data, info["type_offset"], TYPE_REMAP[info["type"]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(data))

    remapped, _ = parse_tensor_infos(bytes(data))
    if any(info["type"] in TYPE_REMAP for info in remapped):
        raise SystemExit("remap did not take effect")
    if [info["data_offset"] for info in remapped] != [info["data_offset"] for info in infos]:
        raise SystemExit("remap perturbed tensor data offsets")

    counts = {}
    for info in remapped:
        counts[info["type"]] = counts.get(info["type"], 0) + 1
    summary = ", ".join(f"{type_name(t)}={counts[t]}" for t in sorted(counts))
    print(f"wrote {args.output} ({len(data)} bytes, alignment {alignment}, {summary})")
    print(f"remapped {len(stale)} tensor type ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
