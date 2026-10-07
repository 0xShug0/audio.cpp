"""Extract Whistle's mel filterbank, permutations, and custom tokenizer blob from .cact."""

import argparse
import hashlib
import struct
from pathlib import Path

HEADER = struct.Struct("<48If")
RECORD = struct.Struct("<BBHIIIIQQII")
ASSETS = {
    226: ("hadamard_perm1.f32", 2, (512,), 2048,
          "b075b0817b651b2890601816fb1bd99609b815abdeeb4263f32c0456d082eabd"),
    227: ("hadamard_perm2.f32", 2, (512,), 2048,
          "c4b5af55b06cf07dee8b8acbfd7eef2e749ffa5dc4e27f12864435a7533b2ed9"),
    679: ("mel_filterbank_80.f32", 2, (257, 80), 82240,
          "9d4bff0db28af35ccdacf7fedf9ef0ae8dfb95e8e812484015ba78151edadfcb"),
    680: ("tokenizer.blob", 4, (), 109231,
          "3e606cf88ea3a3446c615459ae3dbc1ba241e32f6bd72f85796dfaabcbc58bf0"),
}


def extract(source: Path, destination: Path) -> None:
    with source.open("rb") as stream:
        header = HEADER.unpack(stream.read(HEADER.size))
        if header[0] != 0x05E12A84 or header[1] != 681:
            raise ValueError("Unexpected Whistle container header")
        stream.seek(HEADER.size + header[2] * 4)
        records = [RECORD.unpack(stream.read(RECORD.size)) for _ in range(header[1])]
        for index, (name, dtype, shape, size, digest) in ASSETS.items():
            record = records[index]
            if (record[0] != dtype or record[3:3 + record[1]] != shape or
                    record[8] != size or record[7] + size > source.stat().st_size):
                raise ValueError(f"Unexpected Whistle asset record {index}")
            stream.seek(record[7])
            data = stream.read(size)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Whistle asset record {index} has an invalid checksum")
            if index in (226, 227) and sorted(struct.unpack("<512f", data)) != list(range(512)):
                raise ValueError(f"Whistle asset record {index} is not a permutation")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    extract(args.source, args.destination)
