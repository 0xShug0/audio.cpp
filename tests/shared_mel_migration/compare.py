#!/usr/bin/env python3
"""Compare complete frontend feature captures from before and after migration."""

import argparse
from pathlib import Path
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--allow-extra-after", action="store_true")
    args = parser.parse_args()

    names = {path.stem for path in args.before.glob("*.f32")}
    after_names = {path.stem for path in args.after.glob("*.f32")}
    if not names or (not names.issubset(after_names) if args.allow_extra_after else names != after_names):
        raise SystemExit("capture cases differ or are empty")
    total = 0
    for name in sorted(names):
        before_meta = (args.before / f"{name}.meta").read_bytes()
        after_meta = (args.after / f"{name}.meta").read_bytes()
        if before_meta != after_meta:
            raise SystemExit(f"{name}: feature shape changed")
        before = (args.before / f"{name}.f32").read_bytes()
        after = (args.after / f"{name}.f32").read_bytes()
        if len(before) != len(after) or len(before) % 4:
            raise SystemExit(f"{name}: feature byte count changed")
        if before != after:
            first = next(i for i in range(0, len(before), 4) if before[i:i + 4] != after[i:i + 4])
            old = struct.unpack_from("<f", before, first)[0]
            new = struct.unpack_from("<f", after, first)[0]
            raise SystemExit(f"{name}: first mismatch at F32 index {first // 4}: {old} != {new}")
        count = len(before) // 4
        total += count
        print(f"{name}: {count}/{count} F32 values identical")
    print(f"total: {total}/{total} F32 values identical")


if __name__ == "__main__":
    main()
