#!/usr/bin/env python3
"""Compare complete production frontend tensors captured by probe.cpp."""

import argparse
from pathlib import Path
import numpy as np


def compare(before: Path, after: Path, max_abs: float) -> None:
    old_meta = before.with_suffix(before.suffix + ".meta").read_text().strip()
    new_meta = after.with_suffix(after.suffix + ".meta").read_text().strip()
    if old_meta != new_meta:
        raise SystemExit(f"FAIL metadata: {old_meta!r} != {new_meta!r}")
    old = np.fromfile(before.with_suffix(before.suffix + ".f32"), dtype="<f4")
    new = np.fromfile(after.with_suffix(after.suffix + ".f32"), dtype="<f4")
    if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise SystemExit("FAIL tensor shape or finite values")
    delta = np.abs(old - new)
    worst = float(delta.max(initial=0))
    mean = float(delta.mean())
    print(f"{before.name}: {old_meta}; values={old.size}; max_abs={worst:.9g}; mean_abs={mean:.9g}; mismatches={np.count_nonzero(old != new)}")
    if worst > max_abs:
        raise SystemExit(f"FAIL max_abs {worst:.9g} > {max_abs:.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--max-abs", type=float, default=0.0)
    args = parser.parse_args()
    compare(args.before, args.after, args.max_abs)
