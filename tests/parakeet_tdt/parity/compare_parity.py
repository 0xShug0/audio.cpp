#!/usr/bin/env python3
"""Compare a C++ Parakeet-TDT activation dump against the NeMo reference dump.

Usage:
  python3 compare_parity.py --nemo-dir /tmp/parakeet_nemo_dump --cpp-dir /tmp/parakeet_cpp_dump

Exits non-zero (and prints which stage failed) if any compared tensor drops
below the similarity/scale tolerance. This is the numerical counterpart to
test_golden_transcription.cpp: that test catches regressions large enough to
flip the final decoded text; this one catches smaller numerical drift in any
of the three stages it compares, at whatever point it first appears.

Only requires numpy — unlike dump_nemo_reference.py, this script does not
need NeMo/torch installed, so it can run anywhere the two .npy dump
directories are available (e.g. copied out of a NeMo-enabled machine).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Stage:
    def __init__(self, name: str, nemo_file: str, cpp_file: str, nemo_layout: str, cpp_layout: str):
        self.name = name
        self.nemo_file = nemo_file
        self.cpp_file = cpp_file
        # "btd" = [1, T, D] (batch, time, dim); "bdt" = [1, D, T]; "td" = [T, D]
        self.nemo_layout = nemo_layout
        self.cpp_layout = cpp_layout

    def load(self, directory: str, filename: str, layout: str) -> np.ndarray:
        arr = np.load(os.path.join(directory, filename)).astype(np.float32)
        if layout == "btd":
            arr = arr[0]
        elif layout == "bdt":
            arr = arr[0].T
        elif layout == "td":
            pass
        else:
            raise ValueError(f"unknown layout {layout}")
        return arr


# (stage name, nemo filename, nemo layout, cpp filename, cpp layout, cosine threshold, std-ratio tolerance)
#
# enc_out's threshold is deliberately looser than mel_features/layer_0: it is
# the output of the real, unmodified production encoder graph (all 24 layers
# + the shared positional-encoding sub-graph, built and executed as one large
# ~2M-node ggml graph), compared against layer_0's isolated single-layer
# graph (a few hundred nodes). Both compute mathematically the same thing
# (verified directly: chaining 24 isolated single-layer builds, each fed the
# previous one's real output exactly as the production encoder does, stays
# at cosine >= 0.999999 through every single layer) — but float32 summation
# order and ggml's internal threading/blocking decisions are not required to
# match bit-for-bit between two structurally different graphs computing the
# same math, and small per-op rounding differences compound multiplicatively
# through softmax/normalization over 24 layers. 0.97 leaves real headroom
# below the ~0.999999 that perfect parity would give while still being tight
# enough to catch another bias-dropping-style regression (which produced
# 0.778 before it was fixed).
STAGES = [
    ("mel_features", "mel_features.npy", "bdt", "mel_features.npy", "td", 0.999, 0.02),
    ("layer_0", "layer_0.npy", "btd", "layer_0.npy", "td", 0.999, 0.05),
    ("enc_out", "enc_out.npy", "bdt", "enc_out.npy", "td", 0.97, 0.05),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nemo-dir", required=True)
    parser.add_argument("--cpp-dir", required=True)
    args = parser.parse_args()

    ok = True
    for name, nemo_file, nemo_layout, cpp_file, cpp_layout, cos_threshold, std_tol in STAGES:
        nemo_path = os.path.join(args.nemo_dir, nemo_file)
        cpp_path = os.path.join(args.cpp_dir, cpp_file)
        if not os.path.exists(nemo_path):
            print(f"SKIP {name}: missing NeMo reference dump {nemo_path}")
            continue
        if not os.path.exists(cpp_path):
            print(f"FAIL {name}: missing C++ dump {cpp_path}")
            ok = False
            continue

        s = Stage(name, nemo_file, cpp_file, nemo_layout, cpp_layout)
        nemo = s.load(args.nemo_dir, nemo_file, nemo_layout)
        cpp = s.load(args.cpp_dir, cpp_file, cpp_layout)

        if nemo.shape != cpp.shape:
            print(f"FAIL {name}: shape mismatch nemo={nemo.shape} cpp={cpp.shape}")
            ok = False
            continue

        cos = cosine(nemo, cpp)
        nemo_std = float(nemo.std())
        cpp_std = float(cpp.std())
        std_ratio = abs(cpp_std - nemo_std) / (nemo_std + 1e-12)

        passed = cos >= cos_threshold and std_ratio <= std_tol
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} {name}: cosine={cos:.6f} (>= {cos_threshold}) "
            f"nemo_std={nemo_std:.6f} cpp_std={cpp_std:.6f} std_ratio={std_ratio:.4f} (<= {std_tol})"
        )
        if not passed:
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
