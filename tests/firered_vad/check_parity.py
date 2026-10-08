#!/usr/bin/env python3
"""Check exported native FireRed VAD outputs against the official Python fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--probability-tolerance", type=float, default=0.002)
    parser.add_argument("--timestamp-tolerance", type=float, default=0)
    args = parser.parse_args()
    metrics = {}
    for suffix in (".features.f32", ".probs.f32", ".reference_features_probs.f32", ".chunked_probs.f32"):
        native_path = Path(args.native + suffix)
        if suffix == ".chunked_probs.f32" and not native_path.exists():
            continue
        reference_suffix = ".probs.f32" if suffix == ".reference_features_probs.f32" else suffix
        reference = np.fromfile(args.reference + reference_suffix, dtype="<f4")
        native = np.fromfile(native_path, dtype="<f4")
        if native.shape != reference.shape or not np.isfinite(native).all():
            raise AssertionError(f"Invalid output shape or non-finite values: {suffix}")
        maximum = float(np.max(np.abs(native - reference), initial=0))
        metrics[suffix] = {"max_abs": maximum, "rmse": float(np.sqrt(np.mean((native - reference) ** 2))) if native.size else 0}
        # The framework and kaldi-native-fbank use different FFT/filterbank
        # rounding. Bound both the worst bin and the overall normalized error.
        tolerance = 0.001 if suffix == ".features.f32" else args.probability_tolerance
        if maximum > tolerance:
            raise AssertionError(f"{suffix}: max_abs={maximum} exceeds {tolerance}")
        if suffix == ".features.f32" and metrics[suffix]["rmse"] > 1e-5:
            raise AssertionError(f"Frontend RMSE exceeds 1e-5: {metrics[suffix]}")
    reference = np.asarray(json.loads(Path(args.reference + ".segments.json").read_text()))
    native = np.asarray(json.loads(Path(args.native + ".segments.json").read_text()))
    if native.shape != reference.shape or not np.allclose(native, reference, rtol=0, atol=args.timestamp_tolerance):
        raise AssertionError(f"Timestamp mismatch: reference={reference.tolist()}, native={native.tolist()}")
    metrics["timestamps"] = "passed"
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
