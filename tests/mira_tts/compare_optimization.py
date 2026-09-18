#!/usr/bin/env python3
"""Require exact before/after MiraTTS output parity and report request timings.

Run from the repository root so relative audio_out paths in benchmark summaries
resolve in the same way as they did for mira_tts_warm_bench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = {}
    for item in payload["results"]:
        key = (item["name"], item["iteration"], item["mode"])
        if key in results:
            raise ValueError(f"duplicate request in {path}: {key}")
        results[key] = item
    if not results:
        raise ValueError(f"empty results: {path}")
    return payload, results


def compare(before_path: Path, after_path: Path) -> list[str]:
    before_meta, before = load(before_path)
    after_meta, after = load(after_path)
    for field in ("family", "backend", "model", "mode"):
        if before_meta[field] != after_meta[field]:
            raise ValueError(f"different {field}: cannot compare like-for-like runs")
    if before.keys() != after.keys():
        raise ValueError("before/after request sets differ")
    rows = []
    for key, left in before.items():
        right = after[key]
        for field in ("audio_hash", "samples", "sample_rate", "channels",
                      "audio_seconds", "event_count"):
            if left[field] != right[field]:
                raise ValueError(f"{key}: {field} changed")
        if left["samples"] <= 0:
            raise ValueError(f"{key}: empty audio")
        hashes = [hashlib.sha256(Path(item["audio_out"]).read_bytes()).digest()
                  for item in (left, right)]
        if hashes[0] != hashes[1]:
            raise ValueError(f"{key}: WAV bytes differ")
        old, new = float(left["wall_ms"]), float(right["wall_ms"])
        if not all(math.isfinite(value) and value > 0 for value in (old, new)):
            raise ValueError(f"{key}: invalid timing")
        rows.append(f"| {key[0]} ({key[1]}) | {old / 1000:.3f} | "
                    f"{new / 1000:.3f} | {(1 - new / old) * 100:.1f}% | identical |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    args = parser.parse_args()
    # Validate every result before printing a success table.
    rows = compare(args.before, args.after)
    print("| Request | Before (s) | After (s) | Time reduction | WAV |")
    print("|---|---:|---:|---:|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
