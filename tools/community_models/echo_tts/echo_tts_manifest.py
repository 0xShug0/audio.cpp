#!/usr/bin/env python3
"""Dump the tensor manifest (names, shapes, dtypes) of the Echo-TTS checkpoints.

Safetensors stores a JSON header at the front of the file: an 8-byte
little-endian length, then that many bytes of JSON. So the full tensor listing
can be read with a couple of HTTP range requests -- no weights are downloaded.

Usage:

    # remote, no download (default: all three Echo-TTS checkpoints)
    python3 echo_tts_manifest.py -o echo_manifest.json

    # a checkpoint you already have on disk
    python3 echo_tts_manifest.py --local /path/to/pytorch_model.safetensors -o out.json

Set HF_TOKEN in the environment if a repo needs auth. Requires only the standard
library.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

DEFAULT_TARGETS: List[Tuple[str, str, str]] = [
    ("echo_dit", "jordand/echo-tts-base", "pytorch_model.safetensors"),
    ("echo_pca", "jordand/echo-tts-base", "pca_state.safetensors"),
    ("fish_ae", "jordand/fish-s1-dac-min", "pytorch_model.safetensors"),
]

# A safetensors header is JSON; 64 MiB is far more than any real one needs and
# still bounds a malformed-length read.
MAX_HEADER_BYTES = 64 * 1024 * 1024


def _request(url: str, byte_range: Tuple[int, int]) -> bytes:
    start, end = byte_range
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "echo-tts-manifest/1.0",
    }
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.status
        data = response.read(end - start + 1)
    if status != 206:
        raise RuntimeError(
            f"server ignored the range request (HTTP {status}); refusing to "
            f"download the whole file"
        )
    return data


def read_remote_header(repo_id: str, filename: str, revision: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    prefix = _request(url, (0, 7))
    if len(prefix) != 8:
        raise RuntimeError(f"short read on the length prefix of {filename}")
    (header_length,) = struct.unpack("<Q", prefix)
    if not 0 < header_length <= MAX_HEADER_BYTES:
        raise RuntimeError(f"implausible header length {header_length} in {filename}")
    raw = _request(url, (8, 8 + header_length - 1))
    return json.loads(raw.decode("utf-8"))


def read_local_header(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"short read on the length prefix of {path}")
        (header_length,) = struct.unpack("<Q", prefix)
        if not 0 < header_length <= MAX_HEADER_BYTES:
            raise RuntimeError(f"implausible header length {header_length} in {path}")
        raw = handle.read(header_length)
    return json.loads(raw.decode("utf-8"))


def normalize(header: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a safetensors header to name -> {dtype, shape, nbytes}."""
    tensors: Dict[str, Any] = {}
    total_bytes = 0
    total_params = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        offsets = entry.get("data_offsets", [0, 0])
        nbytes = int(offsets[1]) - int(offsets[0])
        shape = [int(dim) for dim in entry.get("shape", [])]
        count = 1
        for dim in shape:
            count *= dim
        tensors[name] = {
            "dtype": entry.get("dtype"),
            "shape": shape,
            "nbytes": nbytes,
        }
        total_bytes += nbytes
        total_params += count
    return {
        "metadata": header.get("__metadata__", {}),
        "tensor_count": len(tensors),
        "total_parameters": total_params,
        "total_bytes": total_bytes,
        "tensors": tensors,
    }


def group_key(name: str) -> str:
    """Collapse numeric path segments so repeated blocks fold into one entry."""
    parts = []
    for part in name.split("."):
        parts.append("{N}" if part.isdigit() else part)
    return ".".join(parts)


def summarize(label: str, manifest: Dict[str, Any]) -> str:
    groups: Dict[str, Dict[str, Any]] = {}
    for name, info in manifest["tensors"].items():
        key = group_key(name)
        bucket = groups.setdefault(
            key, {"count": 0, "shape": info["shape"], "dtype": info["dtype"]}
        )
        bucket["count"] += 1

    lines = [
        f"=== {label} ===",
        f"  tensors={manifest['tensor_count']} "
        f"params={manifest['total_parameters']:,} "
        f"bytes={manifest['total_bytes']:,}",
    ]
    if manifest["metadata"]:
        lines.append(f"  metadata={manifest['metadata']}")
    for key in sorted(groups):
        bucket = groups[key]
        suffix = f" x{bucket['count']}" if bucket["count"] > 1 else ""
        lines.append(f"  {key:<62} {bucket['dtype']:<8} {bucket['shape']}{suffix}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="PATH",
        help="read a local .safetensors file instead of querying the Hub "
        "(repeatable)",
    )
    parser.add_argument(
        "--revision", default="main", help="git revision to resolve (default: main)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="echo_manifest.json",
        help="where to write the full JSON manifest",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {}
    failures: List[str] = []

    if args.local:
        for path in args.local:
            label = os.path.basename(path)
            try:
                results[label] = normalize(read_local_header(path))
            except Exception as error:  # noqa: BLE001 - report and continue
                failures.append(f"{label}: {error}")
    else:
        for label, repo_id, filename in DEFAULT_TARGETS:
            try:
                header = read_remote_header(repo_id, filename, args.revision)
                manifest = normalize(header)
                manifest["source"] = f"{repo_id}/{filename}@{args.revision}"
                results[label] = manifest
            except Exception as error:  # noqa: BLE001 - report and continue
                failures.append(f"{label} ({repo_id}/{filename}): {error}")

    for label in sorted(results):
        print(summarize(label, results[label]))
        print()

    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)

    if results:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1, sort_keys=True)
        print(f"wrote {args.output}")

    return 1 if failures and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
