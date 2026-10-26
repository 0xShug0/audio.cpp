#!/usr/bin/env python3
"""Compare the audio.cpp Earshot VAD port against upstream earshot.

Builds nothing; pass the binaries:

    cargo build --release --manifest-path tests/earshot_vad/reference_rs/Cargo.toml
    python tests/earshot_vad/check_parity.py \
        --probe build/<preset>/bin/earshot_vad_probe \
        --reference tests/earshot_vad/reference_rs/target/release/earshot_reference \
        [--cli build/<preset>/bin/audiocpp_cli] [--bench 20] wav [wav ...]

Checks, per 16 kHz mono wav:
  * per-frame scores: max |audio.cpp - upstream| <= --score-tol (default 1e-3)
  * segments with keep_trailing_segment=false equal upstream segments.rs after
    compensating its frame-counter lag (segment i is reported i frames early
    upstream, because the frame that returns a segment is not counted)
  * the batched offline scorer equals the frame-by-frame streaming detector
  * with --cli: `--task vad` offline and streaming segments equal the probe
Exit status is non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FRAME = 256
REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "assets" / "framework" / "models" / "earshot_vad"


def run_json(cmd: list[str]) -> dict:
    return json.loads(subprocess.check_output(cmd, text=True))


def cli_segments(cli: Path, wav: Path, mode: str) -> list[list[int]]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "segments.json"
        subprocess.run(
            [str(cli), "--task", "vad", "--model", str(MODEL), "--backend", "cpu", "--mode", mode,
             "--audio", str(wav), "--segments-out", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not out.exists():
            return []  # the CLI writes --segments-out only when it found speech
        return [[s["start_sample"], s["end_sample"]] for s in json.loads(out.read_text())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--score-tol", type=float, default=1e-3)
    parser.add_argument("--bench", type=int, default=0, help="repeat count for speed comparison")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("wavs", nargs="+", type=Path)
    args = parser.parse_args()

    failures = 0
    print("| wav | seconds | frames | max score diff | segments | upstream segments | match |"
          + (" upstream ms | offline ms | streaming ms |" if args.bench else ""))
    print("|---|---:|---:|---:|---:|---:|---|" + ("---:|---:|---:|" if args.bench else ""))
    for wav in args.wavs:
        bench = ["--bench", str(args.bench)] if args.bench else []
        ref = run_json([str(args.reference), str(wav), *bench])
        ours = run_json([str(args.probe), str(MODEL), str(wav), "--upstream-segments",
                         "--threads", str(args.threads), *bench])
        problems = []
        if len(ref["scores"]) != len(ours["scores"]):
            problems.append("frame count")
        diff = max((abs(a - b) for a, b in zip(ref["scores"], ours["scores"])), default=0.0)
        if diff > args.score_tol:
            problems.append(f"score diff {diff:.2e}")
        if ours["max_stream_diff"] > 1e-6:
            problems.append("stream != offline")
        expected = [[s + i * FRAME, e + i * FRAME] for i, (s, e) in enumerate(ref["segments"])]
        if ours["segments"] != expected:
            problems.append("segments")
        if args.cli is not None:
            full = run_json([str(args.probe), str(MODEL), str(wav)])["segments"]
            # The CLI keeps a segment that is still open at the end and extends it to the last sample.
            if full and full[-1][1] == len(ours["scores"]) * FRAME:
                full[-1][1] = ours["num_samples"]
            for mode in ("offline", "streaming"):
                if cli_segments(args.cli, wav, mode) != full:
                    problems.append(f"cli {mode}")
        failures += bool(problems)
        seconds = ours["num_samples"] / 16000
        row = (f"| {wav.name} | {seconds:.1f} | {len(ours['scores'])} | {diff:.2e} | {len(ours['segments'])} | "
               f"{len(ref['segments'])} | {'yes' if not problems else ', '.join(problems)} |")
        if args.bench:
            row += f" {ref['bench_ms']:.2f} | {ours['offline_ms']:.2f} | {ours['streaming_ms']:.2f} |"
        print(row)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
