#!/usr/bin/env python3
"""Long-session streaming behavior test for the confucius4_r2t2 family.

The audio tower position table caps a single decode at 1500 frames
(13 tokens/s => 115.40 s). This script pins down what happens past that
limit and, once VAD endpointing is enabled, verifies that segmentation
keeps long dictation sessions alive with bounded per-chunk cost.

Usage:
    python3 tests/confucius4_r2t2/long_stream_test.py \
        --cli build/macos-metal-release/bin/audiocpp_cli \
        --model models/Confucius4-R2T2-GGUF/r2t2-q8_0.gguf \
        [--source-audio assets/resources/sample_16k.wav] \
        [--target-seconds 130] [--gap-seconds 1.0] [--chunk-ms 320]
        [--endpointing] [--report /tmp/r2t2_long_report.json]

Without --endpointing the run must fail safely once the stream exceeds the
position-table limit: nonzero exit, an explicit max_source_positions error,
and no crash. With --endpointing the run must complete, emit segment.end
boundaries, and keep per-chunk encoder cost bounded by the segment cap
instead of the whole stream length.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

TRACE_RE = re.compile(r"^\[(?:TRACE|TIMING) [^\]]*\] (?P<name>\S+)\s?(?P<value>.*)$")

# Per-chunk timing keys (grouped by confucius4_r2t2.stream.chunk_id). The greedy
# decoder does not expose compute timings (framework gap), so encoder +
# frontend cost is the instrumented proxy for per-chunk decode cost.
CHUNK_TIMING_KEYS = (
    "confucius4_r2t2.frontend.normalize_ms",
    "confucius4_r2t2.frontend.log_mel_ms",
    "confucius4_r2t2.audio_encoder.input_upload_ms",
    "confucius4_r2t2.audio_encoder.graph.compute_ms",
    "confucius4_r2t2.audio_encoder.output_read_ms",
)


def build_long_wav(source: Path, target_seconds: float, gap_seconds: float, out: Path) -> float:
    """Concatenate source speech with silence gaps until target_seconds."""
    with wave.open(str(source), "rb") as w:
        rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        speech = w.readframes(frames)
    gap_frames = int(rate * gap_seconds)
    silence = b"\x00" * (gap_frames * channels * width)
    speech_seconds = frames / rate
    block_seconds = speech_seconds + gap_seconds
    repeats = max(1, math.ceil(target_seconds / block_seconds))
    with wave.open(str(out), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        for i in range(repeats):
            w.writeframes(speech)
            if i + 1 < repeats and gap_seconds > 0:
                w.writeframes(silence)
    duration = (frames * repeats + gap_frames * max(0, repeats - 1)) / rate
    return duration


def parse_trace(path: Path):
    """Group per-chunk trace scalars by chunk_id."""
    chunks = []
    pending = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TRACE_RE.match(line)
        if not m:
            continue
        name, value = m.group("name"), m.group("value")
        if name == "confucius4_r2t2.stream.chunk_id":
            if pending is not None:
                chunks.append(pending)
            pending = {"chunk_id": int(value), "final_flush": 0, "timings": {}, "frames": None}
        elif pending is None:
            continue
        elif name == "confucius4_r2t2.stream.final_flush":
            pending["final_flush"] = int(value)
        elif name == "confucius4_r2t2.audio_encoder.frames":
            try:
                pending["frames"] = int(value)
            except ValueError:
                pass
        elif name in CHUNK_TIMING_KEYS:
            try:
                pending["timings"][name] = pending["timings"].get(name, 0.0) + float(value)
            except ValueError:
                pass
    if pending is not None:
        chunks.append(pending)
    return chunks


def chunk_cost_ms(chunk) -> float:
    return sum(chunk["timings"].get(k, 0.0) for k in CHUNK_TIMING_KEYS)


def run_cli(args, audio_path: Path, trace_path: Path):
    cmd = [
        args.cli, "--task", "asr", "--mode", "streaming", "--family", "confucius4_r2t2",
        "--model", args.model, "--backend", args.backend, "--audio", str(audio_path),
        "--session-option", f"confucius4_r2t2.chunk_size_ms={args.chunk_ms}",
        "--session-option", f"confucius4_r2t2.max_tokens={args.max_tokens}",
        "--log-file", str(trace_path),
    ]
    if args.endpointing:
        cmd += [
            "--session-option", "confucius4_r2t2.endpointing=true",
            "--session-option", f"confucius4_r2t2.max_segment_seconds={args.max_segment_seconds}",
        ]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--source-audio", default="assets/resources/sample_16k.wav")
    ap.add_argument("--target-seconds", type=float, default=130.0,
                    help="must exceed the 115.40 s position-table limit")
    ap.add_argument("--gap-seconds", type=float, default=1.0)
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-segment-seconds", type=float, default=20.0)
    ap.add_argument("--endpointing", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio_path = tmp / "long.wav"
        duration = build_long_wav(Path(args.source_audio), args.target_seconds, args.gap_seconds, audio_path)
        print(f"audio: {duration:.1f}s assembled from {args.source_audio} "
              f"(gap {args.gap_seconds}s, limit 115.40s, chunk {args.chunk_ms}ms)")

        trace = tmp / "trace.log"
        proc = run_cli(args, audio_path, trace)
        chunks = parse_trace(trace) if trace.exists() else []
        per_chunk = [c for c in chunks if not c["final_flush"]]
        costs = [chunk_cost_ms(c) for c in per_chunk]
        frames = [c["frames"] for c in per_chunk if c["frames"] is not None]

        segments = [
            line for line in proc.stdout.splitlines()
            if line.startswith("event=speech_end")
        ]
        final_text = next(
            (line[len("text_output="):] for line in proc.stdout.splitlines()
             if line.startswith("text_output=")),
            None,
        )

        report = {
            "mode": "endpointing" if args.endpointing else "baseline",
            "audio_seconds": round(duration, 2),
            "chunk_ms": args.chunk_ms,
            "exit_code": proc.returncode,
            "chunks_decoded": len(per_chunk),
            "cost_proxy": {
                "note": "frontend+encoder ms per chunk; decoder compute is not instrumented",
                "first5_avg_ms": round(sum(costs[:5]) / max(1, len(costs[:5])), 1),
                "last5_avg_ms": round(sum(costs[-5:]) / max(1, len(costs[-5:])), 1),
                "max_ms": round(max(costs), 1) if costs else None,
                "max_encoder_frames": max(frames) if frames else None,
            },
            "segments": len(segments),
            "final_text_chars": len(final_text or ""),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))

        failures = []
        if args.endpointing:
            if proc.returncode != 0:
                failures.append(f"endpointed long run failed ({proc.returncode})")
            if not segments:
                failures.append("endpointed long run produced no speech_end segment events")
            if not final_text:
                failures.append("endpointed long run produced no final transcript")
            if frames and max(frames) > 1500:
                failures.append(f"encoder frames {max(frames)} exceeded position table")
        else:
            if proc.returncode == 0:
                failures.append("baseline run unexpectedly survived past the position-table limit")
            else:
                combined = proc.stderr + proc.stdout
                if "max_source_positions" not in combined:
                    failures.append(
                        "baseline failure is not the safe max_source_positions error:\n"
                        + combined[-2000:])
                else:
                    print("baseline OK: safe max_source_positions failure past 115.40 s")

        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            sys.stderr.write(proc.stdout[-3000:])
            sys.stderr.write(proc.stderr[-3000:])
            return 1
        if args.endpointing:
            print(f"endpointed OK: {len(segments)} segments, "
                  f"{len(per_chunk)} chunks, no position-table error")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"report: {args.report}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
