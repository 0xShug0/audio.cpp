#!/usr/bin/env python3
"""Validate deterministic MiraTTS warm-benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def load_results(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in payload["results"]}


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_audio(item: dict[str, object], failures: list[str]) -> None:
    path = Path(str(item["audio_out"]))
    require(path.is_file(), f"missing WAV: {path}", failures)
    require(int(item["sample_rate"]) == 48000, f"unexpected sample rate: {path}", failures)
    require(int(item["channels"]) == 1, f"unexpected channel count: {path}", failures)
    require(int(item["samples"]) > 0, f"empty audio: {path}", failures)
    require(float(item["audio_seconds"]) > 0.0, f"zero duration: {path}", failures)
    require(float(item["rtf"]) > 0.0, f"invalid RTF: {path}", failures)
    if path.is_file() and shutil.which("ffprobe"):
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(process.returncode == 0, f"ffprobe rejected WAV: {path}", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-summary", type=Path, required=True)
    parser.add_argument("--streaming-summary", type=Path, required=True)
    args = parser.parse_args()

    offline = load_results(args.offline_summary)
    streaming = load_results(args.streaming_summary)
    failures: list[str] = []

    expected_offline = {
        "clone_cold",
        "clone_repeat",
        "short_second_prompt",
        "longform",
        "clone_repeat_after_longform",
    }
    expected_streaming = {"stream_cold", "stream_repeat"}
    require(expected_offline <= set(offline), "offline cases are incomplete", failures)
    require(expected_streaming <= set(streaming), "streaming cases are incomplete", failures)

    for item in [*offline.values(), *streaming.values()]:
        validate_audio(item, failures)

    if expected_offline <= set(offline):
        repeat_hashes = {
            str(offline[name]["audio_hash"])
            for name in (
                "clone_cold",
                "clone_repeat",
                "clone_repeat_after_longform",
            )
        }
        require(len(repeat_hashes) == 1, "offline deterministic hashes differ", failures)
        require(
            offline["short_second_prompt"]["audio_hash"]
            != offline["clone_cold"]["audio_hash"],
            "different prompts unexpectedly produced the same audio hash",
            failures,
        )
        require(
            int(offline["longform"]["samples"])
            > int(offline["clone_cold"]["samples"]),
            "long-form case is not longer than the short case",
            failures,
        )

    if expected_streaming <= set(streaming):
        require(
            streaming["stream_cold"]["audio_hash"]
            == streaming["stream_repeat"]["audio_hash"],
            "streaming deterministic hashes differ",
            failures,
        )
        require(
            int(streaming["stream_cold"]["event_count"]) > 1,
            "streaming cold case emitted fewer than two events",
            failures,
        )
        require(
            streaming["stream_cold"]["event_count"]
            == streaming["stream_repeat"]["event_count"],
            "streaming repeat event counts differ",
            failures,
        )
        for name in expected_streaming:
            require(
                float(streaming[name]["first_event_ms"]) > 0.0,
                f"{name} has no valid first-event latency",
                failures,
            )

    report = {
        "passed": not failures,
        "failures": failures,
        "offline_cases": sorted(offline),
        "streaming_cases": sorted(streaming),
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
