#!/usr/bin/env python3
"""Compare each public Kroko language package with the ONNX reference."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import psutil


PACKAGE_CODE = {
    "de": "DE",
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "it": "IT",
    "he": "IW",
    "nl": "NL",
    "pt": "PT",
    "sv": "SV",
    "tr": "TR",
}


def normalized_words(text: str) -> list[str]:
    return "".join(
        character.lower()
        if unicodedata.category(character)[0] in {"L", "N"}
        else " "
        for character in text
    ).split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, start=1):
        next_row = [index]
        for column, actual in enumerate(hypothesis, start=1):
            next_row.append(
                min(
                    row[column] + 1,
                    next_row[column - 1] + 1,
                    row[column - 1] + (expected != actual),
                )
            )
        row = next_row
    return row[-1]


def reference_word_starts(
    tokens: list[str], timestamps: list[float]
) -> list[float]:
    starts: list[float] = []
    for index, token in enumerate(tokens):
        if index == 0 or token.startswith(" "):
            starts.append(timestamps[index])
    return starts


def run_with_peak_rss(command: list[str]) -> tuple[str, int, float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    peak_rss = 0
    monitored = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, monitored.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.01)
    output = process.communicate()[0]
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{output}"
        )
    return output, peak_rss, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(
        (args.samples / "manifest.json").read_text(encoding="utf-8")
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for language, package_code in PACKAGE_CODE.items():
        package = args.models_root / (
            f"Kroko-{package_code}-Community-64-L-Streaming-001.data"
        )
        model = args.models_root / (
            f"Kroko-{package_code}-Community-64-L-Native"
        )
        audio = args.samples / manifest[language]["audio"]
        reference_path = args.output / f"{language}_reference.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("kroko_reference_transcribe.py")),
                str(package),
                str(audio),
                "--threads",
                "4",
                "--output",
                str(reference_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        text_path = args.output / f"{language}_audiocpp.txt"
        words_path = args.output / f"{language}_audiocpp_words.json"
        log, peak_rss, process_ms = run_with_peak_rss(
            [
                str(args.cli),
                "--task",
                "asr",
                "--family",
                "kroko_asr",
                "--model",
                str(model),
                "--backend",
                args.backend,
                "--audio",
                str(audio),
                "--language",
                language,
                "--text-out",
                str(text_path),
                "--words-out",
                str(words_path),
                "--log",
            ]
        )
        actual = text_path.read_text(encoding="utf-8").strip()
        words = json.loads(words_path.read_text(encoding="utf-8"))
        expected_words = normalized_words(manifest[language]["text"])
        actual_words = normalized_words(actual)
        reference_words = normalized_words(reference["text"])
        timing = re.search(r"kroko_asr\.session_ms ([0-9.]+)", log)
        actual_starts = [
            item["start_sample"] / 16000.0 for item in words
        ]
        reference_starts = reference_word_starts(
            reference["tokens"], reference["timestamps_seconds"]
        )
        timestamp_pairs = min(len(actual_starts), len(reference_starts))
        timestamp_max_abs = (
            max(
                abs(actual_starts[index] - reference_starts[index])
                for index in range(timestamp_pairs)
            )
            if timestamp_pairs
            else None
        )
        duration = reference["duration_seconds"]
        rows.append(
            {
                "language": language,
                "audio_duration_seconds": duration,
                "ground_truth": manifest[language]["text"],
                "reference_text": reference["text"],
                "audiocpp_text": actual,
                "reference_match": actual == reference["text"],
                "reference_wer": edit_distance(
                    reference_words, actual_words
                )
                / max(1, len(reference_words)),
                "ground_truth_wer_reference": edit_distance(
                    expected_words, reference_words
                )
                / max(1, len(expected_words)),
                "ground_truth_wer_audiocpp": edit_distance(
                    expected_words, actual_words
                )
                / max(1, len(expected_words)),
                "reference_ms": reference["elapsed_ms"],
                "reference_rtf": reference["rtf"],
                "audiocpp_session_ms": (
                    float(timing.group(1)) if timing else None
                ),
                "audiocpp_process_ms": process_ms,
                "audiocpp_rtf": (
                    float(timing.group(1)) / 1000.0 / duration
                    if timing
                    else None
                ),
                "peak_rss_mib": peak_rss / (1024 * 1024),
                "word_count_reference": len(reference_starts),
                "word_count_audiocpp": len(actual_starts),
                "timestamp_start_max_abs_seconds": timestamp_max_abs,
            }
        )
    report = {
        "backend": args.backend,
        "model_variant": "Kroko Community 64-L",
        "reference": "sherpa-onnx greedy_search CPU",
        "requests": rows,
    }
    report_path = args.output / "multilingual_parity.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
