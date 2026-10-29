#!/usr/bin/env python3
"""Compare native Bark GGUF/packed weights to original references in one reused session.

The native default is checked unless --offline-mode is explicitly selected.
Bounded-window decoding is a different policy and must not be presented as
exact whole-recording reference parity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "vulkan", "metal"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--offline-mode", choices=["full_context", "auto", "long_form"])
    args = parser.parse_args()
    cases = json.loads(args.reference.read_text(encoding="utf-8"))
    if not cases:
        parser.error("reference manifest must not be empty")
    # Include short/long/short and silence transitions without reconstructing
    # the model or preparing a new session for each utterance.
    sequence = cases + [cases[0], max(cases, key=lambda c: c["audio_seconds"]), cases[0]]
    if any("," in c["path"] for c in sequence):
        parser.error("warm-bench CSV audio paths cannot contain commas")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary = args.bin_dir / ("parakeet_warm_bench.exe" if os.name == "nt" else "parakeet_warm_bench")
    command = [str(binary.resolve()), "--model", str(args.model.resolve()),
               "--audio", cases[0]["path"], "--audio-sequence", ",".join(c["path"] for c in sequence),
               "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads),
               "--warmup", "0", "--iterations", "1",
               "--timing-file", str((args.output_dir / "timing.log").resolve())]
    if args.model.suffix.lower() == ".gguf":
        command += ["--model-spec-override", "embedded"]
    if args.offline_mode:
        command += ["--offline-mode", args.offline_mode]
    with (args.output_dir / "native.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    raw = (args.output_dir / "native.log").read_text(encoding="utf-8")
    summary = json.loads(raw.split("summary_json=", 1)[1])
    steps = summary["sequence_steps"]
    if len(steps) != len(sequence):
        raise AssertionError("native request count differs from reference")
    checks = []
    for expected, actual in zip(sequence, steps):
        limit = round(expected["audio_seconds"] * 16000)
        checks.append(dict(case=expected["name"], exact_text=actual["text_output"] == expected["text"],
                           timestamps_bounded=all(0 <= w["start_sample"] <= w["end_sample"] <= limit
                                                  for w in actual["word_timestamps"]),
                           actual_text=actual["text_output"], expected_text=expected["text"]))
    passed = all(c["exact_text"] and c["timestamps_bounded"] for c in checks)
    report = dict(passed=passed, command=command, checks=checks, summary=summary)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for check in checks:
        print(check["case"], "text", check["exact_text"], "timestamps", check["timestamps_bounded"])
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
