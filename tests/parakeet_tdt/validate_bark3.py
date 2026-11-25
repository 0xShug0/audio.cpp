#!/usr/bin/env python3
"""Compare a native Bark GGUF to original-runner references in one reused session.

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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = [{"id": f"request_{i}", "audio": str(Path(case["path"]).resolve())}
                for i, case in enumerate(sequence)]
    sequence_path = (args.output_dir / "sequence.json").resolve()
    sequence_path.write_text(json.dumps({"requests": requests}, indent=2), encoding="utf-8")
    binary = args.bin_dir / ("audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli")
    command = [str(binary.resolve()), "--task", "asr", "--family", "parakeet_tdt",
               "--model", str(args.model.resolve()), "--model-spec-override", "embedded",
               "--request-sequence", str(sequence_path), "--out-dir", str(args.output_dir.resolve()),
               "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads),
               "--log"]
    if args.offline_mode:
        command += ["--session-option", f"parakeet_tdt.offline_mode={args.offline_mode}"]
    with (args.output_dir / "native.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    raw = (args.output_dir / "native.log").read_text(encoding="utf-8")
    steps = []
    for line in raw.splitlines():
        if line.startswith("request_id="):
            steps.append({"request_id": line.removeprefix("request_id="), "word_timestamps": []})
        elif line.startswith("text_output="):
            if not steps:
                raise AssertionError("text output appeared before request id")
            if "text_output" in steps[-1]:
                raise AssertionError("duplicate text output for request")
            steps[-1]["text_output"] = line.removeprefix("text_output=")
        elif line.startswith("word_timestamps="):
            if not steps:
                raise AssertionError("timestamps appeared before request id")
            steps[-1]["word_timestamps"] = json.loads(line.removeprefix("word_timestamps="))
    if [step["request_id"] for step in steps] != [request["id"] for request in requests]:
        raise AssertionError("native request ids/count differ from reference")
    summary = {"sequence_steps": steps}
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
