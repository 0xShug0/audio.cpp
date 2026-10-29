#!/usr/bin/env python3
"""Compare buffered Bark streaming text/timestamps against an earlier binary.

Both binaries must contain the warm-bench start_stream() lifecycle correction.
Use identical GGUFs and context settings. This checks streaming-to-streaming
parity, not equality to the original model's whole-recording context policy.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-bin", type=Path, required=True)
    parser.add_argument("--after-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=3200)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("chunk size must be positive")
    if any("," in str(path) for path in args.audio):
        parser.error("audio paths cannot contain commas")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, binary in [("before", args.before_bin), ("after", args.after_bin)]:
        command = [str(binary.resolve()), "--model", str(args.model.resolve()),
                   "--model-spec-override", "embedded", "--run-mode", "streaming",
                   "--audio", str(args.audio[0].resolve()),
                   "--audio-sequence", ",".join(str(path.resolve()) for path in args.audio),
                   "--backend", args.backend, "--device", str(args.device),
                   "--threads", str(args.threads), "--chunk-size", str(args.chunk_size),
                   "--warmup", "0", "--iterations", "1", "--timing-file",
                   str((args.output_dir / (name + ".timing.log")).resolve())]
        log_path = args.output_dir / (name + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        summary = json.loads(log_path.read_text(encoding="utf-8").split("summary_json=", 1)[1])
        results.append(dict(command=command, summary=summary))
    before, after = (result["summary"]["sequence_steps"] for result in results)
    if len(before) != len(args.audio) or len(after) != len(args.audio):
        raise AssertionError("unexpected request count")
    checks = [dict(audio=str(path), exact_text=old["text_output"] == new["text_output"],
                   exact_timestamps=old["word_timestamps"] == new["word_timestamps"])
              for path, old, new in zip(args.audio, before, after)]
    passed = all(check["exact_text"] and check["exact_timestamps"] for check in checks)
    report = dict(passed=passed, checks=checks, results=results)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.backend, args.model.name, sum(c["exact_text"] and c["exact_timestamps"] for c in checks),
          "/", len(checks), "exact text/timestamps")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
