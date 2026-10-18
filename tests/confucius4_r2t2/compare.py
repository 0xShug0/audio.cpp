#!/usr/bin/env python3
"""Compare audiocpp R2T2 output against the macOS MPS golden reference.

Usage:
    python3 tests/confucius4_r2t2/compare.py \
        --cli build/macos-metal-release/bin/audiocpp_cli \
        --model models/Confucius4-R2T2 \
        --audio <test.wav> \
        --golden tests/confucius4_r2t2/golden.json \
        [--backend metal] [--chunk-ms 320]

Runs the offline CLI, then the streaming CLI with trace logging enabled, and
diffs both against golden.json produced by make_golden.py.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TRACE_RE = re.compile(r"^\[TRACE [^\]]*\] (?P<name>\S+)\s?(?P<value>.*)$")


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def parse_trace(path):
    chunks = []
    pending = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = TRACE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        value = m.group("value")
        if name == "confucius4_r2t2.stream.chunk_id":
            if pending:
                chunks.append(pending)
            pending = {"chunk_id": int(value), "final_flush": 0, "fixed_text": None, "text": None, "raw_decoded": None}
        elif name == "confucius4_r2t2.stream.final_flush" and pending:
            pending["final_flush"] = int(value)
        elif name == "confucius4_r2t2.stream.fixed_text" and pending:
            pending["fixed_text"] = value
        elif name == "confucius4_r2t2.stream.raw_decoded" and pending:
            pending["raw_decoded"] = value
        elif name == "confucius4_r2t2.stream.text" and pending:
            pending["text"] = value
    if pending:
        chunks.append(pending)
    return chunks


def offline_text_from_golden(field):
    """Extract the text from the Python repr of ASRTranscription(...)."""
    m = re.search(r"""text=(['"])(.*)\1,\s*time_stamps=""", field, re.DOTALL)
    if m:
        return m.group(2)
    return field


def transcript_from_stdout(stdout):
    for line in stdout.splitlines():
        if line.startswith("text_output="):
            return line[len("text_output="):]
    for line in stdout.splitlines():
        if line.startswith("text="):
            return line[len("text="):]
    return None


def transcript_fixed_text(chunk, language):
    """Sanitize reference metadata fragments without changing the raw golden."""
    fixed = chunk["fixed_text"]
    if not language or language.lower() == "auto":
        raw = chunk["raw_decoded"]
        metadata = raw.split("<asr_text>", 1)[0]
        if "<asr_text>" not in raw or (fixed and metadata.startswith(fixed)):
            return ""
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--chunk-ms", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--offline-only", action="store_true")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    failures = []

    # --- offline -----------------------------------------------------------
    offline_cmd = [
        args.cli, "--task", "asr", "--family", "confucius4_r2t2", "--model", args.model,
        "--backend", args.backend, "--audio", args.audio,
    ]
    offline = run(offline_cmd)
    got_offline = transcript_from_stdout(offline.stdout)
    want_offline_text = offline_text_from_golden(golden["offline"])
    if got_offline != want_offline_text:
        failures.append(("offline", want_offline_text, got_offline))
        print(f"offline MISMATCH\n  want: {want_offline_text!r}\n  got:  {got_offline!r}")
    else:
        print(f"offline OK: {got_offline!r}")

    if args.offline_only:
        return 1 if failures else 0

    # --- streaming ---------------------------------------------------------
    chunk_ms = args.chunk_ms if args.chunk_ms is not None else golden["chunk_ms"]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else golden["max_new_tokens"]
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        trace_path = tmp.name
    stream_cmd = [
        args.cli, "--task", "asr", "--mode", "streaming", "--family", "confucius4_r2t2",
        "--model", args.model, "--backend", args.backend, "--audio", args.audio,
        "--session-option", f"confucius4_r2t2.chunk_size_ms={chunk_ms}",
        "--session-option", f"confucius4_r2t2.max_tokens={max_new_tokens}",
        "--session-option", f"confucius4_r2t2.unfixed_chunk_num={golden['unfixed_chunk_num']}",
        "--session-option", f"confucius4_r2t2.unfixed_token_num={golden['unfixed_token_num']}",
        "--log-file", trace_path,
    ]
    if golden.get("language"):
        stream_cmd += ["--language", golden["language"]]
    if golden.get("context"):
        stream_cmd += ["--text", golden["context"]]
    streams = run(stream_cmd)
    got_chunks = parse_trace(trace_path)
    got_final = transcript_from_stdout(streams.stdout)

    # Match transcript-only stable prefixes, excluding reference metadata artifacts.
    last, expected_stream = "", []
    for chunk in golden["stream_chunks"]:
        fixed = transcript_fixed_text(chunk, golden.get("language"))
        if len(fixed) > len(last):
            expected_stream.append(fixed[len(last):])
            last = fixed
    expected_stream_text = "".join(expected_stream)
    got_stream_text = "".join(
        line[len("partial_text="):]
        for line in streams.stdout.splitlines()
        if line.startswith("partial_text=")
    )
    if got_stream_text != expected_stream_text:
        failures.append(("committed_stream", expected_stream_text, got_stream_text))
        print(f"committed stream MISMATCH\n  want: {expected_stream_text!r}\n  got:  {got_stream_text!r}")
    else:
        print(f"committed stream OK: {got_stream_text!r}")

    want_chunks = golden["stream_chunks"]
    per_chunk = [c for c in got_chunks if not c["final_flush"]]
    flush = [c for c in got_chunks if c["final_flush"]]
    if len(per_chunk) != len(want_chunks):
        failures.append(("chunk-count", len(want_chunks), len(per_chunk)))
        print(f"chunk count differ: want {len(want_chunks)} got {len(per_chunk)}")
    for i, (want, got) in enumerate(zip(want_chunks, per_chunk)):
        want = dict(want, fixed_text=transcript_fixed_text(want, golden.get("language")))
        if got["fixed_text"] != want["fixed_text"]:
            failures.append((f"chunk[{i}].fixed_text", want["fixed_text"], got["fixed_text"]))
            print(f"chunk[{i}] fixed_text MISMATCH\n  want: {want['fixed_text']!r}\n  got:  {got['fixed_text']!r}")
            print(f"        raw_decoded: {got['raw_decoded']!r}")
    if flush:
        print(f"final flush text: {flush[-1]['text']!r}")
    if got_final != golden["stream_final_text"]:
        failures.append(("stream_final_text", golden["stream_final_text"], got_final))
        print(f"stream final MISMATCH\n  want: {golden['stream_final_text']!r}\n  got:  {got_final!r}")
    else:
        print(f"stream final OK: {got_final!r}")
    if not failures:
        print(f"streaming OK: {len(per_chunk)} chunks match golden per chunk")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
