#!/usr/bin/env python3
"""Compare final native transcripts and emitted segments with upstream results."""

import argparse
import difflib
import json
import logging
import sys
from pathlib import Path

import jiwer
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", nargs="+", required=True, help="audio-stem=CLI-stdout, server-details JSON, SSE file, or streaming-client events JSONL")
    parser.add_argument("--require-text-increments", action="store_true",
                        help="Require text deltas that do not wait for a closing timestamp/speaker delimiter")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-timestamp-drift", type=float, default=0.0)
    parser.add_argument("--max-word-error", type=float, default=0.0)
    parser.add_argument("--allow-segment-reflow", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])
    sys.path.insert(0, str(args.upstream.resolve()))
    from moss_transcribe_diarize import parse_transcript

    references = {Path(item["audio"]).stem: item for item in json.loads(args.reference.read_text())}
    reports = []
    for value in args.actual:
        name, actual_path = value.split("=", 1)
        reference = references[name]
        expected = reference["trials"][0]["segments"]
        if Path(actual_path).suffix in {".sse", ".jsonl"}:
            if Path(actual_path).suffix == ".jsonl":
                records = [json.loads(line) for line in Path(actual_path).read_text(encoding="utf-8").splitlines()]
                if not records or records[-1].get("data") != "[DONE]":
                    raise AssertionError("Streaming client did not receive completion")
                events = [record["event"] for record in records[:-1]]
                if len(records) < 3 or records[0]["elapsed_ms"] >= records[-2]["elapsed_ms"]:
                    raise AssertionError("No incremental event arrived before the final transcript")
            else:
                data = [line[6:] for line in Path(actual_path).read_text(encoding="utf-8").splitlines() if line.startswith("data: ")]
                if not data or data[-1] != "[DONE]":
                    raise AssertionError("Streaming response did not finish")
                events = [json.loads(line) for line in data[:-1]]
            if (not events or events[-1]["type"] != "transcript.text.done"
                    or any(event["type"] != "transcript.text.delta" for event in events[:-1])):
                raise AssertionError("Unexpected streaming event order")
            text = events[-1]["text"]
            if len(events) < 2 or "".join(event["delta"] for event in events[:-1]) != text:
                raise AssertionError("Streaming deltas do not reproduce the final transcript")
            if args.require_text_increments and not any(
                    "]" not in event["delta"] and any(ch.isalpha() for ch in event["delta"])
                    for event in events[:-1]):
                raise AssertionError("Text streaming still waits for timestamp/speaker delimiters")
            for event in events[:-1]:
                event["delta"].encode("utf-8", errors="strict")
            if events[-1]["timing"]["ttft_ms"] <= 0:
                raise AssertionError("Missing streaming first-token timing")
            turns = segments = None
        elif Path(actual_path).suffix == ".json":
            payload = json.loads(Path(actual_path).read_text())
            text = payload["text"]
            turns = payload["speaker_turns"]
            segments = payload["segments"]
            if payload["sample_rate"] != sf.info(reference["audio"]).samplerate:
                raise AssertionError("Server result sample rate differs from fixture")
        else:
            lines = Path(actual_path).read_text().splitlines()
            fields = dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("["))
            if fields.get("family") != "moss_transcribe_diarize" or fields.get("task") != "asr":
                raise AssertionError("Actual output is not a completed MOSS ASR result")
            text = fields["text_output"]
            turns = json.loads(fields["speaker_turns"])
            segments = json.loads(fields["speech_segments"])
        actual = [dict(start=s.start, end=s.end, speaker=s.speaker, text=s.text)
                  for s in parse_transcript(text)]
        if not expected or not actual:
            raise AssertionError("Parity fixtures must contain speech segments")
        expected_words = " ".join(s["text"] for s in expected)
        actual_words = " ".join(s["text"] for s in actual)
        words = jiwer.process_words(expected_words, actual_words)
        matches = difflib.SequenceMatcher(a=[s["text"] for s in expected],
                                         b=[s["text"] for s in actual], autojunk=False)
        matched = [(expected[a + i], actual[b + i])
                   for a, b, count in matches.get_matching_blocks() for i in range(count)]
        timestamps = [abs(a[k] - b[k]) for a, b in matched for k in ("start", "end")]
        expected_speakers = [s["speaker"] for s in expected for _ in s["text"].split()]
        actual_speakers = [s["speaker"] for s in actual for _ in s["text"].split()]
        speaker_mismatches = sum(
            a != b for alignment in words.alignments[0]
            for a, b in zip(expected_speakers[alignment.ref_start_idx:alignment.ref_end_idx],
                            actual_speakers[alignment.hyp_start_idx:alignment.hyp_end_idx]))
        reflow = [dict(expected=expected[a:b], actual=actual[c:d])
                  for kind, a, b, c, d in matches.get_opcodes() if kind != "equal"]

        # Verify the C++ parser's public outputs, not only its raw model text.
        rate = sf.info(reference["audio"]).samplerate
        if turns is not None and (len(turns) != len(actual) or len(segments) != len(actual)):
            raise AssertionError("C++ parser emitted a different segment count than upstream")
        emitted = zip(actual, turns, segments, strict=True) if turns is not None else []
        for parsed, turn, segment in emitted:
            for emitted in (turn, segment):
                if emitted["text"] != parsed["text"]:
                    raise AssertionError("C++ parser changed segment text")
                for key, time_key in (("start_sample", "start"), ("end_sample", "end")):
                    if abs(emitted[key] - round(parsed[time_key] * rate)) > 1:
                        raise AssertionError("C++ parser changed a timestamp")
            if turn["speaker_id"] != parsed["speaker"]:
                raise AssertionError("C++ parser changed a speaker label")

        report = dict(audio=name, reference_dtype=reference["dtype"],
                      raw_equal=text == reference["trials"][0]["text"],
                      word_error=words.wer, substitutions=words.substitutions,
                      insertions=words.insertions, deletions=words.deletions,
                      expected_segments=len(expected), actual_segments=len(actual),
                      matched_segments=len(matched), speaker_mismatches=speaker_mismatches,
                      max_matched_timestamp_drift=max(timestamps, default=0),
                      timestamp_differences=[dict(text=a["text"], expected=[a["start"], a["end"]],
                                                  actual=[b["start"], b["end"]])
                                             for a, b in matched if a["start"] != b["start"] or a["end"] != b["end"]],
                      changed_segments=reflow)
        report["passed"] = (words.wer <= args.max_word_error and speaker_mismatches == 0
                            and report["max_matched_timestamp_drift"] <= args.max_timestamp_drift + 1e-9
                            and (not reflow or args.allow_segment_reflow))
        reports.append(report)
        logging.info("%s", json.dumps(report, ensure_ascii=False))
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    if not all(r["passed"] for r in reports):
        raise AssertionError("Final-output parity failed; see report")


if __name__ == "__main__":
    main()
