#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_WORD_ERROR_RATE = 0.05
DEFAULT_MAX_ABS_TIMESTAMP_ERROR_MS = 100.0
DEFAULT_MAX_MEAN_ABS_TIMESTAMP_ERROR_MS = 50.0

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class WordSpan:
    word: str
    norm: str
    start_s: float
    end_s: float
    source_index: int


@dataclass(frozen=True)
class AlignmentOp:
    kind: str
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def words_from_json(data: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("word_timestamps"), list):
            return data["word_timestamps"]
        if isinstance(data.get("words"), list):
            return data["words"]
        steps = data.get("sequence_steps")
        if isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], dict):
            return words_from_json(steps[0], label)
    raise ValueError(f"{label} JSON must be a word list or contain word_timestamps")


def item_seconds(item: dict[str, Any], label: str, index: int, sample_rate: float) -> tuple[float, float]:
    if "start" in item and "end" in item:
        start = float(item["start"])
        end = float(item["end"])
    elif "start_time" in item and "end_time" in item:
        start = float(item["start_time"])
        end = float(item["end_time"])
    elif "start_sample" in item and "end_sample" in item:
        start = float(item["start_sample"]) / sample_rate
        end = float(item["end_sample"]) / sample_rate
    else:
        raise ValueError(f"{label}[{index}] is missing start/end or start_sample/end_sample")
    if start < 0 or end < start:
        raise ValueError(f"{label}[{index}] has invalid timestamp span: start={start}, end={end}")
    return start, end


def normalize_phrase(text: str) -> str:
    text = text.lower().replace("’", "'")
    return " ".join(WORD_RE.findall(text))


def parse_word_spans(path: Path, label: str, sample_rate: float) -> list[WordSpan]:
    raw_words = words_from_json(load_json(path), label)
    spans: list[WordSpan] = []
    for index, item in enumerate(raw_words):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        word = str(item.get("word", item.get("text", ""))).strip()
        norm = normalize_phrase(word)
        if not norm:
            continue
        start, end = item_seconds(item, label, index, sample_rate)
        spans.append(WordSpan(word=word, norm=norm, start_s=start, end_s=end, source_index=index))
    return spans


def phrase_norm(words: list[WordSpan], start: int, end: int) -> str:
    return normalize_phrase(" ".join(item.word for item in words[start:end]))


def phrase_text(words: list[WordSpan], start: int, end: int) -> str:
    return " ".join(item.word for item in words[start:end])


def align_words(ref: list[WordSpan], hyp: list[WordSpan], max_phrase_words: int) -> list[AlignmentOp]:
    rows = len(ref) + 1
    cols = len(hyp) + 1
    costs = [[0] * cols for _ in range(rows)]
    choices: list[list[AlignmentOp | None]] = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
        choices[i][0] = AlignmentOp("delete", i - 1, i, 0, 0)
    for j in range(1, cols):
        costs[0][j] = j
        choices[0][j] = AlignmentOp("insert", 0, 0, j - 1, j)

    for i in range(1, rows):
        for j in range(1, cols):
            best_cost = costs[i - 1][j - 1] + (0 if ref[i - 1].norm == hyp[j - 1].norm else 1)
            best_op = AlignmentOp("match" if ref[i - 1].norm == hyp[j - 1].norm else "substitute", i - 1, i, j - 1, j)

            delete_cost = costs[i - 1][j] + 1
            if delete_cost < best_cost:
                best_cost = delete_cost
                best_op = AlignmentOp("delete", i - 1, i, j, j)

            insert_cost = costs[i][j - 1] + 1
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_op = AlignmentOp("insert", i, i, j - 1, j)

            for ref_len in range(1, min(max_phrase_words, i) + 1):
                for hyp_len in range(1, min(max_phrase_words, j) + 1):
                    if ref_len == 1 and hyp_len == 1:
                        continue
                    if phrase_norm(ref, i - ref_len, i) != phrase_norm(hyp, j - hyp_len, j):
                        continue
                    phrase_cost = costs[i - ref_len][j - hyp_len]
                    if phrase_cost < best_cost:
                        best_cost = phrase_cost
                        best_op = AlignmentOp("match", i - ref_len, i, j - hyp_len, j)

            costs[i][j] = best_cost
            choices[i][j] = best_op

    ops: list[AlignmentOp] = []
    i = len(ref)
    j = len(hyp)
    while i > 0 or j > 0:
        op = choices[i][j]
        if op is None:
            raise RuntimeError("word alignment backtrace failed")
        ops.append(op)
        i = op.ref_start
        j = op.hyp_start
    ops.reverse()
    return ops


def endpoint_errors_ms(ref: list[WordSpan], hyp: list[WordSpan], op: AlignmentOp) -> tuple[float, float, float]:
    ref_start_s = ref[op.ref_start].start_s
    ref_end_s = ref[op.ref_end - 1].end_s
    hyp_start_s = hyp[op.hyp_start].start_s
    hyp_end_s = hyp[op.hyp_end - 1].end_s
    start_ms = abs(ref_start_s - hyp_start_s) * 1000.0
    end_ms = abs(ref_end_s - hyp_end_s) * 1000.0
    max_ms = max(start_ms, end_ms)
    return start_ms, end_ms, max_ms


def print_word_mismatches(ops: list[AlignmentOp], ref: list[WordSpan], hyp: list[WordSpan], limit: int) -> None:
    printed = 0
    for op in ops:
        if op.kind == "match":
            continue
        ref_text = phrase_text(ref, op.ref_start, op.ref_end)
        hyp_text = phrase_text(hyp, op.hyp_start, op.hyp_end)
        print(
            f"word_mismatch type={op.kind} "
            f"ref[{op.ref_start}:{op.ref_end}]={ref_text!r} "
            f"hyp[{op.hyp_start}:{op.hyp_end}]={hyp_text!r}"
        )
        printed += 1
        if printed >= limit:
            break


def print_timestamp_mismatches(
    rows: list[tuple[AlignmentOp, float, float, float]],
    ref: list[WordSpan],
    hyp: list[WordSpan],
    limit: int,
) -> None:
    for op, start_ms, end_ms, max_ms in rows[:limit]:
        print(
            f"timestamp_mismatch ref[{op.ref_start}:{op.ref_end}]={phrase_text(ref, op.ref_start, op.ref_end)!r} "
            f"hyp[{op.hyp_start}:{op.hyp_end}]={phrase_text(hyp, op.hyp_start, op.hyp_end)!r} "
            f"start_error_ms={start_ms:.3f} end_error_ms={end_ms:.3f} max_error_ms={max_ms:.3f} "
            f"ref_span_s=({ref[op.ref_start].start_s:.3f},{ref[op.ref_end - 1].end_s:.3f}) "
            f"hyp_span_s=({hyp[op.hyp_start].start_s:.3f},{hyp[op.hyp_end - 1].end_s:.3f})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare word/timestamp JSON against ground truth using edit-aligned word spans."
    )
    parser.add_argument("--ref", required=True, type=Path, help="Ground-truth word timestamp JSON.")
    parser.add_argument("--hyp", required=True, type=Path, help="Result word timestamp JSON.")
    parser.add_argument("--sample-rate", type=float, default=16000.0, help="Sample rate for *_sample timestamp fields.")
    parser.add_argument("--max-word-error-rate", type=float, default=DEFAULT_MAX_WORD_ERROR_RATE)
    parser.add_argument("--max-abs-timestamp-error-ms", type=float, default=DEFAULT_MAX_ABS_TIMESTAMP_ERROR_MS)
    parser.add_argument("--max-mean-abs-timestamp-error-ms", type=float, default=DEFAULT_MAX_MEAN_ABS_TIMESTAMP_ERROR_MS)
    parser.add_argument("--max-phrase-words", type=int, default=3, help="Maximum phrase length for matching multi-word JSON items.")
    parser.add_argument("--max-report", type=int, default=20, help="Maximum mismatch rows printed per category.")
    args = parser.parse_args()

    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.max_phrase_words <= 0:
        raise ValueError("--max-phrase-words must be positive")

    ref = parse_word_spans(args.ref, "ref", args.sample_rate)
    hyp = parse_word_spans(args.hyp, "hyp", args.sample_rate)
    ops = align_words(ref, hyp, args.max_phrase_words)

    substitutions = sum(1 for op in ops if op.kind == "substitute")
    deletions = sum(1 for op in ops if op.kind == "delete")
    insertions = sum(1 for op in ops if op.kind == "insert")
    edit_distance = substitutions + deletions + insertions
    word_error_rate = edit_distance / max(1, len(ref))

    timestamp_rows: list[tuple[AlignmentOp, float, float, float]] = []
    endpoint_errors: list[float] = []
    for op in ops:
        if op.kind != "match":
            continue
        start_ms, end_ms, max_ms = endpoint_errors_ms(ref, hyp, op)
        endpoint_errors.extend([start_ms, end_ms])
        if max_ms > args.max_abs_timestamp_error_ms:
            timestamp_rows.append((op, start_ms, end_ms, max_ms))

    mean_abs_timestamp_error_ms = sum(endpoint_errors) / len(endpoint_errors) if endpoint_errors else 0.0
    max_abs_timestamp_error_ms = max(endpoint_errors) if endpoint_errors else 0.0
    word_ok = word_error_rate <= args.max_word_error_rate
    timestamp_ok = (
        not timestamp_rows and
        mean_abs_timestamp_error_ms <= args.max_mean_abs_timestamp_error_ms
    )
    ok = word_ok and timestamp_ok

    print(
        "summary "
        f"ok={str(ok).lower()} "
        f"ref_words={len(ref)} hyp_words={len(hyp)} "
        f"edit_distance={edit_distance} substitutions={substitutions} deletions={deletions} insertions={insertions} "
        f"word_error_rate={word_error_rate:.6f} "
        f"matched_blocks={sum(1 for op in ops if op.kind == 'match')} "
        f"mean_abs_timestamp_error_ms={mean_abs_timestamp_error_ms:.3f} "
        f"max_abs_timestamp_error_ms={max_abs_timestamp_error_ms:.3f} "
        f"timestamp_failures={len(timestamp_rows)}"
    )
    print(
        "thresholds "
        f"max_word_error_rate={args.max_word_error_rate:.6f} "
        f"max_abs_timestamp_error_ms={args.max_abs_timestamp_error_ms:.3f} "
        f"max_mean_abs_timestamp_error_ms={args.max_mean_abs_timestamp_error_ms:.3f}"
    )
    if not ok and edit_distance > 0:
        print_word_mismatches(ops, ref, hyp, args.max_report)
    if not timestamp_ok:
        print_timestamp_mismatches(timestamp_rows, ref, hyp, args.max_report)
        if not timestamp_rows and mean_abs_timestamp_error_ms > args.max_mean_abs_timestamp_error_ms:
            print(
                "timestamp_mean_failure "
                f"mean_abs_timestamp_error_ms={mean_abs_timestamp_error_ms:.3f} "
                f"threshold_ms={args.max_mean_abs_timestamp_error_ms:.3f}"
            )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
