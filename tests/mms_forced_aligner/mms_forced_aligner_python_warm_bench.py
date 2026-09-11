#!/usr/bin/env python3
# The ML/reference deps below live only in the harness test venv
# (audiocpp-experiments/mms/.venv), not this repo's Python env. Guard them as
# optional environment dependencies so static analysis treats their absence as
# a runtime requirement rather than a code error.
"""MMS-1130 forced-aligner reference warmbench.

Runs the pinned upstream model with the ctc-forced-aligner text/alignment
logic (uroman + torchaudio.functional.forced_align) and emits the same
summary_json schema as the native C++ warmbench. Reference words are mapped
back onto the original transcript words so hyphenated compounds align with
the native output contract (e.g. "Twenty-two" stays one word).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

try:
    import numpy as np  # pyright: ignore[reportMissingImports] - resolved in harness venv
    import soundfile  # pyright: ignore[reportMissingImports] - resolved in harness venv
    import torch  # pyright: ignore[reportMissingImports] - resolved in harness venv
    import torchaudio.functional as F  # pyright: ignore[reportMissingImports] - resolved in harness venv
    from transformers import AutoModelForCTC, AutoTokenizer  # pyright: ignore[reportMissingImports] - resolved in harness venv
    from uroman import Uroman  # pyright: ignore[reportMissingImports] - resolved in harness venv
except ImportError as exc:  # pragma: no cover - requires the harness venv
    raise RuntimeError(
        "mms_forced_aligner_python_warm_bench requires the reference venv "
        "with numpy/soundfile/torch/torchaudio/transformers/uroman"
    ) from exc

SR = 16000
uroman = Uroman()


# ---------- text (mirrors the pinned ctc-forced-aligner norm_dutch) ----------
def norm_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"\([^\)]*\d[^\)]*\)", " ", t)
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def romanize_and_tokenize(text: str, iso: str) -> tuple[list[str], list[str]]:
    roman = uroman.romanize_string(norm_text(text), lcode=iso).strip()
    roman = re.sub("[^a-z' ]", " ", roman.lower())
    roman = re.sub(r"\s+", " ", roman).strip()
    words = roman.split()
    tokens: list[str] = []
    text_parts: list[str] = []
    for word in words:
        tokens.append("<star>")
        text_parts.append("\u2605")
        tokens.append(" ".join(list(word)))
        text_parts.append(word)
    return tokens, text_parts


# ---------- emissions (mirrors generate_emissions) ----------
def generate_emissions(model: Any, audio: torch.Tensor, window: float = 30.0, context: float = 2.0) -> tuple[torch.Tensor, float]:
    ratio = model.config.inputs_to_logits_ratio or 320
    win = int(window * SR)
    ctx = int(context * SR)
    with torch.inference_mode():
        if audio.size(0) < win:
            emissions = model(audio.unsqueeze(0)).logits
            emissions = emissions.flatten(0, 1)
        else:
            ext = math.ceil(audio.size(0) / win) * win - audio.size(0)
            pad = torch.nn.functional.pad(audio, (ctx, ctx + ext))
            x = pad.unfold(0, win + 2 * ctx, win)
            outs = [model(x[i : i + 1]).logits for i in range(0, x.size(0), 1)]
            emissions = torch.cat(outs, dim=0)
            cf = ctx // ratio
            wf = win // ratio
            emissions = emissions[:, cf : cf + wf].flatten(0, 1)
            if ext > 0:
                emissions = emissions[: -(ext // ratio)]
        emissions = torch.log_softmax(emissions, dim=-1)
        emissions = torch.cat([emissions, torch.zeros(emissions.size(0), 1)], dim=1)
    return emissions, ratio * 1000 / SR


# ---------- alignment + spans (mirrors get_segments/get_spans) ----------
def forced_align(lp_np: np.ndarray, tgt_np: np.ndarray, blank: int) -> tuple[np.ndarray, np.ndarray]:
    lp = torch.from_numpy(lp_np).float()
    tgt = torch.from_numpy(tgt_np)
    batch, t_frames, _ = lp.shape
    input_lengths = torch.full((batch,), t_frames, dtype=torch.int32)
    target_lengths = torch.full((batch,), tgt.shape[1], dtype=torch.int32)
    paths, scores = F.forced_align(lp, tgt, input_lengths, target_lengths, blank)
    return paths.numpy(), scores.numpy()


def align_words(model: Any, tokenizer: Any, audio: torch.Tensor, text: str, iso: str) -> list[tuple[str, float, float, float]]:
    tokens, text_parts = romanize_and_tokenize(text, iso)
    emissions, stride = generate_emissions(model, audio)
    vocab = {key.lower(): value for key, value in tokenizer.get_vocab().items()}
    vocab["<star>"] = len(vocab)
    idxs = [vocab[token] for token in " ".join(tokens).split(" ") if token in vocab]
    blank = vocab.get("<blank>", tokenizer.pad_token_id or 0)
    path, _scores = forced_align(emissions.unsqueeze(0).float().cpu().numpy(), np.asarray([idxs], dtype=np.int64), blank)
    idx2tok = {value: key for key, value in vocab.items()}
    blank_str = idx2tok[blank]

    segments: list[list[Any]] = []
    path_labels = path.squeeze()
    previous = None
    for index, label in enumerate(path_labels):
        if label != previous:
            if previous is not None:
                segments[-1][-1] = index - 1
            segments.append([idx2tok[int(label)], index, index])
            previous = label
    segments[-1][-1] = len(path_labels) - 1

    letter_index = 0
    target_index = 0
    intervals: list[tuple[int, int]] = []
    start = 0
    for segment_index, segment in enumerate(segments):
        if target_index >= len(tokens):
            break
        current = tokens[target_index].split(" ")
        if segment[0] == blank_str:
            continue
        if segment[0] != current[letter_index]:
            raise RuntimeError(f"alignment label mismatch at segment {segment_index}: {segment[0]} vs {current[letter_index]}")
        if letter_index == 0:
            start = segment_index
        if letter_index == len(current) - 1:
            letter_index = 0
            target_index += 1
            intervals.append((start, segment_index))
            while target_index < len(tokens) and len(tokens[target_index]) == 0:
                intervals.append((segment_index, segment_index))
                target_index += 1
        else:
            letter_index += 1

    spans: list[list[list[Any]]] = []
    for index, (a, b) in enumerate(intervals):
        span = segments[a : b + 1]
        if a > 0 and segments[a - 1][0] == blank_str:
            start = segments[a - 1][1] if index == 0 else int((segments[a - 1][1] + segments[a - 1][2]) / 2)
            span = [[blank_str, start, span[0][1]]] + span
        if b + 1 < len(segments) and segments[b + 1][0] == blank_str:
            end = segments[b + 1][2] if index == len(intervals) - 1 else int((segments[b + 1][1] + segments[b + 1][2]) / 2)
            span = span + [[blank_str, span[-1][2], end]]
        spans.append(span)

    words: list[tuple[str, float, float, float]] = []
    for index, text_part in enumerate(text_parts):
        if text_part == "\u2605" or index >= len(spans):
            continue
        span = spans[index]
        # Mean log-probability of the aligned path over the word's frames,
        # matching the native confidence definition (mean path log-prob).
        conf_frames = [f for seg in span for f in range(seg[1], seg[2] + 1)]
        conf = 0.0
        if conf_frames:
            conf = float(emissions[conf_frames, path_labels[conf_frames]].mean().item())
        words.append((text_part, span[0][1] * stride / 1000, (span[-1][2] + 1) * stride / 1000, conf))
    return words


# ---------- native output contract mapping ----------
def normalize_native_word(word: str) -> str:
    return "".join(char for char in word.lower() if char in "abcdefghijklmnopqrstuvwxyz'")


def map_to_original_words(transcript: str, ref_words: list[tuple[str, float, float, float]]) -> list[dict[str, Any]]:
    """Maps reference words onto the original transcript words, merging
    hyphen-split compounds (e.g. twenty + two -> Twenty-two) exactly like the
    native text processor. Confidence is the first constituent's mean."""
    original_words = transcript.split()
    out: list[dict[str, Any]] = []
    ref_index = 0
    for original in original_words:
        expected = normalize_native_word(original)
        if not expected:
            continue
        accumulated = ""
        start_sec = None
        end_sec = None
        confidence = 0.0
        while ref_index < len(ref_words) and len(accumulated) < len(expected):
            word, start, end, conf = ref_words[ref_index]
            accumulated += word
            if start_sec is None:
                start_sec = start
                confidence = conf
            end_sec = end
            ref_index += 1
            if accumulated == expected:
                break
        if accumulated != expected:
            raise RuntimeError(
                f"reference words do not reconstruct original '{original}' (got '{accumulated}', want '{expected}')"
            )
        out.append({
            "word": original,
            "start_sample": int(round((start_sec or 0.0) * SR)),
            "end_sample": int(round((end_sec or 0.0) * SR)),
            "confidence": confidence,
        })
    return out


# ---------- main ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-file", default="")
    parser.add_argument("--model", default="models/mms-300m-1130-forced-aligner")
    parser.add_argument("--audio", default="resources/sample.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--warmup-language", default="")
    parser.add_argument("--warmup-transcript", default="")
    parser.add_argument("--request-language", action="append", default=[])
    parser.add_argument("--request-transcript", action="append", default=[])
    args = parser.parse_args()

    device = torch.device("cpu" if args.backend.startswith("cpu") else f"cuda:{args.device}")
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCTC.from_pretrained(args.model, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    audio_paths = [Path(item) for item in args.audio_sequence.split(",") if item] or [Path(args.audio)]
    languages = [args.request_language[index] if index < len(args.request_language) else args.language for index in range(len(audio_paths))]
    transcripts = [args.request_transcript[index] if index < len(args.request_transcript) else args.transcript for index in range(len(audio_paths))]

    def load_audio(path: Path) -> torch.Tensor:
        # Proper WAV decode (not raw PCM memmap): soundfile handles headers,
        # channels and non-16-bit encodings; linearly resample to 16 kHz to
        # match the native runtime resampler.
        audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SR and audio.size > 1:
            out_frames = max(1, int(round(audio.size * SR / sample_rate)))
            indices = np.linspace(0.0, float(audio.size - 1), out_frames)
            audio = np.interp(indices, np.arange(audio.size), audio).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(audio)).to(device)

    warmup_path = Path(args.warmup_audio) if args.warmup_audio else audio_paths[0]
    warmup_transcript = args.warmup_transcript or (transcripts[0] if transcripts else "")
    warmup_language = args.warmup_language or (languages[0] if languages else args.language)
    warmup_audio = load_audio(warmup_path)
    for _ in range(args.warmup):
        align_words(model, tokenizer, warmup_audio, warmup_transcript, warmup_language)

    steps: list[dict[str, Any]] = []
    timing_lines: list[str] = []
    for index, (audio_path, language, transcript) in enumerate(zip(audio_paths, languages, transcripts)):
        audio = load_audio(audio_path)
        total_ms = 0.0
        last_words: list[dict[str, Any]] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            ref_words = align_words(model, tokenizer, audio, transcript, language)
            last_words = map_to_original_words(transcript, ref_words)
            total_ms += (time.perf_counter() - started) * 1000.0
        wall_ms = total_ms / max(1, args.iterations)
        steps.append({
            "request_index": index,
            "audio": str(audio_path),
            "requested_language": language,
            "text_output": transcript,
            "language": language,
            "word_timestamps": last_words,
            "metrics": {"wall_ms": wall_ms},
        })
        print(f"average[{index}]")
        print(f"mms_forced_aligner.wall_ms={wall_ms}")
        timing_lines.append(f"mms_forced_aligner.align_wall_ms {wall_ms:.6f}")

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    summary = {"family": "mms_forced_aligner", "backend": args.backend, "sequence_steps": steps}
    print("summary_json=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
