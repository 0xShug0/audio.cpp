#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPS_ROOT = REPO_ROOT / "reference" / "nemotron-asr-python-deps"
TRANSFORMERS_ROOT = REPO_ROOT / "reference" / "transformers" / "src"
sys.path.insert(0, str(DEPS_ROOT))
sys.path.insert(0, str(TRANSFORMERS_ROOT))

from transformers import AutoModelForRNNT, AutoProcessor  # noqa: E402


def parse_csv_keep_empty(value: str) -> list[str]:
    return value.split(",") if value else []


def parse_csv_paths(value: str, fallback: Path) -> list[Path]:
    return [Path(item) for item in value.split(",") if item] if value else [fallback]


def repeated_arg(values: list[str], index: int, fallback: str) -> str:
    return values[index] if index < len(values) and values[index] else fallback


def repeated_int(values: list[str], index: int, fallback: int) -> int:
    return int(repeated_arg(values, index, str(fallback)))


def repeated_bool(values: list[str], index: int, fallback: bool) -> bool:
    value = repeated_arg(values, index, "true" if fallback else "false").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise RuntimeError(f"invalid bool option: {value}")


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def read_audio_mono_f32(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != target_sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    return audio, sample_rate


def move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> Any:
    inputs = inputs.to(device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=dtype)
    return inputs


def normalize_decoded(decoded: Any) -> str:
    if isinstance(decoded, str):
        return decoded
    if isinstance(decoded, (list, tuple)):
        if not decoded:
            return ""
        return str(decoded[0])
    return str(decoded)


def word_timestamps_json(items: Any, sample_rate: int) -> list[dict[str, Any]]:
    if isinstance(items, (list, tuple)) and items and isinstance(items[0], (list, tuple)):
        items = items[0]
    out = []
    for item in items or []:
        out.append({
            "start_sample": int(round(float(item["start"]) * sample_rate)),
            "end_sample": int(round(float(item["end"]) * sample_rate)),
            "word": str(item["token"]),
            "confidence": 0.0,
        })
    return out


def decode_output(processor: Any, output: Any, keep_language_tags: bool, sample_rate: int) -> tuple[str, list[dict[str, Any]]]:
    sequences = getattr(output, "sequences", output)
    durations = getattr(output, "durations", None)
    if durations is None:
        return normalize_decoded(processor.decode(sequences, skip_special_tokens=not keep_language_tags)), []
    decoded, timestamps = processor.decode(
        sequences,
        durations=durations,
        skip_special_tokens=not keep_language_tags,
    )
    return normalize_decoded(decoded), word_timestamps_json(timestamps, sample_rate)


def transcribe_offline(
    model: Any,
    processor: Any,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    keep_language_tags: bool,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, list[dict[str, Any]]]:
    inputs = processor(audio, sampling_rate=sample_rate, language=language, return_tensors="pt")
    inputs = move_inputs(inputs, device, dtype)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
        )
    return decode_output(processor, output, keep_language_tags, sample_rate)


def transcribe_streaming(
    model: Any,
    processor: Any,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    keep_language_tags: bool,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, list[dict[str, Any]]]:
    first_samples = int(processor.num_samples_first_audio_chunk)
    first_chunk_inputs = processor(
        audio[:first_samples],
        sampling_rate=sample_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    )
    first_chunk_inputs = move_inputs(first_chunk_inputs, device, dtype)

    def input_features_generator():
        yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel_frame_idx = int(processor.num_mel_frames_first_audio_chunk)
        hop_length = int(processor.feature_extractor.hop_length)
        n_fft = int(processor.feature_extractor.n_fft)
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        chunk_samples = int(processor.num_samples_per_audio_chunk)
        while (end_idx := start_idx + chunk_samples) < audio.shape[0]:
            inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=sample_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=language,
                return_tensors="pt",
            )
            inputs = move_inputs(inputs, device, dtype)
            yield inputs.input_features
            mel_frame_idx += int(processor.num_mel_frames_per_audio_chunk)
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_features_generator(),
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
    }
    with torch.inference_mode():
        output = model.generate(**generate_kwargs)
    return decode_output(processor, output, keep_language_tags, sample_rate)


def run_request(
    model: Any,
    processor: Any,
    audio_path: Path,
    language: str,
    keep_language_tags: bool,
    max_new_tokens: int,
    lookahead_tokens: int,
    streaming: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, list[dict[str, Any]]]:
    processor.set_num_lookahead_tokens(lookahead_tokens)
    sample_rate = int(processor.feature_extractor.sampling_rate)
    audio, sample_rate = read_audio_mono_f32(audio_path, sample_rate)
    if streaming:
        return transcribe_streaming(
            model,
            processor,
            audio,
            sample_rate,
            language,
            keep_language_tags,
            max_new_tokens,
            device,
            dtype,
        )
    return transcribe_offline(
        model,
        processor,
        audio,
        sample_rate,
        language,
        keep_language_tags,
        max_new_tokens,
        device,
        dtype,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Python reference Nemotron 3.5 ASR warmbench.")
    parser.add_argument("--model", default="models/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--audio", default="resources/sample_16k.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--language-sequence", default="")
    parser.add_argument("--request-language", action="append", default=[])
    parser.add_argument("--warmup-language", default="")
    parser.add_argument("--lookahead-tokens", type=int, default=3)
    parser.add_argument("--lookahead-tokens-sequence", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens-sequence", default="")
    parser.add_argument("--streaming", choices=["true", "false"], default="false")
    parser.add_argument("--streaming-sequence", default="")
    parser.add_argument("--keep-language-tags", choices=["true", "false"], default="false")
    parser.add_argument("--keep-language-tags-sequence", default="")
    parser.add_argument("--timing-file", default="")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    dtype = torch.float32

    model_path = resolve_path(args.model)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(str(model_path), local_files_only=True).to(device).eval()

    warmup_audio = resolve_path(args.warmup_audio) if args.warmup_audio else resolve_path(args.audio)
    request_paths = parse_csv_paths(args.audio_sequence, Path(args.audio))
    request_languages = parse_csv_keep_empty(args.language_sequence)
    lookahead_sequence = parse_csv_keep_empty(args.lookahead_tokens_sequence)
    max_tokens_sequence = parse_csv_keep_empty(args.max_tokens_sequence)
    streaming_sequence = parse_csv_keep_empty(args.streaming_sequence)
    keep_language_tags_sequence = parse_csv_keep_empty(args.keep_language_tags_sequence)
    warmup_language = args.warmup_language or args.language
    keep_language_tags = args.keep_language_tags == "true"

    timing_lines: list[str] = [
        "nemotron_asr.python_dtype float32",
        f"nemotron_asr.model_root {model_path}",
        f"nemotron_asr.backend {args.backend}",
    ]
    for _ in range(args.warmup):
        run_request(
            model,
            processor,
            warmup_audio,
            warmup_language,
            keep_language_tags,
            args.max_tokens,
            args.lookahead_tokens,
            args.streaming == "true",
            device,
            dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    steps = []
    for request_index, audio in enumerate(request_paths):
        audio_path = resolve_path(audio)
        language = (
            request_languages[request_index]
            if request_index < len(request_languages) and request_languages[request_index]
            else repeated_arg(args.request_language, request_index, args.language)
        )
        lookahead_tokens = repeated_int(lookahead_sequence, request_index, args.lookahead_tokens)
        max_new_tokens = repeated_int(max_tokens_sequence, request_index, args.max_tokens)
        streaming = repeated_bool(streaming_sequence, request_index, args.streaming == "true")
        keep_tags = repeated_bool(keep_language_tags_sequence, request_index, keep_language_tags)
        text = ""
        word_timestamps: list[dict[str, Any]] = []
        total_ms = 0.0
        for _ in range(args.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            text, word_timestamps = run_request(
                model,
                processor,
                audio_path,
                language,
                keep_tags,
                max_new_tokens,
                lookahead_tokens,
                streaming,
                device,
                dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - start) * 1000.0
        wall_ms = total_ms / args.iterations
        print(f"average[{request_index}]")
        print(f"nemotron_asr.wall_ms={wall_ms}")
        timing_lines.append(f"nemotron_asr.request{request_index}.wall_ms {wall_ms:.6f}")
        timing_lines.append(f"nemotron_asr.request{request_index}.language {language}")
        timing_lines.append(f"nemotron_asr.request{request_index}.lookahead_tokens {lookahead_tokens}")
        timing_lines.append(f"nemotron_asr.request{request_index}.streaming {1 if streaming else 0}")
        timing_lines.append(f"nemotron_asr.request{request_index}.keep_language_tags {1 if keep_tags else 0}")
        steps.append({
            "request_index": request_index,
            "audio": str(audio),
            "requested_language": language,
            "language": language,
            "text_output": text,
            "word_timestamps": word_timestamps,
            "metrics": {
                "wall_ms": wall_ms,
                "lookahead_tokens": lookahead_tokens,
                "streaming": streaming,
            },
        })

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print("summary_json=" + json.dumps(
        {"family": "nemotron_asr", "backend": args.backend, "sequence_steps": steps},
        separators=(",", ":"),
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
