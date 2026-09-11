#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_csv_keep_empty(value: str) -> list[str]:
    return value.split(",") if value else []


def parse_csv_paths(value: str, fallback: Path) -> list[Path]:
    return [Path(item) for item in value.split(",") if item] if value else [fallback]


def repeated_arg(values: list[str], index: int, fallback: str) -> str:
    return values[index] if index < len(values) and values[index] else fallback


def repeated_int(values: list[str], index: int, fallback: int) -> int:
    return int(repeated_arg(values, index, str(fallback)))


def repeated_float(values: list[str], index: int, fallback: float) -> float:
    return float(repeated_arg(values, index, str(fallback)))


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid bool value: {value}")


def repeated_bool(values: list[str], index: int, fallback: bool) -> bool:
    return parse_bool(repeated_arg(values, index, "true" if fallback else "false"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_audio(path: Path, sampling_rate: int) -> Any:
    audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return audio


def prepare_requests(paths: list[Path], sampling_rate: int) -> list[tuple[Path, Any, float]]:
    requests = []
    for path in paths:
        resolved = resolve_path(path)
        audio = load_audio(resolved, sampling_rate)
        duration_sec = float(len(audio)) / float(sampling_rate)
        requests.append((path, audio, duration_sec))
    return requests


def run_request(
    processor: Any,
    model: Any,
    audio: Any,
    max_new_tokens: int | None,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    streaming: bool,
) -> tuple[str, dict[str, float]]:
    device = model.device
    dtype = model.dtype

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preprocess_start = time.perf_counter()
    if streaming:
        inputs = processor(
            audio[: int(processor.num_samples_first_audio_chunk)],
            sampling_rate=int(processor.feature_extractor.sampling_rate),
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(audio, return_tensors="pt")
    inputs = inputs.to(device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    def input_features_generator():
        yield inputs.input_features
        mel_frame_idx = int(processor.num_mel_frames_first_audio_chunk)
        hop_length = int(processor.feature_extractor.hop_length)
        n_fft = int(processor.feature_extractor.n_fft)
        chunk_samples = int(processor.num_samples_per_audio_chunk)
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        while (end_idx := start_idx + chunk_samples) < audio.shape[0]:
            chunk_inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=int(processor.feature_extractor.sampling_rate),
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            chunk_inputs = chunk_inputs.to(device, dtype=dtype)
            yield chunk_inputs.input_features
            mel_frame_idx += int(processor.audio_length_per_tok)
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    generate_start = time.perf_counter()
    seed_everything(seed)
    with torch.inference_mode():
        generate_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        generate_kwargs["do_sample"] = do_sample
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            generate_kwargs["top_k"] = top_k
        if streaming:
            model_inputs = dict(inputs)
            model_inputs["input_features"] = input_features_generator()
            outputs = model.generate(**model_inputs, **generate_kwargs)
        else:
            outputs = model.generate(**inputs, **generate_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generate_ms = (time.perf_counter() - generate_start) * 1000.0

    decode_start = time.perf_counter()
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    return decoded[0], {
        "preprocess_ms": preprocess_ms,
        "generate_ms": generate_ms,
        "decode_ms": decode_ms,
        "wall_ms": preprocess_ms + generate_ms + decode_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Python reference VoxTral realtime ASR warmbench.")
    parser.add_argument("--model", default="models/Voxtral-Mini-4B-Realtime-2602")
    parser.add_argument("--audio", default="resources/sample.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=["cuda"], default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-tokens-sequence", default="")
    parser.add_argument("--do-sample", default="false")
    parser.add_argument("--do-sample-sequence", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-sequence", default="")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-p-sequence", default="")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-k-sequence", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seed-sequence", default="")
    parser.add_argument("--streaming", default="false")
    parser.add_argument("--timing-file", default="")
    parser.add_argument("--summary-file", default="")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{args.device}")
    model_path = resolve_path(args.model)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        local_files_only=True,
    ).eval()

    sampling_rate = int(processor.feature_extractor.sampling_rate)
    warmup_path = Path(args.warmup_audio) if args.warmup_audio else Path(args.audio)
    warmup_audio = load_audio(resolve_path(warmup_path), sampling_rate)
    request_paths = parse_csv_paths(args.audio_sequence, Path(args.audio))
    request_inputs = prepare_requests(request_paths, sampling_rate)
    max_tokens_sequence = parse_csv_keep_empty(args.max_tokens_sequence)
    do_sample_sequence = parse_csv_keep_empty(args.do_sample_sequence)
    temperature_sequence = parse_csv_keep_empty(args.temperature_sequence)
    top_p_sequence = parse_csv_keep_empty(args.top_p_sequence)
    top_k_sequence = parse_csv_keep_empty(args.top_k_sequence)
    seed_sequence = parse_csv_keep_empty(args.seed_sequence)

    timing_lines: list[str] = [
        "voxtral_realtime.model_load_excluded 1",
        "voxtral_realtime.backend cuda",
        "voxtral_realtime.dtype bfloat16",
        "voxtral_realtime.tf32_disabled 1",
        f"voxtral_realtime.threads {max(1, args.threads)}",
        f"voxtral_realtime.sampling_rate {sampling_rate}",
    ]

    for _ in range(args.warmup):
        run_request(
            processor,
            model,
            warmup_audio,
            args.max_tokens if args.max_tokens > 0 else None,
            parse_bool(args.do_sample),
            args.temperature,
            args.top_p,
            args.top_k,
            args.seed,
            parse_bool(args.streaming),
        )

    steps = []
    for request_index, (audio_path, audio, duration_sec) in enumerate(request_inputs):
        max_new_tokens_value = repeated_int(max_tokens_sequence, request_index, args.max_tokens)
        max_new_tokens = max_new_tokens_value if max_new_tokens_value > 0 else None
        do_sample = repeated_bool(do_sample_sequence, request_index, parse_bool(args.do_sample))
        temperature = repeated_float(temperature_sequence, request_index, args.temperature)
        top_p = repeated_float(top_p_sequence, request_index, args.top_p)
        top_k = repeated_int(top_k_sequence, request_index, args.top_k)
        seed = repeated_int(seed_sequence, request_index, args.seed)
        text = ""
        totals = {
            "preprocess_ms": 0.0,
            "generate_ms": 0.0,
            "decode_ms": 0.0,
            "wall_ms": 0.0,
        }
        for iteration in range(args.iterations):
            text, metrics = run_request(
                processor,
                model,
                audio,
                max_new_tokens,
                do_sample,
                temperature,
                top_p,
                top_k,
                seed + iteration,
                parse_bool(args.streaming),
            )
            for key, value in metrics.items():
                totals[key] += value
        averaged = {key: value / float(args.iterations) for key, value in totals.items()}
        averaged["duration_sec"] = duration_sec
        if max_new_tokens is not None:
            averaged["max_new_tokens"] = float(max_new_tokens)
        print(f"average[{request_index}]")
        print(f"voxtral_realtime.wall_ms={averaged['wall_ms']}")
        print(f"voxtral_realtime.generate_ms={averaged['generate_ms']}")
        timing_lines.extend([
            f"voxtral_realtime.request{request_index}.wall_ms {averaged['wall_ms']:.6f}",
            f"voxtral_realtime.request{request_index}.preprocess_ms {averaged['preprocess_ms']:.6f}",
            f"voxtral_realtime.request{request_index}.generate_ms {averaged['generate_ms']:.6f}",
            f"voxtral_realtime.request{request_index}.decode_ms {averaged['decode_ms']:.6f}",
            f"voxtral_realtime.request{request_index}.duration_sec {duration_sec:.6f}",
            f"voxtral_realtime.request{request_index}.max_new_tokens {max_new_tokens if max_new_tokens is not None else 'default'}",
            f"voxtral_realtime.request{request_index}.do_sample {str(do_sample).lower()}",
            f"voxtral_realtime.request{request_index}.temperature {temperature:.6f}",
            f"voxtral_realtime.request{request_index}.top_p {top_p:.6f}",
            f"voxtral_realtime.request{request_index}.top_k {top_k}",
            f"voxtral_realtime.request{request_index}.seed {seed}",
        ])
        steps.append({
            "request_index": request_index,
            "audio": str(audio_path),
            "language": "",
            "text_output": text,
            "word_timestamps": [],
            "metrics": {
                **averaged,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
            },
        })

    summary = {
        "family": "voxtral_realtime",
        "backend": args.backend,
        "sequence_steps": steps,
    }
    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary_json=" + json.dumps(summary, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
