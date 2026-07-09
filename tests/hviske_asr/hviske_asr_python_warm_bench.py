#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


def install_transformers457_alias() -> Any:
    transformers = importlib.import_module("transformers457")
    sys.modules["transformers"] = transformers
    for name in ("transformers457.dynamic_module_utils", "transformers.dynamic_module_utils"):
        module = importlib.import_module(name)
        module.transformers = transformers
    return transformers


TRANSFORMERS = install_transformers457_alias()

from transformers import AutoProcessor  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_csv_keep_empty(value: str) -> list[str]:
    return value.split(",") if value else []


def parse_csv_paths(value: str, fallback: Path) -> list[Path]:
    return [Path(item) for item in value.split(",") if item] if value else [fallback]


def repeated_arg(values: list[str], index: int, fallback: str) -> str:
    return values[index] if index < len(values) else fallback


def repeated_int(values: list[str], index: int, fallback: int) -> int:
    return int(repeated_arg(values, index, str(fallback)))


def repeated_float(values: list[str], index: int, fallback: float) -> float:
    return float(repeated_arg(values, index, str(fallback)))


def repeated_bool(values: list[str], index: int, fallback: bool) -> bool:
    value = repeated_arg(values, index, "true" if fallback else "false").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise RuntimeError(f"invalid bool option: {value}")


def read_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    else:
        raise RuntimeError(f"unsupported WAV bit depth: {sample_width * 8}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def transcribe(
    model: Any,
    processor: Any,
    audio_path: Path,
    language: str,
    punctuation: bool,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> str:
    audio, sample_rate = read_wav_mono_f32(audio_path)
    if (
        max_new_tokens == 256
        and num_beams == 1
        and abs(length_penalty - 1.0) < 1e-6
        and not do_sample
    ):
        result = model.transcribe(
            processor=processor,
            language=language,
            audio_arrays=[audio],
            sample_rates=[sample_rate],
            punctuation=punctuation,
        )
        if not isinstance(result, list) or not result:
            raise RuntimeError("Hviske ASR Python transcribe returned no results")
        return str(result[0])

    if do_sample:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    prompt_text = model.build_prompt(language=language, punctuation=punctuation)
    inputs = processor(audio=[audio], text=[prompt_text], sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    if "input_ids" in inputs and "decoder_input_ids" not in inputs:
        inputs["decoder_input_ids"] = inputs.pop("input_ids")
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if "decoder_input_ids" in inputs and "decoder_attention_mask" not in inputs:
        if pad_token_id is None:
            inputs["decoder_attention_mask"] = torch.ones(
                inputs["decoder_input_ids"].shape,
                dtype=torch.long,
                device=inputs["decoder_input_ids"].device,
            )
        else:
            inputs["decoder_attention_mask"] = inputs["decoder_input_ids"].ne(pad_token_id).long()
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "do_sample": do_sample,
        "decoder_start_token_id": int(inputs["decoder_input_ids"][0, 0].item()),
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs.update({
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)
    if "decoder_attention_mask" in inputs:
        prompt_len = int(inputs["decoder_attention_mask"].sum(dim=1)[0].item())
    else:
        prompt_len = int(inputs["decoder_input_ids"].shape[1])
    token_ids = generated_ids[0].detach().cpu().tolist()
    prompt_ids = inputs["decoder_input_ids"][0, :prompt_len].detach().cpu().tolist()
    if prompt_len > 0 and len(token_ids) >= prompt_len and token_ids[:prompt_len] == prompt_ids:
        token_ids = token_ids[prompt_len:]
    if eos_token_id is not None:
        try:
            token_ids = token_ids[:token_ids.index(eos_token_id)]
        except ValueError:
            pass
    return str(tokenizer.batch_decode([token_ids], skip_special_tokens=True)[0]).strip()


def load_hviske_model(model_path: Path, dtype: torch.dtype) -> Any:
    model_class = get_class_from_dynamic_module(
        "modeling_cohere_asr.CohereAsrForConditionalGeneration",
        str(model_path),
        local_files_only=True,
    )
    unexpected_ignore = getattr(model_class, "_keys_to_ignore_on_load_unexpected", None)
    if isinstance(unexpected_ignore, set):
        model_class._keys_to_ignore_on_load_unexpected = sorted(unexpected_ignore)
    return model_class.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Python reference Hviske ASR warmbench.")
    parser.add_argument("--model", default="models/hviske-v5.3")
    parser.add_argument("--audio", default="resources/sample.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--language", default="da")
    parser.add_argument("--language-sequence", default="")
    parser.add_argument("--request-language", action="append", default=[])
    parser.add_argument("--warmup-language", default="")
    parser.add_argument("--punctuation", choices=["true", "false"], default="true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens-sequence", default="")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-beams-sequence", default="")
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty-sequence", default="")
    parser.add_argument("--do-sample", choices=["true", "false"], default="false")
    parser.add_argument("--do-sample-sequence", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-sequence", default="")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-k-sequence", default="")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-p-sequence", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-sequence", default="")
    parser.add_argument("--timing-file", default="")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device(f"cuda:{args.device}")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model_path = resolve_path(args.model)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = load_hviske_model(model_path, dtype).to(device).eval()

    punctuation = args.punctuation == "true"
    warmup_audio = resolve_path(args.warmup_audio) if args.warmup_audio else resolve_path(args.audio)
    request_paths = parse_csv_paths(args.audio_sequence, Path(args.audio))
    request_languages = parse_csv_keep_empty(args.language_sequence)
    max_tokens_sequence = parse_csv_keep_empty(args.max_tokens_sequence)
    num_beams_sequence = parse_csv_keep_empty(args.num_beams_sequence)
    length_penalty_sequence = parse_csv_keep_empty(args.length_penalty_sequence)
    do_sample_sequence = parse_csv_keep_empty(args.do_sample_sequence)
    temperature_sequence = parse_csv_keep_empty(args.temperature_sequence)
    top_k_sequence = parse_csv_keep_empty(args.top_k_sequence)
    top_p_sequence = parse_csv_keep_empty(args.top_p_sequence)
    seed_sequence = parse_csv_keep_empty(args.seed_sequence)
    warmup_language = args.warmup_language or args.language

    timing_lines: list[str] = [
        f"hviske_asr.python_dtype {str(dtype).replace('torch.', '')}",
        f"hviske_asr.transformers_version {TRANSFORMERS.__version__}",
    ]
    for _ in range(args.warmup):
        transcribe(
            model,
            processor,
            warmup_audio,
            warmup_language,
            punctuation,
            args.max_tokens,
            args.num_beams,
            args.length_penalty,
            args.do_sample == "true",
            args.temperature,
            args.top_k,
            args.top_p,
            args.seed,
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
        max_new_tokens = repeated_int(max_tokens_sequence, request_index, args.max_tokens)
        num_beams = repeated_int(num_beams_sequence, request_index, args.num_beams)
        length_penalty = repeated_float(length_penalty_sequence, request_index, args.length_penalty)
        do_sample = repeated_bool(do_sample_sequence, request_index, args.do_sample == "true")
        temperature = repeated_float(temperature_sequence, request_index, args.temperature)
        top_k = repeated_int(top_k_sequence, request_index, args.top_k)
        top_p = repeated_float(top_p_sequence, request_index, args.top_p)
        seed = repeated_int(seed_sequence, request_index, args.seed)
        text = ""
        total_ms = 0.0
        for _ in range(args.iterations):
            start = time.perf_counter()
            text = transcribe(
                model,
                processor,
                audio_path,
                language,
                punctuation,
                max_new_tokens,
                num_beams,
                length_penalty,
                do_sample,
                temperature,
                top_k,
                top_p,
                seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - start) * 1000.0
        wall_ms = total_ms / args.iterations
        print(f"average[{request_index}]")
        print(f"hviske_asr.wall_ms={wall_ms}")
        timing_lines.append(f"hviske_asr.transcribe_wall_ms {wall_ms:.6f}")
        steps.append({
            "request_index": request_index,
            "audio": str(audio),
            "requested_language": language,
            "language": language,
            "text_output": text,
            "word_timestamps": [],
            "metrics": {"wall_ms": wall_ms},
        })

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print("summary_json=" + json.dumps(
        {"family": "hviske_asr", "backend": args.backend, "sequence_steps": steps},
        separators=(",", ":"),
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
