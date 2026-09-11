#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "VibeVoice"


def install_transformers457_alias() -> Any:
    transformers = importlib.import_module("transformers457")
    sys.modules["transformers"] = transformers
    for name in (
        "configuration_utils",
        "dynamic_module_utils",
        "generation",
        "generation.utils",
        "modeling_outputs",
        "modeling_utils",
        "models",
        "models.auto",
        "models.auto.auto_factory",
        "models.auto.configuration_auto",
        "models.auto.modeling_auto",
        "models.auto.modeling_flax_auto",
        "models.auto.modeling_tf_auto",
        "models.qwen2",
        "models.qwen2.configuration_qwen2",
        "models.qwen2.modeling_qwen2",
        "models.qwen2.tokenization_qwen2",
        "models.qwen2.tokenization_qwen2_fast",
        "tokenization_utils",
        "tokenization_utils_base",
        "tokenization_utils_fast",
        "utils",
        "utils.logging",
    ):
        try:
            sys.modules[f"transformers.{name}"] = importlib.import_module(f"transformers457.{name}")
        except ModuleNotFoundError:
            pass
    return transformers


TRANSFORMERS = install_transformers457_alias()


def make_register_idempotent() -> None:
    from transformers.models.auto import AutoModel, AutoModelForCausalLM

    for cls in (AutoModel, AutoModelForCausalLM):
        original = cls.register

        def register(config_class: Any, model_class: Any, exist_ok: bool = False, *, _original: Any = original) -> Any:
            return _original(config_class, model_class, exist_ok=True)

        cls.register = register


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_attention_impl(value: str, device: torch.device) -> str:
    if value != "auto":
        return value
    if device.type == "cuda":
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"


def load_reference(model_path: Path, tokenizer_path: Path, device: torch.device, attention_impl: str) -> tuple[Any, Any]:
    sys.path.insert(0, str(REFERENCE_ROOT))
    make_register_idempotent()
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )
    processor = VibeVoiceASRProcessor(
        tokenizer=tokenizer,
        audio_processor=VibeVoiceTokenizerProcessor(
            sampling_rate=24000,
            normalize_audio=True,
            target_dB_FS=-25,
            eps=1e-6,
        ),
        speech_tok_compress_ratio=3200,
        target_sample_rate=24000,
        normalize_audio=True,
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        attn_implementation=attention_impl,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device).eval()
    return model, processor


def generation_kwargs(
    processor: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if num_beams > 1:
        kwargs["num_beams"] = num_beams
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = temperature > 0.0
        if temperature > 0.0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            kwargs["top_k"] = top_k
    return kwargs


def normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        out.append({
            "start_time": str(segment.get("start_time", "")),
            "end_time": str(segment.get("end_time", "")),
            "speaker_id": str(segment.get("speaker_id", "")),
            "text": text,
        })
    return out


def transcribe_request(
    model: Any,
    processor: Any,
    audio_path: Path,
    context_info: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    seed: int,
    device: torch.device,
) -> tuple[str, str, list[dict[str, Any]]]:
    seed_everything(seed)
    inputs = processor(
        audio=str(audio_path),
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info or None,
    )
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs(processor, max_new_tokens, temperature, top_p, top_k, num_beams),
        )
    input_length = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[0, input_length:]
    eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        generated_ids = generated_ids[: int(eos_positions[0]) + 1]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    segments = normalize_segments(processor.post_process_transcription(raw_text))
    text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
    if not text:
        text = raw_text.strip()
    return raw_text, text, segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Python reference VibeVoice-ASR warmbench.")
    parser.add_argument("--model", default="models/VibeVoice-ASR")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--audio", default="resources/sample_16k.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens-sequence", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature-sequence", default="")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-p-sequence", default="")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-k-sequence", default="")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-beams-sequence", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--context-sequence", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seed-sequence", default="")
    parser.add_argument("--attn-implementation", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto")
    parser.add_argument("--timing-file", default="")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    model_path = resolve_path(args.model)
    tokenizer_path = resolve_path(args.tokenizer or args.model)
    attention_impl = pick_attention_impl(args.attn_implementation, device)
    model, processor = load_reference(model_path, tokenizer_path, device, attention_impl)

    warmup_audio = resolve_path(args.warmup_audio) if args.warmup_audio else resolve_path(args.audio)
    request_paths = parse_csv_paths(args.audio_sequence, Path(args.audio))
    context_sequence = parse_csv_keep_empty(args.context_sequence)
    max_tokens_sequence = parse_csv_keep_empty(args.max_tokens_sequence)
    temperature_sequence = parse_csv_keep_empty(args.temperature_sequence)
    top_p_sequence = parse_csv_keep_empty(args.top_p_sequence)
    top_k_sequence = parse_csv_keep_empty(args.top_k_sequence)
    num_beams_sequence = parse_csv_keep_empty(args.num_beams_sequence)
    seed_sequence = parse_csv_keep_empty(args.seed_sequence)

    timing_lines: list[str] = [
        f"vibevoice_asr.model_root {model_path}",
        f"vibevoice_asr.tokenizer_root {tokenizer_path}",
        f"vibevoice_asr.reference_root {REFERENCE_ROOT}",
        f"vibevoice_asr.backend {args.backend}",
        f"vibevoice_asr.python_dtype {'bfloat16' if device.type == 'cuda' else 'float32'}",
        f"vibevoice_asr.attn_implementation {attention_impl}",
        f"vibevoice_asr.transformers_version {TRANSFORMERS.__version__}",
    ]

    for warmup_index in range(args.warmup):
        transcribe_request(
            model,
            processor,
            warmup_audio,
            args.context,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.num_beams,
            args.seed + warmup_index,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    steps = []
    for request_index, audio in enumerate(request_paths):
        audio_path = resolve_path(audio)
        context_info = repeated_arg(context_sequence, request_index, args.context)
        max_new_tokens = repeated_int(max_tokens_sequence, request_index, args.max_tokens)
        temperature = repeated_float(temperature_sequence, request_index, args.temperature)
        top_p = repeated_float(top_p_sequence, request_index, args.top_p)
        top_k = repeated_int(top_k_sequence, request_index, args.top_k)
        num_beams = repeated_int(num_beams_sequence, request_index, args.num_beams)
        seed = repeated_int(seed_sequence, request_index, args.seed + request_index)
        raw_text = ""
        text = ""
        segments: list[dict[str, Any]] = []
        total_ms = 0.0
        for iteration in range(args.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            raw_text, text, segments = transcribe_request(
                model,
                processor,
                audio_path,
                context_info,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                num_beams,
                seed + iteration,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - start) * 1000.0
        wall_ms = total_ms / args.iterations
        print(f"average[{request_index}]")
        print(f"vibevoice_asr.wall_ms={wall_ms}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.wall_ms {wall_ms:.6f}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.max_tokens {max_new_tokens}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.temperature {temperature:.6f}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.top_p {top_p:.6f}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.top_k {top_k}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.num_beams {num_beams}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.seed {seed}")
        timing_lines.append(f"vibevoice_asr.request{request_index}.context {context_info}")
        steps.append({
            "request_index": request_index,
            "audio": str(audio),
            "language": "",
            "text_output": text,
            "raw_text_output": raw_text,
            "word_timestamps": [],
            "speech_segments": segments,
            "metrics": {
                "wall_ms": wall_ms,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_beams": num_beams,
                "seed": seed,
                "segment_count": len(segments),
            },
        })

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print("summary_json=" + json.dumps(
        {"family": "vibevoice_asr", "backend": args.backend, "sequence_steps": steps},
        separators=(",", ":"),
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
