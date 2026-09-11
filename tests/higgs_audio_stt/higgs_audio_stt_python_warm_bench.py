#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."


def install_transformers452_alias() -> Any:
    transformers = importlib.import_module("transformers452")
    sys.modules["transformers"] = transformers
    for name in (
        "configuration_utils",
        "dynamic_module_utils",
        "generation",
        "generation.utils",
        "models",
        "models.auto",
        "models.auto.configuration_auto",
        "models.auto.modeling_auto",
        "models.whisper",
        "models.whisper.modeling_whisper",
        "models.whisper.processing_whisper",
        "models.qwen3",
        "models.qwen3.configuration_qwen3",
        "models.qwen3.modeling_qwen3",
    ):
        try:
            sys.modules[f"transformers.{name}"] = importlib.import_module(f"transformers452.{name}")
        except ModuleNotFoundError:
            pass
    return transformers


TRANSFORMERS = install_transformers452_alias()


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


def read_audio_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def patch_whisper_processor_path(transcribe_module: Any, processor_path: Path) -> None:
    original = transcribe_module.WhisperProcessor.from_pretrained

    def from_pretrained(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai/whisper-large-v3":
            kwargs.setdefault("local_files_only", True)
            return original(str(processor_path), *args, **kwargs)
        return original(name, *args, **kwargs)

    transcribe_module.WhisperProcessor.from_pretrained = from_pretrained


def patch_generation_config_kwargs(model: Any) -> None:
    original = model._prepare_generation_config

    def prepare_generation_config(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        generation_config, model_kwargs = original(*args, **kwargs)
        if not hasattr(generation_config, "generation_kwargs"):
            generation_config.generation_kwargs = {}
        return generation_config, model_kwargs

    if hasattr(model.generation_config, "generation_kwargs"):
        model.generation_config.generation_kwargs.clear()
    else:
        model.generation_config.generation_kwargs = {}
    model._prepare_generation_config = prepare_generation_config


def patch_sample_signature(model: Any) -> None:
    model_class = type(model)
    if getattr(model_class, "_minitts_higgs_sample_patched", False):
        return
    original = model_class._sample

    def sample(
        self: Any,
        input_ids: Any,
        logits_processor: Any,
        stopping_criteria: Any,
        generation_config: Any,
        synced_gpus: bool,
        streamer: Any = None,
        past_key_values_buckets: Any = None,
        **model_kwargs: Any,
    ) -> Any:
        model_kwargs.pop("tokenizer", None)
        return original(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            streamer,
            past_key_values_buckets,
            **model_kwargs,
        )

    model_class._sample = sample
    model_class._minitts_higgs_sample_patched = True


def load_reference(model_path: Path, processor_path: Path, device: torch.device) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(model_path))
    from transformers.configuration_utils import PretrainedConfig  # noqa: E402
    from transformers import AutoModel, AutoTokenizer  # noqa: E402
    import transcribe as higgs_transcribe  # noqa: E402

    patch_whisper_processor_path(higgs_transcribe, processor_path)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    old_repr = PretrainedConfig.__repr__
    PretrainedConfig.__repr__ = lambda self: f"{self.__class__.__name__}(...)"
    try:
        model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map=str(device) if device.type == "cuda" else None,
            local_files_only=True,
        ).eval()
    finally:
        PretrainedConfig.__repr__ = old_repr
    if device.type == "cpu":
        model = model.to(device)
    patch_generation_config_kwargs(model)
    patch_sample_signature(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
    model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
    return model, tokenizer, higgs_transcribe


def transcribe_request(
    model: Any,
    tokenizer: Any,
    transcribe_module: Any,
    audio_path: Path,
    prompt: str,
    enable_thinking: bool,
    max_new_tokens: int,
) -> str:
    audio, sample_rate = read_audio_mono_f32(audio_path)
    return transcribe_module.transcribe(
        model,
        tokenizer,
        audio,
        sample_rate=sample_rate,
        user_prompt=prompt,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Python reference Higgs Audio v3 STT warmbench.")
    parser.add_argument("--model", default="models/higgs-audio-v3-stt")
    parser.add_argument("--whisper-processor", default="models/whisper-large-v3")
    parser.add_argument("--audio", default="resources/sample_16k.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-sequence", default="")
    parser.add_argument("--enable-thinking", choices=["true", "false"], default="true")
    parser.add_argument("--enable-thinking-sequence", default="")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens-sequence", default="")
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
    processor_path = resolve_path(args.whisper_processor)
    model, tokenizer, higgs_transcribe = load_reference(model_path, processor_path, device)

    warmup_audio = resolve_path(args.warmup_audio) if args.warmup_audio else resolve_path(args.audio)
    request_paths = parse_csv_paths(args.audio_sequence, Path(args.audio))
    prompt_sequence = parse_csv_keep_empty(args.prompt_sequence)
    thinking_sequence = parse_csv_keep_empty(args.enable_thinking_sequence)
    max_tokens_sequence = parse_csv_keep_empty(args.max_tokens_sequence)

    base_enable_thinking = args.enable_thinking == "true"
    timing_lines: list[str] = [
        f"higgs_audio_stt.model_root {model_path}",
        f"higgs_audio_stt.whisper_processor {processor_path}",
        f"higgs_audio_stt.backend {args.backend}",
        f"higgs_audio_stt.python_dtype {'bfloat16' if device.type == 'cuda' else 'float32'}",
        f"higgs_audio_stt.transformers_version {TRANSFORMERS.__version__}",
    ]

    for _ in range(args.warmup):
        transcribe_request(
            model,
            tokenizer,
            higgs_transcribe,
            warmup_audio,
            args.prompt,
            base_enable_thinking,
            args.max_tokens,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    steps = []
    for request_index, audio in enumerate(request_paths):
        audio_path = resolve_path(audio)
        prompt = repeated_arg(prompt_sequence, request_index, args.prompt)
        enable_thinking = repeated_bool(thinking_sequence, request_index, base_enable_thinking)
        max_new_tokens = repeated_int(max_tokens_sequence, request_index, args.max_tokens)
        text = ""
        total_ms = 0.0
        for _ in range(args.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            text = transcribe_request(
                model,
                tokenizer,
                higgs_transcribe,
                audio_path,
                prompt,
                enable_thinking,
                max_new_tokens,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - start) * 1000.0
        wall_ms = total_ms / args.iterations
        print(f"average[{request_index}]")
        print(f"higgs_audio_stt.wall_ms={wall_ms}")
        timing_lines.append(f"higgs_audio_stt.request{request_index}.wall_ms {wall_ms:.6f}")
        timing_lines.append(f"higgs_audio_stt.request{request_index}.enable_thinking {1 if enable_thinking else 0}")
        timing_lines.append(f"higgs_audio_stt.request{request_index}.max_tokens {max_new_tokens}")
        steps.append({
            "request_index": request_index,
            "audio": str(audio),
            "language": "",
            "text_output": text,
            "word_timestamps": [],
            "metrics": {
                "wall_ms": wall_ms,
                "enable_thinking": enable_thinking,
                "max_tokens": max_new_tokens,
            },
        })

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print("summary_json=" + json.dumps(
        {"family": "higgs_audio_stt", "backend": args.backend, "sequence_steps": steps},
        separators=(",", ":"),
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
