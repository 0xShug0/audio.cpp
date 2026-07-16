from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import transformers457
import transformers457.dynamic_module_utils as transformers457_dynamic_module_utils
from transformers457 import AutoConfig, AutoModelForCausalLM

sys.modules.setdefault("transformers", transformers457)
transformers457_dynamic_module_utils.transformers = transformers457


REPO_ROOT = Path(__file__).resolve().parents[2]
kDefaultWarmupText = "This is a fixed warmup request for the MOSS-TTS-Nano session benchmark."


def timestamp_seconds_local() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm repeated Python benchmark for MOSS-TTS-Nano.")
    parser.add_argument("--model", type=Path, default=Path("models/MOSS-TTS-Nano-100M"))
    parser.add_argument("--audio-tokenizer-model", type=Path, default=Path("models/MOSS-Audio-Tokenizer-Nano"))
    parser.add_argument("--text", action="append", dest="texts", default=[])
    parser.add_argument("--clone-audio", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-text", default=kDefaultWarmupText)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-new-frames", type=int, default=300)
    parser.add_argument("--active-codebooks", type=int, default=16)
    parser.add_argument("--do-sample", choices=("true", "false"), default="true")
    parser.add_argument("--text-temperature", type=float, default=1.5)
    parser.add_argument("--text-top-p", type=float, default=1.0)
    parser.add_argument("--text-top-k", type=int, default=50)
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--audio-top-p", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--use-kv-cache", choices=("true", "false"), default="true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=50)
    parser.add_argument("--voice-clone-max-memory-per-sample-gb", type=float, default=1.0)
    parser.add_argument("--tts-max-batch-size", type=int, default=0)
    parser.add_argument("--codec-max-batch-size", type=int, default=0)
    parser.add_argument("--audio-out", type=Path, default=Path("moss_tts_nano_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--timing-file", type=Path)
    parser.add_argument("--artifact-stamp", default="")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"invalid boolean: {value}")


def summarize(audio: np.ndarray, sample_rate: int, text: str) -> dict[str, object]:
    audio = audio.astype(np.float32, copy=False)
    return {
        "sample_rate": sample_rate,
        "channels": 1 if audio.ndim == 1 else int(audio.shape[0]),
        "samples": int(audio.shape[-1]),
        "sum": float(audio.sum(dtype=np.float64)),
        "mean_abs": float(np.abs(audio).mean(dtype=np.float64)) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
        "min": float(audio.min()) if audio.size else 0.0,
        "max": float(audio.max()) if audio.size else 0.0,
        "request_char_count": len(text),
        "first_samples": audio.reshape(-1)[:32].tolist(),
    }


def timing_line(timestamp: str, key: str, value: object) -> str:
    if isinstance(value, str):
        return f"[TIMING ts={timestamp}] {key} {value}"
    if isinstance(value, bool):
        return f"[TIMING ts={timestamp}] {key} {1 if value else 0}"
    if isinstance(value, int):
        return f"[TIMING ts={timestamp}] {key} {value}"
    return f"[TIMING ts={timestamp}] {key} {float(value):.6f}"


def write_sectioned_timing_log(path: Path, sections: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, (name, lines) in enumerate(sections):
            output.write(f"[{name}]\n")
            for line in lines:
                output.write(f"{line}\n")
            if index + 1 < len(sections):
                output.write("\n")


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    elif waveform.ndim == 2 and waveform.shape[0] <= waveform.shape[1]:
        waveform = waveform.T
    sf.write(str(path), waveform, int(sample_rate))


def force_attention_implementation(module: object, value: str) -> None:
    if hasattr(module, "attn_implementation"):
        setattr(module, "attn_implementation", value)
    if hasattr(module, "config"):
        config = getattr(module, "config")
        if hasattr(config, "attn_implementation"):
            setattr(config, "attn_implementation", value)
        if hasattr(config, "local_transformer_attn_implementation"):
            setattr(config, "local_transformer_attn_implementation", value)
        if hasattr(config, "gpt2_config") and getattr(config, "gpt2_config") is not None:
            gpt2_config = getattr(config, "gpt2_config")
            if hasattr(gpt2_config, "_attn_implementation"):
                setattr(gpt2_config, "_attn_implementation", value)
            if hasattr(gpt2_config, "attn_implementation"):
                setattr(gpt2_config, "attn_implementation", value)
    named_children = getattr(module, "named_children", None)
    if callable(named_children):
        for _, child in named_children():
            force_attention_implementation(child, value)


def main() -> int:
    args = parse_args()
    texts = list(args.texts)
    if not texts:
        texts = ["Hello from MOSS-TTS-Nano. This benchmark should produce stable cloned speech for comparison."]

    stamp = args.artifact_stamp or timestamp_seconds_local()
    timing_path = args.timing_file or (
        REPO_ROOT / "build" / "logs" / "parity" / "moss_tts_nano" / f"moss_tts_python_{args.backend}-{stamp}.log"
    )

    import torch
    torch.set_num_threads(max(1, args.threads))
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")

    model_config = AutoConfig.from_pretrained(
        str(args.model.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    attention_impl = "sdpa" if args.backend == "cuda" else "eager"
    model_config.attn_implementation = attention_impl
    model_config.local_transformer_attn_implementation = attention_impl
    if getattr(model_config, "gpt2_config", None) is not None:
        model_config.gpt2_config._attn_implementation = attention_impl
    use_safetensors = not (args.model.resolve() / "pytorch_model.bin").is_file()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model.resolve()),
        config=model_config,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=use_safetensors,
    )
    force_attention_implementation(model, attention_impl)
    model = model.to(device)
    model.eval()

    clone_audio = str((REPO_ROOT / args.clone_audio).resolve()) if not args.clone_audio.is_absolute() else str(args.clone_audio)
    resolved_model_path = args.model.resolve() if not args.model.is_absolute() else args.model
    tokenizer_model_path = resolved_model_path / "tokenizer.model"
    text_tokenizer_path = str(tokenizer_model_path if tokenizer_model_path.is_file() else resolved_model_path)
    audio_tokenizer_path = (
        str((REPO_ROOT / args.audio_tokenizer_model).resolve())
        if not args.audio_tokenizer_model.is_absolute()
        else str(args.audio_tokenizer_model)
    )

    def run_voice_clone_request(text: str, output_audio_path: Path) -> dict[str, object]:
        return model.inference(
            text=text,
            output_audio_path=str(output_audio_path),
            mode="voice_clone",
            prompt_audio_path=clone_audio,
            text_tokenizer_path=text_tokenizer_path,
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_path,
            device=device,
            nq=args.active_codebooks,
            max_new_frames=args.max_new_frames,
            do_sample=parse_bool(args.do_sample),
            text_temperature=args.text_temperature,
            text_top_p=args.text_top_p,
            text_top_k=args.text_top_k,
            audio_temperature=args.audio_temperature,
            audio_top_p=args.audio_top_p,
            audio_top_k=args.audio_top_k,
            audio_repetition_penalty=args.audio_repetition_penalty,
            use_kv_cache=parse_bool(args.use_kv_cache),
            voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
            voice_clone_max_memory_per_sample_gb=args.voice_clone_max_memory_per_sample_gb,
            tts_max_batch_size=args.tts_max_batch_size,
            codec_max_batch_size=args.codec_max_batch_size,
        )

    log_sections: list[tuple[str, list[str]]] = []
    warmup_outputs: list[tuple[np.ndarray, int]] = []
    for warmup_index in range(max(0, args.warmup)):
        warmup_audio_out = (
            args.audio_out.parent / f"{args.audio_out.stem}_warmup_{warmup_index}{args.audio_out.suffix}"
        )
        torch.cuda.synchronize(args.device) if args.backend == "cuda" else None
        torch.manual_seed(args.seed)
        if args.backend == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        started = time.perf_counter()
        result = run_voice_clone_request(args.warmup_text, warmup_audio_out)
        torch.cuda.synchronize(args.device) if args.backend == "cuda" else None
        wall_ms = (time.perf_counter() - started) * 1000.0
        audio, sample_rate = sf.read(str(warmup_audio_out), always_2d=True, dtype="float32")
        warmup_outputs.append((audio.T, int(sample_rate)))
        ts = timestamp_seconds_local()
        log_sections.append((
            f"warmup{warmup_index + 1}",
            [
                timing_line(ts, "moss_tts_nano.request_char_count", len(args.warmup_text)),
                timing_line(ts, "moss_tts_nano.request_wall_ms", wall_ms),
                timing_line(ts, "moss_tts_nano.generated_frames", int(result["audio_token_ids"].shape[0])),
            ],
        ))

    last_audios: list[np.ndarray] = []
    last_sample_rates: list[int] = []
    wall_sums = [0.0 for _ in texts]
    last_wall_ms = [0.0 for _ in texts]
    for request_index, text in enumerate(texts):
        request_audio_out = (
            args.audio_out_dir / f"request_{request_index}.wav"
            if args.audio_out_dir is not None
            else args.audio_out.parent / f"{args.audio_out.stem}_{request_index}{args.audio_out.suffix}"
        )
        request_audio_out.parent.mkdir(parents=True, exist_ok=True)
        for iteration in range(max(1, args.iterations)):
            torch.cuda.synchronize(args.device) if args.backend == "cuda" else None
            torch.manual_seed(args.seed)
            if args.backend == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            started = time.perf_counter()
            result = run_voice_clone_request(text, request_audio_out)
            torch.cuda.synchronize(args.device) if args.backend == "cuda" else None
            last_wall_ms[request_index] = (time.perf_counter() - started) * 1000.0
            wall_sums[request_index] += last_wall_ms[request_index]
            audio, sample_rate = sf.read(str(request_audio_out), always_2d=True, dtype="float32")
            if request_index >= len(last_audios):
                last_audios.append(audio.T)
                last_sample_rates.append(int(sample_rate))
            else:
                last_audios[request_index] = audio.T
                last_sample_rates[request_index] = int(sample_rate)
            ts = timestamp_seconds_local()
            log_sections.append((
                f"iteration{iteration + 1}.request{request_index + 1}",
                [
                    timing_line(ts, "moss_tts_nano.request_char_count", len(text)),
                    timing_line(ts, "moss_tts_nano.request_wall_ms", last_wall_ms[request_index]),
                    timing_line(ts, "moss_tts_nano.generated_frames", int(result["audio_token_ids"].shape[0])),
                ],
            ))

    write_sectioned_timing_log(timing_path, log_sections)

    for warmup_index, (audio, sample_rate) in enumerate(warmup_outputs):
        print(f"warmup_text[{warmup_index}]={args.warmup_text}")
        print(f"warmup_summary_json[{warmup_index}]={json.dumps(summarize(audio, sample_rate, args.warmup_text), ensure_ascii=False)}")
    for request_index, text in enumerate(texts):
        print(f"text[{request_index}]={text}")
        print(f"summary_json[{request_index}]={json.dumps(summarize(last_audios[request_index], last_sample_rates[request_index], text), ensure_ascii=False)}")
        if len(texts) == 1 and request_index == 0:
            print(f"text={text}")
            print(f"summary_json={json.dumps(summarize(last_audios[request_index], last_sample_rates[request_index], text), ensure_ascii=False)}")
    print(f"timing_out={timing_path}")

    if args.audio_out_dir is not None:
        args.audio_out_dir.mkdir(parents=True, exist_ok=True)
        for request_index in range(len(last_audios)):
            print(f"audio_out[{request_index}]={args.audio_out_dir / f'request_{request_index}.wav'}")

    final_out = args.audio_out
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if args.audio_out_dir is not None:
        shutil.copyfile(args.audio_out_dir / f"request_{len(last_audios) - 1}.wav", final_out)
    else:
        write_audio(final_out, last_audios[-1], last_sample_rates[-1])
    print(f"audio_out={final_out}")

    for request_index, wall_sum in enumerate(wall_sums):
        print(f"average[{request_index}]")
        print(f"moss_tts_nano.request_wall_ms={wall_sum / float(max(1, args.iterations))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
