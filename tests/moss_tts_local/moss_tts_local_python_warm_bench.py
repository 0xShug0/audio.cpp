from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WARMUP_TEXT = "This is a fixed warmup request for the MOSS-TTS-Local session benchmark."


def timestamp_seconds_local() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm repeated Python benchmark for MOSS-TTS-Local.")
    parser.add_argument("--model", type=Path, default=Path("models/MOSS-TTS-Local-Transformer-v1.5"))
    parser.add_argument("--text", action="append", dest="texts", default=[])
    parser.add_argument("--request-file", type=Path, default=None)
    parser.add_argument("--clone-audio", type=Path, default=Path(""))
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-text", default=DEFAULT_WARMUP_TEXT)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-new-frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--do-sample", choices=("true", "false"), default="true")
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--audio-top-p", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--use-kv-cache", choices=("true", "false"), default="true")
    parser.add_argument("--audio-out", type=Path, default=Path("moss_tts_local_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--timing-file", type=Path)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"invalid boolean: {value}")


def timing_line(timestamp: str, key: str, value: object) -> str:
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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def mix_hash(key: int, value: int) -> int:
    key ^= int(value) & 0xFFFFFFFF
    key = (key * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return key


def prompt_hash(input_ids: torch.Tensor) -> int:
    flat = input_ids.detach().to(torch.int64).cpu().reshape(-1)
    key = 1469598103934665603
    for value in flat.tolist():
        key = mix_hash(key, int(value))
    return key & 0x7FFFFFFFFFFFFFFF


def prompt_audio_nonpad_count(input_ids: torch.Tensor, audio_pad_token_id: int) -> int:
    audio = input_ids[..., 1:].detach().cpu()
    return int(audio.ne(int(audio_pad_token_id)).sum().item())


def summarize(audio: torch.Tensor, sample_rate: int, text: str) -> dict[str, object]:
    audio = audio.detach().to(torch.float32).cpu()
    flat = audio.reshape(-1)
    min_value = float(flat.min().item()) if flat.numel() else 0.0
    max_value = float(flat.max().item()) if flat.numel() else 0.0
    channels = int(audio.shape[0]) if audio.ndim == 2 else 1
    samples = int(audio.shape[-1]) if audio.ndim >= 1 else 0
    seconds = float(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": samples,
        "audio_seconds": seconds,
        "sum": float(flat.sum(dtype=torch.float64).item()),
        "mean_abs": float(flat.abs().mean(dtype=torch.float64).item()) if flat.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(flat.to(torch.float64).square())).item()) if flat.numel() else 0.0,
        "min": min_value,
        "max": max_value,
        "request_char_count": len(text),
        "first_samples": flat[:32].tolist(),
    }


def load_request_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_file is not None:
        payload = json.loads(resolve_path(args.request_file).read_text(encoding="utf-8"))
        requests = payload.get("requests")
        if not isinstance(requests, list) or not requests:
            raise RuntimeError(f"MOSS-TTS-Local request file has no requests: {args.request_file}")
        specs = [dict(request) for request in requests]
    else:
        texts = list(args.texts) or ["Hello from MOSS-TTS-Local. This benchmark should produce stable speech for comparison."]
        specs = [{"text": text} for text in texts]
    for spec in specs:
        if not spec.get("text"):
            raise RuntimeError("MOSS-TTS-Local request is missing text")
    return specs


def main() -> int:
    args = parse_args()
    request_specs = load_request_specs(args)

    torch.set_num_threads(max(1, args.threads))
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if args.dtype == "fp32" and args.backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    attn = "sdpa" if args.backend == "cuda" else "eager"
    if args.backend == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    from transformers import AutoModel, AutoProcessor

    model_path = resolve_path(args.model)
    processor_kwargs: dict[str, Any] = {}
    if args.dtype == "fp32":
        processor_kwargs["codec_weight_dtype"] = "fp32"
        processor_kwargs["codec_compute_dtype"] = "fp32"
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, **processor_kwargs)
    if args.dtype == "fp32":
        processor.audio_tokenizer = processor.audio_tokenizer.to(device=device, dtype=torch.float32)
        if hasattr(processor.audio_tokenizer, "set_compute_dtype"):
            processor.audio_tokenizer.set_compute_dtype("fp32")
    else:
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        attn_implementation=attn,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    timing_path = args.timing_file or (
        REPO_ROOT / "build" / "logs" / "parity" / "moss_tts_local" / f"moss_tts_local_python_{args.backend}-{timestamp_seconds_local()}.log"
    )
    log_sections: list[tuple[str, list[str]]] = []
    sampling_rate = int(processor.model_config.sampling_rate)

    def request_value(spec: dict[str, Any], key: str, fallback: Any) -> Any:
        return spec[key] if key in spec else fallback

    def run_request(spec: dict[str, Any], use_default_clone_audio: bool) -> tuple[torch.Tensor, dict[str, int]]:
        text = str(spec["text"])
        language = spec.get("language")
        message_kwargs: dict[str, Any] = {"text": text, "language": language}
        clone_audio_path = str(spec.get("voice_ref") or spec.get("clone_audio") or (args.clone_audio if use_default_clone_audio else ""))
        if clone_audio_path:
            message_kwargs["reference"] = [str(resolve_path(Path(clone_audio_path)))]
        conversation = [processor.build_user_message(**message_kwargs)]
        batch = processor([conversation], mode="generation")
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        prompt_stats = {
            "moss_tts_local.prefix_len": int(input_ids.shape[1]),
            "moss_tts_local.prefix_hash": int(prompt_hash(input_ids)),
            "moss_tts_local.prefix_audio_nonpad": int(
                prompt_audio_nonpad_count(input_ids, int(processor.model_config.audio_pad_token_id))
            ),
        }
        do_sample = parse_bool(str(request_value(spec, "do_sample", args.do_sample)).lower())
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(request_value(spec, "max_tokens", request_value(spec, "max_new_frames", args.max_new_frames))),
            "do_sample": do_sample,
            "use_kv_cache": parse_bool(args.use_kv_cache),
            "audio_temperature": float(request_value(spec, "temperature", request_value(spec, "audio_temperature", args.audio_temperature))) if do_sample else 0.0,
            "text_temperature": float(request_value(spec, "text_temperature", args.text_temperature)) if do_sample else 0.0,
            "audio_top_p": float(request_value(spec, "top_p", request_value(spec, "audio_top_p", args.audio_top_p))),
            "audio_top_k": int(request_value(spec, "top_k", request_value(spec, "audio_top_k", args.audio_top_k))),
            "audio_repetition_penalty": float(request_value(spec, "repetition_penalty", request_value(spec, "audio_repetition_penalty", args.audio_repetition_penalty))),
        }
        with torch.no_grad():
            gen_out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            for message in processor.decode(gen_out):
                if message is not None:
                    return message.audio_codes_list[0], prompt_stats
        raise RuntimeError("MOSS-TTS-Local Python warm bench produced no audio")

    warmup_outputs: list[torch.Tensor] = []
    for warmup_index in range(max(0, args.warmup)):
        warmup_spec = {"text": args.warmup_text}
        torch.manual_seed(args.seed)
        if args.backend == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        audio, prompt_stats = run_request(warmup_spec, args.request_file is None)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        seconds = float(audio.shape[-1]) / float(sampling_rate) if sampling_rate > 0 else 0.0
        ts = timestamp_seconds_local()
        log_sections.append((
            f"warmup{warmup_index + 1}",
            [
                timing_line(ts, "moss_tts_local.request_char_count", len(args.warmup_text)),
                timing_line(ts, "moss_tts_local.prefix_len", prompt_stats["moss_tts_local.prefix_len"]),
                timing_line(ts, "moss_tts_local.prefix_hash", prompt_stats["moss_tts_local.prefix_hash"]),
                timing_line(ts, "moss_tts_local.prefix_audio_nonpad", prompt_stats["moss_tts_local.prefix_audio_nonpad"]),
                timing_line(ts, "moss_tts_local.request_wall_ms", wall_ms),
                timing_line(ts, "moss_tts_local.audio_seconds", seconds),
                timing_line(ts, "moss_tts_local.rtf", (wall_ms / 1000.0) / seconds if seconds > 0.0 else 0.0),
            ],
        ))
        warmup_outputs.append(audio.detach().to(torch.float32).cpu())

    last_audios: list[torch.Tensor] = []
    wall_sums = [0.0 for _ in request_specs]
    for request_index, spec in enumerate(request_specs):
        text = str(spec["text"])
        request_audio_out = (
            args.audio_out_dir / f"request_{request_index}.wav"
            if args.audio_out_dir is not None
            else args.audio_out.parent / f"{args.audio_out.stem}_{request_index}{args.audio_out.suffix}"
        )
        request_audio_out.parent.mkdir(parents=True, exist_ok=True)
        last_audio = torch.empty(0)
        for iteration in range(max(1, args.iterations)):
            seed = int(spec.get("seed", args.seed))
            torch.manual_seed(seed)
            if args.backend == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            last_audio, prompt_stats = run_request(spec, args.request_file is None)
            if args.backend == "cuda":
                torch.cuda.synchronize(args.device)
            wall_ms = (time.perf_counter() - started) * 1000.0
            wall_sums[request_index] += wall_ms
            seconds = float(last_audio.shape[-1]) / float(sampling_rate) if sampling_rate > 0 else 0.0
            ts = timestamp_seconds_local()
            log_sections.append((
                f"iteration{iteration + 1}.request{request_index + 1}",
                [
                    timing_line(ts, "moss_tts_local.request_char_count", len(text)),
                    timing_line(ts, "moss_tts_local.prefix_len", prompt_stats["moss_tts_local.prefix_len"]),
                    timing_line(ts, "moss_tts_local.prefix_hash", prompt_stats["moss_tts_local.prefix_hash"]),
                    timing_line(ts, "moss_tts_local.prefix_audio_nonpad", prompt_stats["moss_tts_local.prefix_audio_nonpad"]),
                    timing_line(ts, "moss_tts_local.request_wall_ms", wall_ms),
                    timing_line(ts, "moss_tts_local.audio_seconds", seconds),
                    timing_line(ts, "moss_tts_local.rtf", (wall_ms / 1000.0) / seconds if seconds > 0.0 else 0.0),
                ],
            ))
        torchaudio.save(str(request_audio_out), last_audio.to(torch.float32).cpu(), sampling_rate)
        last_audios.append(last_audio.detach().to(torch.float32).cpu())

    write_sectioned_timing_log(timing_path, log_sections)

    for warmup_index, audio in enumerate(warmup_outputs):
        print(f"warmup_text[{warmup_index}]={args.warmup_text}")
        print(f"warmup_summary_json[{warmup_index}]=" + json.dumps(summarize(audio, sampling_rate, args.warmup_text), ensure_ascii=False))
    for request_index, spec in enumerate(request_specs):
        text = str(spec["text"])
        print(f"text[{request_index}]={text}")
        print(f"summary_json[{request_index}]=" + json.dumps(summarize(last_audios[request_index], sampling_rate, text), ensure_ascii=False))
        if args.audio_out_dir is not None:
            print(f"audio_out[{request_index}]={args.audio_out_dir / f'request_{request_index}.wav'}")
    print(f"timing_out={timing_path}")

    final_out = args.audio_out
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if args.audio_out_dir is not None:
        shutil.copyfile(args.audio_out_dir / f"request_{len(last_audios) - 1}.wav", final_out)
    else:
        torchaudio.save(str(final_out), last_audios[-1], sampling_rate)
    print(f"audio_out={final_out}")

    for request_index, wall_sum in enumerate(wall_sums):
        print(f"average[{request_index}]")
        print(f"moss_tts_local.request_wall_ms={wall_sum / float(max(1, args.iterations))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
