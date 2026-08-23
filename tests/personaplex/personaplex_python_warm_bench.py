#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "PersonaPlex" / "moshi"
DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
DEFAULT_INPUT_WAV = REPO_ROOT / "reference" / "PersonaPlex" / "assets" / "test" / "input_assistant.wav"
DEFAULT_ASSISTANT_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
)


@dataclass
class Request:
    request_id: str
    input_wav: Path
    text_prompt: str
    voice_prompt: str
    seed: int
    temp_audio: float
    temp_text: float
    topk_audio: int
    topk_text: int
    greedy: bool


@dataclass
class Runtime:
    mimi: Any
    other_mimi: Any
    lm_gen: Any
    text_tokenizer: Any
    sample_rate: int
    frame_size: int
    device: str
    offline: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference PersonaPlex warmbench.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--moshi-weight", type=Path, default=None)
    parser.add_argument("--mimi-weight", type=Path, default=None)
    parser.add_argument("--voice-prompt-dir", type=Path, default=None)
    parser.add_argument("--voice-prompt", default="NATF2.pt")
    parser.add_argument("--text-prompt", default=DEFAULT_ASSISTANT_PROMPT)
    parser.add_argument("--input-wav", type=Path, default=DEFAULT_INPUT_WAV)
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-input-wav", type=Path, default=DEFAULT_INPUT_WAV)
    parser.add_argument("--warmup-text-prompt", default=DEFAULT_ASSISTANT_PROMPT)
    parser.add_argument("--warmup-voice-prompt", default="NATF2.pt")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--topk-audio", type=int, default=250)
    parser.add_argument("--topk-text", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--audio-out", type=Path, default=Path("personaplex_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--text-out-dir", type=Path, default=None)
    parser.add_argument("--timing-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return path
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_voice_prompt_path(voice_prompt_dir: str, voice_prompt: str) -> Path:
    path = Path(voice_prompt)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.is_file():
        return repo_path
    return Path(voice_prompt_dir) / path


def import_reference() -> Any:
    if not (REFERENCE_ROOT / "moshi" / "offline.py").is_file():
        raise RuntimeError(f"missing PersonaPlex reference code: {REFERENCE_ROOT}")
    sys.path.insert(0, str(REFERENCE_ROOT))
    from moshi import offline

    return offline


def seed_all(seed: int, device: str) -> None:
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def optional_file(path: Path | None, model_dir: Path | None, filename: str) -> str | None:
    if path is not None:
        resolved = resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"missing file: {resolved}")
        return str(resolved)
    if model_dir is not None:
        candidate = resolve_path(model_dir) / filename
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_voice_prompt_dir(offline: Any, args: argparse.Namespace) -> str:
    if args.voice_prompt_dir:
        resolved = resolve_path(args.voice_prompt_dir)
        if not resolved.is_dir():
            raise FileNotFoundError(f"missing voice prompt directory: {resolved}")
        return str(resolved)
    model_dir = resolve_path(args.model) if args.model is not None else None
    if model_dir is not None:
        for candidate in (model_dir / "voices", model_dir / "voice_prompts"):
            if candidate.is_dir():
                return str(candidate)
    return offline._get_voice_prompt_dir(None, args.hf_repo)


def load_runtime(args: argparse.Namespace, voice_prompt_dir: str, offline: Any) -> Runtime:
    loaders = offline.loaders
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PersonaPlex warmbench requested CUDA, but torch.cuda.is_available() is false")
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    model_dir = resolve_path(args.model) if args.model is not None else None
    tokenizer_path = optional_file(args.tokenizer, model_dir, loaders.TEXT_TOKENIZER_NAME)
    moshi_weight = optional_file(args.moshi_weight, model_dir, loaders.MOSHI_NAME)
    mimi_weight = optional_file(args.mimi_weight, model_dir, loaders.MIMI_NAME)

    # Download config.json to mirror the reference offline path and verify repo access before timing.
    offline.hf_hub_download(args.hf_repo, "config.json")
    if mimi_weight is None:
        mimi_weight = offline.hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if tokenizer_path is None:
        tokenizer_path = offline.hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    if moshi_weight is None:
        moshi_weight = offline.hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)

    print(f"[TRACE] personaplex.reference_root {REFERENCE_ROOT}")
    print(f"[TRACE] personaplex.hf_repo {args.hf_repo}")
    print(f"[TRACE] personaplex.device {device}")
    print(f"[TRACE] personaplex.voice_prompt_dir {voice_prompt_dir}")
    print(f"[TRACE] personaplex.moshi_weight {moshi_weight}")
    print(f"[TRACE] personaplex.mimi_weight {mimi_weight}")
    print(f"[TRACE] personaplex.tokenizer {tokenizer_path}")

    started = time.perf_counter()
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    text_tokenizer = offline.sentencepiece.SentencePieceProcessor(tokenizer_path)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=args.cpu_offload)
    lm.eval()
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = offline.LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        use_sampling=not args.greedy,
        temp=args.temp_audio,
        temp_text=args.temp_text,
        top_k=args.topk_audio,
        top_k_text=args.topk_text,
    )
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    synchronize(device)
    load_ms = (time.perf_counter() - started) * 1000.0
    print(f"[TIMING] personaplex.model_load_ms {load_ms:.6f}")

    started = time.perf_counter()
    offline.warmup(mimi, other_mimi, lm_gen, device, frame_size)
    synchronize(device)
    warmup_ms = (time.perf_counter() - started) * 1000.0
    print(f"[TIMING] personaplex.graph_warmup_ms {warmup_ms:.6f}")

    return Runtime(
        mimi=mimi,
        other_mimi=other_mimi,
        lm_gen=lm_gen,
        text_tokenizer=text_tokenizer,
        sample_rate=int(mimi.sample_rate),
        frame_size=frame_size,
        device=device,
        offline=offline,
    )


def load_user_audio(runtime: Runtime, input_wav: Path) -> np.ndarray:
    resolved = resolve_path(input_wav)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing input WAV: {resolved}")
    return runtime.offline.lm_load_audio(str(resolved), runtime.sample_rate)


def request_from_dict(args: argparse.Namespace, value: dict[str, Any], index: int) -> Request:
    return Request(
        request_id=str(value.get("id") or value.get("request_id") or f"request{index + 1}"),
        input_wav=Path(value.get("input_wav") or value.get("audio") or str(args.input_wav)),
        text_prompt=str(value.get("text_prompt") or args.text_prompt),
        voice_prompt=str(value.get("voice_prompt") or args.voice_prompt),
        seed=int(value.get("seed", args.seed)),
        temp_audio=float(value.get("temp_audio", args.temp_audio)),
        temp_text=float(value.get("temp_text", args.temp_text)),
        topk_audio=int(value.get("topk_audio", args.topk_audio)),
        topk_text=int(value.get("topk_text", args.topk_text)),
        greedy=bool(value.get("greedy", args.greedy)),
    )


def load_requests(args: argparse.Namespace) -> list[Request]:
    if args.request_sequence_json:
        payload = json.loads(args.request_sequence_json)
        if not isinstance(payload, list):
            raise RuntimeError("--request-sequence-json must decode to a list")
        return [request_from_dict(args, item, index) for index, item in enumerate(payload)]
    if args.request_json:
        payload = json.loads(args.request_json)
        if not isinstance(payload, dict):
            raise RuntimeError("--request-json must decode to an object")
        return [request_from_dict(args, payload, 0)]
    return [
        Request(
            request_id="request1",
            input_wav=args.input_wav,
            text_prompt=args.text_prompt,
            voice_prompt=args.voice_prompt,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=args.greedy,
        )
    ]


def decode_text_token(runtime: Runtime, token_id: int) -> str:
    if token_id in (0, 1, 2, 3):
        return ("EPAD", "BOS", "EOS", "PAD")[token_id]
    return runtime.text_tokenizer.id_to_piece(token_id).replace("▁", " ")


def run_request(
    runtime: Runtime,
    request: Request,
    voice_prompt_dir: str,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    voice_prompt_path = resolve_voice_prompt_path(voice_prompt_dir, request.voice_prompt)
    if not voice_prompt_path.is_file():
        raise FileNotFoundError(f"missing voice prompt: {voice_prompt_path}")
    user_audio = load_user_audio(runtime, request.input_wav)
    total_target_samples = int(user_audio.shape[-1])

    runtime.lm_gen.use_sampling = not request.greedy
    runtime.lm_gen.temp = request.temp_audio
    runtime.lm_gen.temp_text = request.temp_text
    runtime.lm_gen.top_k = request.topk_audio
    runtime.lm_gen.top_k_text = request.topk_text
    seed_all(request.seed, runtime.device)

    synchronize(runtime.device)
    started = time.perf_counter()
    if str(voice_prompt_path).endswith(".pt"):
        runtime.lm_gen.load_voice_prompt_embeddings(str(voice_prompt_path))
    else:
        runtime.lm_gen.load_voice_prompt(str(voice_prompt_path))
    runtime.lm_gen.text_prompt_tokens = runtime.text_tokenizer.encode(
        runtime.offline.wrap_with_system_tags(request.text_prompt)
    )

    runtime.mimi.reset_streaming()
    runtime.other_mimi.reset_streaming()
    runtime.lm_gen.reset_streaming()
    runtime.lm_gen.step_system_prompts(runtime.mimi)
    runtime.mimi.reset_streaming()
    synchronize(runtime.device)
    prompt_ms = (time.perf_counter() - started) * 1000.0

    generated_frames: list[np.ndarray] = []
    generated_text_tokens: list[str] = []
    started = time.perf_counter()
    for user_encoded in runtime.offline.lm_encode_from_sphn(
        runtime.mimi,
        runtime.offline.lm_iterate_audio(
            user_audio,
            sample_interval_size=runtime.frame_size,
            pad=True,
        ),
        max_batch=1,
    ):
        steps = int(user_encoded.shape[-1])
        for step_index in range(steps):
            tokens = runtime.lm_gen.step(user_encoded[:, :, step_index : step_index + 1])
            if tokens is None:
                continue
            pcm = runtime.offline.decode_tokens_to_pcm(
                runtime.mimi,
                runtime.other_mimi,
                runtime.lm_gen,
                tokens,
            )
            generated_frames.append(pcm)
            generated_text_tokens.append(decode_text_token(runtime, int(tokens[0, 0, 0].item())))
    synchronize(runtime.device)
    generate_ms = (time.perf_counter() - started) * 1000.0

    if not generated_frames:
        raise RuntimeError("PersonaPlex produced no audio frames")
    output_pcm = np.concatenate(generated_frames, axis=-1)
    if output_pcm.shape[-1] > total_target_samples:
        output_pcm = output_pcm[:total_target_samples]
    elif output_pcm.shape[-1] < total_target_samples:
        pad_len = total_target_samples - output_pcm.shape[-1]
        output_pcm = np.concatenate([output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1)

    wall_ms = prompt_ms + generate_ms
    duration_ms = 1000.0 * total_target_samples / float(runtime.sample_rate)
    return output_pcm.astype(np.float32, copy=False), generated_text_tokens, {
        "prompt_ms": prompt_ms,
        "generate_ms": generate_ms,
        "wall_ms": wall_ms,
        "audio_duration_ms": duration_ms,
        "rtf": wall_ms / max(duration_ms, 1e-6),
        "x_realtime": duration_ms / max(wall_ms, 1e-6),
    }


def write_timing(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def output_path(args: argparse.Namespace, request_id: str, suffix: str) -> Path:
    if suffix == ".wav" and args.audio_out_dir:
        return resolve_path(args.audio_out_dir) / f"{request_id}.wav"
    if suffix == ".json" and args.text_out_dir:
        return resolve_path(args.text_out_dir) / f"{request_id}.json"
    if suffix == ".wav":
        return resolve_path(args.audio_out)
    return resolve_path(args.audio_out).with_suffix(".json")


def main() -> int:
    args = parse_args()
    offline = import_reference()
    voice_prompt_dir = resolve_voice_prompt_dir(offline, args)
    runtime = load_runtime(args, voice_prompt_dir, offline)

    warmup_request = Request(
        request_id="warmup",
        input_wav=args.warmup_input_wav,
        text_prompt=args.warmup_text_prompt,
        voice_prompt=args.warmup_voice_prompt,
        seed=args.seed,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        topk_audio=args.topk_audio,
        topk_text=args.topk_text,
        greedy=args.greedy,
    )
    for index in range(max(0, args.warmup)):
        _, _, timing = run_request(runtime, warmup_request, voice_prompt_dir)
        print(
            f"[TIMING] warmup{index + 1}.wall_ms {timing['wall_ms']:.6f} "
            f"rtf {timing['rtf']:.6f} x_realtime {timing['x_realtime']:.6f}"
        )

    requests = load_requests(args)
    timing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for iteration in range(max(1, args.iterations)):
        for request in requests:
            request_id = request.request_id if args.iterations == 1 else f"{request.request_id}_iter{iteration + 1}"
            audio, text_tokens, timing = run_request(runtime, request, voice_prompt_dir)
            wav_path = output_path(args, request_id, ".wav")
            text_path = output_path(args, request_id, ".json")
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            runtime.offline.sphn.write_wav(str(wav_path), audio, runtime.sample_rate)
            text_path.write_text(json.dumps(text_tokens, ensure_ascii=False) + "\n", encoding="utf-8")

            row = {
                "request": request_id,
                "input_wav": str(resolve_path(request.input_wav)),
                "voice_prompt": request.voice_prompt,
                "sample_rate": runtime.sample_rate,
                **timing,
            }
            timing_rows.append(row)
            summary_rows.append(
                {
                    **row,
                    "output_wav": str(wav_path),
                    "output_text": str(text_path),
                    "samples": int(audio.shape[-1]),
                    "text_token_count": len(text_tokens),
                    "mean_abs": float(np.abs(audio).mean()) if audio.size else 0.0,
                    "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
                }
            )
            print(
                f"[TIMING] request.{request_id}.prompt_ms {timing['prompt_ms']:.6f}\n"
                f"[TIMING] request.{request_id}.generate_ms {timing['generate_ms']:.6f}\n"
                f"[TIMING] request.{request_id}.wall_ms {timing['wall_ms']:.6f}\n"
                f"[TIMING] request.{request_id}.audio_duration_ms {timing['audio_duration_ms']:.6f}\n"
                f"[TIMING] request.{request_id}.rtf {timing['rtf']:.6f}\n"
                f"[TIMING] request.{request_id}.x_realtime {timing['x_realtime']:.6f}"
            )

    write_timing(args.timing_file, timing_rows)
    write_summary(args.summary_file, summary_rows)
    sequence_steps: list[dict[str, Any]] = []
    for row in summary_rows:
        sequence_steps.append({
            "id": row["request"],
            "stems": [{
                "name": "audio",
                "audio": row["output_wav"],
                "summary": {
                    "sample_rate": row["sample_rate"],
                    "channels": 1,
                    "frames": row["samples"],
                    "samples": row["samples"],
                    "mean_abs": row["mean_abs"],
                    "rms": row["rms"],
                },
            }],
            "metrics": {
                "prompt_ms": row["prompt_ms"],
                "generate_ms": row["generate_ms"],
                "wall_ms": row["wall_ms"],
                "audio_duration_ms": row["audio_duration_ms"],
                "rtf": row["rtf"],
                "x_realtime": row["x_realtime"],
            },
        })
    print("summary_json=" + json.dumps(
        {"family": "personaplex", "backend": args.backend, "sequence_steps": sequence_steps},
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
