#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "DramaBox"
DEFAULT_MODEL = REPO_ROOT / "models" / "Dramabox"
DEFAULT_GEMMA = REPO_ROOT / "models" / "gemma-3-12b-it-bnb-4bit"
DEFAULT_VOICE = REFERENCE_ROOT / "assets" / "voices" / "female_american.wav"
DEFAULT_PROMPT = (
    'A woman speaks warmly, "Hello, how are you today?" '
    'She laughs, "Hahaha, it is so good to see you!"'
)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def add_reference_paths(reference_root: Path) -> None:
    root = resolve_path(reference_root)
    server_path = root / "src" / "inference_server.py"
    if not server_path.is_file():
        raise RuntimeError(f"missing DramaBox reference inference server: {server_path}")
    sys.path.insert(0, str(root / "ltx2"))
    sys.path.insert(0, str(root / "src"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference DramaBox warmbench.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--gemma-root", default=str(DEFAULT_GEMMA))
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--voice-ref", default=str(DEFAULT_VOICE))
    parser.add_argument("--no-ref", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--stg-scale", type=float, default=1.5)
    parser.add_argument("--duration-multiplier", type=float, default=1.1)
    parser.add_argument("--gen-duration", type=float, default=0.0)
    parser.add_argument("--ref-duration", type=float, default=10.0)
    parser.add_argument("--rescale-scale", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--denoise-ref", action="store_true")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max-chunk-duration", type=float, default=45.0)
    parser.add_argument("--target-chunk-duration", type=float, default=37.0)
    parser.add_argument("--crossfade-ms", type=float, default=50.0)
    parser.add_argument("--audio-out", type=Path, default=Path("dramabox_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timing-file", type=Path, default=Path("dramabox_python_timing.log"))
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--request-sequence-json", default="")
    return parser.parse_args()


def request_sequence(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_sequence_json:
        root = json.loads(args.request_sequence_json)
        if not isinstance(root, list) or not root:
            raise RuntimeError("DramaBox request sequence must be a non-empty JSON array")
        return root
    return [{
        "prompt": args.prompt,
        "voice_ref": None if args.no_ref else args.voice_ref,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "stg_scale": args.stg_scale,
        "duration_multiplier": args.duration_multiplier,
        "gen_duration": args.gen_duration,
        "ref_duration": args.ref_duration,
        "rescale_scale": args.rescale_scale,
        "denoise_ref": args.denoise_ref,
        "watermark": args.watermark,
        "max_chunk_duration": args.max_chunk_duration,
        "target_chunk_duration": args.target_chunk_duration,
        "crossfade_ms": args.crossfade_ms,
    }]


def case_value(case: dict[str, Any], key: str, fallback: Any) -> Any:
    return case[key] if key in case and case[key] is not None else fallback


def case_voice_ref(case: dict[str, Any], args: argparse.Namespace) -> str | None:
    if bool(case_value(case, "no_ref", args.no_ref)):
        return None
    voice_ref = case_value(case, "voice_ref", args.voice_ref)
    if voice_ref is None or voice_ref == "":
        return None
    path = resolve_path(voice_ref)
    if not path.is_file():
        raise RuntimeError(f"missing DramaBox voice reference: {path}")
    return str(path)


def audio_summary(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_sec": frames / float(sample_rate) if sample_rate > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DramaBox warmbench requested CUDA, but CUDA is not available")
        torch.cuda.set_device(args.device)
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    model_root = resolve_path(args.model)
    checkpoint = model_root / "dramabox-dit-v1.safetensors"
    audio_components = model_root / "dramabox-audio-components.safetensors"
    gemma_root = resolve_path(args.gemma_root)
    for path in (checkpoint, audio_components, gemma_root):
        if not path.exists():
            raise RuntimeError(f"missing DramaBox model path: {path}")

    add_reference_paths(args.reference_root)
    from inference_server import TTSServer

    load_start = time.perf_counter()
    server = TTSServer(
        checkpoint=str(checkpoint),
        full_checkpoint=str(audio_components),
        gemma_root=str(gemma_root),
        device=device,
        dtype=args.dtype,
        compile_model=args.compile_model,
        bnb_4bit=True,
    )
    sync_device(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    requests = request_sequence(args)
    output_dir_arg = args.output_dir if args.output_dir is not None else args.audio_out_dir
    output_dir = resolve_path(output_dir_arg) if output_dir_arg is not None else None
    audio_out = resolve_path(args.audio_out)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        audio_out.parent.mkdir(parents=True, exist_ok=True)
    if args.timing_file:
        timing_file = resolve_path(args.timing_file)
        timing_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        timing_file = None

    warmup_case = requests[0]
    for _ in range(args.warmup):
        seed_all(args.seed)
        server.generate_to_file(
            prompt=case_value(warmup_case, "prompt", args.prompt),
            output=str(audio_out.with_name(audio_out.stem + "_warmup.wav")),
            voice_ref=case_voice_ref(warmup_case, args),
            cfg_scale=float(case_value(warmup_case, "cfg_scale", args.cfg_scale)),
            stg_scale=float(case_value(warmup_case, "stg_scale", args.stg_scale)),
            duration_multiplier=float(case_value(warmup_case, "duration_multiplier", args.duration_multiplier)),
            seed=int(case_value(warmup_case, "seed", args.seed)),
            ref_duration=float(case_value(warmup_case, "ref_duration", args.ref_duration)),
            gen_duration=float(case_value(warmup_case, "gen_duration", case_value(warmup_case, "duration_seconds", args.gen_duration))),
            rescale_scale=case_value(warmup_case, "rescale_scale", args.rescale_scale),
            denoise_ref=bool(case_value(warmup_case, "denoise_ref", args.denoise_ref)),
            watermark=bool(case_value(warmup_case, "watermark", args.watermark)),
            max_chunk_duration=float(case_value(warmup_case, "max_chunk_duration", args.max_chunk_duration)),
            target_chunk_duration=float(case_value(warmup_case, "target_chunk_duration", args.target_chunk_duration)),
            crossfade_ms=float(case_value(warmup_case, "crossfade_ms", args.crossfade_ms)),
        )
        sync_device(device)

    steps: list[dict[str, Any]] = []
    timing_lines = [
        f"dramabox.load_wall_ms {load_ms:.6f}",
        f"dramabox.backend {args.backend}",
        f"dramabox.dtype {args.dtype}",
    ]
    for request_index, case in enumerate(requests):
        prompt = case_value(case, "prompt", args.prompt)
        voice_ref = case_voice_ref(case, args)
        seed = int(case_value(case, "seed", args.seed))
        cfg_scale = float(case_value(case, "cfg_scale", args.cfg_scale))
        negative_prompt = case_value(case, "negative_prompt", args.negative_prompt)
        stg_scale = float(case_value(case, "stg_scale", args.stg_scale))
        duration_multiplier = float(case_value(case, "duration_multiplier", args.duration_multiplier))
        ref_duration = float(case_value(case, "ref_duration", args.ref_duration))
        gen_duration = float(case_value(case, "gen_duration", case_value(case, "duration_seconds", args.gen_duration)))
        rescale_scale = case_value(case, "rescale_scale", args.rescale_scale)
        denoise_ref = bool(case_value(case, "denoise_ref", args.denoise_ref))
        watermark = bool(case_value(case, "watermark", args.watermark))
        max_chunk_duration = float(case_value(case, "max_chunk_duration", args.max_chunk_duration))
        target_chunk_duration = float(case_value(case, "target_chunk_duration", args.target_chunk_duration))
        crossfade_ms = float(case_value(case, "crossfade_ms", args.crossfade_ms))
        total_ms = 0.0
        last_output = audio_out
        for iteration in range(args.iterations):
            seed_all(seed)
            if output_dir is not None:
                last_output = output_dir / f"request_{request_index}.wav"
            elif args.iterations == 1 and len(requests) == 1:
                last_output = audio_out
            else:
                last_output = audio_out.with_name(f"{audio_out.stem}_request_{request_index}_{iteration}.wav")
            start = time.perf_counter()
            server.generate_to_file(
                prompt=prompt,
                output=str(last_output),
                voice_ref=voice_ref,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                stg_scale=stg_scale,
                duration_multiplier=duration_multiplier,
                seed=seed,
                ref_duration=ref_duration,
                gen_duration=gen_duration,
                rescale_scale=rescale_scale,
                denoise_ref=denoise_ref,
                watermark=watermark,
                max_chunk_duration=max_chunk_duration,
                target_chunk_duration=target_chunk_duration,
                crossfade_ms=crossfade_ms,
            )
            sync_device(device)
            wall_ms = (time.perf_counter() - start) * 1000.0
            total_ms += wall_ms
            timing_lines.append(f"dramabox.generate_wall_ms {wall_ms:.6f}")
        avg_ms = total_ms / float(max(1, args.iterations))
        audio_seconds = wav_duration_seconds(last_output)
        print(f"dramabox.wall_ms={avg_ms}")
        print(f"dramabox.audio_seconds={audio_seconds}")
        timing_lines.append(f"dramabox.audio_seconds {audio_seconds:.6f}")
        steps.append({
            "request_index": request_index,
            "prompt": prompt,
            "voice_ref": voice_ref,
            "stems": [{
                "name": "audio",
                "summary": audio_summary(last_output),
                "audio": str(last_output),
            }],
            "metrics": {"wall_ms": avg_ms},
        })

    if timing_file is not None:
        timing_file.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    summary = {
        "family": "dramabox",
        "backend": args.backend,
        "model": str(model_root),
        "gemma_root": str(gemma_root),
        "load_ms": load_ms,
        "sequence_steps": steps,
    }
    if args.summary_file:
        summary_file = resolve_path(args.summary_file)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary_json=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
