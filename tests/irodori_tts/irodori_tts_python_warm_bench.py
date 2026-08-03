#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "Irodori-TTS"
REFERENCE_DEPS_ROOT = REPO_ROOT / "reference" / "irodori-python-deps"
DEFAULT_MODEL = REPO_ROOT / "models" / "Irodori-TTS-500M-v3"
DEFAULT_VOICE_DESIGN_MODEL = REPO_ROOT / "models" / "Irodori-TTS-600M-v3-VoiceDesign"
DEFAULT_CODEC = REPO_ROOT / "models" / "Semantic-DACVAE-Japanese-32dim" / "weights.pth"
SHORT_EMOJI_TEXT = "今日は短い確認です。😊やさしく、聞き取りやすい声でお願いします。"
STYLE_SOUND_TEXT = "📞お電話ありがとうございます。少しお待ちください。⏸️ただいま担当者へおつなぎします。"
CLONE_TEXT = "この声の雰囲気を保ったまま、落ち着いて自然に読み上げます。"
VOICE_DESIGN_TEXT = "本日はお越しいただき、誠にありがとうございます。どうぞごゆっくりお過ごしください。"
VOICE_DESIGN_CAPTION = "落ち着いた大人の男性。フォーマルな場で、深く響く声で丁寧かつ歓迎の意を込めて話している。"
VOICE_DESIGN_EMOJI_TEXT = "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。"
VOICE_DESIGN_EMOJI_CAPTION = "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。"


TEST_CASES: dict[str, dict[str, Any]] = {
    "short_emoji": {
        "text": SHORT_EMOJI_TEXT,
        "no_ref": True,
    },
    "style_sound": {
        "text": STYLE_SOUND_TEXT,
        "no_ref": True,
    },
    "clone_reference": {
        "text": CLONE_TEXT,
        "ref_wav": str(DEFAULT_MODEL / "samples" / "clone_ref1.wav"),
        "no_ref": False,
    },
    "voice_design_caption": {
        "text": VOICE_DESIGN_TEXT,
        "caption": VOICE_DESIGN_CAPTION,
        "no_ref": True,
    },
    "voice_design_ref_caption_emoji": {
        "text": VOICE_DESIGN_EMOJI_TEXT,
        "caption": VOICE_DESIGN_EMOJI_CAPTION,
        "ref_wav": str(DEFAULT_VOICE_DESIGN_MODEL / "samples" / "clone_ref2.wav"),
        "no_ref": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference Irodori-TTS warmbench.")
    parser.add_argument("--family", default="irodori_tts")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--reference-deps-root", type=Path, default=REFERENCE_DEPS_ROOT)
    parser.add_argument("--codec-repo", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--case", choices=tuple(TEST_CASES), default="short_emoji")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--text", action="append", dest="texts", default=[])
    parser.add_argument("--caption", default="")
    parser.add_argument("--ref-wav", default="")
    parser.add_argument("--no-ref", action="store_true")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--cfg-guidance-mode", choices=("independent", "joint", "alternating"), default="independent")
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--decode-mode", choices=("sequential", "batch"), default="sequential")
    parser.add_argument("--model-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--codec-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-out", type=Path, default=Path("irodori_tts_python_audio.wav"))
    parser.add_argument("--timing-file", type=Path, default=Path("irodori_tts_python_timing.log"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def add_reference_paths(reference_root: Path, deps_root: Path) -> Path:
    root = resolve_repo_path(reference_root).resolve()
    deps = resolve_repo_path(deps_root).resolve()
    package = root / "irodori_tts" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing Irodori-TTS reference package: {package}")
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    sys.path.insert(0, str(root))
    return root


def load_reference_symbols(reference_root: Path, deps_root: Path) -> tuple[Any, Any, Any, Any, Any, Path]:
    root = add_reference_paths(reference_root, deps_root)
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        resolve_cfg_scales,
        save_wav,
    )
    import irodori_tts

    module_path = Path(irodori_tts.__file__).resolve()
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Irodori-TTS imported from {module_path}, expected under {root}") from exc
    return InferenceRuntime, RuntimeKey, SamplingRequest, resolve_cfg_scales, save_wav, module_path


def normalize_device(args: argparse.Namespace) -> str:
    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Irodori-TTS warmbench requested CUDA, but torch.cuda.is_available() is false")
        torch.cuda.set_device(args.device)
        return f"cuda:{args.device}"
    return "cpu"


def sync_device(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def seed_all(seed: int, backend: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    if backend == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_sequence_json:
        payload = json.loads(args.request_sequence_json)
        if not isinstance(payload, list):
            raise RuntimeError("--request-sequence-json must decode to a list")
        return payload
    if args.request_json:
        payload = json.loads(args.request_json)
        if not isinstance(payload, dict):
            raise RuntimeError("--request-json must decode to an object")
        return [payload]
    if args.texts:
        return [
            {
                "text": text,
                "caption": args.caption,
                "ref_wav": args.ref_wav,
                "no_ref": bool(args.no_ref or not args.ref_wav),
            }
            for text in args.texts
        ]
    return [dict(TEST_CASES[args.case])]


def summarize_audio(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
        flat = audio.reshape(-1)
    else:
        flat = audio
    if flat.size == 0:
        raise RuntimeError("Irodori-TTS warmbench received empty audio")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "samples": int(flat.size),
        "frames": int(frames),
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def request_value(request: dict[str, Any], primary: str, fallback: str, default: Any) -> Any:
    if primary in request:
        return request[primary]
    if fallback in request:
        return request[fallback]
    return default


def run_request(
    runtime: Any,
    SamplingRequest: Any,
    resolve_cfg_scales: Any,
    save_wav: Any,
    request: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> tuple[dict[str, Any], list[str]]:
    text = str(request.get("text", "")).strip()
    if not text:
        raise RuntimeError("Irodori-TTS warmbench request missing text")
    caption = str(request.get("caption", "")).strip()
    ref_wav = str(request.get("ref_wav", "")).strip()
    no_ref = bool(request.get("no_ref", not bool(ref_wav)))
    seed = int(request.get("seed", args.seed))
    seed_all(seed, args.backend)

    use_caption = bool(runtime.model_cfg.use_caption_condition and caption)
    use_speaker = bool(runtime.model_cfg.use_speaker_condition_resolved and not no_ref)
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode=str(request_value(request, "guidance_mode", "cfg_guidance_mode", args.cfg_guidance_mode)),
        cfg_scale_text=float(request_value(request, "text_guidance_scale", "cfg_scale_text", args.cfg_scale_text)),
        cfg_scale_caption=float(request_value(request, "caption_guidance_scale", "cfg_scale_caption", args.cfg_scale_caption)),
        cfg_scale_speaker=float(request_value(request, "speaker_guidance_scale", "cfg_scale_speaker", args.cfg_scale_speaker)),
        cfg_scale=(
            float(request_value(request, "guidance_scale", "cfg_scale", args.cfg_scale))
            if float(request_value(request, "guidance_scale", "cfg_scale", args.cfg_scale)) > 0.0
            else None
        ),
        use_caption_condition=use_caption,
        use_speaker_condition=use_speaker,
    )

    req = SamplingRequest(
        text=text,
        caption=caption or None,
        ref_wav=ref_wav or None,
        no_ref=no_ref,
        num_candidates=1,
        decode_mode=str(request.get("decode_mode", args.decode_mode)),
        seconds=(
            None
            if float(request_value(request, "duration_sec", "seconds", args.seconds)) <= 0.0
            else float(request_value(request, "duration_sec", "seconds", args.seconds))
        ),
        duration_scale=float(request.get("duration_scale", args.duration_scale)),
        num_steps=int(request_value(request, "num_inference_steps", "num_steps", args.num_steps)),
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=str(request_value(request, "guidance_mode", "cfg_guidance_mode", args.cfg_guidance_mode)),
        cfg_min_t=float(request_value(request, "guidance_min_t", "cfg_min_t", args.cfg_min_t)),
        cfg_max_t=float(request_value(request, "guidance_max_t", "cfg_max_t", args.cfg_max_t)),
        seed=seed,
        context_kv_cache=True,
        trim_tail=True,
    )
    sync_device(device)
    started = time.perf_counter()
    result = runtime.synthesize(req, log_fn=None)
    sync_device(device)
    wall_ms = (time.perf_counter() - started) * 1000.0

    audio_path = output_dir / "audio.wav"
    save_wav(audio_path, result.audio, int(result.sample_rate))
    summary = summarize_audio(audio_path)
    timing = [
        f"irodori_tts.wall_ms {wall_ms:.6f}",
        f"irodori_tts.seed {seed}",
        f"irodori_tts.used_seed {int(result.used_seed)}",
    ]
    for name, sec in result.stage_timings:
        timing.append(f"irodori_tts.{name}_ms {float(sec) * 1000.0:.6f}")
    metrics = {"wall_ms": wall_ms}
    if result.total_to_decode > 0.0:
        metrics["total_to_decode_ms"] = float(result.total_to_decode) * 1000.0

    step = {
        "request": request,
        "stems": [
            {
                "name": "audio",
                "audio": str(audio_path),
                "summary": summary,
            }
        ],
        "metrics": metrics,
    }
    return step, timing


def main() -> int:
    args = parse_args()
    if args.family != "irodori_tts":
        raise RuntimeError(f"unsupported Irodori-TTS warmbench family: {args.family}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = normalize_device(args)
    seed_all(args.seed, args.backend)

    InferenceRuntime, RuntimeKey, SamplingRequest, resolve_cfg_scales, save_wav, module_path = load_reference_symbols(
        args.reference_root,
        args.reference_deps_root,
    )
    model_root = resolve_repo_path(args.model).resolve()
    checkpoint = model_root / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing Irodori-TTS checkpoint: {checkpoint}")
    codec_path = resolve_repo_path(args.codec_repo).resolve()
    if not codec_path.is_file():
        raise RuntimeError(f"missing Irodori-TTS codec checkpoint: {codec_path}")
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=device,
            codec_repo=str(codec_path),
            model_precision=args.model_precision,
            codec_device=device,
            codec_precision=args.codec_precision,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    requests = load_requests(args)
    if not requests:
        raise RuntimeError("Irodori-TTS warmbench request sequence is empty")

    output_root = args.output_dir if args.output_dir is not None else args.audio_out.parent
    output_root = resolve_repo_path(output_root).resolve()
    timing_path = resolve_repo_path(args.timing_file)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_lines = [
        f"irodori_tts.python_reference_module {module_path}",
        f"irodori_tts.model_root {model_root}",
        f"irodori_tts.codec {codec_path}",
        f"irodori_tts.backend {args.backend}",
        "irodori_tts.python_tf32_disabled 1",
    ]

    for warmup_index in range(max(0, args.warmup)):
        _, warmup_timing = run_request(
            runtime,
            SamplingRequest,
            resolve_cfg_scales,
            save_wav,
            requests[0],
            output_root / "warmup" / f"{warmup_index:02d}",
            args,
            device,
        )
        timing_lines.extend(warmup_timing)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        total_ms = 0.0
        last_step: dict[str, Any] | None = None
        for iteration in range(max(1, args.iterations)):
            step, run_timing = run_request(
                runtime,
                SamplingRequest,
                resolve_cfg_scales,
                save_wav,
                request,
                output_root / f"request_{request_index:02d}" / f"iter_{iteration:02d}",
                args,
                device,
            )
            total_ms += float(step["metrics"]["wall_ms"])
            last_step = step
            timing_lines.extend(run_timing)
        if last_step is None:
            raise RuntimeError("Irodori-TTS warmbench produced no request step")
        last_step = dict(last_step)
        last_step["request_index"] = request_index
        last_step["metrics"] = dict(last_step["metrics"])
        last_step["metrics"]["wall_ms"] = total_ms / float(max(1, args.iterations))
        print(f"irodori_tts.wall_ms={last_step['metrics']['wall_ms']}")
        steps.append(last_step)

    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    summary = {
        "family": "irodori_tts",
        "backend": args.backend,
        "sequence_steps": steps,
    }
    if args.summary_file is not None:
        summary_path = resolve_repo_path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
