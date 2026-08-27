#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "controlfoley"
DEFAULT_MODEL = REPO_ROOT / "models" / "ControlFoley"

sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(REFERENCE_ROOT / "lib"))

from lib.flow_matching import FlowMatching  # noqa: E402
from controlfoley.audio_model import AudioGenerationNetwork, create_audio_generation_model  # noqa: E402
from controlfoley.feature_extractor import FeaturesUtils  # noqa: E402
from controlfoley.inference_utils import ModelConfig, all_model_cfg, generate, load_video, setup_eval_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference ControlFoley warmbench.")
    parser.add_argument("--family", default="controlfoley")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timing-file", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


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
    raise RuntimeError("ControlFoley warmbench requires --request-json or --request-sequence-json")


def seed_all(seed: int, device: torch.device) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return rng


def summarize_audio(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError(f"ControlFoley reference produced empty audio: {path}")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "samples": int(flat.size),
        "frames": int(audio.shape[0]),
        "duration_sec": float(audio.shape[0] / float(sample_rate)),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def patch_bigvgan_hub_kwargs() -> None:
    from lib.bigvgan_v2.bigvgan import BigVGAN

    fn = BigVGAN._from_pretrained.__func__
    kw = dict(fn.__kwdefaults__ or {})
    kw.setdefault("proxies", None)
    kw.setdefault("resume_download", False)
    fn.__kwdefaults__ = kw


def load_model(model_root: Path, device: torch.device, dtype: torch.dtype) -> tuple[ModelConfig, AudioGenerationNetwork, FeaturesUtils]:
    if not model_root.is_dir():
        raise RuntimeError(f"ControlFoley model root not found: {model_root}")
    model = all_model_cfg["large_44k"]
    net = create_audio_generation_model(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_root / "weights" / "controlfoley.pth", map_location=device, weights_only=True))
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=str(model_root / "ext_weights" / "v1-44.pth"),
        synchformer_ckpt=str(model_root / "ext_weights" / "synchformer_state_dict.pth"),
        cav_mae_ckpt=str(model_root / "ext_weights" / "cav_mae_st.pth"),
        clap_ckpt=str(model_root / "ext_weights" / "music_speech_audioset_epoch_15_esc_89.98.pt"),
        mode=model.mode,
        enable_conditions=True,
        need_vae_encoder=False,
    ).to(device, dtype).eval()
    return model, net, feature_utils


def request_value(request: dict[str, Any], *keys: str, fallback: Any = None) -> Any:
    for key in keys:
        if key in request:
            return request[key]
    return fallback


def prepare_request_inputs(
    request: dict[str, Any],
    model: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    duration = float(request_value(request, "duration_sec", "duration", fallback=8.0))
    video_path = request.get("video")
    if video_path:
        video_info = load_video(resolve_path(str(video_path)), duration)
        duration = min(duration, float(video_info.total_duration))
        clip_frames = video_info.clip_embeddings
        visual_frames = video_info.visual_features
        sync_frames = video_info.sync_embeddings
        if bool(request.get("mask_away_clip", False)):
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        visual_frames = visual_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
    else:
        clip_frames = None
        visual_frames = None
        sync_frames = None

    audio_path = request.get("audio")
    if audio_path:
        audio_frames, sampling_rate = torchaudio.load(resolve_path(str(audio_path)))
        audio_frames = audio_frames.to(device, dtype)
        timbre_frames = audio_frames
        if sampling_rate != 16000:
            audio_frames = torchaudio.functional.resample(audio_frames, sampling_rate, 16000)
        audio_frames = audio_frames.mean(dim=0, keepdim=True).reshape(1, -1).unsqueeze(0)
        if sampling_rate != 32000:
            timbre_frames = torchaudio.functional.resample(timbre_frames, sampling_rate, 32000)
        min_length = 2 * 32000
        max_length = 4 * 32000
        samples = timbre_frames.shape[-1]
        if samples < min_length:
            timbre_frames = torch.nn.functional.pad(timbre_frames, (0, min_length - samples), mode="constant", value=0)
        elif samples > max_length:
            timbre_frames = timbre_frames[..., :max_length]
        timbre_duration = float(timbre_frames.shape[-1]) / 32000.0
        timbre_frames = timbre_frames.mean(dim=0, keepdim=True).reshape(1, -1).unsqueeze(0)
    else:
        audio_frames = None
        timbre_frames = None
        timbre_duration = 0.0

    seq_cfg = model.seq_cfg
    seq_cfg.total_time_seconds = duration
    return {
        "duration": duration,
        "seq_cfg": seq_cfg,
        "clip_frames": clip_frames,
        "visual_frames": visual_frames,
        "sync_frames": sync_frames,
        "audio_frames": audio_frames,
        "timbre_frames": timbre_frames,
        "timbre_duration": timbre_duration,
        "prompt": str(request.get("prompt", "")),
        "negative_prompt": str(request_value(request, "negative_prompt", "negative_text", fallback="")),
        "num_steps": int(request_value(request, "num_inference_steps", "num_steps", fallback=25)),
        "cfg_strength": float(request.get("cfg_strength", 4.5)),
    }


def run_one(
    request: dict[str, Any],
    model: ModelConfig,
    net: AudioGenerationNetwork,
    feature_utils: FeaturesUtils,
    output_path: Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, float]:
    prepared = prepare_request_inputs(request, model, device, dtype)
    seq_cfg = prepared["seq_cfg"]
    net.update_seq_lengths(
        seq_cfg.latent_sequence_length,
        seq_cfg.clip_sequence_length,
        seq_cfg.visual_sequence_length,
        seq_cfg.sync_sequence_length,
    )
    rng = seed_all(int(request.get("seed", 42)), device)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=prepared["num_steps"])
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        audios = generate(
            prepared["clip_frames"],
            prepared["visual_frames"],
            prepared["sync_frames"],
            prepared["audio_frames"],
            prepared["timbre_frames"],
            prepared["timbre_duration"],
            [prepared["prompt"]],
            negative_text=[prepared["negative_prompt"]],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=prepared["cfg_strength"],
        )
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    audio = audios.detach().float().cpu()[0]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        values = audio.transpose(0, 1).numpy() if audio.ndim == 2 else audio.numpy()
        sf.write(str(output_path), values, int(seq_cfg.audio_sample_rate), subtype="PCM_16")
    return audio, int(seq_cfg.audio_sample_rate), wall_ms


def main() -> int:
    args = parse_args()
    if args.family != "controlfoley":
        raise RuntimeError(f"unsupported ControlFoley warmbench family: {args.family}")
    if args.backend != "cuda":
        raise RuntimeError("ControlFoley Python warmbench is CUDA-only")
    if not torch.cuda.is_available():
        raise RuntimeError("ControlFoley Python warmbench requires CUDA")

    os.environ["OMP_NUM_THREADS"] = str(max(1, args.threads))
    os.environ["MKL_NUM_THREADS"] = str(max(1, args.threads))
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    setup_eval_logging()
    patch_bigvgan_hub_kwargs()
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32
    model, net, feature_utils = load_model(resolve_path(args.model), device, dtype)
    requests = load_requests(args)
    output_root = resolve_path(args.output_dir) if args.output_dir else REPO_ROOT / "build/logs/controlfoley/python_warmbench"
    output_root.mkdir(parents=True, exist_ok=True)

    timing_lines = [f"controlfoley.backend {args.backend}"]
    for warmup_index in range(max(0, args.warmup)):
        _, _, wall_ms = run_one(
            dict(requests[0]),
            model,
            net,
            feature_utils,
            None,
            device,
            dtype,
        )
        timing_lines.append(f"controlfoley.warmup{warmup_index}.wall_ms {wall_ms:.6f}")

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        total_ms = 0.0
        last_audio_path = output_root / f"request_{request_index:02d}_{request.get('id', 'request')}" / "output.wav"
        for _ in range(max(1, args.iterations)):
            _, _, wall_ms = run_one(
                request,
                model,
                net,
                feature_utils,
                last_audio_path,
                device,
                dtype,
            )
            total_ms += wall_ms
        mean_wall_ms = total_ms / float(max(1, args.iterations))
        timing_lines.append(f"controlfoley.request{request_index}.wall_ms {mean_wall_ms:.6f}")
        print(f"controlfoley.wall_ms={mean_wall_ms}")
        steps.append({
            "request_index": request_index,
            "request": request,
            "stems": [{"name": "audio", "audio": str(last_audio_path), "summary": summarize_audio(last_audio_path)}],
            "metrics": {"wall_ms": mean_wall_ms},
        })

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    summary = {"family": "controlfoley", "backend": args.backend, "sequence_steps": steps}
    print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
