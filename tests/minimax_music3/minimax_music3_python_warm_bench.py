#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/home/leo/Desktop/MiniMax-Music")

# Known-good local run, using the installed diffusers in conda qwen3-tts
# and the model snapshot moved to Desktop:
#
# REQUEST=$(python -c "import json; d=json.load(open('tests/minimax_music3/minimax_music3_warm_bench_cases.json'))['default']['requests'][0]; print(json.dumps(d))")
# conda run -n qwen3-tts python tests/minimax_music3/minimax_music3_python_warm_bench.py \
#   --model /home/leo/Desktop/MiniMax-Music --backend cuda --device 0 --threads 8 \
#   --warmup 1 --iterations 1 --request-json "$REQUEST" \
#   --output-dir build/logs/minimax_music3/python_desktop_rtf \
#   --timing-file build/logs/minimax_music3/python_desktop_rtf/timing.log \
#   --summary-file build/logs/minimax_music3/python_desktop_rtf/summary.json


DEFAULT_REQUEST = {
    "lyrics": "[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe",
    "prompt": (
        "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
        "Vocals: soft female lead, close and breathy. Arrangement: fingerpicked guitar, soft piano, brushed drums, and upright bass."
    ),
    "audio_duration": 20.0,
    "seed": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference MiniMax Music 3 warmbench.")
    parser.add_argument("--family", default="minimax_music3")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--diffusers-root", default="")
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--group-offload", action="store_true")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-out", type=Path, default=Path("minimax_music3_python_audio.wav"))
    parser.add_argument("--timing-file", type=Path, default=Path("minimax_music3_python_timing.log"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def add_diffusers_path(diffusers_root: str) -> None:
    if not diffusers_root:
        return
    root = resolve_path(diffusers_root).resolve()
    src = root / "src"
    if not (src / "diffusers" / "__init__.py").is_file():
        raise RuntimeError(f"missing Diffusers checkout with MiniMax Music 3 support: {src}")
    sys.path.insert(0, str(src))


def materialize_local_modular_index(model_path: Path) -> Path:
    source_index = model_path / "modular_model_index.json"
    payload = json.loads(source_index.read_text(encoding="utf-8"))
    for value in payload.values():
        if (
            isinstance(value, list)
            and len(value) >= 3
            and isinstance(value[2], dict)
            and "pretrained_model_name_or_path" in value[2]
        ):
            value[2]["pretrained_model_name_or_path"] = str(model_path)
    local_root = Path(tempfile.mkdtemp(prefix="minimax_music3_local_index_"))
    (local_root / "modular_model_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config = model_path / "config.json"
    if config.is_file():
        shutil.copy2(config, local_root / "config.json")
    return local_root


def normalize_device(args: argparse.Namespace) -> str:
    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax Music 3 warmbench requested CUDA, but torch.cuda.is_available() is false")
    torch.cuda.set_device(args.device)
    return f"cuda:{args.device}"


def sync_device(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_sequence_json:
        payload = json.loads(args.request_sequence_json)
        if not isinstance(payload, list):
            raise RuntimeError("--request-sequence-json must decode to a list")
        return [dict(item) for item in payload]
    if args.request_json:
        payload = json.loads(args.request_json)
        if not isinstance(payload, dict):
            raise RuntimeError("--request-json must decode to an object")
        return [dict(payload)]
    return [dict(DEFAULT_REQUEST)]


def audio_to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2:
        raise RuntimeError(f"MiniMax Music 3 expected 2-D audio, got shape {audio.shape}")
    if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        audio = audio.T
    return np.ascontiguousarray(audio)


def summarize_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if audio.size == 0:
        raise RuntimeError("MiniMax Music 3 warmbench received empty audio")
    flat = audio.reshape(-1).astype(np.float64, copy=False)
    frames = int(audio.shape[0])
    channels = int(audio.shape[1])
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": int(flat.size),
        "frames": frames,
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat), dtype=np.float64))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return {"name": "audio", "audio": str(path), "summary": summarize_audio(audio, sample_rate)}


def request_seed(request: dict[str, Any], fallback: int) -> int:
    return int(request.get("seed", fallback))


def run_request(
    pipe: Any,
    request: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> tuple[dict[str, Any], str]:
    seed = request_seed(request, 7)
    seed_all(seed)
    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(generator_device).manual_seed(seed)

    lyrics = str(request.get("lyrics", request.get("input", "")))
    prompt = str(request.get("prompt", request.get("instructions", "")))
    if not lyrics:
        raise RuntimeError("MiniMax Music 3 request missing lyrics/input")
    if not prompt:
        raise RuntimeError("MiniMax Music 3 request missing prompt/instructions")
    audio_duration = float(request.get("audio_duration", request.get("duration", 20.0)))
    if audio_duration <= 0.0:
        raise RuntimeError(f"MiniMax Music 3 audio_duration must be positive: {audio_duration}")

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_duration": audio_duration,
        "generator": generator,
        "output": "audios",
    }
    if "max_new_tokens" in request:
        kwargs["max_new_tokens"] = int(request["max_new_tokens"])

    sync_device(device)
    print(f"minimax_music3.request.start duration={audio_duration:.3f} seed={seed}", flush=True)
    started = time.perf_counter()
    audio = pipe(**kwargs)[0]
    sync_device(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    print(f"minimax_music3.request.done wall_ms={wall_ms:.6f}", flush=True)

    audio_np = audio_to_numpy(audio)
    sample_rate = int(pipe.sampling_rate)
    stem = save_audio(output_dir / "audio.wav", audio_np, sample_rate)
    timing_line = f"minimax_music3.wall_ms {wall_ms:.6f}"
    return {
        "request": request,
        "stems": [stem],
        "metrics": {
            "wall_ms": wall_ms,
            "audio_duration_ms": float(stem["summary"]["duration_sec"]) * 1000.0,
            "rtf": wall_ms / max(1.0, float(stem["summary"]["duration_sec"]) * 1000.0),
        },
    }, timing_line


def load_pipeline(args: argparse.Namespace, device: str) -> Any:
    print("minimax_music3.load.import_diffusers.start", flush=True)
    import transformers.utils as transformers_utils
    if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.hooks import apply_group_offloading
    print("minimax_music3.load.import_diffusers.done", flush=True)

    model_path = resolve_path(args.model).resolve()
    if not (model_path / "modular_model_index.json").is_file():
        raise RuntimeError(f"missing MiniMax Music 3 model snapshot: {model_path}")
    pipeline_path = materialize_local_modular_index(model_path)
    dtype = torch_dtype(args.dtype)

    if args.cpu_offload:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=device)
        print("minimax_music3.load.pipeline.start", flush=True)
        pipe = ModularPipeline.from_pretrained(str(pipeline_path), components_manager=manager, local_files_only=True)
        shutil.rmtree(pipeline_path, ignore_errors=True)
        print("minimax_music3.load.pipeline.done", flush=True)
        print(f"minimax_music3.load.components.start dtype={args.dtype}", flush=True)
        pipe.load_components(dtype=dtype)
        print("minimax_music3.load.components.done", flush=True)
    else:
        print("minimax_music3.load.pipeline.start", flush=True)
        pipe = ModularPipeline.from_pretrained(str(pipeline_path), local_files_only=True)
        shutil.rmtree(pipeline_path, ignore_errors=True)
        print("minimax_music3.load.pipeline.done", flush=True)
        print(f"minimax_music3.load.components.start dtype={args.dtype}", flush=True)
        pipe.load_components(dtype=dtype)
        print("minimax_music3.load.components.done", flush=True)
        print(f"minimax_music3.load.to_device.start device={device}", flush=True)
        pipe.to(device)
        print("minimax_music3.load.to_device.done", flush=True)

    if args.group_offload:
        print("minimax_music3.load.group_offload.start", flush=True)
        apply_group_offloading(
            pipe.language_model,
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=str(device).startswith("cuda"),
        )
        print("minimax_music3.load.group_offload.done", flush=True)
    return pipe


def main() -> int:
    args = parse_args()
    if args.family != "minimax_music3":
        raise RuntimeError(f"unsupported MiniMax Music 3 warmbench family: {args.family}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    add_diffusers_path(args.diffusers_root)
    device = normalize_device(args)
    seed_all(7)

    pipe = load_pipeline(args, device)
    requests = load_requests(args)
    if not requests:
        raise RuntimeError("MiniMax Music 3 warmbench request sequence is empty")

    output_root = args.output_dir if args.output_dir is not None else args.audio_out.parent
    output_root = resolve_path(output_root).resolve()
    timing_path = resolve_path(args.timing_file)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_lines: list[str] = []

    for warmup_index in range(max(0, args.warmup)):
        _, timing_line = run_request(
            pipe,
            requests[0],
            output_root / "warmup" / f"{warmup_index:02d}",
            args,
            device,
        )
        timing_lines.append(timing_line)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        total_ms = 0.0
        last_step: dict[str, Any] | None = None
        for iteration in range(max(1, args.iterations)):
            step, timing_line = run_request(
                pipe,
                request,
                output_root / f"request_{request_index:02d}" / f"iter_{iteration:02d}",
                args,
                device,
            )
            total_ms += float(step["metrics"]["wall_ms"])
            last_step = step
            timing_lines.append(timing_line)
        if last_step is None:
            raise RuntimeError("MiniMax Music 3 warmbench produced no request step")
        last_step = dict(last_step)
        last_step["request_index"] = request_index
        last_step["metrics"] = dict(last_step["metrics"])
        last_step["metrics"]["wall_ms"] = total_ms / float(max(1, args.iterations))
        print(f"minimax_music3.wall_ms={last_step['metrics']['wall_ms']}")
        steps.append(last_step)

    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    summary = {
        "family": "minimax_music3",
        "backend": args.backend,
        "model": str(resolve_path(args.model).resolve()),
        "sequence_steps": steps,
    }
    if args.summary_file is not None:
        summary_path = resolve_path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
