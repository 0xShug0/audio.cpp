#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "MeanVC2"
MODEL_ROOT = REPO_ROOT / "models" / "MeanVC2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference MeanVC2 warmbench.")
    parser.add_argument("--family", default="meanvc2")
    parser.add_argument("--model", default=str(MODEL_ROOT))
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timing-file", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--cases", default=str(REPO_ROOT / "tests" / "meanvc2" / "meanvc2_warm_bench_cases.json"))
    parser.add_argument("--only", default="quality_120ms_default")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_case_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
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

    with open(resolve_path(args.cases), "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if args.only not in cases:
        raise RuntimeError(f"MeanVC2 warmbench case not found: {args.only}")
    requests = cases[args.only].get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"MeanVC2 warmbench case has no requests: {args.only}")
    return requests


def load_reference_runtime(model_root: Path):
    runtime_dir = REFERENCE_ROOT / "runtime"
    if not runtime_dir.exists():
        raise RuntimeError(f"MeanVC2 reference runtime not found: {runtime_dir}")
    sys.path.insert(0, str(runtime_dir))
    spec = importlib.util.spec_from_file_location("meanvc2_runtime_run_rt", runtime_dir / "run_rt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to import MeanVC2 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.VOCODER_PATH = str(model_root / "ckpts" / "vocos" / "vocos.pt")
    module.SPEAKER_MODEL_PATH = str(model_root / "preprocess" / "ckpts" / "wavlm_large_finetune.pth")
    module.WAVLM_CONFIG_PATH = str(model_root / "preprocess" / "ckpts" / "wavlm_large_cfg.pt")
    module.MODEL_PATHS["120ms"]["ckpt"] = str(model_root / "ckpts" / "pretrained_models" / "meanvc2_120ms_40ms.safetensors")
    module.MODEL_PATHS["120ms"]["config"] = str(REFERENCE_ROOT / "src" / "config" / "config_120ms_40ms.json")
    module.MODEL_PATHS["120ms"]["asr_ckpt"] = str(model_root / "preprocess" / "ckpts" / "fastu2pp_160ms.pt")
    return module


def reseed(seed: int, backend: str) -> None:
    import torch

    torch.manual_seed(seed)
    if backend == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def synchronize(backend: str) -> None:
    if backend == "cuda":
        import torch

        torch.cuda.synchronize()


def summarize_audio(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError(f"MeanVC2 reference produced empty audio: {path}")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "samples": int(flat.size),
        "duration_sec": float(audio.shape[0]) / float(sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat), dtype=np.float64))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def source_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise RuntimeError(f"MeanVC2 source audio has invalid sample rate: {path}")
    return float(info.frames) / float(info.samplerate)


def main() -> None:
    args = parse_args()
    model_root = resolve_path(args.model)
    output_root = resolve_path(args.output_dir) if args.output_dir else REPO_ROOT / "build" / "logs" / "meanvc2" / "python_baseline"
    output_root.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "cuda":
        torch.cuda.set_device(args.device)
        device = "cuda"
    else:
        device = "cpu"

    requests = load_case_requests(args)
    runtime = load_reference_runtime(model_root)
    runners: dict[tuple[str, str], Any] = {}
    results: list[dict[str, Any]] = []

    def runner_for(model: str, target_audio: Path):
        key = (model, str(target_audio))
        if key not in runners:
            runners[key] = runtime.VCRunner(target_wav=str(target_audio), device=device, model=model)
        return runners[key]

    def run_request(request: dict[str, Any], output_path: Path, record: bool, ordinal: int) -> dict[str, Any] | None:
        model = str(request.get("model", "120ms"))
        if model != "120ms":
            raise RuntimeError("MeanVC2 warmbench currently prepares only the quality 120ms_40ms path")
        source_audio = resolve_path(str(request["source_audio"]))
        target_audio = resolve_path(str(request["target_audio"]))
        seed = int(request.get("seed", 42))
        reseed(seed, args.backend)
        runner = runner_for(model, target_audio)
        synchronize(args.backend)
        start = time.perf_counter()
        runner.process_file(str(source_audio), str(output_path), seed=seed)
        synchronize(args.backend)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not record:
            return None
        summary = summarize_audio(output_path)
        source_duration = source_duration_sec(source_audio)
        return {
            "request_index": ordinal,
            "model": model,
            "source_audio": str(source_audio),
            "target_audio": str(target_audio),
            "seed": seed,
            "output": str(output_path),
            "wall_ms": elapsed_ms,
            "source_duration_sec": source_duration,
            "rtf": elapsed_ms / 1000.0 / source_duration,
            "audio": summary,
        }

    for request_index, request in enumerate(requests):
        for warmup_index in range(args.warmup):
            output_path = output_root / f"warmup_{request_index}_{warmup_index}.wav"
            run_request(request, output_path, record=False, ordinal=request_index)
        for iteration_index in range(args.iterations):
            output_path = output_root / f"request_{request_index}_iter_{iteration_index}.wav"
            result = run_request(request, output_path, record=True, ordinal=request_index)
            if result is not None:
                results.append(result)

    payload = {
        "family": args.family,
        "backend": args.backend,
        "device": args.device,
        "threads": args.threads,
        "output_dir": str(output_root),
        "results": results,
    }
    if args.timing_file:
        timing_file = resolve_path(args.timing_file)
        timing_file.parent.mkdir(parents=True, exist_ok=True)
        timing_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
