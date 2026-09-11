#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "MiDashengLM-Gen"
DEFAULT_CASES = REPO_ROOT / "tests" / "midashenglm_gen" / "midashenglm_gen_warm_bench_cases.json"


def timestamp_seconds_local() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference MiDashengLM-Gen warmbench.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-set", default="default")
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact-stamp", default="")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sync_device() -> None:
    torch.cuda.synchronize()


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def load_cases(path: Path, case_set: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if case_set not in payload:
        raise RuntimeError(f"missing MiDashengLM-Gen case set: {case_set}")
    cases = payload[case_set]
    if "requests" not in cases or not isinstance(cases["requests"], list):
        raise RuntimeError(f"MiDashengLM-Gen case set {case_set} must contain a requests list")
    return cases


def apply_seq_len(model: Any, seq_len: int) -> None:
    model.config.seq_len = int(seq_len)
    model.model.seq_len = int(seq_len)


def seq_len_from_duration(duration_sec: float, sample_rate: int = 16000, hop: int = 320) -> int:
    frames_per_second = sample_rate / float(2 * hop)
    return max(1, int(np.ceil(float(duration_sec) * frames_per_second)))


def summarize_audio(audio: np.ndarray, sample_rate: int, text: str) -> dict[str, Any]:
    waveform = np.asarray(audio, dtype=np.float32)
    flat = waveform.reshape(-1)
    if flat.size == 0:
        raise RuntimeError("MiDashengLM-Gen generated empty audio")
    return {
        "sample_rate": int(sample_rate),
        "channels": 1 if waveform.ndim == 1 else int(waveform.shape[1]),
        "samples": int(flat.size),
        "duration_sec": float(flat.size / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "request_char_count": len(text),
        "first_samples": flat[:32].tolist(),
    }


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    sf.write(str(path), waveform, int(sample_rate))


def run_generation(model: Any, request: dict[str, Any]) -> dict[str, Any]:
    duration_sec = float(request["duration_sec"])
    seq_len = seq_len_from_duration(duration_sec)
    seed = int(request.get("seed", 0))
    guidance_scale = float(request.get("guidance_scale", 2.0))
    stop_threshold = float(request.get("stop_threshold", 0.5))
    min_stop_step = int(request.get("min_stop_step", 5))

    apply_seq_len(model, seq_len)
    seed_all(seed)
    sync_device()
    started = time.perf_counter()
    if request.get("mode") == "batch":
        result = model.generate(
            list(request["texts"]),
            eval_cfg=guidance_scale,
            stop_threshold=stop_threshold,
            min_stop_step=min_stop_step,
            seed=seed,
        )
    else:
        result = model.generate(
            str(request["text"]),
            eval_cfg=guidance_scale,
            stop_threshold=stop_threshold,
            min_stop_step=min_stop_step,
            seed=seed,
        )
    sync_device()
    return {"result": result, "wall_ms": elapsed_ms(started)}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MiDashengLM-Gen warmbench requires CUDA")
    torch.set_num_threads(max(1, args.threads))
    torch.cuda.set_device(args.device)

    cases_path = resolve_repo_path(args.cases)
    model_path = resolve_repo_path(args.model)
    cases = load_cases(cases_path, args.case_set)

    stamp = args.artifact_stamp or timestamp_seconds_local()
    output_dir = args.output_dir or (
        REPO_ROOT / "build" / "logs" / "midashenglm_gen" / f"python_warmbench_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = {
        "argv": sys.argv,
        "model": str(model_path),
        "cases": str(cases_path),
        "case_set": args.case_set,
        "backend": args.backend,
        "device": args.device,
        "threads": args.threads,
    }
    (output_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    sync_device()
    load_started = time.perf_counter()
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = model.to(f"cuda:{args.device}")
    model.eval()
    sync_device()
    load_ms = elapsed_ms(load_started)

    timing_lines = [f"[TIMING] load_ms {load_ms:.6f}"]
    summaries: list[dict[str, Any]] = []

    warmup = cases.get("warmup")
    if warmup is not None:
        warmup_request = dict(warmup)
        warmup_request.setdefault("id", "warmup")
        warmup_request.setdefault("mode", "single")
        run = run_generation(model, warmup_request)
        timing_lines.append(f"[TIMING] warmup.wall_ms {run['wall_ms']:.6f}")

    for index, request in enumerate(cases["requests"]):
        request_id = str(request.get("id", f"request_{index:02d}"))
        run = run_generation(model, request)
        result = run["result"]
        sample_rate = int(result["sample_rate"])
        timing_lines.append(f"[TIMING] {request_id}.wall_ms {run['wall_ms']:.6f}")

        if request.get("mode") == "batch":
            audios = list(result["audio"])
            texts = list(request["texts"])
        else:
            audios = [result["audio"]]
            texts = [str(request["text"])]

        for item_index, audio in enumerate(audios):
            suffix = f"_{item_index:02d}" if len(audios) > 1 else ""
            audio_path = output_dir / f"{request_id}{suffix}.wav"
            write_audio(audio_path, audio, sample_rate)
            summary = summarize_audio(audio, sample_rate, texts[item_index])
            summary.update(
                {
                    "id": request_id,
                    "item": item_index,
                    "audio": str(audio_path),
                    "duration_sec": float(request["duration_sec"]),
                    "internal_seq_len": seq_len,
                    "guidance_scale": float(request.get("guidance_scale", 2.0)),
                    "seed": int(request.get("seed", 0)),
                    "wall_ms": float(run["wall_ms"]),
                }
            )
            summaries.append(summary)
            timing_lines.append(f"[TIMING] {request_id}{suffix}.duration_sec {summary['duration_sec']:.6f}")

    (output_dir / "timing.log").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"MiDashengLM-Gen Python warmbench output: {output_dir}")
    for line in timing_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
