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


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "YuE"
DEFAULT_CASES = REPO_ROOT / "tests" / "yue2" / "yue2_warm_bench_cases.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "build" / "logs" / "yue2" / "python_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference YuE2 warmbench.")
    parser.add_argument("--family", default="yue2")
    parser.add_argument("--model", default="/home/leo/Desktop/YuE2/YuE2-3B")
    parser.add_argument("--vae", default="/home/leo/Desktop/YuE2/YuE2-Vae")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timing-file", type=Path, default=DEFAULT_OUTPUT_ROOT / "python_baseline.log")
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_OUTPUT_ROOT / "summary.json")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def add_reference_path(reference_root: Path) -> None:
    src = resolve_path(reference_root) / "src"
    if not (src / "yue2" / "__init__.py").is_file():
        raise RuntimeError(f"missing YuE2 reference package under {src}")
    sys.path.insert(0, str(src.resolve()))


def configure_runtime(args: argparse.Namespace) -> str:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.set_num_threads(max(1, args.threads))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("YuE2 warmbench requested CUDA, but torch.cuda.is_available() is false")
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(0)
        return f"cuda:{args.device}"
    return "cpu"


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("YuE2 warmbench cases must be a JSON object")
    return payload


def selected_cases(args: argparse.Namespace, cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = args.case or list(cases)
    out = []
    for name in names:
        if name not in cases:
            raise RuntimeError(f"unknown YuE2 warmbench case: {name}")
        case = dict(cases[name])
        case.setdefault("id", name)
        out.append(case)
    return out


def make_generation_config(case: dict[str, Any]):
    from yue2.protocol import GenerationConfig, Sampling

    defaults = GenerationConfig()
    abc = Sampling(
        temperature=float(case.get("abc_temperature", defaults.abc.temperature)),
        top_p=float(case.get("abc_top_p", defaults.abc.top_p)),
        top_k=int(case.get("abc_top_k", defaults.abc.top_k)),
        repetition_penalty=float(case.get("abc_repetition_penalty", defaults.abc.repetition_penalty)),
        penalty_window=int(case.get("abc_penalty_window", defaults.abc.penalty_window)),
        min_tokens=int(case.get("abc_min_tokens", defaults.abc.min_tokens)),
        max_tokens=int(case.get("abc_max_tokens", defaults.abc.max_tokens)),
    )
    semantic = Sampling(
        temperature=float(case.get("semantic_temperature", defaults.semantic.temperature)),
        top_p=float(case.get("semantic_top_p", defaults.semantic.top_p)),
        top_k=int(case.get("semantic_top_k", defaults.semantic.top_k)),
        repetition_penalty=float(case.get("semantic_repetition_penalty", defaults.semantic.repetition_penalty)),
        penalty_window=int(case.get("semantic_penalty_window", defaults.semantic.penalty_window)),
        min_tokens=int(case.get("semantic_min_tokens", defaults.semantic.min_tokens)),
        max_tokens=int(case.get("semantic_max_tokens", defaults.semantic.max_tokens)),
    )
    return GenerationConfig(
        abc=abc,
        semantic=semantic,
        ode_steps=int(case.get("ode_steps", defaults.ode_steps)),
        ode_method=str(case.get("ode_method", defaults.ode_method)),
        context=int(case.get("context", defaults.context)),
        version=str(case.get("version", defaults.version)),
    )


def request_from_case(case: dict[str, Any]) -> dict[str, Any]:
    request = {
        "style": str(case["style"]),
        "lyrics": str(case["lyrics"]),
        "cot": str(case.get("cot", "full")),
        "seed": int(case.get("seed", 831001)),
        "id": str(case.get("id", "song")),
    }
    if "cfg_scale" in case:
        request["cfg_scale"] = float(case["cfg_scale"])
    if "abc" in case:
        request["abc"] = str(case["abc"])
    if "abc_file" in case:
        request["abc"] = resolve_path(case["abc_file"]).read_text(encoding="utf-8")
    return request


def summarize_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError("YuE2 warmbench produced empty audio")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "frames": int(audio.shape[0]) if audio.ndim >= 1 else 0,
        "samples": int(flat.size),
        "duration_sec": float((audio.shape[0] if audio.ndim >= 1 else 0) / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat), dtype=np.float64))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def run_case(pipe: Any, case: dict[str, Any], iteration: int, output_root: Path) -> dict[str, Any]:
    request = request_from_case(case)
    case_dir = output_root / f"{request['id']}_iter{iteration}"
    case_dir.mkdir(parents=True, exist_ok=True)
    config = make_generation_config(case)
    pipe.generation_config = config

    start = time.perf_counter()
    plan_start = time.perf_counter()
    plan = pipe.plan(**request)
    sync_device(str(pipe.device))
    plan_wall = time.perf_counter() - plan_start

    semantic_start = time.perf_counter()
    semantic = pipe.generate_semantic(plan)
    sync_device(str(pipe.device))
    semantic_wall = time.perf_counter() - semantic_start

    nar_start = time.perf_counter()
    latents = pipe.synthesize(semantic)
    sync_device(str(pipe.device))
    nar_wall = time.perf_counter() - nar_start

    decode_start = time.perf_counter()
    audio = pipe.decode(latents)
    sync_device(str(pipe.device))
    decode_wall = time.perf_counter() - decode_start
    wall = time.perf_counter() - start

    audio_path = case_dir / "audio.wav"
    sf.write(audio_path, audio, 48000, subtype="FLOAT")
    plan.save(case_dir / "plan")
    np.save(case_dir / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
    np.save(case_dir / "latent.npy", latents.astype(np.float32))

    result = {
        "case": request["id"],
        "iteration": iteration,
        "request": request,
        "truncated": {"abc": bool(plan.truncated), "semantic": bool(semantic.truncated)},
        "token_counts": {
            "abc": len(plan.abc_ids),
            "prefix": len(plan.prefix),
            "semantic": len(semantic.tokens),
            "latent_frames": int(latents.shape[0]) if getattr(latents, "ndim", 0) == 2 else None,
        },
        "timing_sec": {
            "plan_wall": plan_wall,
            "semantic_wall": semantic_wall,
            "nar_wall": nar_wall,
            "decode_wall": decode_wall,
            "wall": wall,
        },
        "audio": summarize_audio(audio, 48000),
        "paths": {
            "case_dir": str(case_dir),
            "audio": str(audio_path),
        },
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    add_reference_path(args.reference_root)
    device = configure_runtime(args)
    from yue2 import YuE2Pipeline

    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    args.timing_file = resolve_path(args.timing_file)
    args.summary_file = resolve_path(args.summary_file)
    args.timing_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    cases = selected_cases(args, load_cases(args.cases))
    results: list[dict[str, Any]] = []
    model = str(resolve_path(args.model)) if Path(args.model).exists() else args.model
    vae = str(resolve_path(args.vae)) if Path(args.vae).exists() else args.vae

    with YuE2Pipeline.from_pretrained(
        model,
        vae=vae,
        device=device,
        backend="torch",
        local_files_only=args.local_files_only,
        progress=False,
        generation_config=make_generation_config(cases[0]),
    ) as pipe:
        for i in range(args.warmup):
            run_case(pipe, cases[0], -(i + 1), output_root / "warmup")
        for iteration in range(args.iterations):
            for case in cases:
                result = run_case(pipe, case, iteration, output_root)
                results.append(result)
                with args.timing_file.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "family": args.family,
        "model": model,
        "vae": vae,
        "backend": args.backend,
        "device": device,
        "threads": args.threads,
        "cases": [r["case"] for r in results],
        "results": results,
    }
    args.summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
