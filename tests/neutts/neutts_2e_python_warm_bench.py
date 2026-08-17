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
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "reference" / "neutts"
DEFAULT_CASES = REPO_ROOT / "tests" / "neutts" / "neutts_2e_warm_bench_cases.json"
DEFAULT_BACKBONE = REPO_ROOT / "models" / "NeuTTS-2E"
DEFAULT_OUT_ROOT = REPO_ROOT / "build" / "logs" / "neutts" / "python_baseline"
FAMILY = "neutts"
SAMPLE_RATE = 24_000


def load_cases(path: Path) -> list[dict[str, Any]]:
    root = json.loads(path.read_text(encoding="utf-8"))
    cases = root.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(f"case file has no cases: {path}")
    return cases


def select_cases(cases: list[dict[str, Any]], only: set[str]) -> list[dict[str, Any]]:
    selected = []
    for case in cases:
        if case.get("family") != FAMILY:
            continue
        if only and str(case.get("id")) not in only:
            continue
        selected.append(case)
    if only:
        found = {str(case.get("id")) for case in selected}
        missing = sorted(only - found)
        if missing:
            raise RuntimeError(f"unknown {FAMILY} case id(s): {', '.join(missing)}")
    if not selected:
        raise RuntimeError(f"no {FAMILY} cases selected")
    return selected


def sync_cuda(torch_mod: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch_mod.cuda.synchronize(torch_mod.device(device))


def reset_peak(torch_mod: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch_mod.cuda.reset_peak_memory_stats(torch_mod.device(device))


def memory_snapshot(torch_mod: Any, device: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        import psutil

        out["rss_mb"] = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    if device.startswith("cuda"):
        dev = torch_mod.device(device)
        out["cuda_alloc_mb"] = torch_mod.cuda.memory_allocated(dev) / (1024.0 * 1024.0)
        out["cuda_peak_mb"] = torch_mod.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
        out["cuda_reserved_mb"] = torch_mod.cuda.memory_reserved(dev) / (1024.0 * 1024.0)
    return out


def configure_torch_runtime(torch_mod: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch_mod.backends.cuda.matmul.allow_tf32 = False
        torch_mod.backends.cudnn.allow_tf32 = False
        torch_mod.backends.cudnn.benchmark = False
    torch_mod.use_deterministic_algorithms(True, warn_only=True)


def seed_runtime(torch_mod: Any, device: str, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch_mod.manual_seed(seed)
    if device.startswith("cuda"):
        torch_mod.cuda.manual_seed_all(seed)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def audio_summary(audio: Any, sample_rate: int) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim not in (1, 2):
        raise RuntimeError(f"NeuTTS warmbench expected 1D or 2D audio, got shape {values.shape}")
    flat = values.reshape(-1)
    if flat.size == 0:
        raise RuntimeError("NeuTTS warmbench received empty audio")
    frames = int(values.shape[0])
    channels = 1 if values.ndim == 1 else int(values.shape[1])
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": int(flat.size),
        "frames": frames,
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def case_text(request: dict[str, Any], key: str, fallback: str) -> str:
    value = request.get(key, fallback)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"request requires non-empty string field '{key}'")
    return value


def case_int(request: dict[str, Any], key: str, fallback: int) -> int:
    return int(request[key]) if key in request and request[key] is not None else fallback


def case_float(request: dict[str, Any], key: str, fallback: float) -> float:
    return float(request[key]) if key in request and request[key] is not None else fallback


def run_request(tts: Any, torch_mod: Any, device: str, request: dict[str, Any], output_path: Path) -> dict[str, Any]:
    request_id = case_text(request, "id", "request")
    speaker = case_text(request, "speaker", "emily")
    emotion = case_text(request, "emotion", "neutral")
    text = case_text(request, "text", "")
    seed = case_int(request, "seed", 1234)
    temperature = case_float(request, "temperature", 1.0)
    top_k = case_int(request, "top_k", 50)

    # NeuTTS keeps the seed on the session object. Update it per request so the
    # model remains loaded while each request stays reproducible.
    tts._seed = seed
    seed_runtime(torch_mod, device, seed)
    reset_peak(torch_mod, device)
    sync_cuda(torch_mod, device)
    start = time.perf_counter()
    wav = tts.infer(text, speaker=speaker, emotion=emotion, temperature=temperature, top_k=top_k)
    sync_cuda(torch_mod, device)
    inference_ms = (time.perf_counter() - start) * 1000.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, SAMPLE_RATE)
    audio_seconds = wav_duration_seconds(output_path)
    rtf = (inference_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0
    result = memory_snapshot(torch_mod, device)
    result.update(
        {
            "request_id": request_id,
            "speaker": speaker,
            "emotion": emotion,
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "inference_ms": inference_ms,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
            "x_realtime": (1.0 / rtf) if rtf > 0.0 else 0.0,
            "audio": str(output_path),
            "summary": audio_summary(wav, SAMPLE_RATE),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuTTS-2E Python reference warmbench.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--backbone", default=str(DEFAULT_BACKBONE))
    parser.add_argument("--codec", default="neuphonic/neucodec")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codec-device", default="cuda")
    parser.add_argument("--only", action="append", default=[], help="case id(s), comma-separated allowed")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--enable-watermark", action="store_true")
    args = parser.parse_args()

    reference_root = args.reference_root.resolve()
    if not (reference_root / "neutts").is_dir():
        raise RuntimeError(f"missing NeuTTS reference checkout: {reference_root}")
    sys.path.insert(0, str(reference_root))

    import torch
    from neutts import NeuTTS2E

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("NeuTTS warmbench requested CUDA, but CUDA is not available")
    configure_torch_runtime(torch, args.device)

    only = {item.strip() for raw in args.only for item in raw.split(",") if item.strip()}
    cases = select_cases(load_cases(args.cases), only)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tts = NeuTTS2E(
        backbone_repo=args.backbone,
        backbone_device=args.device,
        codec_repo=args.codec,
        codec_device=args.codec_device,
        seed=None,
    )
    if not args.enable_watermark:
        tts.watermarker = None
    sync_cuda(torch, args.device)

    for case in cases:
        case_dir = out_root / str(case["id"])
        outputs_dir = case_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        stdout_lines = [
            f"family={FAMILY}",
            f"task={case.get('task', 'tts')}",
            f"mode={case.get('mode', 'offline')}",
            "torch_tf32=0",
            "deterministic_algorithms=warn_only",
        ]
        memory: dict[str, Any] = {
            "case": case["id"],
            "device": args.device,
            "codec_device": args.codec_device,
            "requests": [],
        }
        steps: list[dict[str, Any]] = []

        warmup = case.get("warmup")
        if warmup is not None:
            warm_path = outputs_dir / f"{case_text(warmup, 'id', 'warmup')}.wav"
            warm = run_request(tts, torch, args.device, warmup, warm_path)
            stdout_lines.append(f"warmup_id={warm['request_id']}")
            stdout_lines.append(f"warmup_inference_ms={warm['inference_ms']:.6f}")
            memory["warmup"] = warm

        total_inference_ms = 0.0
        for index, request in enumerate(case["requests"]):
            request_id = case_text(request, "id", f"request_{index}")
            result = run_request(tts, torch, args.device, request, outputs_dir / f"{request_id}.wav")
            total_inference_ms += float(result["inference_ms"])
            memory["requests"].append(result)
            stdout_lines.extend(
                [
                    f"request_index={index}",
                    f"request_id={request_id}",
                    f"text[{index}]={case_text(request, 'text', '')}",
                    f"[TIMING] request.{request_id}.inference_ms {result['inference_ms']:.6f}",
                    f"audio_seconds={result['audio_seconds']:.6f}",
                    f"rtf={result['rtf']:.6f}",
                    f"x_realtime={result['x_realtime']:.6f}",
                    f"audio_out[{index}]={result['audio']}",
                ]
            )
            step = {
                "request_index": index,
                "id": request_id,
                "speaker": result["speaker"],
                "emotion": result["emotion"],
                "seed": result["seed"],
                "temperature": result["temperature"],
                "top_k": result["top_k"],
                "text_length": len(case_text(request, "text", "")),
                "stems": [
                    {
                        "name": "audio",
                        "audio": result["audio"],
                        "summary": result["summary"],
                    }
                ],
                "metrics": {
                    "wall_ms": result["inference_ms"],
                    "inference_ms": result["inference_ms"],
                    "rtf": result["rtf"],
                    "x_realtime": result["x_realtime"],
                },
            }
            steps.append(step)
            stdout_lines.append(f"summary_json[{index}]=" + json.dumps(result["summary"], ensure_ascii=False, separators=(",", ":")))
            print(
                f"[REF] {case['id']}/{request_id}: {result['inference_ms']:.2f} ms, "
                f"{result['audio_seconds']:.2f}s audio, RTF={result['rtf']:.3f}, "
                f"{result['x_realtime']:.2f}x",
                flush=True,
            )

        stdout_lines.append(f"[TIMING] session.inference_ms {total_inference_ms:.6f}")
        summary = {
            "family": FAMILY,
            "task": case.get("task", "tts"),
            "mode": case.get("mode", "offline"),
            "device": args.device,
            "codec_device": args.codec_device,
            "case_name": case["id"],
            "timing_boundary": "per-request inference only; model load and wav write are excluded",
            "sequence_steps": steps,
        }
        stdout_lines.append("summary_json=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        (case_dir / "command.json").write_text(
            json.dumps(
                {
                    "reference": "neutts_2e_python_warm_bench.py",
                    "case": case["id"],
                    "cases": str(args.cases),
                    "reference_root": str(reference_root),
                    "backbone": args.backbone,
                    "codec": args.codec,
                    "device": args.device,
                    "codec_device": args.codec_device,
                    "enable_watermark": args.enable_watermark,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (case_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        (case_dir / "memory.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
        (case_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print("summary_json=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
