#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
NEMO_ROOT = REPO_ROOT / "reference" / "nemo"
DEFAULT_MODEL = REPO_ROOT / "models" / "magpie_tts_multilingual_357m" / "magpie_tts_multilingual_357m.nemo"
DEFAULT_CASES = REPO_ROOT / "tests" / "magpie_tts" / "magpie_tts_warm_bench_cases.json"
DEFAULT_LOG_DIR = REPO_ROOT / "build" / "logs" / "magpie_tts" / "python_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference MagpieTTS warmbench.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_cases(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return None, payload
    if not isinstance(payload, dict):
        raise RuntimeError("MagpieTTS warmbench cases must be a JSON object or list")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError("MagpieTTS warmbench cases require a non-empty requests array")
    warmup = payload.get("warmup")
    if warmup is not None and not isinstance(warmup, dict):
        raise RuntimeError("MagpieTTS warmbench warmup must be an object")
    return warmup, requests


def request_text(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"MagpieTTS warmbench request missing {key}")
    return value


def request_int(request: dict[str, Any], key: str, fallback: int) -> int:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        raise RuntimeError(f"MagpieTTS warmbench request field {key} must be an integer")
    return int(value)


def request_bool(request: dict[str, Any], key: str, fallback: bool) -> bool:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise RuntimeError(f"MagpieTTS warmbench request field {key} must be boolean")


def request_float(request: dict[str, Any], key: str, fallback: float) -> float:
    return float(request.get(key, fallback))


def seed_all(seed: int, device: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)


def summarize_audio(audio: np.ndarray, sample_rate: int, wall_ms: float) -> dict[str, Any]:
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    count = max(1, int(flat.size))
    return {
        "sample_rate": int(sample_rate),
        "channels": 1,
        "samples": int(flat.size),
        "duration_sec": float(flat.size / sample_rate) if sample_rate else 0.0,
        "sum": float(np.sum(flat, dtype=np.float64)) if flat.size else 0.0,
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)) if flat.size else 0.0,
        "rms": float(np.sqrt(np.sum(flat.astype(np.float64) ** 2) / count)) if flat.size else 0.0,
        "min": float(np.min(flat)) if flat.size else 0.0,
        "max": float(np.max(flat)) if flat.size else 0.0,
        "synthesize_wall_ms": float(wall_ms),
    }


def load_model(model_path: Path, device: str):
    if str(NEMO_ROOT) not in sys.path:
        sys.path.insert(0, str(NEMO_ROOT))
    from nemo.collections.tts.models import MagpieTTSModel

    model = MagpieTTSModel.restore_from(str(model_path), map_location=torch.device(device))
    model.eval().to(device)
    for tokenizer in getattr(model.tokenizer, "tokenizers", {}).values():
        g2p = getattr(tokenizer, "g2p", None)
        if g2p is not None and hasattr(g2p, "phoneme_probability"):
            g2p.phoneme_probability = 1.0
    return model


def prime_text_normalizers(model: Any, requests: list[dict[str, Any]]) -> None:
    for request in requests:
        if request_bool(request, "apply_text_normalization", True):
            language = request_text(request, "language")
            model._get_normalized_text(transcript="Warmup text.", language=language)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MagpieTTS warmbench requires CUDA")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(max(1, args.threads))
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    warmup, requests = load_cases(args.cases)
    output_dir = resolve_repo_path(args.output_dir)
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / "timing.log"
    summary_path = output_dir / "summary.jsonl"
    timing_path.write_text("", encoding="utf-8")
    summary_path.write_text("", encoding="utf-8")

    model_path = resolve_repo_path(args.model)
    device = f"cuda:{args.device}"
    print(f"[magpie_tts] model={model_path} device={device}")
    model = load_model(model_path, device)
    sample_rate = int(getattr(model, "output_sample_rate", getattr(model, "sample_rate", 22050)))
    prime_requests = list(requests)
    if warmup is not None:
        prime_requests.insert(0, warmup)
    prime_text_normalizers(model, prime_requests)

    def run_request(request: dict[str, Any], *, write_output: bool) -> dict[str, Any]:
        request_id = request_text(request, "id")
        seed_all(request_int(request, "seed", 1234), args.device)
        inference = model.inference_parameters
        old_temperature = inference.temperature
        old_topk = inference.topk
        old_cfg_scale = inference.cfg_scale
        old_max_decoder_steps = inference.max_decoder_steps
        inference.temperature = request_float(request, "temperature", old_temperature)
        inference.topk = request_int(request, "top_k", old_topk)
        inference.cfg_scale = request_float(request, "guidance_scale", old_cfg_scale)
        inference.max_decoder_steps = request_int(request, "max_tokens", old_max_decoder_steps)
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        try:
            audio, audio_len = model.do_tts(
                request_text(request, "text"),
                language=request_text(request, "language"),
                apply_TN=request_bool(request, "apply_text_normalization", True),
                use_cfg=request_bool(request, "use_cfg", True),
                speaker_index=request_int(request, "speaker_index", 0),
            )
            torch.cuda.synchronize(args.device)
            ended = time.perf_counter()
        finally:
            inference.temperature = old_temperature
            inference.topk = old_topk
            inference.cfg_scale = old_cfg_scale
            inference.max_decoder_steps = old_max_decoder_steps
        length = int(audio_len[0].item()) if torch.is_tensor(audio_len) else int(audio_len)
        waveform = audio[0, :length].detach().float().cpu().numpy()
        wall_ms = (ended - started) * 1000.0
        result = summarize_audio(waveform, sample_rate, wall_ms)
        result.update(
            {
                "id": request_id,
                "language": request_text(request, "language"),
                "voice_id": request.get("voice_id", ""),
                "speaker_index": request_int(request, "speaker_index", 0),
            }
        )
        if write_output:
            audio_path = outputs_dir / f"{request_id}.wav"
            sf.write(str(audio_path), waveform, sample_rate)
            result["output_audio"] = str(audio_path)
        return result

    if warmup is not None:
        print(f"[magpie_tts] warmup={warmup.get('id', 'warmup')}")
        run_request(warmup, write_output=False)

    with timing_path.open("a", encoding="utf-8") as timing_file, summary_path.open("a", encoding="utf-8") as summary_file:
        for iteration in range(args.iterations):
            for request in requests:
                result = run_request(request, write_output=True)
                timing_file.write(f"[TIMING] request.{result['id']}.wall_ms {result['synthesize_wall_ms']:.4f}\n")
                summary_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                timing_file.flush()
                summary_file.flush()
                print(
                    f"[magpie_tts] {result['id']} iter={iteration} "
                    f"wall_ms={result['synthesize_wall_ms']:.3f} "
                    f"duration_sec={result['duration_sec']:.3f}"
                )
    print(f"[magpie_tts] outputs={outputs_dir}")
    print(f"[magpie_tts] timing={timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
