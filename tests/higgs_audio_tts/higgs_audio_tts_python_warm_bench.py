#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "HiggsAudioV3TTS" / "sglang-omni"
DEFAULT_MODEL = REPO_ROOT / "models" / "higgs-audio-v3-tts-4b"
DEFAULT_REFERENCE_AUDIO = REPO_ROOT / "resources" / "sample.wav"
DEFAULT_REFERENCE_TEXT = (
    "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. "
    "22,500 times longer than you."
)
DEFAULT_TEXT = (
    "The control room confirmed the overnight checks. The backup recorder is stable, the relay stayed online, "
    "and the field team can restart the survey after one final timestamp review."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference Higgs Audio v3 TTS warmbench.")
    parser.add_argument("--family", default="higgs_audio_tts")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--server-timeout-sec", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-out", type=Path, default=Path("higgs_audio_tts_python_audio.wav"))
    parser.add_argument("--timing-file", type=Path, default=Path("higgs_audio_tts_python_timing.log"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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
    return [
        {
            "text": args.text or DEFAULT_TEXT,
            "reference_audio": str(args.reference_audio),
            "reference_text": args.reference_text,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        }
    ]


def request_text(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Higgs TTS warmbench request missing {key}")
    return value


def request_int(request: dict[str, Any], key: str, fallback: int) -> int:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        raise RuntimeError(f"Higgs TTS warmbench request field {key} must be an integer")
    return int(value)


def request_float(request: dict[str, Any], key: str, fallback: float) -> float:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        raise RuntimeError(f"Higgs TTS warmbench request field {key} must be numeric")
    return float(value)


def summarize_audio(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    flat = audio.reshape(-1)
    if flat.size == 0:
        raise RuntimeError("Higgs TTS warmbench summary received empty audio")
    if audio.ndim == 1:
        frames = int(audio.shape[0])
        channels = 1
    elif audio.ndim == 2:
        frames = int(audio.shape[0])
        channels = int(audio.shape[1])
    else:
        raise RuntimeError(f"Higgs TTS warmbench expected 1D or 2D audio, got shape {audio.shape}")
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError as exc:
        raise RuntimeError("Higgs TTS warmbench requires torch in the reference environment") from exc


def server_env(args: argparse.Namespace) -> dict[str, str]:
    reference_root = resolve_repo_path(args.reference_root).resolve()
    package = reference_root / "sglang_omni" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing Higgs reference package: {package}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(reference_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env["OMP_NUM_THREADS"] = str(args.threads)
    torch_lib = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "torch" / "lib"
    ld_paths = [str(torch_lib), "/usr/local/cuda-12.9/lib64", env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(path for path in ld_paths if path)
    return env


def start_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    model_path = resolve_repo_path(args.model).resolve()
    log_path = args.timing_file.with_name("python.server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--allowed-local-media-path",
        str(REPO_ROOT),
        "--log-level",
        "warning",
        "--decode-mode",
        "sync",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs",
        "1",
        "stages.1.factory_args.dtype",
        "float32",
        "stages.3.factory_args.dtype",
        "float32",
    ]
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=server_env(args),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()
    return process


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def wait_for_server(args: argparse.Namespace, process: subprocess.Popen[str]) -> None:
    deadline = time.perf_counter() + args.server_timeout_sec
    url = f"http://{args.host}:{args.port}/health"
    last_error = ""
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Higgs TTS reference server exited before readiness with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= int(response.status) < 500:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Higgs TTS reference server exited during readiness with code {process.returncode}"
                        )
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"Higgs TTS reference server did not become ready: {last_error}")


def speech_payload(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    reference_audio = request.get("reference_audio", request.get("voice_ref"))
    if not isinstance(reference_audio, str) or not reference_audio.strip():
        raise RuntimeError("Higgs TTS warmbench request missing reference_audio")
    reference_path = resolve_repo_path(Path(reference_audio)).resolve()
    if not reference_path.is_file():
        raise RuntimeError(f"Higgs TTS warmbench reference audio is missing: {reference_path}")
    seed = request_int(request, "seed", args.seed)
    payload = {
        "model": str(resolve_repo_path(args.model).resolve()),
        "input": request_text(request, "text"),
        "voice": str(request.get("voice", "warmbench")),
        "response_format": "wav",
        "ref_audio": str(reference_path),
        "ref_text": request_text(request, "reference_text"),
        "max_new_tokens": request_int(request, "max_tokens", args.max_tokens),
        "temperature": request_float(request, "temperature", args.temperature),
        "top_p": request_float(request, "top_p", args.top_p),
        "top_k": request_int(request, "top_k", args.top_k),
        "seed": seed,
    }
    if "repetition_penalty" in request:
        payload["repetition_penalty"] = request_float(request, "repetition_penalty", None)
    return payload


def post_speech(args: argparse.Namespace, payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/v1/audio/speech",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600.0) as response:
            body = response.read()
            if int(response.status) != 200:
                raise RuntimeError(f"Higgs TTS speech request failed with HTTP {response.status}: {body[:4096]!r}")
            return body
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"Higgs TTS speech request failed with HTTP {exc.code}: {body[:4096]!r}") from exc


def run_request(args: argparse.Namespace, request: dict[str, Any], audio_path: Path) -> tuple[dict[str, Any], float]:
    payload = speech_payload(args, request)
    seed_all(int(payload["seed"]))
    started = time.perf_counter()
    audio_bytes = post_speech(args, payload)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if not audio_bytes:
        raise RuntimeError("Higgs TTS speech response was empty")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(audio_bytes)
    summary = summarize_audio(audio_path)
    return summary, wall_ms


def write_timing(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, 0))
        args.port = int(sock.getsockname()[1])
    requests = load_requests(args)
    if not requests:
        raise RuntimeError("Higgs TTS warmbench request sequence is empty")
    output_dir = args.output_dir or args.audio_out.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_lines = ["higgs_audio_tts.python.model_load_excluded=1"]
    server = start_server(args)
    timing_lines.append(f"higgs_audio_tts.python.server_pid={server.pid}")
    try:
        wait_for_server(args, server)
        for _ in range(max(0, args.warmup)):
            run_request(args, requests[0], output_dir / "warmup.wav")
        steps: list[dict[str, Any]] = []
        for request_index, request in enumerate(requests):
            total_ms = 0.0
            summary: dict[str, Any] | None = None
            audio_path = output_dir / f"audio_{request_index}.wav"
            for _ in range(max(1, args.iterations)):
                summary, wall_ms = run_request(args, request, audio_path)
                total_ms += wall_ms
            average_ms = total_ms / float(max(1, args.iterations))
            timing_lines.append(f"higgs_audio_tts.python.request_{request_index}.wall_ms={average_ms:.6f}")
            print(f"higgs_audio_tts.python.wall_ms={average_ms}")
            steps.append(
                {
                    "request_index": request_index,
                    "stems": [
                        {
                            "name": "audio",
                            "audio": str(audio_path),
                            "summary": summary,
                        }
                    ],
                    "metrics": {"wall_ms": average_ms},
                }
            )
        summary_payload = {"family": "higgs_audio_tts", "backend": args.backend, "sequence_steps": steps}
        if args.summary_file:
            args.summary_file.parent.mkdir(parents=True, exist_ok=True)
            args.summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"summary_json={json.dumps(summary_payload, ensure_ascii=False)}")
    finally:
        stop_server(server)
        write_timing(args.timing_file, timing_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
