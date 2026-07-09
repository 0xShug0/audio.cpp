#!/usr/bin/env python3
"""Measure streaming TTS with an auditable incremental-text client."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "streaming_test"
DEFAULT_SERVER_BIN = REPO_ROOT / "build" / "debug" / "bin" / "audiocpp_server"
DEFAULT_SAMPLE_RATE = 48000


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def split_incremental_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise RuntimeError("--max-chars must be positive")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence
        while len(current) > max_chars:
            split_at = current.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(current[:split_at].strip())
            current = current[split_at:].strip()
    if current:
        chunks.append(current)
    if not chunks:
        raise RuntimeError("input text produced no chunks")
    return chunks


def gpu_memory_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return max(values)


class VramSampler:
    def __init__(self, path: Path, interval_s: float) -> None:
        self.path = path
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "VramSampler":
        if self.interval_s <= 0.0:
            raise RuntimeError("--vram-sample-ms must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        start = time.perf_counter()
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_ms", "vram_mib"])
            while not self.stop_event.is_set():
                writer.writerow([round((time.perf_counter() - start) * 1000.0, 3), gpu_memory_mib()])
                handle.flush()
                time.sleep(self.interval_s)


def read_vram_summary(path: Path) -> dict[str, Any]:
    values: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append(int(row["vram_mib"]))
    if not values:
        raise RuntimeError(f"VRAM sampler wrote no samples: {path}")
    return {
        "vram_start_mib": values[0],
        "vram_peak_mib": max(values),
        "vram_end_mib": values[-1],
        "vram_samples": len(values),
    }


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server health timeout for {base_url}: {last_error}")


def stream_speech_request(base_url: str, model: str, text: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": text,
        "response_format": "pcm",
        "stream_format": "sse",
        "options": {"retry_badcase": False},
    }
    request = urllib.request.Request(
        base_url + "/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    start = time.perf_counter()
    pcm = bytearray()
    delta_events = 0
    first_delta_ms: float | None = None
    done_ms: float | None = None
    ttft_ms: float | None = None
    done_event: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                event_ms = (time.perf_counter() - start) * 1000.0
                event_type = event.get("type")
                if event_type == "error":
                    errors.append(event.get("error", {}).get("message", json.dumps(event)))
                elif event_type == "speech.audio.delta":
                    delta_events += 1
                    if first_delta_ms is None:
                        first_delta_ms = event_ms
                    pcm.extend(base64.b64decode(event["audio"]))
                elif event_type == "speech.audio.done":
                    done_event = event
                    done_ms = event_ms
                    ttft_ms = event.get("timing", {}).get("ttft_ms")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if errors:
        raise RuntimeError("; ".join(errors))
    if ttft_ms is None:
        raise RuntimeError("streaming speech response did not include ttft_ms")
    if not pcm:
        raise RuntimeError("streaming speech response produced no audio")
    return {
        "request": payload,
        "elapsed_ms": elapsed_ms,
        "ttft_ms": ttft_ms,
        "client_first_delta_ms": first_delta_ms,
        "client_done_ms": done_ms,
        "audio_duration_s": len(pcm) / (2.0 * DEFAULT_SAMPLE_RATE),
        "delta_events": delta_events,
        "pcm_bytes": len(pcm),
        "done_event": done_event,
        "pcm": bytes(pcm),
    }


def write_wav(path: Path, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("percentile requires values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if not path_is_under(output_dir, LOG_ROOT):
        raise RuntimeError(f"--output-dir must be under {LOG_ROOT}")
    if args.max_chars <= 0:
        raise RuntimeError("--max-chars must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.server_config)
    host = args.host or config.get("host", "127.0.0.1")
    port = int(args.port or config["port"])
    models = config.get("models") or []
    if not models and not args.model:
        raise RuntimeError("server config contains no models and --model was not provided")
    model = args.model or models[0]["id"]
    base_url = f"http://{host}:{port}"
    write_json(output_dir / "server_config_snapshot.json", config)

    input_text = args.input_text.read_text(encoding="utf-8")
    chunks = split_incremental_text(input_text, args.max_chars)
    (output_dir / "input_text.txt").write_text(input_text, encoding="utf-8")
    write_json(
        output_dir / "input_chunks.json",
        {
            "source": str(args.input_text),
            "input_chars": len(input_text),
            "max_chars": args.max_chars,
            "chunks": [{"index": index, "chars": len(chunk), "text": chunk} for index, chunk in enumerate(chunks)],
        },
    )

    requests_path = output_dir / "requests.jsonl"
    if requests_path.exists():
        requests_path.unlink()

    server: subprocess.Popen[str] | None = None
    server_log_handle = None
    server_command: list[str] | None = None
    try:
        if args.start_server:
            server_command = [
                str(args.server_bin),
                "--config",
                str(args.server_config),
                "--log-file",
                str(output_dir / "framework.log"),
            ]
            server_log_handle = (output_dir / "server_stdout.log").open("w", encoding="utf-8")
            server = subprocess.Popen(
                server_command,
                cwd=args.repo_root,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wait_for_health(base_url, args.health_timeout_s)
        warmup = stream_speech_request(base_url, model, args.warmup_text)
        write_wav(output_dir / "warmup_discard.wav", warmup["pcm"])
        warmup_meta = {key: value for key, value in warmup.items() if key != "pcm"}
        write_json(output_dir / "warmup_discard.json", warmup_meta)

        measured: list[dict[str, Any]] = []
        merged_pcm = bytearray()
        sequence_start = time.perf_counter()
        with VramSampler(output_dir / "measured_vram.csv", args.vram_sample_ms / 1000.0):
            for index, chunk in enumerate(chunks):
                result = stream_speech_request(base_url, model, chunk)
                chunk_pcm = result.pop("pcm")
                merged_pcm.extend(chunk_pcm)
                write_wav(output_dir / f"chunk_{index:03d}.wav", chunk_pcm)
                record = {
                    "index": index,
                    "input_chars": len(chunk),
                    "input_text": chunk,
                    **result,
                }
                measured.append(record)
                with requests_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        sequence_elapsed_ms = (time.perf_counter() - sequence_start) * 1000.0
        write_wav(output_dir / "merged.wav", bytes(merged_pcm))

        ttfts = [float(item["ttft_ms"]) for item in measured]
        summary = {
            "measurement_kind": "incremental_client_text_chunks_to_streaming_sse_tts",
            "repo": str(args.repo_root),
            "server_config": str(args.server_config),
            "server_command": server_command,
            "model": model,
            "memory_saver": args.memory_saver_label,
            "host": host,
            "port": port,
            "input_text": str(output_dir / "input_text.txt"),
            "input_text_source": str(args.input_text),
            "input_chars": len(input_text),
            "chunk_max_chars": args.max_chars,
            "chunk_count": len(chunks),
            "warmup": warmup_meta,
            "measured_sequence_elapsed_ms": sequence_elapsed_ms,
            "total_audio_duration_s": len(merged_pcm) / (2.0 * DEFAULT_SAMPLE_RATE),
            "ttft_first_chunk_ms": ttfts[0],
            "ttft_min_ms": min(ttfts),
            "ttft_p50_ms": percentile(ttfts, 0.50),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "ttft_max_ms": max(ttfts),
            "ttft_mean_ms": sum(ttfts) / len(ttfts),
            "client_first_delta_first_chunk_ms": measured[0]["client_first_delta_ms"],
            "chunks": measured,
            "artifacts": {
                "summary": str(output_dir / "summary.json"),
                "server_config_snapshot": str(output_dir / "server_config_snapshot.json"),
                "framework_log": str(output_dir / "framework.log") if args.start_server else None,
                "server_stdout": str(output_dir / "server_stdout.log") if args.start_server else None,
                "input_text": str(output_dir / "input_text.txt"),
                "chunks": str(output_dir / "input_chunks.json"),
                "requests": str(requests_path),
                "warmup_wav": str(output_dir / "warmup_discard.wav"),
                "warmup_json": str(output_dir / "warmup_discard.json"),
                "merged_wav": str(output_dir / "merged.wav"),
                "vram_csv": str(output_dir / "measured_vram.csv"),
            },
        }
        summary.update(read_vram_summary(output_dir / "measured_vram.csv"))
        write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if server_log_handle is not None:
            server_log_handle.close()


def parse_args() -> argparse.Namespace:
    default_output = LOG_ROOT / f"tts_streaming_{timestamp_slug()}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--input-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--memory-saver-label", required=True)
    parser.add_argument("--max-chars", type=int, default=480)
    parser.add_argument("--warmup-text", default="Short warmup request for streaming timing.")
    parser.add_argument("--health-timeout-s", type=float, default=180.0)
    parser.add_argument("--vram-sample-ms", type=float, default=200.0)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run_measurement(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    keys = (
        "memory_saver",
        "chunk_count",
        "measured_sequence_elapsed_ms",
        "total_audio_duration_s",
        "ttft_first_chunk_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_max_ms",
        "client_first_delta_first_chunk_ms",
        "vram_peak_mib",
    )
    print(json.dumps({key: summary[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
