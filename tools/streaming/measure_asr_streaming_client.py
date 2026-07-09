#!/usr/bin/env python3
"""Measure ASR streaming with an auditable incremental-upload client."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "streaming_test"
DEFAULT_SERVER_BIN = REPO_ROOT / "build" / "debug" / "bin" / "audiocpp_server"


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


def audio_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_s": frames / float(sample_rate),
        "audio_bytes_per_second": channels * sample_width * sample_rate,
    }


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


def wait_for_health(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    request = (
        f"GET /health HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("utf-8")
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0) as sock:
                sock.sendall(request)
                response = sock.recv(256)
                if b"200 OK" in response:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server health timeout for {host}:{port}: {last_error}")


def multipart_preamble(boundary: str, model: str, language: str | None, filename: str) -> bytes:
    fields = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n'
            "\r\n"
            f"{model}\r\n"
        ),
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="stream"\r\n'
            "\r\n"
            "true\r\n"
        ),
    ]
    if language:
        fields.append(
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="language"\r\n'
            "\r\n"
            f"{language}\r\n"
        )
    fields.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n"
        "\r\n"
    )
    return "".join(fields).encode("utf-8")


def multipart_trailer(boundary: str) -> bytes:
    return f"\r\n--{boundary}--\r\n".encode("utf-8")


def read_http_headers(reader: Any) -> tuple[str, dict[str, str]]:
    status = reader.readline().decode("iso-8859-1", "replace").strip()
    headers: dict[str, str] = {}
    while True:
        line = reader.readline().decode("iso-8859-1", "replace")
        if line in ("\r\n", "\n", ""):
            break
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return status, headers


def parse_sse_data(buffer: str) -> tuple[list[str], str]:
    events: list[str] = []
    while "\n\n" in buffer:
        block, buffer = buffer.split("\n\n", 1)
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            events.append("\n".join(data_lines))
    return events, buffer


def iter_chunked_response(reader: Any) -> Iterator[bytes]:
    while True:
        size_line = reader.readline().decode("ascii", "replace").strip()
        if not size_line:
            continue
        size = int(size_line.split(";", 1)[0], 16)
        if size == 0:
            reader.readline()
            return
        chunk = reader.read(size)
        reader.read(2)
        yield chunk


def stream_asr_request(
    host: str,
    port: int,
    model: str,
    audio_path: Path,
    output_dir: Path,
    request_label: str,
    audio_chunk_ms: float,
    pace_factor: float,
    language: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    info = audio_info(audio_path)
    boundary = "audiocpp-stream-" + uuid.uuid4().hex
    preamble = multipart_preamble(boundary, model, language, audio_path.name)
    trailer = multipart_trailer(boundary)
    file_bytes = audio_path.read_bytes()
    content_length = len(preamble) + len(file_bytes) + len(trailer)
    bytes_per_second = max(1, int(info["audio_bytes_per_second"]))
    audio_chunk_bytes = max(1, int(bytes_per_second * audio_chunk_ms / 1000.0))

    manifest = {
        "label": request_label,
        "model": model,
        "endpoint": "/v1/audio/transcriptions",
        "method": "POST",
        "stream": True,
        "language": language,
        "audio": info,
        "audio_chunk_ms": audio_chunk_ms,
        "audio_chunk_bytes": audio_chunk_bytes,
        "pace_factor": pace_factor,
        "content_length": content_length,
        "client_input_streaming": "multipart body sent as multiple socket writes",
        "client_response_reading": "SSE response reader runs concurrently with upload",
        "server_body_handling": "current server reads full HTTP body before invoking ASR streaming handler",
    }
    write_json(output_dir / f"{request_label}_request_manifest.json", manifest)

    upload_log_path = output_dir / f"{request_label}_upload_chunks.csv"
    events_path = output_dir / f"{request_label}_events.jsonl"
    transcript_path = output_dir / f"{request_label}_transcript.txt"
    raw_response_path = output_dir / f"{request_label}_response_headers.json"

    request_headers = (
        "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        "Accept: text/event-stream\r\n"
        f"Content-Length: {content_length}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("utf-8")

    start = time.perf_counter()
    upload_complete_ms: float | None = None
    first_event_ms: float | None = None
    first_delta_ms: float | None = None
    done_ms: float | None = None
    server_ttft_ms: float | None = None
    delta_events = 0
    final_text = ""
    errors: list[str] = []
    response_events: list[dict[str, Any]] = []
    response_headers: dict[str, Any] = {}
    reader_error: list[Exception] = []

    def read_response(reader: Any) -> None:
        nonlocal first_event_ms
        nonlocal first_delta_ms
        nonlocal done_ms
        nonlocal server_ttft_ms
        nonlocal delta_events
        nonlocal final_text
        try:
            status, headers = read_http_headers(reader)
            response_headers.update({"status": status, "headers": headers})
            write_json(raw_response_path, response_headers)
            if not status.startswith("HTTP/1.1 200"):
                body = reader.read().decode("utf-8", "replace")
                raise RuntimeError(f"{request_label}: {status}: {body}")

            sse_buffer = ""
            with events_path.open("w", encoding="utf-8") as events_handle:
                transfer_encoding = headers.get("transfer-encoding", "").lower()
                chunks = iter_chunked_response(reader) if "chunked" in transfer_encoding else [reader.read()]
                for raw_chunk in chunks:
                    if not raw_chunk:
                        continue
                    sse_buffer += raw_chunk.decode("utf-8", "replace").replace("\r\n", "\n")
                    events, sse_buffer = parse_sse_data(sse_buffer)
                    for data in events:
                        event_ms = (time.perf_counter() - start) * 1000.0
                        if first_event_ms is None:
                            first_event_ms = event_ms
                        if data == "[DONE]":
                            done_ms = event_ms
                            events_handle.write(json.dumps({"elapsed_ms": event_ms, "data": data}, ensure_ascii=False) + "\n")
                            events_handle.flush()
                            continue
                        event = json.loads(data)
                        event_type = event.get("type")
                        if event_type == "transcript.text.delta":
                            delta_events += 1
                            if first_delta_ms is None:
                                first_delta_ms = event_ms
                        elif event_type == "transcript.text.done":
                            final_text = event.get("text", "")
                            server_ttft_ms = event.get("timing", {}).get("ttft_ms")
                            done_ms = event_ms
                        elif event_type == "error":
                            errors.append(event.get("error", {}).get("message", json.dumps(event)))
                        response_events.append({"elapsed_ms": event_ms, "event": event})
                        events_handle.write(json.dumps(response_events[-1], ensure_ascii=False) + "\n")
                        events_handle.flush()
        except Exception as exc:
            reader_error.append(exc)

    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        reader = sock.makefile("rb")
        reader_thread = threading.Thread(target=read_response, args=(reader,), daemon=True)
        reader_thread.start()
        with upload_log_path.open("w", newline="", encoding="utf-8") as upload_handle:
            writer = csv.writer(upload_handle)
            writer.writerow(["index", "kind", "elapsed_ms", "bytes", "file_offset", "audio_time_end_s"])

            def send_part(index: int, kind: str, data: bytes, file_offset: int = 0) -> None:
                sock.sendall(data)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                audio_time_end_s = min(file_offset, len(file_bytes)) / float(bytes_per_second)
                writer.writerow([index, kind, round(elapsed_ms, 3), len(data), file_offset, round(audio_time_end_s, 6)])
                upload_handle.flush()

            send_part(0, "headers", request_headers)
            send_part(1, "multipart_preamble", preamble)
            index = 2
            offset = 0
            while offset < len(file_bytes):
                end = min(len(file_bytes), offset + audio_chunk_bytes)
                send_part(index, "audio", file_bytes[offset:end], end)
                offset = end
                index += 1
                if pace_factor > 0.0 and offset < len(file_bytes):
                    time.sleep((audio_chunk_ms / 1000.0) * pace_factor)
            send_part(index, "multipart_trailer", trailer, len(file_bytes))
            upload_complete_ms = (time.perf_counter() - start) * 1000.0

        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        reader_thread.join(timeout=timeout_s)
        if reader_thread.is_alive():
            raise RuntimeError(f"{request_label}: timed out waiting for streaming ASR response")
        if reader_error:
            raise reader_error[0]

    if errors:
        raise RuntimeError(f"{request_label}: " + "; ".join(errors))
    if not final_text:
        raise RuntimeError(f"{request_label}: streaming ASR response did not include final text")
    if server_ttft_ms is None:
        raise RuntimeError(f"{request_label}: streaming ASR response did not include server ttft_ms")

    transcript_path.write_text(final_text, encoding="utf-8")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "label": request_label,
        "audio": info,
        "elapsed_ms": elapsed_ms,
        "upload_complete_ms": upload_complete_ms,
        "first_event_ms": first_event_ms,
        "first_delta_ms": first_delta_ms,
        "done_ms": done_ms,
        "server_ttft_ms": server_ttft_ms,
        "client_observed_ttft_ms": first_delta_ms if first_delta_ms is not None else done_ms,
        "client_observed_ttft_after_upload_ms": (
            (first_delta_ms if first_delta_ms is not None else done_ms) - upload_complete_ms
            if upload_complete_ms is not None and (first_delta_ms is not None or done_ms is not None)
            else None
        ),
        "first_event_before_upload_complete": (
            first_event_ms < upload_complete_ms
            if first_event_ms is not None and upload_complete_ms is not None
            else None
        ),
        "first_delta_before_upload_complete": (
            first_delta_ms < upload_complete_ms
            if first_delta_ms is not None and upload_complete_ms is not None
            else None
        ),
        "delta_events": delta_events,
        "text_chars": len(final_text),
        "text_words": len(final_text.split()),
        "artifacts": {
            "request_manifest": str(output_dir / f"{request_label}_request_manifest.json"),
            "upload_chunks": str(upload_log_path),
            "events": str(events_path),
            "response_headers": str(raw_response_path),
            "transcript": str(transcript_path),
        },
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if not path_is_under(output_dir, LOG_ROOT):
        raise RuntimeError(f"--output-dir must be under {LOG_ROOT}")
    if args.audio_chunk_ms <= 0.0:
        raise RuntimeError("--audio-chunk-ms must be positive")
    if args.pace_factor < 0.0:
        raise RuntimeError("--pace-factor must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.server_config)
    host = args.host or config.get("host", "127.0.0.1")
    port = int(args.port or config["port"])
    models = config.get("models") or []
    if not models and not args.model:
        raise RuntimeError("server config contains no models and --model was not provided")
    model = args.model or models[0]["id"]
    write_json(output_dir / "server_config_snapshot.json", config)

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
        wait_for_health(host, port, args.health_timeout_s)

        requests: list[dict[str, Any]] = []
        if args.warmup_audio is not None:
            warmup = stream_asr_request(
                host,
                port,
                model,
                args.warmup_audio,
                output_dir,
                "warmup_discard",
                args.audio_chunk_ms,
                args.pace_factor,
                args.language,
                args.request_timeout_s,
            )
            requests.append(warmup)

        with VramSampler(output_dir / "measured_vram.csv", args.vram_sample_ms / 1000.0):
            measured = stream_asr_request(
                host,
                port,
                model,
                args.audio,
                output_dir,
                "measured",
                args.audio_chunk_ms,
                args.pace_factor,
                args.language,
                args.request_timeout_s,
            )
        requests.append(measured)

        summary = {
            "measurement_kind": "client_incremental_upload_to_streaming_sse_asr",
            "repo": str(args.repo_root),
            "server_command": server_command,
            "server_config": str(args.server_config),
            "model": model,
            "host": host,
            "port": port,
            "audio": str(args.audio),
            "warmup_audio": str(args.warmup_audio) if args.warmup_audio is not None else None,
            "audio_chunk_ms": args.audio_chunk_ms,
            "pace_factor": args.pace_factor,
            "client_input_streaming": "multipart body sent as multiple socket writes",
            "server_body_handling": "current server reads full HTTP body before invoking ASR streaming handler",
            "requests": requests,
            "measured": measured,
            "artifacts": {
                "summary": str(output_dir / "summary.json"),
                "server_config_snapshot": str(output_dir / "server_config_snapshot.json"),
                "framework_log": str(output_dir / "framework.log") if args.start_server else None,
                "server_stdout": str(output_dir / "server_stdout.log") if args.start_server else None,
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
    default_output = LOG_ROOT / f"asr_streaming_{timestamp_slug()}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--warmup-audio", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--audio-chunk-ms", type=float, default=200.0)
    parser.add_argument(
        "--pace-factor",
        type=float,
        default=0.0,
        help="sleep audio_chunk_ms * pace_factor between audio chunk writes; 1.0 approximates real time",
    )
    parser.add_argument("--health-timeout-s", type=float, default=180.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--vram-sample-ms", type=float, default=200.0)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run_measurement(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    measured = summary["measured"]
    print(json.dumps(
        {
            "output_dir": summary["artifacts"]["summary"].rsplit("/", 1)[0],
            "model": summary["model"],
            "audio_duration_s": measured["audio"]["duration_s"],
            "upload_complete_ms": measured["upload_complete_ms"],
            "server_ttft_ms": measured["server_ttft_ms"],
            "client_observed_ttft_ms": measured["client_observed_ttft_ms"],
            "client_observed_ttft_after_upload_ms": measured["client_observed_ttft_after_upload_ms"],
            "first_event_before_upload_complete": measured["first_event_before_upload_complete"],
            "first_delta_before_upload_complete": measured["first_delta_before_upload_complete"],
            "delta_events": measured["delta_events"],
            "text_words": measured["text_words"],
            "vram_peak_mib": summary["vram_peak_mib"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
