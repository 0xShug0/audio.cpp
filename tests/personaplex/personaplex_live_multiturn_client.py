#!/usr/bin/env python3
"""Run a true PersonaPlex speech/live multi-turn conversation.

Example turns JSON:

[
  {
    "label": "turn_01_part_availability",
    "audio": "questions/turn_01.wav",
    "user_question": "My dishwasher stopped draining. Can you tell me if the replacement pump is in stock?",
    "assistant_instruction": "Answer as a helpful appliance repair dispatcher. Be concise and specific."
  },
  {
    "label": "turn_02_delay",
    "audio": "questions/turn_02.wav",
    "user_question": "If the part is not in stock, how long would the alternative take?",
    "assistant_instruction": "Continue the same support call and explain the delay clearly."
  },
  {
    "label": "turn_03_cost",
    "audio": "questions/turn_03.wav",
    "user_question": "What should I expect for labor cost, and can I still book a morning appointment?",
    "assistant_instruction": "Continue the same support call and answer both cost and scheduling."
  }
]

The request body is raw PCM over HTTP/1.1 chunked transfer encoding, and the
SSE response reader runs concurrently with upload. This validates the live path;
it is not equivalent to uploading a complete WAV first.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
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


def load_json(path: Path) -> Any:
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


def wav_info(path: Path) -> dict[str, Any]:
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


def load_pcm16_wav(path: Path, max_seconds: float | None) -> tuple[bytes, dict[str, Any]]:
    info = wav_info(path)
    if info["sample_width_bytes"] != 2:
        raise RuntimeError(f"{path}: expected 16-bit PCM WAV, got sample width {info['sample_width_bytes']}")
    if info["channels"] <= 0 or info["sample_rate"] <= 0:
        raise RuntimeError(f"{path}: invalid WAV format")
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        if max_seconds is not None:
            frames = min(frames, int(max_seconds * wav.getframerate()))
        pcm = wav.readframes(frames)
    info["streamed_frames"] = frames
    info["streamed_duration_s"] = frames / float(info["sample_rate"])
    info["streamed_pcm_bytes"] = len(pcm)
    return pcm, info


def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def analyze_pcm16(pcm: bytes, sample_rate: int) -> dict[str, Any]:
    if not pcm:
        return {
            "duration_s": 0.0,
            "max_peak": 0.0,
            "first_peak_gt_0_01_s": None,
            "last_peak_gt_0_01_s": None,
            "silent_spans_peak_lt_0_01": [],
        }
    sample_count = len(pcm) // 2
    samples = struct.unpack("<" + "h" * sample_count, pcm[:sample_count * 2])
    win = max(1, sample_rate // 10)
    first: float | None = None
    last: float | None = None
    silent_start: float | None = None
    silent_spans: list[tuple[float, float]] = []
    max_peak = 0.0
    for offset in range(0, sample_count, win):
        chunk = samples[offset:offset + win]
        peak = max(abs(value) for value in chunk) / 32768.0 if chunk else 0.0
        max_peak = max(max_peak, peak)
        t = offset / float(sample_rate)
        if peak > 0.01:
            if first is None:
                first = t
            last = t
        if peak < 0.01:
            if silent_start is None:
                silent_start = t
        elif silent_start is not None:
            if t - silent_start >= 0.5:
                silent_spans.append((round(silent_start, 2), round(t, 2)))
            silent_start = None
    duration_s = sample_count / float(sample_rate)
    if silent_start is not None and duration_s - silent_start >= 0.5:
        silent_spans.append((round(silent_start, 2), round(duration_s, 2)))
    return {
        "duration_s": duration_s,
        "max_peak": max_peak,
        "first_peak_gt_0_01_s": first,
        "last_peak_gt_0_01_s": last,
        "silent_spans_peak_lt_0_01": silent_spans,
    }


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


def normalize_turns(turns_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    raw = load_json(turns_path)
    if not isinstance(raw, list):
        raise RuntimeError("--turns-json must contain a JSON array")
    if not raw:
        raise RuntimeError("--turns-json must contain at least one turn")
    turns: list[dict[str, Any]] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise RuntimeError(f"turn {index}: expected object")
        label = str(item.get("label") or f"turn_{index:02d}")
        audio_raw = item.get("audio")
        question = str(item.get("user_question") or "").strip()
        instruction = str(item.get("assistant_instruction") or "").strip()
        if not audio_raw:
            raise RuntimeError(f"{label}: missing audio")
        if not question:
            raise RuntimeError(f"{label}: missing user_question")
        if not instruction:
            raise RuntimeError(f"{label}: missing assistant_instruction")
        audio_path = Path(str(audio_raw))
        if not audio_path.is_absolute():
            audio_path = (turns_path.parent / audio_path).resolve()
            if not audio_path.exists():
                audio_path = (repo_root / str(audio_raw)).resolve()
        if not audio_path.exists():
            raise RuntimeError(f"{label}: audio path does not exist: {audio_path}")
        turns.append({
            "label": label,
            "audio": audio_path,
            "user_question": question,
            "assistant_instruction": instruction,
            "expected_answer_points": item.get("expected_answer_points", []),
        })
    return turns


def stream_turn(
    host: str,
    port: int,
    model: str,
    turn: dict[str, Any],
    output_dir: Path,
    audio_chunk_ms: float,
    pace_factor: float,
    max_input_seconds: float | None,
    request_timeout_s: float,
    extra_options: dict[str, str],
) -> dict[str, Any]:
    pcm_input, input_info = load_pcm16_wav(turn["audio"], max_input_seconds)
    sample_rate = int(input_info["sample_rate"])
    channels = int(input_info["channels"])
    bytes_per_second = max(1, int(input_info["audio_bytes_per_second"]))
    chunk_bytes = max(channels * 2, int(bytes_per_second * audio_chunk_ms / 1000.0))
    frame_bytes = channels * 2
    chunk_bytes = max(frame_bytes, (chunk_bytes // frame_bytes) * frame_bytes)

    query = {
        "model": model,
        "input": turn["assistant_instruction"],
        "sample_rate": str(sample_rate),
        "channels": str(channels),
        "sample_format": "s16le",
        "stream_format": "sse",
        "response_format": "pcm",
    }
    query.update(extra_options)
    target = "/v1/audio/speech/live?" + urllib.parse.urlencode(query)

    turn_dir = output_dir / turn["label"]
    turn_dir.mkdir(parents=True, exist_ok=True)
    upload_log_path = turn_dir / "upload_chunks.csv"
    events_path = turn_dir / "events.jsonl"
    response_headers_path = turn_dir / "response_headers.json"
    output_wav = turn_dir / "assistant_response.wav"
    request_manifest_path = turn_dir / "request_manifest.json"

    request_headers = (
        f"POST {target} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Accept: text/event-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n"
        "Expect:\r\n"
        "\r\n"
    ).encode("utf-8")

    manifest = {
        "label": turn["label"],
        "endpoint": "/v1/audio/speech/live",
        "model": model,
        "user_question": turn["user_question"],
        "assistant_instruction": turn["assistant_instruction"],
        "expected_answer_points": turn["expected_answer_points"],
        "input_audio": input_info,
        "audio_chunk_ms": audio_chunk_ms,
        "audio_chunk_bytes": chunk_bytes,
        "pace_factor": pace_factor,
        "client_input_streaming": "raw PCM sent with HTTP/1.1 Transfer-Encoding: chunked",
        "client_response_reading": "SSE response reader runs concurrently with upload",
    }
    write_json(request_manifest_path, manifest)

    start = time.perf_counter()
    upload_complete_ms: float | None = None
    first_event_ms: float | None = None
    first_delta_ms: float | None = None
    done_ms: float | None = None
    ttft_ms: float | None = None
    first_audio_before_input_end: bool | None = None
    request_start_to_first_audio_ms: float | None = None
    input_end_ms: float | None = None
    overlap_ms: float | None = None
    delta_events = 0
    output_pcm = bytearray()
    response_headers: dict[str, Any] = {}
    response_events: list[dict[str, Any]] = []
    errors: list[str] = []
    reader_error: list[Exception] = []

    def read_response(reader: Any) -> None:
        nonlocal first_event_ms, first_delta_ms, done_ms, ttft_ms, first_audio_before_input_end
        nonlocal request_start_to_first_audio_ms, input_end_ms, overlap_ms, delta_events
        try:
            status, headers = read_http_headers(reader)
            response_headers.update({"status": status, "headers": headers})
            write_json(response_headers_path, response_headers)
            if not status.startswith("HTTP/1.1 200"):
                body = reader.read().decode("utf-8", "replace")
                raise RuntimeError(f"{turn['label']}: {status}: {body}")

            sse_buffer = ""
            with events_path.open("w", encoding="utf-8") as events_handle:
                response_chunks = (
                    iter_chunked_response(reader)
                    if "chunked" in headers.get("transfer-encoding", "").lower()
                    else [reader.read()]
                )
                for raw_chunk in response_chunks:
                    if not raw_chunk:
                        continue
                    sse_buffer += raw_chunk.decode("utf-8", "replace").replace("\r\n", "\n")
                    events, sse_buffer = parse_sse_data(sse_buffer)
                    for data in events:
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        if first_event_ms is None:
                            first_event_ms = elapsed_ms
                        if data == "[DONE]":
                            done_ms = elapsed_ms
                            events_handle.write(json.dumps({"elapsed_ms": elapsed_ms, "data": data}) + "\n")
                            events_handle.flush()
                            continue
                        event = json.loads(data)
                        event_type = event.get("type")
                        if event_type == "speech.audio.delta":
                            delta_events += 1
                            if first_delta_ms is None:
                                first_delta_ms = elapsed_ms
                            output_pcm.extend(base64.b64decode(event["audio"]))
                        elif event_type == "speech.audio.done":
                            done_ms = elapsed_ms
                            timing = event.get("timing", {})
                            ttft_ms = timing.get("ttft_ms")
                            first_audio_before_input_end = timing.get("first_audio_before_input_end")
                            request_start_to_first_audio_ms = timing.get("request_start_to_first_audio_ms")
                            input_end_ms = timing.get("input_end_ms")
                            overlap_ms = timing.get("overlap_ms")
                            if request_start_to_first_audio_ms is None:
                                request_start_to_first_audio_ms = ttft_ms
                        elif event_type == "error":
                            errors.append(event.get("error", {}).get("message", json.dumps(event)))
                        response_events.append({"elapsed_ms": elapsed_ms, "event": event})
                        events_handle.write(json.dumps(response_events[-1], ensure_ascii=False) + "\n")
                        events_handle.flush()
        except Exception as exc:
            reader_error.append(exc)

    with socket.create_connection((host, port), timeout=request_timeout_s) as sock:
        sock.settimeout(request_timeout_s)
        reader = sock.makefile("rb")
        reader_thread = threading.Thread(target=read_response, args=(reader,), daemon=True)
        reader_thread.start()
        with upload_log_path.open("w", newline="", encoding="utf-8") as upload_handle:
            writer = csv.writer(upload_handle)
            writer.writerow(["index", "elapsed_ms", "bytes", "audio_offset_bytes", "audio_time_end_s"])
            sock.sendall(request_headers)
            offset = 0
            index = 0
            while offset < len(pcm_input):
                end = min(len(pcm_input), offset + chunk_bytes)
                chunk = pcm_input[offset:end]
                sock.sendall(f"{len(chunk):X}\r\n".encode("ascii"))
                sock.sendall(chunk)
                sock.sendall(b"\r\n")
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                writer.writerow([
                    index,
                    round(elapsed_ms, 3),
                    len(chunk),
                    end,
                    round(end / float(bytes_per_second), 6),
                ])
                upload_handle.flush()
                offset = end
                index += 1
                if pace_factor > 0.0 and offset < len(pcm_input):
                    time.sleep((audio_chunk_ms / 1000.0) * pace_factor)
            sock.sendall(b"0\r\n\r\n")
            upload_complete_ms = (time.perf_counter() - start) * 1000.0

        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        reader_thread.join(timeout=request_timeout_s)
        if reader_thread.is_alive():
            raise RuntimeError(f"{turn['label']}: timed out waiting for live speech response")
        if reader_error:
            raise reader_error[0]

    if errors:
        raise RuntimeError(f"{turn['label']}: " + "; ".join(errors))
    if request_start_to_first_audio_ms is None:
        raise RuntimeError(f"{turn['label']}: response did not include first-audio timing")
    if ttft_ms is None and first_audio_before_input_end is not True:
        raise RuntimeError(f"{turn['label']}: response did not include live TTFT timing")
    if not output_pcm:
        raise RuntimeError(f"{turn['label']}: response produced no audio")

    write_pcm16_wav(output_wav, bytes(output_pcm), sample_rate=sample_rate, channels=1)
    audio_stats = analyze_pcm16(bytes(output_pcm), sample_rate)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = {
        "label": turn["label"],
        "user_question": turn["user_question"],
        "assistant_instruction": turn["assistant_instruction"],
        "expected_answer_points": turn["expected_answer_points"],
        "elapsed_ms": elapsed_ms,
        "upload_complete_ms": upload_complete_ms,
        "first_event_ms": first_event_ms,
        "first_delta_ms": first_delta_ms,
        "done_ms": done_ms,
        "ttft_ms": ttft_ms,
        "first_audio_before_input_end": first_audio_before_input_end,
        "request_start_to_first_audio_ms": request_start_to_first_audio_ms,
        "input_end_ms": input_end_ms,
        "overlap_ms": overlap_ms,
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
        "client_observed_ttft_after_upload_ms": (
            first_delta_ms - upload_complete_ms
            if first_delta_ms is not None and upload_complete_ms is not None
            else None
        ),
        "delta_events": delta_events,
        "input_audio": input_info,
        "output_audio": audio_stats,
        "artifacts": {
            "request_manifest": str(request_manifest_path),
            "upload_chunks": str(upload_log_path),
            "events": str(events_path),
            "response_headers": str(response_headers_path),
            "assistant_response_wav": str(output_wav),
        },
    }
    write_json(turn_dir / "summary.json", result)
    return result


def stream_continuous_conversation(
    host: str,
    port: int,
    model: str,
    turns: list[dict[str, Any]],
    output_dir: Path,
    system_prompt: str,
    audio_chunk_ms: float,
    pace_factor: float,
    turn_gap_seconds: float,
    final_gap_seconds: float,
    max_input_seconds: float | None,
    request_timeout_s: float,
    extra_options: dict[str, str],
) -> dict[str, Any]:
    if turn_gap_seconds < 0.0:
        raise RuntimeError("--turn-gap-seconds must be non-negative")
    if final_gap_seconds < 0.0:
        raise RuntimeError("--final-gap-seconds must be non-negative")

    input_parts: list[bytes] = []
    input_infos: list[dict[str, Any]] = []
    sample_rate: int | None = None
    channels: int | None = None
    sample_width = 2
    timeline: list[dict[str, Any]] = []
    cursor_s = 0.0
    for index, turn in enumerate(turns):
        pcm, info = load_pcm16_wav(turn["audio"], max_input_seconds)
        current_sample_rate = int(info["sample_rate"])
        current_channels = int(info["channels"])
        if sample_rate is None:
            sample_rate = current_sample_rate
            channels = current_channels
        if current_sample_rate != sample_rate or current_channels != channels:
            raise RuntimeError("continuous PersonaPlex test requires all turn WAVs to share sample rate and channels")
        input_parts.append(pcm)
        input_infos.append(info)
        duration_s = len(pcm) / float(max(1, current_sample_rate * current_channels * sample_width))
        timeline.append({
            "label": turn["label"],
            "user_question": turn["user_question"],
            "start_s": round(cursor_s, 6),
            "end_s": round(cursor_s + duration_s, 6),
            "audio": info,
            "expected_answer_points": turn["expected_answer_points"],
        })
        cursor_s += duration_s
        if index + 1 < len(turns) and turn_gap_seconds > 0.0:
            silence_frames = int(round(turn_gap_seconds * current_sample_rate))
            silence = b"\x00" * silence_frames * current_channels * sample_width
            input_parts.append(silence)
            timeline.append({
                "label": f"gap_after_{turn['label']}",
                "start_s": round(cursor_s, 6),
                "end_s": round(cursor_s + silence_frames / float(current_sample_rate), 6),
                "silence_s": silence_frames / float(current_sample_rate),
            })
            cursor_s += silence_frames / float(current_sample_rate)

    if final_gap_seconds > 0.0:
        silence_frames = int(round(final_gap_seconds * sample_rate))
        silence = b"\x00" * silence_frames * channels * sample_width
        input_parts.append(silence)
        timeline.append({
            "label": "final_response_gap",
            "start_s": round(cursor_s, 6),
            "end_s": round(cursor_s + silence_frames / float(sample_rate), 6),
            "silence_s": silence_frames / float(sample_rate),
        })
        cursor_s += silence_frames / float(sample_rate)

    if sample_rate is None or channels is None:
        raise RuntimeError("continuous PersonaPlex test requires at least one turn")

    pcm_input = b"".join(input_parts)
    bytes_per_second = max(1, sample_rate * channels * sample_width)
    chunk_bytes = max(channels * sample_width, int(bytes_per_second * audio_chunk_ms / 1000.0))
    frame_bytes = channels * sample_width
    chunk_bytes = max(frame_bytes, (chunk_bytes // frame_bytes) * frame_bytes)

    query = {
        "model": model,
        "input": system_prompt,
        "sample_rate": str(sample_rate),
        "channels": str(channels),
        "sample_format": "s16le",
        "stream_format": "sse",
        "response_format": "pcm",
    }
    query.update(extra_options)
    target = "/v1/audio/speech/live?" + urllib.parse.urlencode(query)

    conversation_dir = output_dir / "continuous"
    conversation_dir.mkdir(parents=True, exist_ok=True)
    upload_log_path = conversation_dir / "upload_chunks.csv"
    events_path = conversation_dir / "events.jsonl"
    response_headers_path = conversation_dir / "response_headers.json"
    output_wav = conversation_dir / "assistant_response.wav"
    request_manifest_path = conversation_dir / "request_manifest.json"
    input_wav = conversation_dir / "user_input_stream.wav"
    write_pcm16_wav(input_wav, pcm_input, sample_rate=sample_rate, channels=channels)

    request_headers = (
        f"POST {target} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Accept: text/event-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n"
        "Expect:\r\n"
        "\r\n"
    ).encode("utf-8")

    manifest = {
        "endpoint": "/v1/audio/speech/live",
        "model": model,
        "system_prompt": system_prompt,
        "input_wav": str(input_wav),
        "input_duration_s": len(pcm_input) / float(bytes_per_second),
        "timeline": timeline,
        "audio_chunk_ms": audio_chunk_ms,
        "audio_chunk_bytes": chunk_bytes,
        "pace_factor": pace_factor,
        "turn_gap_seconds": turn_gap_seconds,
        "final_gap_seconds": final_gap_seconds,
        "client_input_streaming": "all user turns sent over one HTTP/1.1 chunked live request",
        "client_response_reading": "one SSE response reader runs concurrently with upload",
    }
    write_json(request_manifest_path, manifest)

    start = time.perf_counter()
    upload_complete_ms: float | None = None
    first_event_ms: float | None = None
    first_delta_ms: float | None = None
    done_ms: float | None = None
    ttft_ms: float | None = None
    first_audio_before_input_end: bool | None = None
    request_start_to_first_audio_ms: float | None = None
    input_end_ms: float | None = None
    overlap_ms: float | None = None
    delta_events = 0
    output_pcm = bytearray()
    response_headers: dict[str, Any] = {}
    response_events: list[dict[str, Any]] = []
    errors: list[str] = []
    reader_error: list[Exception] = []

    def read_response(reader: Any) -> None:
        nonlocal first_event_ms, first_delta_ms, done_ms, ttft_ms, first_audio_before_input_end
        nonlocal request_start_to_first_audio_ms, input_end_ms, overlap_ms, delta_events
        try:
            status, headers = read_http_headers(reader)
            response_headers.update({"status": status, "headers": headers})
            write_json(response_headers_path, response_headers)
            if not status.startswith("HTTP/1.1 200"):
                body = reader.read().decode("utf-8", "replace")
                raise RuntimeError(f"continuous: {status}: {body}")

            sse_buffer = ""
            with events_path.open("w", encoding="utf-8") as events_handle:
                response_chunks = (
                    iter_chunked_response(reader)
                    if "chunked" in headers.get("transfer-encoding", "").lower()
                    else [reader.read()]
                )
                for raw_chunk in response_chunks:
                    if not raw_chunk:
                        continue
                    sse_buffer += raw_chunk.decode("utf-8", "replace").replace("\r\n", "\n")
                    events, sse_buffer = parse_sse_data(sse_buffer)
                    for data in events:
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        if first_event_ms is None:
                            first_event_ms = elapsed_ms
                        if data == "[DONE]":
                            done_ms = elapsed_ms
                            events_handle.write(json.dumps({"elapsed_ms": elapsed_ms, "data": data}) + "\n")
                            events_handle.flush()
                            continue
                        event = json.loads(data)
                        event_type = event.get("type")
                        if event_type == "speech.audio.delta":
                            delta_events += 1
                            if first_delta_ms is None:
                                first_delta_ms = elapsed_ms
                            output_pcm.extend(base64.b64decode(event["audio"]))
                        elif event_type == "speech.audio.done":
                            done_ms = elapsed_ms
                            timing = event.get("timing", {})
                            ttft_ms = timing.get("ttft_ms")
                            first_audio_before_input_end = timing.get("first_audio_before_input_end")
                            request_start_to_first_audio_ms = timing.get("request_start_to_first_audio_ms")
                            input_end_ms = timing.get("input_end_ms")
                            overlap_ms = timing.get("overlap_ms")
                            if request_start_to_first_audio_ms is None:
                                request_start_to_first_audio_ms = ttft_ms
                        elif event_type == "error":
                            errors.append(event.get("error", {}).get("message", json.dumps(event)))
                        response_events.append({"elapsed_ms": elapsed_ms, "event": event})
                        events_handle.write(json.dumps(response_events[-1], ensure_ascii=False) + "\n")
                        events_handle.flush()
        except Exception as exc:
            reader_error.append(exc)

    with socket.create_connection((host, port), timeout=request_timeout_s) as sock:
        sock.settimeout(request_timeout_s)
        reader = sock.makefile("rb")
        reader_thread = threading.Thread(target=read_response, args=(reader,), daemon=True)
        reader_thread.start()
        with upload_log_path.open("w", newline="", encoding="utf-8") as upload_handle:
            writer = csv.writer(upload_handle)
            writer.writerow(["index", "elapsed_ms", "bytes", "audio_offset_bytes", "audio_time_end_s"])
            sock.sendall(request_headers)
            offset = 0
            index = 0
            while offset < len(pcm_input):
                end = min(len(pcm_input), offset + chunk_bytes)
                chunk = pcm_input[offset:end]
                sock.sendall(f"{len(chunk):X}\r\n".encode("ascii"))
                sock.sendall(chunk)
                sock.sendall(b"\r\n")
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                writer.writerow([
                    index,
                    round(elapsed_ms, 3),
                    len(chunk),
                    end,
                    round(end / float(bytes_per_second), 6),
                ])
                upload_handle.flush()
                offset = end
                index += 1
                if pace_factor > 0.0 and offset < len(pcm_input):
                    time.sleep((audio_chunk_ms / 1000.0) * pace_factor)
            sock.sendall(b"0\r\n\r\n")
            upload_complete_ms = (time.perf_counter() - start) * 1000.0

        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        reader_thread.join(timeout=request_timeout_s)
        if reader_thread.is_alive():
            raise RuntimeError("continuous: timed out waiting for live speech response")
        if reader_error:
            raise reader_error[0]

    if errors:
        raise RuntimeError("continuous: " + "; ".join(errors))
    if request_start_to_first_audio_ms is None:
        raise RuntimeError("continuous: response did not include first-audio timing")
    if ttft_ms is None and first_audio_before_input_end is not True:
        raise RuntimeError("continuous: response did not include live TTFT timing")
    if not output_pcm:
        raise RuntimeError("continuous: response produced no audio")

    write_pcm16_wav(output_wav, bytes(output_pcm), sample_rate=sample_rate, channels=1)
    audio_stats = analyze_pcm16(bytes(output_pcm), sample_rate)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = {
        "label": "continuous",
        "elapsed_ms": elapsed_ms,
        "upload_complete_ms": upload_complete_ms,
        "first_event_ms": first_event_ms,
        "first_delta_ms": first_delta_ms,
        "done_ms": done_ms,
        "ttft_ms": ttft_ms,
        "first_audio_before_input_end": first_audio_before_input_end,
        "request_start_to_first_audio_ms": request_start_to_first_audio_ms,
        "input_end_ms": input_end_ms,
        "overlap_ms": overlap_ms,
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
        "client_observed_ttft_after_upload_ms": (
            first_delta_ms - upload_complete_ms
            if first_delta_ms is not None and upload_complete_ms is not None
            else None
        ),
        "delta_events": delta_events,
        "input_audio": {
            "sample_rate": sample_rate,
            "channels": channels,
            "streamed_duration_s": len(pcm_input) / float(bytes_per_second),
            "streamed_pcm_bytes": len(pcm_input),
            "turns": input_infos,
        },
        "output_audio": audio_stats,
        "timeline": timeline,
        "artifacts": {
            "request_manifest": str(request_manifest_path),
            "input_wav": str(input_wav),
            "upload_chunks": str(upload_log_path),
            "events": str(events_path),
            "response_headers": str(response_headers_path),
            "assistant_response_wav": str(output_wav),
        },
    }
    write_json(conversation_dir / "summary.json", result)
    return result


def run_client(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if not path_is_under(output_dir, LOG_ROOT):
        raise RuntimeError(f"--output-dir must be under {LOG_ROOT}")
    if args.audio_chunk_ms <= 0.0:
        raise RuntimeError("--audio-chunk-ms must be positive")
    if args.pace_factor < 0.0:
        raise RuntimeError("--pace-factor must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)

    server_config = load_json(args.server_config)
    host = args.host or server_config.get("host", "127.0.0.1")
    port = int(args.port or server_config["port"])
    models = server_config.get("models") or []
    model = args.model or (models[0]["id"] if models else "")
    if not model:
        raise RuntimeError("--model is required when server config has no models")
    write_json(output_dir / "server_config_snapshot.json", server_config)

    option_pairs: dict[str, str] = {}
    for raw in args.request_option:
        key, sep, value = raw.partition("=")
        if not sep or not key:
            raise RuntimeError(f"--request-option must be key=value, got: {raw}")
        option_pairs[key] = value

    turns = normalize_turns(args.turns_json, args.repo_root)
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
        wait_for_health(str(host), port, args.health_timeout_s)

        effective_system_prompt = args.system_prompt
        if not effective_system_prompt:
            effective_system_prompt = turns[0]["assistant_instruction"]
        results = [stream_continuous_conversation(
            str(host),
            port,
            model,
            turns,
            output_dir,
            effective_system_prompt,
            args.audio_chunk_ms,
            args.pace_factor,
            args.turn_gap_seconds,
            args.final_gap_seconds,
            args.max_input_seconds,
            args.request_timeout_s,
            option_pairs,
        )]

        summary = {
            "measurement_kind": "personaplex_true_speech_live_continuous_multiturn",
            "repo": str(args.repo_root),
            "server_command": server_command,
            "server_config": str(args.server_config),
            "turns_json": str(args.turns_json),
            "model": model,
            "host": host,
            "port": port,
            "system_prompt": effective_system_prompt,
            "audio_chunk_ms": args.audio_chunk_ms,
            "pace_factor": args.pace_factor,
            "turn_gap_seconds": args.turn_gap_seconds,
            "final_gap_seconds": args.final_gap_seconds,
            "max_input_seconds": args.max_input_seconds,
            "request_options": option_pairs,
            "turns": results,
            "artifacts": {
                "summary": str(output_dir / "summary.json"),
                "server_config_snapshot": str(output_dir / "server_config_snapshot.json"),
                "framework_log": str(output_dir / "framework.log") if args.start_server else None,
                "server_stdout": str(output_dir / "server_stdout.log") if args.start_server else None,
            },
        }
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
    default_output = LOG_ROOT / f"personaplex_live_multiturn_{timestamp_slug()}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--turns-json", type=Path, required=True)
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--audio-chunk-ms", type=float, default=200.0)
    parser.add_argument("--turn-gap-seconds", type=float, default=1.2)
    parser.add_argument("--final-gap-seconds", type=float, default=6.0)
    parser.add_argument(
        "--system-prompt",
        default="",
    )
    parser.add_argument(
        "--pace-factor",
        type=float,
        default=1.0,
        help="sleep audio_chunk_ms * pace_factor between chunks; 1.0 approximates real time",
    )
    parser.add_argument("--max-input-seconds", type=float)
    parser.add_argument("--request-option", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--health-timeout-s", type=float, default=180.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run_client(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output_dir": summary["artifacts"]["summary"].rsplit("/", 1)[0],
        "model": summary["model"],
        "turns": [
            {
                "label": turn["label"],
                "input_duration_s": turn["input_audio"]["streamed_duration_s"],
                "output_duration_s": turn["output_audio"]["duration_s"],
                "ttft_ms": turn["ttft_ms"],
                "first_audio_before_input_end": turn["first_audio_before_input_end"],
                "request_start_to_first_audio_ms": turn["request_start_to_first_audio_ms"],
                "overlap_ms": turn["overlap_ms"],
                "first_delta_ms": turn["first_delta_ms"],
                "upload_complete_ms": turn["upload_complete_ms"],
                "first_delta_before_upload_complete": turn["first_delta_before_upload_complete"],
                "delta_events": turn["delta_events"],
                "silent_spans": turn["output_audio"]["silent_spans_peak_lt_0_01"],
                "wav": turn["artifacts"]["assistant_response_wav"],
                "input_wav": turn["artifacts"].get("input_wav"),
            }
            for turn in summary["turns"]
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
