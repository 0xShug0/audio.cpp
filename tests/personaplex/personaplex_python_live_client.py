#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.parse
import wave
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import sphn


def read_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if width != 2:
        raise RuntimeError(f"{path}: expected 16-bit PCM WAV")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if audio.size == 0:
        return {"duration_s": 0.0, "silent_spans_peak_lt_0_01": [], "max_peak": 0.0}
    win = max(1, sample_rate // 10)
    spans: list[list[float]] = []
    start: float | None = None
    max_peak = 0.0
    for offset in range(0, audio.size, win):
        peak = float(np.max(np.abs(audio[offset:offset + win]))) if offset < audio.size else 0.0
        max_peak = max(max_peak, peak)
        t = offset / float(sample_rate)
        if peak < 0.01:
            if start is None:
                start = t
        elif start is not None:
            if t - start >= 0.5:
                spans.append([round(start, 2), round(t, 2)])
            start = None
    duration = audio.size / float(sample_rate)
    if start is not None and duration - start >= 0.5:
        spans.append([round(start, 2), round(duration, 2)])
    return {
        "duration_s": duration,
        "silent_spans_peak_lt_0_01": spans,
        "max_peak": max_peak,
    }


async def run_turn(
    url: str,
    audio_path: Path,
    output_wav: Path,
    chunk_ms: float,
    pace_factor: float,
    final_gap_seconds: float,
) -> dict[str, Any]:
    audio, sample_rate = read_wav_mono_f32(audio_path)
    if sample_rate != 24000:
        raise RuntimeError(f"{audio_path}: expected 24000 Hz input for Python PersonaPlex live, got {sample_rate}")
    if final_gap_seconds < 0.0:
        raise RuntimeError("--final-gap-seconds must be non-negative")
    if final_gap_seconds > 0.0:
        audio = np.concatenate([
            audio,
            np.zeros(int(round(final_gap_seconds * sample_rate)), dtype=np.float32),
        ])

    opus_writer = sphn.OpusStreamWriter(sample_rate)
    opus_reader = sphn.OpusStreamReader(sample_rate)
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000.0))
    output_parts: list[np.ndarray] = []
    text_parts: list[str] = []
    first_audio_ms: float | None = None
    upload_done_ms: float | None = None
    handshake_ms: float | None = None
    last_message_time = time.perf_counter()
    start = time.perf_counter()

    async def receive(ws: aiohttp.ClientWebSocketResponse) -> None:
        nonlocal first_audio_ms, handshake_ms, last_message_time
        async for msg in ws:
            last_message_time = time.perf_counter()
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            data = msg.data
            if not data:
                continue
            kind = data[0]
            payload = data[1:]
            now_ms = (time.perf_counter() - start) * 1000.0
            if kind == 0:
                handshake_ms = now_ms
            elif kind == 1:
                opus_reader.append_bytes(payload)
                while True:
                    pcm = opus_reader.read_pcm()
                    if pcm.shape[-1] == 0:
                        break
                    if first_audio_ms is None:
                        first_audio_ms = now_ms
                    output_parts.append(np.asarray(pcm, dtype=np.float32).reshape(-1))
            elif kind == 2:
                text_parts.append(payload.decode("utf-8", "replace"))

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, max_msg_size=0) as ws:
            receive_task = asyncio.create_task(receive(ws))
            deadline = time.perf_counter() + 120.0
            while handshake_ms is None:
                if time.perf_counter() > deadline:
                    raise RuntimeError("timed out waiting for Python live handshake")
                await asyncio.sleep(0.01)
            for offset in range(0, audio.size, chunk_samples):
                chunk = audio[offset:offset + chunk_samples]
                opus_writer.append_pcm(chunk)
                while True:
                    payload = opus_writer.read_bytes()
                    if not payload:
                        break
                    await ws.send_bytes(b"\x01" + payload)
                if pace_factor > 0 and offset + chunk_samples < audio.size:
                    await asyncio.sleep((chunk_ms / 1000.0) * pace_factor)
            upload_done_ms = (time.perf_counter() - start) * 1000.0
            wait_deadline = time.perf_counter() + 12.0
            while time.perf_counter() < wait_deadline:
                idle_s = time.perf_counter() - last_message_time
                if first_audio_ms is not None and idle_s > 2.0:
                    break
                await asyncio.sleep(0.05)
            await ws.close()
            await receive_task

    output = np.concatenate(output_parts) if output_parts else np.zeros(0, dtype=np.float32)
    write_wav(output_wav, output, sample_rate)
    return {
        "input_wav": str(audio_path),
        "output_wav": str(output_wav),
        "input_duration_s": audio.size / float(sample_rate),
        "output": analyze_audio(output, sample_rate),
        "handshake_ms": handshake_ms,
        "first_audio_ms": first_audio_ms,
        "upload_done_ms": upload_done_ms,
        "first_audio_before_upload_done": (
            first_audio_ms < upload_done_ms
            if first_audio_ms is not None and upload_done_ms is not None
            else None
        ),
        "text": "".join(text_parts),
    }


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--turns-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--voice-prompt", default="NATF2.pt")
    parser.add_argument("--chunk-ms", type=float, default=200.0)
    parser.add_argument("--pace-factor", type=float, default=1.0)
    parser.add_argument("--final-gap-seconds", type=float, default=0.0)
    args = parser.parse_args()

    turns = json.loads(args.turns_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for turn in turns:
        query = urllib.parse.urlencode({
            "text_prompt": turn["assistant_instruction"],
            "voice_prompt": args.voice_prompt,
        })
        url = f"http://{args.host}:{args.port}/api/chat?{query}".replace("http://", "ws://", 1)
        audio = Path(turn["audio"])
        if not audio.is_absolute():
            audio = (args.turns_json.parent / audio).resolve()
        turn_dir = args.out_dir / turn["label"]
        summary = await run_turn(
            url,
            audio,
            turn_dir / "assistant_response.wav",
            args.chunk_ms,
            args.pace_factor,
            args.final_gap_seconds,
        )
        summary["label"] = turn["label"]
        summary["user_question"] = turn["user_question"]
        summary["assistant_instruction"] = turn["assistant_instruction"]
        (turn_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summaries.append(summary)
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
