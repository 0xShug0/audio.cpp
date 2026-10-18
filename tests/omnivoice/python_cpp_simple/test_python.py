#!/usr/bin/env python3

"""Sends one OpenAI-compatible speech request to a running audiocpp_server.

Model inference happens in C++ inside the server. This client uses only the
Python standard library.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

SERVER_URL = "http://127.0.0.1:8080/v1/audio/speech"

TEXT = (
    "আমি একটা টিটিএস মডেল, নাম OmniVoice। "
    "কিভাবে আপনাকে সাহায্য করতে পারি?"
)

OUTPUT = (
    REPO_ROOT
    / "tests/omnivoice/outputs/python_cpp_simple/python_output.wav"
)

payload = {
    "model": "omnivoice",
    "input": TEXT,
    "language": "Bengali",
    "seed": 42,
    "num_inference_steps": 20,
    "guidance_scale": 2.0,
    "response_format": "wav",
    "options": {
        "speed": 1.2
    }
}

request = urllib.request.Request(
    SERVER_URL,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("Sending request to audio.cpp server...")

start = time.perf_counter()

try:
    with urllib.request.urlopen(request, timeout=600) as response:
        audio_bytes = response.read()
except urllib.error.HTTPError as error:
    body = error.read().decode("utf-8", errors="replace")
    raise RuntimeError(
        f"Server returned HTTP {error.code}: {body}"
    ) from error
except urllib.error.URLError as error:
    raise RuntimeError(
        "Could not connect to the audio.cpp server. "
        "Run tests/omnivoice/python_cpp_simple/run_server.sh first."
    ) from error

elapsed = time.perf_counter() - start

if not audio_bytes.startswith(b"RIFF"):
    preview = audio_bytes[:300].decode("utf-8", errors="replace")
    raise RuntimeError(f"Expected WAV response, received: {preview}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_bytes(audio_bytes)

print(f"Python output: {OUTPUT}")
print(f"Response size: {len(audio_bytes)} bytes")
print(f"Request time: {elapsed:.3f} seconds")
