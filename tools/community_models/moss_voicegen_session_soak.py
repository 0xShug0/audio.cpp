"""Soak one long-lived MOSS-VoiceGenerator session with repeated requests.

The community-model bar asks for a long-lived session rather than repeated process
launches: it is the only way to see whether graphs and caches are reused and whether
memory grows request over request. This starts audiocpp_server once, fires N requests at
the same session, and reports per-request latency next to GPU memory sampled from sysfs.

    python3 tools/community_models/moss_voicegen_session_soak.py \
        --server build_hip/bin/audiocpp_server --model /path/to/moss-voicegen.gguf \
        --backend hip --device 0 --card /sys/class/drm/card1/device --requests 12
"""

import argparse
import json
import pathlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

TEXTS = [
    "Good evening, and welcome back to the late show.",
    "The weather stays mild tonight, with a light breeze from the west.",
    "That was Miles Davis, and you are listening to the night programme.",
    "We continue with three quiet pieces, and the news at the top of the hour.",
]
INSTRUCTION = "A warm male radio voice in his fifties, calm, never shrill."


def vram_mib(card: pathlib.Path | None) -> int:
    if card is None:
        return 0
    try:
        return int((card / "mem_info_vram_used").read_text()) // (1024 * 1024)
    except OSError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--card", help="sysfs device dir, e.g. /sys/class/drm/card1/device")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    card = pathlib.Path(args.card) if args.card else None
    config = {
        "host": "127.0.0.1",
        "port": args.port,
        "backend": args.backend,
        "device": args.device,
        "threads": 8,
        "lazy_load": False,
        "models": [
            {
                "id": "voicegen",
                "family": "moss_voicegen",
                "path": args.model,
                "task": "vdes",
                "mode": "offline",
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
        config_path = handle.name

    print(f"idle VRAM: {vram_mib(card)} MiB")
    server = subprocess.Popen(
        [args.server, "--config", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{args.port}/v1/audio/speech"
    try:
        for _ in range(120):                      # wait for the model to load
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2).read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if server.poll() is not None:
                    raise SystemExit("server exited before becoming ready")
                time.sleep(2)
        loaded = vram_mib(card)
        print(f"after load: {loaded} MiB\n")
        print(f"{'req':>4} {'seconds':>8} {'bytes':>9} {'VRAM MiB':>9} {'growth':>8}")

        first = None
        for index in range(args.requests):
            payload = json.dumps({
                "model": "voicegen",
                "input": TEXTS[index % len(TEXTS)],
                "instructions": INSTRUCTION,
                "seed": index,
                "response_format": "wav",
            }).encode()
            request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            start = time.monotonic()
            try:
                body = urllib.request.urlopen(request, timeout=600).read()
            except urllib.error.HTTPError as error:
                print(f"{index:4d} HTTP {error.code}: {error.read()[:200]!r}")
                continue
            elapsed = time.monotonic() - start
            used = vram_mib(card)
            first = used if first is None else first
            print(f"{index:4d} {elapsed:8.2f} {len(body):9d} {used:9d} {used-first:+8d}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        pathlib.Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
