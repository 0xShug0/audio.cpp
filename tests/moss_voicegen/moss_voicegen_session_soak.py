"""Soak one long-lived MOSS-VoiceGenerator session with repeated requests.

The community-model bar asks for a long-lived session rather than repeated process
launches: it is the only way to see whether graphs and caches are reused and whether
memory grows request over request. This starts audiocpp_server once, fires N requests at
the same session, and reports per-request latency next to GPU memory sampled from sysfs.

    python3 tests/moss_voicegen/moss_voicegen_session_soak.py \
        --server build/debug/bin/audiocpp_server --model /path/to/moss-voicegen.gguf \
        --backend cuda --device 0 --requests 12 --server-log soak-server.log
"""

import argparse
import io
import wave
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
    "Coming up after the break, an interview recorded earlier this week in Rotterdam.",
    "It is just past two in the morning, and the lines are open.",
    "That track came out in nineteen fifty nine, and it has not aged a day.",
    "Stay with us. There is more music, and a longer story, on the other side of this.",
]
INSTRUCTION = "A warm male radio voice in his fifties, calm, never shrill."


def vram_mib(card: pathlib.Path | None) -> float | None:
    if card is None:
        return None
    try:
        value = int((card / "mem_info_vram_used").read_text())
        if value < 0:
            raise ValueError("negative VRAM telemetry")
        return value / (1024 * 1024)
    except OSError as error:
        raise RuntimeError(f"cannot read requested telemetry from {card}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--card", help="sysfs device dir, e.g. /sys/class/drm/card1/device")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--server-log", required=True, help="save server logs; --log is always enabled")
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be positive")
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

    idle = vram_mib(card)
    print(f"idle VRAM: {str(idle) + ' MiB' if idle is not None else 'unavailable'}")
    log_handle = open(args.server_log, "w") if args.server_log else subprocess.DEVNULL
    server = subprocess.Popen(
        [args.server, "--config", config_path, "--log"],
        stdout=log_handle,
        stderr=subprocess.STDOUT if args.server_log else subprocess.DEVNULL,
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
        else:
            raise RuntimeError("server readiness timed out")
        loaded = vram_mib(card)
        print(f"after load VRAM: {loaded if loaded is not None else 'unavailable'}\n")
        print(f"{'req':>4} {'seconds':>8} {'bytes':>9} {'VRAM MiB':>9} {'growth':>8}")

        first = None
        durations = []
        latencies = []
        failures = {}
        vram_samples = []
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
                message = error.read().decode("utf-8", "replace")
                kind = "no audio" if "answered in text" in message else message[:60]
                failures[kind] = failures.get(kind, 0) + 1
                print(f"{index:4d} FAILED: {kind}")
                continue
            elapsed = time.monotonic() - start
            used = vram_mib(card)
            first = used if first is None else first
            with wave.open(io.BytesIO(body), "rb") as audio:
                frames, rate = audio.getnframes(), audio.getframerate()
                channels, width = audio.getnchannels(), audio.getsampwidth()
                samples = audio.readframes(frames)
                if (rate != 24000 or channels != 1 or width != 2 or frames <= 0 or
                        audio.getcomptype() != "NONE" or len(samples) != frames * channels * width):
                    raise ValueError(f"request {index}: expected complete nonempty 24 kHz mono PCM16 WAV")
            seconds = frames / rate
            durations.append(seconds)
            latencies.append(elapsed)
            if used is not None:
                vram_samples.append(used)
            memory = f"{used:9.2f} {used-first:+8.2f}" if used is not None else "unavailable unavailable"
            print(f"{index:4d} {elapsed:8.2f} {len(body):9d} {memory}")

        print()
        ok = len(latencies)
        print(f"succeeded: {ok}/{args.requests}")
        for kind, count in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"  failed {count:3d} ({100*count/args.requests:4.1f}%): {kind}")
        if ok:
            # The first request carries the one-time runtime build, so it is reported apart.
            steady_rtf = sorted(l / d for l, d in list(zip(latencies, durations))[1:] if d > 0)
            print(f"audio produced: {sum(durations):.1f} s in {sum(latencies):.1f} s wall")
            if steady_rtf:
                mid = steady_rtf[len(steady_rtf) // 2]
                print(f"RTF after the first request: min {steady_rtf[0]:.2f}  median {mid:.2f}  max {steady_rtf[-1]:.2f}")
            print(f"first request RTF: {latencies[0]/durations[0]:.2f} (includes the one-time build)")
            if vram_samples:
                print(f"VRAM after requests: {vram_samples[0]} -> {vram_samples[-1]} MiB; max sampled endpoint {max(vram_samples)} MiB")
            else:
                print("VRAM: unavailable; memory stability was not validated")
        if failures or ok != args.requests:
            raise RuntimeError(f"soak failed: {ok}/{args.requests} requests succeeded")
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        if args.server_log:
            log_handle.close()
        pathlib.Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
