"""Compare two Higgs CUDA servers: sampled VRAM, latency, and exact WAV parity.

Requires Python 3 and NVIDIA's NVML library (included with the NVIDIA driver).
Run on an otherwise idle GPU. Both servers are started/stopped by this script.
"""

import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


class Memory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


def check_nvml(status):
    if status != 0:
        raise RuntimeError(f"NVML error {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-server", type=Path, required=True)
    parser.add_argument("--after-server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    for path in (args.before_server, args.after_server, args.model, args.model_spec, args.reference):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    nvml = ctypes.WinDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
    handle = ctypes.c_void_p()
    check_nvml(nvml.nvmlInit_v2())
    check_nvml(nvml.nvmlDeviceGetHandleByIndex_v2(args.device, ctypes.byref(handle)))
    nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory)]

    def used_memory():
        memory = Memory()
        check_nvml(nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
        return memory.used

    idle_bytes = used_memory()

    def wait_idle():
        deadline = time.monotonic() + 30
        consecutive = 0
        while time.monotonic() < deadline:
            consecutive = consecutive + 1 if used_memory() <= idle_bytes + 1024**2 else 0
            if consecutive >= 10:
                return
            time.sleep(.1)
        raise RuntimeError("GPU did not return to its initial idle memory usage")

    polish = "Ptaszki ćwierkają, że Apple chce wrócić na rynek serwerów dostarczając chłonnemu rynkowi AI swoje własne rozwiązania bazujące na zmodyfikowanych układach z serii M."
    english = "The model checkpoint contains the diffusion transformer and layer-fusion weights. The MLLM encoder and VAE are loaded from separate files at runtime, so missing text encoder keys during checkpoint loading are expected."
    request = {"text": polish, "voice_ref": str(args.reference.resolve()), "reference_text": args.reference_text, "seed": 1234, "max_tokens": 1024}
    cases = [
        ("polish_cold", request),
        ("polish_warm", request),
        ("english_same_reference", {**request, "text": english, "seed": 42}),
        ("short_small_cache", {**request, "text": "Hello, this is a memory allocation test.", "seed": 7, "max_tokens": 128}),
        ("english_after_cache_resize", {**request, "text": english, "seed": 123}),
        ("unconditioned_cold", {"text": english, "seed": 42, "max_tokens": 256}),
        ("unconditioned_repeat", {"text": english, "seed": 42, "max_tokens": 256}),
        ("reference_after_unconditioned", {**request, "seed": 4321}),
        ("long_generation", {**request, "text": english + " " + english, "seed": 42, "max_tokens": 1536}),
        ("polish_after_long", request),
    ]
    (root / "requests.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    variants = {}
    try:
        for label, executable in (("before", args.before_server), ("after", args.after_server)):
            wait_idle()
            out = root / label
            out.mkdir(exist_ok=True)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            config = {"host": "127.0.0.1", "port": port, "backend": "cuda", "device": args.device, "threads": 8, "lazy_load": False,
                      "models": [{"id": "higgs", "family": "higgs_audio_tts", "path": str(args.model.resolve()), "model_spec_override": str(args.model_spec.resolve()), "task": "tts", "mode": "offline"}]}
            (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            url = f"http://127.0.0.1:{port}"
            results = []
            with (out / "server.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen([str(executable.resolve()), "--config", str(out / "config.json"), "--no-ui"], stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    deadline = time.monotonic() + 90
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError((out / "server.log").read_text(encoding="utf-8"))
                        try:
                            with urllib.request.urlopen(url + "/health", timeout=1):
                                break
                        except (urllib.error.URLError, TimeoutError):
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(.1)
                    time.sleep(1)
                    loaded_bytes = used_memory()
                    for name, body in cases:
                        samples, monitor_errors = [], []
                        stop = threading.Event()

                        def monitor():
                            try:
                                while not stop.is_set():
                                    samples.append(used_memory())
                                    stop.wait(.01)
                            except Exception as error:
                                monitor_errors.append(error)

                        thread = threading.Thread(target=monitor)
                        thread.start()
                        started = time.perf_counter()
                        try:
                            http_request = urllib.request.Request(url + "/v1/tasks/run", data=json.dumps({"model": "higgs", "request": body}).encode(), headers={"Content-Type": "application/json"})
                            try:
                                with urllib.request.urlopen(http_request, timeout=180) as response:
                                    result = json.loads(response.read())
                                wav = base64.b64decode(result.pop("audio"))
                                (out / (name + ".wav")).write_bytes(wav)
                                result["sha256"] = hashlib.sha256(wav).hexdigest()
                            except urllib.error.HTTPError as error:
                                message = error.read().decode()
                                if not name.startswith("unconditioned_") or "reached max_tokens" not in message:
                                    raise RuntimeError(message) from error
                                result = {"error": message, "sha256": None}
                        finally:
                            http_s = time.perf_counter() - started
                            stop.set()
                            thread.join()
                        if monitor_errors:
                            raise monitor_errors[0]
                        result.update(name=name, http_s=http_s, peak_bytes=max(samples), extra_peak_bytes=max(samples) - loaded_bytes, sample_count=len(samples))
                        results.append(result)
                        print(label, name, f"extra peak {result['extra_peak_bytes'] / 2**30:.3f} GiB", result.get("timing", {}).get("wall_ms", "token-limit error"), flush=True)
                finally:
                    proc.terminate()
                    proc.wait(timeout=20)
            variants[label] = {"initial_idle_bytes": idle_bytes, "loaded_bytes": loaded_bytes, "results": results}
            (out / "results.json").write_text(json.dumps(variants[label], indent=2), encoding="utf-8")
        comparisons = [{"name": a["name"], "audio": a["sha256"] is not None,
                        "identical": a["sha256"] == b["sha256"] and a.get("error") == b.get("error")}
                       for a, b in zip(variants["before"]["results"], variants["after"]["results"])]
        (root / "comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
        if not all(item["identical"] for item in comparisons):
            raise RuntimeError("Output parity failed; see comparison.json")
        print("PASS:", sum(item["audio"] for item in comparisons), "identical WAVs; all responses match")
        wait_idle()
    finally:
        check_nvml(nvml.nvmlShutdown())


if __name__ == "__main__":
    main()
