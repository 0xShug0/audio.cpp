"""Compare shared-model slots with independent single-slot CUDA servers.

Requires an otherwise idle NVIDIA GPU, NVML and a CUDA audiocpp_server build.
Uses only the Python standard library. Input is a task request JSON with fixed
seed and reference audio; all generated WAVs and sampled observations are saved.
"""

import argparse
import base64
import concurrent.futures
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


class GpuMemory:
    class Memory(ctypes.Structure):
        _fields_ = [(field, ctypes.c_ulonglong) for field in ("total", "free", "used")]

    def __init__(self):
        library = "nvml.dll" if os.name == "nt" else ctypes.util.find_library("nvidia-ml")
        if not library:
            raise RuntimeError("NVML is required")
        self.nvml = ctypes.CDLL(library)
        self.handle = ctypes.c_void_p()
        self.check(self.nvml.nvmlInit_v2())
        self.check(self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))
        self.nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Memory)]
        self.idle = self.used()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def used(self):
        value = self.Memory()
        self.check(self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(value)))
        return value.used

    def wait_idle(self):
        deadline = time.monotonic() + 60
        stable = 0
        while time.monotonic() < deadline:
            stable = stable + 1 if self.used() <= self.idle + 1024**2 else 0
            if stable >= 10:
                return
            time.sleep(0.1)
        raise RuntimeError("GPU memory did not return to its initial idle baseline")


def api(url, path, body=None):
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def measure_case(args, gpu, request, kind, count, expected):
    out = args.output / f"{kind}-{count}"
    out.mkdir(parents=True, exist_ok=True)
    resources, urls = [], []
    startup = time.perf_counter()
    try:
        for index in range(count if kind == "instances" else 1):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            model = {
                "id": "higgs", "family": "higgs_audio_tts", "task": "tts",
                "mode": "offline", "path": str(args.model),
                "slots": 1 if kind == "instances" else count,
            }
            if args.spec:
                model["model_spec_override"] = str(args.spec)
            config = {
                "host": "127.0.0.1", "port": port, "backend": "cuda", "device": 0,
                "threads": args.threads, "lazy_load": False, "models": [model],
            }
            config_path = out / f"config-{index}.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            log = (out / f"server-{index}.log").open("w", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    [str(args.server), "--config", str(config_path), "--no-ui"],
                    cwd=args.server.parent, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except BaseException:
                log.close()
                raise
            resources.append((proc, log))
            url = f"http://127.0.0.1:{port}"
            urls.append(url)
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited; see {out / f'server-{index}.log'}")
                try:
                    api(url, "/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
        report = {
            "kind": kind, "requests": count, "server_instances": len(urls),
            "startup_to_ready_s": time.perf_counter() - startup,
            "loaded_bytes_over_idle": gpu.used() - gpu.idle, "phases": [],
        }
        for repeat in range(args.iterations + 1):
            phase = "cold" if repeat == 0 else "warm"
            label = f"{phase}-{repeat}"
            stop, barrier = threading.Event(), threading.Barrier(count + 1)
            samples, active, monitor_errors = [], [], []

            def monitor():
                last_status = 0
                try:
                    while not stop.is_set():
                        now = time.perf_counter()
                        samples.append([now, gpu.used() - gpu.idle])
                        if now - last_status >= 0.05:
                            active.append([now, sum(api(url, "/v1/models")["data"][0]["active_slots"] for url in urls)])
                            last_status = now
                        stop.wait(0.01)
                except Exception as error:
                    monitor_errors.append(str(error))

            def run(index):
                barrier.wait(timeout=30)
                started = time.perf_counter()
                result = api(urls[index if kind == "instances" else 0], "/v1/tasks/run",
                             {"model": "higgs", "request": request})
                elapsed = time.perf_counter() - started
                audio = base64.b64decode(result.pop("audio"))
                (out / f"{label}-{index}.wav").write_bytes(audio)
                return {"index": index, "http_s": elapsed, "sha256": hashlib.sha256(audio).hexdigest(),
                        "bytes": len(audio), "timing": result["timing"]}

            thread = threading.Thread(target=monitor)
            thread.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                    jobs = [pool.submit(run, index) for index in range(count)]
                    started = time.perf_counter()
                    barrier.wait(timeout=30)
                    results = [job.result() for job in jobs]
                    elapsed = time.perf_counter() - started
            finally:
                stop.set()
                thread.join()
            if monitor_errors:
                raise RuntimeError(monitor_errors)
            maximum_active = max(value for _, value in active)
            if maximum_active != count:
                raise RuntimeError(f"Expected {count} concurrent requests; observed {maximum_active}")
            if phase not in expected:
                expected[phase] = results[0]["sha256"]
            if any(result["sha256"] != expected[phase] for result in results):
                raise RuntimeError(f"Audio differs from matching single-slot reference: {kind} {count} {label}")
            entry = {
                "phase": phase, "repeat": repeat, "whole_batch_http_s": elapsed,
                "peak_bytes_over_idle": max(value for _, value in samples),
                "maximum_active": maximum_active, "results": results,
            }
            report["phases"].append(entry)
            (out / f"{label}-samples.json").write_text(json.dumps({"memory": samples, "active": active}))
            print(f"{kind} {count} {label}: {elapsed:.4f}s, {entry['peak_bytes_over_idle'] / 1024**3:.3f} GiB; audio exact", flush=True)
        warm = [phase for phase in report["phases"] if phase["phase"] == "warm"]
        report["warm_median_http_s"] = statistics.median(phase["whole_batch_http_s"] for phase in warm)
        report["warm_max_peak_GiB"] = max(phase["peak_bytes_over_idle"] for phase in warm) / 1024**3
        (out / "results.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        for proc, _ in resources:
            if proc.poll() is None:
                proc.terminate()
        for proc, log in resources:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            finally:
                log.close()
        gpu.wait_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "request-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--iterations", type=int, default=3, help="warm repeats per configuration")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    if args.iterations < 1 or not args.slots or args.slots[0] != 1 or any(n < 1 or n > 16 for n in args.slots):
        parser.error("positive iterations and slots beginning with 1 in the range 1..16 are required")
    for name in ("server", "model", "request_json", "output", "spec"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    if "seed" not in request:
        parser.error("request JSON must contain a fixed seed for parity comparison")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        text=True, creationflags=creationflags,
    ).strip()
    if int(used) != 0:
        raise RuntimeError(f"GPU 0 must be idle before benchmarking; currently {used} MiB used")
    gpu, expected, reports = GpuMemory(), {}, []
    try:
        for count in args.slots:
            reports.append(measure_case(args, gpu, request, "slots", count, expected))
        for count in args.slots:
            if count > 1:
                reports.append(measure_case(args, gpu, request, "instances", count, expected))
        (args.output / "results.json").write_text(json.dumps({
            "server": str(args.server), "model": str(args.model), "request": request,
            "idle_nvml_bytes": gpu.idle, "expected_sha256": expected, "cases": reports,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("PASS all WAVs match their cold/warm single-slot reference", flush=True)
    finally:
        gpu.check(gpu.nvml.nvmlShutdown())


if __name__ == "__main__":
    main()
