#!/usr/bin/env python3
"""Probe C++ audiocpp_cli memory across a long-lived MOSS-TTS-Local session.

Runs one case (default: the multi-request ``moss_tts_local_long_lived_session``)
through ``audiocpp_cli`` while sampling the process RSS and its per-PID CUDA
memory (via ``nvidia-smi``), tagging each sample to the request that was active
when it was taken. Reports peak/final memory per request so you can show the
session is stable (no growth) after the first request or two.

Reuses ``run_audiocpp_cli_path_tests`` to build the exact CLI command, so the
run matches the normal path-test harness.

Usage:
    python moss_tts_local_session_probe.py \
        --audiocpp-cli-bin build/windows-cuda-release/bin/audiocpp_cli.exe \
        --backend cuda --out-root build/logs/moss_session_probe
"""

from __future__ import annotations

import argparse
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import run_audiocpp_cli_path_tests as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMING_RE = re.compile(r"^\[TIMING[^\]]*\]\s+(\S+)\s+([-+0-9.eE]+)\s*$")


def gpu_used_mb(pid: int, device: int = 0) -> Optional[float]:
    """Per-process VRAM if available, else device-total used memory.

    Per-process memory is not exposed under Windows WDDM, so we fall back to the
    device total. On a dedicated run that total is dominated by this process, so
    the curve shape (growth vs plateau) is still meaningful.
    """
    def _nvsmi(query: str) -> str:
        try:
            return subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            ).stdout
        except FileNotFoundError:
            return ""

    for line in _nvsmi("--query-compute-apps=pid,used_memory").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return float(parts[1])
            except ValueError:
                break
    # Fallback: device-total used memory (index-th GPU).
    rows = _nvsmi("--query-gpu=memory.used").splitlines()
    if device < len(rows):
        try:
            return float(rows[device].strip())
        except ValueError:
            return None
    return None


class Sampler(threading.Thread):
    def __init__(self, pid: int, interval: float, device: int = 0) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.device = device
        self.samples: list[dict] = []
        self._stop = threading.Event()
        try:
            import psutil

            self.proc = psutil.Process(pid)
        except Exception:  # noqa: BLE001 - psutil optional / process may exit
            self.proc = None

    def run(self) -> None:
        while not self._stop.is_set():
            rss = None
            if self.proc is not None:
                try:
                    rss = self.proc.memory_info().rss / (1024.0 * 1024.0)
                except Exception:  # noqa: BLE001 - process ended
                    rss = None
            self.samples.append({"t": time.perf_counter(), "rss_mb": rss, "gpu_mb": gpu_used_mb(self.pid, self.device)})
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def reconstruct_windows(stdout_lines: list[str], t0: float, t_end: float) -> list[tuple[str, float, float]]:
    """Rebuild per-request time windows from the CLI's own [TIMING] lines.

    audiocpp_cli block-buffers stdout when piped, so real-time request markers
    are unreliable. Instead we use the per-request wall_ms it reports: requests
    run back-to-back at the tail of the process (after the big model load), so we
    lay them out ending at t_end, oldest first.
    """
    req: list[tuple[str, float]] = []
    for line in stdout_lines:
        m = TIMING_RE.match(line)
        if m and m.group(1).startswith("request.") and m.group(1).endswith(".wall_ms"):
            rid = m.group(1)[len("request."):-len(".wall_ms")]
            req.append((rid, float(m.group(2)) / 1000.0))
    total = sum(d for _, d in req)
    windows: list[tuple[str, float, float]] = []
    # End of the last request ~= process end; walk backwards summing durations.
    end = t_end
    for rid, dur in reversed(req):
        windows.append((rid, end - dur, end))
        end -= dur
    windows.reverse()
    return windows


def summarize(samples: list[dict], windows: list[tuple[str, float, float]]) -> list[dict]:
    """Assign each memory sample to the request whose window contains it."""
    out: dict[str, dict] = {rid: {"rss": [], "gpu": []} for rid, _, _ in windows}
    for s in samples:
        for rid, start, end in windows:
            if start <= s["t"] < end:
                if s["rss_mb"] is not None:
                    out[rid]["rss"].append(s["rss_mb"])
                if s["gpu_mb"] is not None:
                    out[rid]["gpu"].append(s["gpu_mb"])
                break
    rows = []
    for rid, _, _ in windows:
        b = out[rid]
        rows.append({
            "request_id": rid,
            "rss_peak_mb": max(b["rss"]) if b["rss"] else None,
            "rss_final_mb": b["rss"][-1] if b["rss"] else None,
            "gpu_peak_mb": max(b["gpu"]) if b["gpu"] else None,
            "gpu_final_mb": b["gpu"][-1] if b["gpu"] else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=runner.DEFAULT_CASES)
    parser.add_argument("--case-id", default="moss_tts_local_long_lived_session")
    parser.add_argument("--audiocpp-cli-bin", type=Path, default=runner.DEFAULT_AUDIOCPP_CLI_BIN)
    parser.add_argument("--models-root", type=Path, default=runner.DEFAULT_MODELS_ROOT)
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "vulkan", "metal", "best"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--session-option", action="append", default=[], help="key=value, e.g. moss_tts_local.weight_type=f32")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--log", action="store_true", help="pass --log through to audiocpp_cli")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    if not args.models_root.is_absolute():
        args.models_root = REPO_ROOT / args.models_root
    catalog = runner.load_cases(args.cases)
    case = next((c for c in catalog.get("cases", []) if c["id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"case not found: {args.case_id}")
    # Inject any extra session options (e.g. weight_type) without editing the shared JSON.
    for kv in args.session_option:
        key, _, value = kv.partition("=")
        case.setdefault("session_options", {})[key] = value

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    case_dir = out_root / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    command = runner.build_command(args, case, case_dir)
    (case_dir / "command.json").write_text(runner.json.dumps(command, indent=2) + "\n", encoding="utf-8")

    print(f"[PROBE] {args.case_id}: {' '.join(map(str, command))}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sampler = Sampler(proc.pid, args.interval, args.device)
    sampler.start()

    # NOTE: audiocpp_cli block-buffers piped stdout, so these lines arrive in a
    # burst at exit; we correlate memory to requests post-hoc via [TIMING] lines.
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line.rstrip("\n"))
    proc.wait()
    t_end = time.perf_counter()
    sampler.stop()
    sampler.join(timeout=2.0)
    stderr = proc.stderr.read() if proc.stderr else ""

    (case_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (case_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"[FAIL] exit {proc.returncode}; see {case_dir}/stderr.log")
        return 1

    windows = reconstruct_windows(stdout_lines, t0, t_end)
    rows = summarize(sampler.samples, windows)
    report = {"case": args.case_id, "samples": len(sampler.samples), "per_request": rows}
    (case_dir / "memory_cpp.json").write_text(runner.json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nrequest            rss_peak  rss_final  gpu_peak  gpu_final (MB)")
    for r in rows:
        print(f"{r['request_id']:<18} {r['rss_peak_mb'] or 0:8.0f} {r['rss_final_mb'] or 0:9.0f} "
              f"{r['gpu_peak_mb'] or 0:8.0f} {r['gpu_final_mb'] or 0:9.0f}")
    print(f"\n[DONE] memory report -> {case_dir/'memory_cpp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
