#!/usr/bin/env python3
"""Run model path coverage tests through audiocpp_cli only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "tools" / "audiocpp_cli" / "audiocpp_cli_path_cases.json"
DEFAULT_AUDIOCPP_CLI_BIN = REPO_ROOT / "build" / "bin" / "audiocpp_cli"
DEFAULT_MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_THREADS = 8
DEFAULT_RESOURCE_SAMPLE_MS = 100


def option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def append_key_values(command: list[str], flag: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        command.extend([flag, f"{key}={option_value(value)}"])


@dataclass
class ResourceMetrics:
    observed_peak_rss_bytes: int = 0
    observed_peak_vram_bytes: int = 0
    sample_ms: int = DEFAULT_RESOURCE_SAMPLE_MS
    samples: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "observed_peak_rss_bytes": self.observed_peak_rss_bytes,
            "observed_peak_rss_mib": round(self.observed_peak_rss_bytes / 1024.0 / 1024.0, 2),
            "observed_peak_vram_bytes": self.observed_peak_vram_bytes,
            "observed_peak_vram_mib": round(self.observed_peak_vram_bytes / 1024.0 / 1024.0, 2),
            "sample_ms": self.sample_ms,
            "samples": self.samples,
        }


class ResourceMonitor:
    def __init__(self, root_pid: int, device: int, sample_ms: int) -> None:
        if sample_ms <= 0:
            raise RuntimeError("--resource-sample-ms must be positive")
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("--measure-resources requires the psutil Python package") from exc
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("--measure-resources requires the pynvml Python package") from exc

        self.psutil = psutil
        self.pynvml = pynvml
        self.root = psutil.Process(root_pid)
        self.metrics = ResourceMetrics(sample_ms=sample_ms)
        pynvml.nvmlInit()
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:
            pynvml.nvmlShutdown()
            raise

    def close(self) -> None:
        self.pynvml.nvmlShutdown()

    def sample(self) -> None:
        pids = self.process_tree_pids()
        rss = self.process_tree_rss(pids)
        vram = self.process_tree_vram(pids)
        self.metrics.samples += 1
        self.metrics.observed_peak_rss_bytes = max(self.metrics.observed_peak_rss_bytes, rss)
        self.metrics.observed_peak_vram_bytes = max(self.metrics.observed_peak_vram_bytes, vram)

    def process_tree_pids(self) -> set[int]:
        try:
            processes = [self.root, *self.root.children(recursive=True)]
        except self.psutil.NoSuchProcess:
            return set()
        return {process.pid for process in processes}

    def process_tree_rss(self, pids: set[int]) -> int:
        rss = 0
        for pid in pids:
            try:
                rss += self.psutil.Process(pid).memory_info().rss
            except self.psutil.NoSuchProcess:
                continue
        return rss

    def process_tree_vram(self, pids: set[int]) -> int:
        process_memory: dict[int, int] = {}
        for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            getter = getattr(self.pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                gpu_processes = getter(self.handle)
            except self.pynvml.NVMLError_NotSupported:
                continue
            for gpu_process in gpu_processes:
                pid = int(gpu_process.pid)
                if pid not in pids:
                    continue
                used = int(gpu_process.usedGpuMemory)
                process_memory[pid] = max(process_memory.get(pid, 0), used)
        return sum(process_memory.values())


def validate_resource_monitor(args: argparse.Namespace) -> None:
    if not args.measure_resources:
        return
    if args.resource_sample_ms <= 0:
        raise RuntimeError("--resource-sample-ms must be positive")
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("--measure-resources requires the psutil Python package") from exc
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("--measure-resources requires the pynvml Python package") from exc
    pynvml.nvmlInit()
    try:
        pynvml.nvmlDeviceGetHandleByIndex(args.device)
    finally:
        pynvml.nvmlShutdown()


def load_cases(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_cases(catalog: dict[str, Any], only: set[str], family: str | None) -> list[dict[str, Any]]:
    cases = catalog.get("cases", [])
    out: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["id"]
        if only and case_id not in only:
            continue
        if family and case.get("family") != family:
            continue
        out.append(case)
    if only:
        found = {case["id"] for case in out}
        missing = sorted(only - found)
        if missing:
            raise RuntimeError("unknown case id(s): " + ", ".join(missing))
    return out


def request_to_args(request: dict[str, Any], repo_root: Path) -> list[str]:
    command: list[str] = []
    if "text" in request:
        command.extend(["--text", str(request["text"])])
    if "language" in request:
        command.extend(["--language", str(request["language"])])
    if "audio" in request:
        command.extend(["--audio", str(repo_root / request["audio"])])
    if "voice_ref" in request:
        command.extend(["--voice-ref", str(repo_root / request["voice_ref"])])
    if "voice_id" in request:
        command.extend(["--voice-id", str(request["voice_id"])])
    if "reference_text" in request:
        command.extend(["--reference-text", str(request["reference_text"])])
    if "instruct" in request:
        command.extend(["--instruct", str(request["instruct"])])
    if "task_route" in request:
        command.extend(["--task-route", str(request["task_route"])])
    if "route" in request:
        command.extend(["--task-route", str(request["route"])])
    if "source_audio" in request:
        command.extend(["--source-audio", str(repo_root / request["source_audio"])])
    if "target_voice" in request:
        command.extend(["--target-voice", str(repo_root / request["target_voice"])])
    if "prosody_ref" in request:
        command.extend(["--prosody-ref", str(repo_root / request["prosody_ref"])])
    if "style_ref" in request:
        command.extend(["--style-ref", str(repo_root / request["style_ref"])])
    for key, flag in (
        ("target_text", "--target-text"),
        ("style_ref_text", "--style-ref-text"),
        ("lyrics", "--lyrics"),
        ("track_name", "--track-name"),
        ("speaker", "--speaker"),
        ("duration_seconds", "--duration-seconds"),
        ("repaint_start", "--repaint-start"),
        ("repaint_end", "--repaint-end"),
        ("repaint_mode", "--repaint-mode"),
        ("repaint_strength", "--repaint-strength"),
        ("seed", "--seed"),
        ("max_tokens", "--max-tokens"),
        ("max_steps", "--max-steps"),
        ("temperature", "--temperature"),
        ("top_k", "--top-k"),
        ("top_p", "--top-p"),
        ("repetition_penalty", "--repetition-penalty"),
        ("do_sample", "--do-sample"),
        ("guidance_scale", "--guidance-scale"),
        ("num_inference_steps", "--num-inference-steps"),
        ("text_chunk_size", "--text-chunk-size"),
        ("use_prosody_code", "--use-prosody-code"),
        ("predict_target_prosody", "--predict-target-prosody"),
        ("use_pitch_shift", "--use-pitch-shift"),
        ("source_shift_steps", "--source-shift-steps"),
        ("prosody_shift_steps", "--prosody-shift-steps"),
        ("style_shift_steps", "--style-shift-steps"),
        ("target_duration_seconds", "--target-duration-seconds"),
        ("reference_duration_seconds", "--reference-duration-seconds"),
    ):
        if key in request:
            command.extend([flag, option_value(request[key])])
    if "style_language" in request:
        command.extend(["--style-language", str(request["style_language"])])
    if "emotion" in request:
        command.extend(["--emotion", str(request["emotion"])])
    if "speaking_rate" in request:
        command.extend(["--speaking-rate", option_value(request["speaking_rate"])])
    if "pitch_shift" in request:
        command.extend(["--pitch-shift", option_value(request["pitch_shift"])])
    if "energy_scale" in request:
        command.extend(["--energy-scale", option_value(request["energy_scale"])])
    append_key_values(command, "--style-tag", request.get("style_tags", {}))
    append_key_values(command, "--request-option", request.get("options", {}))
    return command


def maybe_absolute_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    if value.startswith(("resources/", "models/", "build/", "reference/")):
        return str(REPO_ROOT / path)
    return value


def maybe_absolute_path_list(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    parts = [item.strip() for item in value.split(",")]
    if not parts:
        return value
    return ",".join(str(maybe_absolute_path(item)) for item in parts if item)


def materialize_request_paths(request: dict[str, Any]) -> dict[str, Any]:
    out = dict(request)
    if "text_file" in out:
        text_path = Path(maybe_absolute_path(out.pop("text_file")))
        text = text_path.read_text(encoding="utf-8")
        repeat = int(out.pop("text_repeat", 1))
        if repeat <= 0:
            raise RuntimeError("text_repeat must be positive")
        out["text"] = "\n".join(text.strip() for _ in range(repeat))
    if "text_repeat" in out and isinstance(out["text_repeat"], dict):
        if "text" in out:
            raise RuntimeError("request cannot contain both text and text_repeat")
        spec = out.pop("text_repeat")
        text = str(spec["text"])
        chars = int(spec["chars"])
        if not text or chars <= 0:
            raise RuntimeError("text_repeat requires non-empty text and positive chars")
        out["text"] = (text * ((chars // len(text)) + 1))[:chars]
    if "emotion_repeat" in out:
        if "emotion" in out:
            raise RuntimeError("request cannot contain both emotion and emotion_repeat")
        spec = out.pop("emotion_repeat")
        text = str(spec["text"])
        chars = int(spec["chars"])
        if not text or chars <= 0:
            raise RuntimeError("emotion_repeat requires non-empty text and positive chars")
        out["emotion"] = (text * ((chars // len(text)) + 1))[:chars]
    for key in ("audio", "voice_ref", "source_audio", "target_voice", "prosody_ref", "style_ref"):
        if key in out:
            out[key] = maybe_absolute_path(out[key])
    if isinstance(out.get("options"), dict):
        options = dict(out["options"])
        for key, value in options.items():
            if key.endswith("_path") or key.endswith("_file") or key.endswith(".path") or key.endswith(".file"):
                options[key] = maybe_absolute_path(value)
            elif key in {"voice_samples", "vibevoice.voice_samples"}:
                options[key] = maybe_absolute_path_list(value)
        out["options"] = options
    return out


def write_sequence_file(case: dict[str, Any], case_dir: Path) -> Path:
    sequence_path = case_dir / "requests.json"
    requests = [materialize_request_paths(request) for request in case["requests"]]
    with sequence_path.open("w", encoding="utf-8") as handle:
        json.dump({"requests": requests}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return sequence_path


def extract_text_outputs(case: dict[str, Any], stdout: str) -> list[dict[str, str]]:
    request_ids = [str(request.get("id", f"request_{index}")) for index, request in enumerate(case.get("requests", []))]
    outputs: list[dict[str, str]] = []
    request_id = ""
    for line in stdout.splitlines():
        if line.startswith("request_id="):
            request_id = line[len("request_id=") :]
            continue
        if not line.startswith("text_output="):
            continue
        fallback_id = request_ids[len(outputs)] if len(outputs) < len(request_ids) else f"text_output_{len(outputs)}"
        outputs.append({
            "id": request_id if request_id else fallback_id,
            "text": line[len("text_output=") :],
        })
        request_id = ""
    return outputs


def write_text_outputs(case: dict[str, Any], case_dir: Path, stdout: str) -> None:
    outputs = extract_text_outputs(case, stdout)
    with (case_dir / "text_outputs.json").open("w", encoding="utf-8") as handle:
        json.dump({"outputs": outputs}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_model_path(models_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "models":
        return models_root.joinpath(*path.parts[1:])
    if path.parts and path.parts[0] == "assets":
        return REPO_ROOT / path
    return models_root / path


def build_command(args: argparse.Namespace, case: dict[str, Any], case_dir: Path) -> list[str]:
    command = [
        str(args.audiocpp_cli_bin),
        "--task",
        case["task"],
        "--family",
        case["family"],
        "--model",
        str(resolve_model_path(args.models_root, case["model"])),
        "--backend",
        case.get("backend", args.backend),
        "--mode",
        case.get("mode", "offline"),
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
    ]
    append_key_values(command, "--load-option", case.get("load_options", {}))
    append_key_values(command, "--session-option", case.get("session_options", {}))
    for override in getattr(args, "session_option", []) or []:
        command.extend(["--session-option", override])
    if args.log:
        command.append("--log")

    outputs = set(case.get("outputs", []))
    mode = case.get("mode", "offline")
    if mode == "offline":
        sequence_path = write_sequence_file(case, case_dir)
        command.extend(["--request-sequence", str(sequence_path), "--out-dir", str(case_dir / "outputs")])
        if "segments" in outputs:
            command.extend(["--segments-out", str(case_dir / "segments.json")])
        if "turns" in outputs:
            command.extend(["--turns-out", str(case_dir / "turns.json")])
        if "words" in outputs:
            command.extend(["--words-out", str(case_dir / "words.json")])
        return command

    requests = case.get("requests", [])
    if len(requests) != 1:
        raise RuntimeError(f"streaming case {case['id']} must contain exactly one request")
    command.extend(request_to_args(requests[0], REPO_ROOT))
    command.extend(["--chunk-size", str(case.get("chunk_size", 512))])
    if "audio" in outputs:
        command.extend(["--out", str(case_dir / "outputs" / "stream.wav")])
    if "named_audio" in outputs:
        command.extend(["--out-dir", str(case_dir / "outputs")])
    if "segments" in outputs:
        command.extend(["--segments-out", str(case_dir / "segments.json")])
    if "turns" in outputs:
        command.extend(["--turns-out", str(case_dir / "turns.json")])
    if "words" in outputs:
        command.extend(["--words-out", str(case_dir / "words.json")])
    return command


def verify_case(case: dict[str, Any], case_dir: Path, stdout: str) -> None:
    outputs = set(case.get("outputs", []))
    wavs = list((case_dir / "outputs").rglob("*.wav"))
    if "audio" in outputs and not any(path.stat().st_size > 44 for path in wavs):
        raise RuntimeError(f"{case['id']} did not produce a non-empty wav")
    if "named_audio" in outputs and len([path for path in wavs if path.stat().st_size > 44]) < 1:
        raise RuntimeError(f"{case['id']} did not produce named wav outputs")
    if "text" in outputs:
        text_outputs = extract_text_outputs(case, stdout)
        if not text_outputs:
            raise RuntimeError(f"{case['id']} did not print text_output")
        text_output_path = case_dir / "text_outputs.json"
        if not text_output_path.exists() or text_output_path.stat().st_size <= 2:
            raise RuntimeError(f"{case['id']} did not write text_outputs.json")
    if "artifact" in outputs and "artifact=" not in stdout:
        raise RuntimeError(f"{case['id']} did not print an artifact")
    if "artifact" in outputs and "artifact_out[" not in stdout:
        raise RuntimeError(f"{case['id']} did not write an artifact json")
    if "artifact" in outputs:
        artifacts = list((case_dir / "outputs").rglob("*.json"))
        if not artifacts or not any(path.stat().st_size > 2 for path in artifacts):
            raise RuntimeError(f"{case['id']} did not produce artifact json")
    for kind, filename in (("segments", "segments"), ("turns", "turns"), ("words", "words")):
        if kind not in outputs:
            continue
        matches = sorted(case_dir.glob(f"{filename}*.json"))
        if not matches or not any(path.stat().st_size > 2 for path in matches):
            raise RuntimeError(f"{case['id']} did not produce {kind} json")


def run_case(args: argparse.Namespace, case: dict[str, Any], out_root: Path) -> dict[str, Any]:
    case_dir = out_root / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, case, case_dir)
    (case_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    print(f"[RUN] {case['id']}: {case.get('coverage', '')}", flush=True)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    metrics: ResourceMetrics | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        monitor: ResourceMonitor | None = None
        try:
            if args.measure_resources:
                try:
                    monitor = ResourceMonitor(proc.pid, args.device, args.resource_sample_ms)
                    while proc.poll() is None:
                        monitor.sample()
                        time.sleep(args.resource_sample_ms / 1000.0)
                    monitor.sample()
                    metrics = monitor.metrics
                except Exception:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    raise
            returncode = proc.wait()
        finally:
            if monitor is not None:
                monitor.close()

    stdout = stdout_path.read_text(encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"{case['id']} failed with exit code {returncode}; see {case_dir}")
    if "text" in set(case.get("outputs", [])):
        write_text_outputs(case, case_dir, stdout)
    verify_case(case, case_dir, stdout)
    result = {
        "id": case["id"],
        "returncode": returncode,
        "dir": str(case_dir),
        "coverage": case.get("coverage", ""),
    }
    if metrics is not None:
        result["resources"] = metrics.to_json()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--audiocpp-cli-bin", type=Path, default=DEFAULT_AUDIOCPP_CLI_BIN)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "vulkan", "metal", "best"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--measure-resources", action="store_true", help="Record observed peak RSS and GPU memory per case")
    parser.add_argument("--resource-sample-ms", type=int, default=DEFAULT_RESOURCE_SAMPLE_MS)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--only", action="append", default=[], help="Case id or comma-separated case ids")
    parser.add_argument(
        "--session-option",
        action="append",
        default=[],
        help="Extra key=value session option appended to every case (e.g. moss_tts_local.weight_type=f32)",
    )
    parser.add_argument("--family", help="Run only cases for one family")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.models_root.is_absolute():
        args.models_root = REPO_ROOT / args.models_root
    catalog = load_cases(args.cases)
    only = {item.strip() for raw in args.only for item in raw.split(",") if item.strip()}
    cases = selected_cases(catalog, only, args.family)
    if args.list:
        for case in cases:
            print(f"{case['id']}\t{case.get('family', '')}\t{case.get('coverage', '')}")
        if catalog.get("audit_gaps"):
            print("\nAudit gaps:")
            for gap in catalog["audit_gaps"]:
                print(f"{gap.get('family', '')}\t{gap.get('model', '')}\t{gap.get('reason', '')}")
        return 0

    validate_resource_monitor(args)
    if not args.audiocpp_cli_bin.exists():
        raise RuntimeError(f"missing audiocpp_cli binary: {args.audiocpp_cli_bin}")
    stamp = datetime.now().strftime("audiocpp_cli_path_%Y%m%d_%H%M%S")
    out_root = args.out_root or (REPO_ROOT / "build" / "logs" / "audiocpp_cli_path_tests" / stamp)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for case in cases:
        summary.append(run_case(args, case, out_root))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] {len(summary)} case(s), log_dir={out_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
