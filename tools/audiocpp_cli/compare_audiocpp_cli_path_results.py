#!/usr/bin/env python3
"""Compare two audiocpp_cli path-test result directories."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import wave
from pathlib import Path

import numpy as np


TIMING_RE = re.compile(r"^\[TIMING[^\]]*\]\s+(\S+)\s+([-+0-9.eE]+)\s*$")
TEXT_OUTPUTS_JSON = "text_outputs.json"

TEXT_OUTPUT_CASES = {
    "citrinet_asr_offline",
    "higgs_audio_stt_paths",
    "hviske_asr_paths",
    "nemotron_asr_offline_paths",
    "nemotron_asr_streaming_paths",
    "parakeet_tdt_offline_long",
    "parakeet_tdt_streaming",
    "qwen3_asr_offline",
    "vibevoice_asr_structured_segments",
    "vibevoice_asr_paths",
}

RELAXED_WAV_HASH_CASES = {
    "miocodec_voice_conversion",
    "miotts_long_text_voice_clone",
}
RELAXED_WAV_COSINE_THRESHOLD = 0.99999999


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a64 = a.astype(np.float64, copy=False).reshape(-1)
    b64 = b.astype(np.float64, copy=False).reshape(-1)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0.0:
        return 1.0 if np.linalg.norm(a64 - b64) == 0.0 else 0.0
    return float(np.dot(a64, b64) / denom)


def read_wav_f32(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if channels <= 0:
        raise RuntimeError(f"{path}: invalid channel count {channels}")
    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_in = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_in[:, 0].astype(np.int32)
            | (bytes_in[:, 1].astype(np.int32) << 8)
            | (bytes_in[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        audio = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"{path}: unsupported WAV sample width {sample_width}")

    return sample_rate, audio.reshape(-1, channels)


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim != 2:
        return audio.reshape(-1)
    return audio.mean(axis=1)


def common_audio(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = min(a.shape[0], b.shape[0])
    channels = min(a.shape[1], b.shape[1])
    return a[:frames, :channels], b[:frames, :channels]


def rms_envelope(samples: np.ndarray, frame: int = 1024, hop: int = 256) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size < frame:
        return np.array([math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))], dtype=np.float32)
    count = 1 + (samples.size - frame) // hop
    out = np.empty(count, dtype=np.float32)
    for index in range(count):
        start = index * hop
        window = samples[start : start + frame].astype(np.float64, copy=False)
        out[index] = math.sqrt(float(np.mean(window * window)))
    return out


def stft_magnitude(samples: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if samples.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[: samples.size] = samples
        samples = padded
    frame_count = 1 + (samples.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.empty((n_fft // 2 + 1, frame_count), dtype=np.float32)
    for index in range(frame_count):
        start = index * hop
        frame = samples[start : start + n_fft] * window
        spec[:, index] = np.abs(np.fft.rfft(frame)).astype(np.float32)
    return spec


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 80) -> np.ndarray:
    freq_bins = n_fft // 2 + 1
    mel_points = np.linspace(float(hz_to_mel(0.0)), float(hz_to_mel(sample_rate / 2.0)), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    filters = np.zeros((n_mels, freq_bins), dtype=np.float32)
    for mel_index in range(n_mels):
        left, center, right = bins[mel_index : mel_index + 3]
        if center > left:
            filters[mel_index, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[mel_index, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def wav_similarity_detail(src_path: Path, baseline_path: Path) -> tuple[float, str]:
    src_rate, src_audio = read_wav_f32(src_path)
    baseline_rate, baseline_audio = read_wav_f32(baseline_path)
    src_common, baseline_common = common_audio(src_audio, baseline_audio)
    src_mono = mono(src_common)
    baseline_mono = mono(baseline_common)
    wav_cos = cosine_similarity(src_common, baseline_common)
    env_cos = cosine_similarity(rms_envelope(src_mono), rms_envelope(baseline_mono))

    src_mag = stft_magnitude(src_mono)
    baseline_mag = stft_magnitude(baseline_mono)
    freq_bins = min(src_mag.shape[0], baseline_mag.shape[0])
    frames = min(src_mag.shape[1], baseline_mag.shape[1])
    src_mag = src_mag[:freq_bins, :frames]
    baseline_mag = baseline_mag[:freq_bins, :frames]
    stft_cos = cosine_similarity(src_mag, baseline_mag)
    log_stft_cos = cosine_similarity(np.log1p(src_mag), np.log1p(baseline_mag))

    if src_rate == baseline_rate and freq_bins > 0 and frames > 0:
        filters = mel_filterbank(src_rate, 1024)
        filters = filters[:, :freq_bins]
        src_mel = np.log1p(filters @ src_mag)
        baseline_mel = np.log1p(filters @ baseline_mag)
        log_mel_cos = cosine_similarity(src_mel, baseline_mel)
        log_mel_text = f"{log_mel_cos:.9f}"
    else:
        log_mel_text = "n/a"

    detail = (
        f"wav_cos={wav_cos:.9f}, rms_env_cos={env_cos:.9f}, "
        f"stft_mag_cos={stft_cos:.9f}, log_stft_mag_cos={log_stft_cos:.9f}, "
        f"log_mel_cos={log_mel_text}, sr={src_rate}/{baseline_rate}, "
        f"frames={src_audio.shape[0]}/{baseline_audio.shape[0]}"
    )
    return wav_cos, detail


def wav_similarity(src_path: Path, baseline_path: Path) -> str:
    _, detail = wav_similarity_detail(src_path, baseline_path)
    return detail


def load_summary(root: Path) -> dict[str, str]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {entry["id"]: entry.get("status", "ok") for entry in data}


def case_dirs(root: Path) -> tuple[dict[str, Path], bool]:
    if (root / "command.json").exists():
        return {root.name: root}, True
    out: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "command.json").exists():
            out[path.name] = path
    return out, False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(case_dir: Path) -> dict[str, tuple[int, str]]:
    hashes: dict[str, tuple[int, str]] = {}
    roots = [case_dir / "outputs"]
    roots.extend(path for path in case_dir.glob("*.json") if path.name not in {"command.json", "requests.json", TEXT_OUTPUTS_JSON})
    for root in roots:
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = [path for path in root.rglob("*") if path.is_file()]
        else:
            continue
        for path in paths:
            rel = path.relative_to(case_dir).as_posix()
            hashes[rel] = (path.stat().st_size, sha256_file(path))
    return hashes


def shorten_text(value: str, limit: int = 180) -> str:
    value = value.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def wav_mismatch_summary(src: Path, baseline: Path, mismatches: list[str]) -> str:
    wavs = [key for key in mismatches if key.lower().endswith(".wav")]
    if not wavs:
        return ""
    parts = []
    for key in wavs[:3]:
        try:
            parts.append(f"{key}: {wav_similarity(src / key, baseline / key)}")
        except Exception as exc:  # noqa: BLE001 - comparison output should report unreadable artifacts.
            parts.append(f"{key}: audio metrics unavailable ({exc})")
    if len(wavs) > 3:
        parts.append(f"{len(wavs) - 3} more wav mismatch(es)")
    return "; ".join(parts)


def relaxed_wav_hash_summary(case_id: str, src: Path, baseline: Path, mismatches: list[str]) -> tuple[bool, str]:
    if case_id not in RELAXED_WAV_HASH_CASES:
        return False, ""
    if any(not key.lower().endswith(".wav") for key in mismatches):
        return False, "non-WAV artifact mismatch is not relaxed"

    parts = []
    for key in mismatches:
        try:
            wav_cos, detail = wav_similarity_detail(src / key, baseline / key)
        except Exception as exc:  # noqa: BLE001 - comparison output should report unreadable artifacts.
            return False, f"{key}: audio metrics unavailable ({exc})"
        parts.append(f"{key}: {detail}")
        if not wav_cos > RELAXED_WAV_COSINE_THRESHOLD:
            return False, (
                f"{key}: wav_cos {wav_cos:.9f} <= "
                f"{RELAXED_WAV_COSINE_THRESHOLD:.9f}"
            )

    threshold_percent = RELAXED_WAV_COSINE_THRESHOLD * 100.0
    return True, f"WAV hash skipped, wav_cos > {threshold_percent:.6f}% required; " + "; ".join(parts)


def json_text_outputs(case_dir: Path) -> dict[str, str] | None:
    path = case_dir / TEXT_OUTPUTS_JSON
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError(f"{path}: expected outputs array")

    result: dict[str, str] = {}
    for index, item in enumerate(outputs):
        if not isinstance(item, dict):
            raise RuntimeError(f"{path}: outputs[{index}] must be an object")
        item_id = item.get("id")
        text = item.get("text")
        if not isinstance(item_id, str) or not isinstance(text, str):
            raise RuntimeError(f"{path}: outputs[{index}] requires string id and text")
        if item_id in result:
            raise RuntimeError(f"{path}: duplicate output id {item_id}")
        result[item_id] = text
    return result


def load_text_outputs(case_dir: Path) -> tuple[dict[str, str] | None, str | None]:
    try:
        return json_text_outputs(case_dir), None
    except Exception as exc:  # noqa: BLE001 - report malformed comparison inputs as data failures.
        return None, str(exc)


def text_mismatch_detail(src_text: dict[str, str], baseline_text: dict[str, str]) -> str:
    src_keys = set(src_text)
    baseline_keys = set(baseline_text)
    if src_keys != baseline_keys:
        missing_src = sorted(baseline_keys - src_keys)
        missing_baseline = sorted(src_keys - baseline_keys)
        parts = [f"text outputs {len(src_keys)} vs {len(baseline_keys)}"]
        if missing_src:
            parts.append("missing current=" + ",".join(missing_src[:3]))
        if missing_baseline:
            parts.append("missing baseline=" + ",".join(missing_baseline[:3]))
        return "; ".join(parts)

    mismatches = [key for key in sorted(src_keys) if src_text[key] != baseline_text[key]]
    if not mismatches:
        return ""
    first = mismatches[0]
    baseline_value = shorten_text(baseline_text[first])
    src_value = shorten_text(src_text[first])
    return (
        f"{len(mismatches)} text_output mismatch(es), first={first}, "
        f"baseline={json.dumps(baseline_value, ensure_ascii=False)}, "
        f"current={json.dumps(src_value, ensure_ascii=False)}"
    )


def timing_summary(case_dir: Path) -> tuple[float, dict[str, float]]:
    stdout_path = case_dir / "stdout.log"
    timings: dict[str, float] = {}
    if not stdout_path.exists():
        return 0.0, timings
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.match(line)
        if not match:
            continue
        timings[match.group(1)] = timings.get(match.group(1), 0.0) + float(match.group(2))
    wall = timings.get("session.wall_ms", 0.0)
    return wall, timings


def timing_detail(src: Path, baseline: Path) -> str:
    src_wall, _ = timing_summary(src)
    baseline_wall, _ = timing_summary(baseline)
    if src_wall <= 0.0 or baseline_wall <= 0.0:
        return "timing unavailable"
    delta = src_wall - baseline_wall
    pct = (delta / baseline_wall) * 100.0
    return f"timing {src_wall:.3f} ms vs {baseline_wall:.3f} ms ({delta:+.3f} ms, {pct:+.2f}%)"


def with_timing(detail: str, src: Path, baseline: Path) -> str:
    return f"{detail}, {timing_detail(src, baseline)}"


def compare_case(case_id: str, src: Path | None, baseline: Path | None, src_status: str, baseline_status: str) -> tuple[str, str]:
    if src is None:
        return "missing-src", "case missing from source"
    if baseline is None:
        return "missing-baseline", "case missing from baseline"
    if src_status != baseline_status:
        return "status", with_timing(f"status {src_status} vs {baseline_status}", src, baseline)

    src_text, src_text_error = load_text_outputs(src)
    baseline_text, baseline_text_error = load_text_outputs(baseline)
    expects_text = case_id in TEXT_OUTPUT_CASES or src_text is not None or baseline_text is not None
    if expects_text:
        if src_text_error is not None:
            return "text-output-json", with_timing("current text_outputs.json malformed: " + src_text_error, src, baseline)
        if baseline_text_error is not None:
            return "text-output-json", with_timing("baseline text_outputs.json malformed: " + baseline_text_error, src, baseline)
        missing = []
        if src_text is None:
            missing.append("current")
        if baseline_text is None:
            missing.append("baseline")
        if missing:
            return "text-output-list", with_timing("missing text_outputs.json in " + "/".join(missing), src, baseline)
        if not src_text and not baseline_text:
            return "text-output-list", with_timing("text_outputs.json has no outputs", src, baseline)
        text_detail = text_mismatch_detail(src_text, baseline_text)
        if text_detail:
            return "text-output", with_timing(text_detail, src, baseline)

    src_artifacts = artifact_hashes(src)
    baseline_artifacts = artifact_hashes(baseline)
    if src_artifacts != baseline_artifacts:
        src_keys = set(src_artifacts)
        baseline_keys = set(baseline_artifacts)
        if src_keys != baseline_keys:
            return "artifact-list", with_timing(f"artifacts {len(src_keys)} vs {len(baseline_keys)}", src, baseline)
        mismatches = [key for key in sorted(src_keys) if src_artifacts[key] != baseline_artifacts[key]]
        audio_summary = wav_mismatch_summary(src, baseline, mismatches)
        detail = f"{len(mismatches)} artifact hash/size mismatch(es), first={mismatches[0]}"
        if audio_summary:
            detail += f"; {audio_summary}"
        relaxed_ok, relaxed_detail = relaxed_wav_hash_summary(case_id, src, baseline, mismatches)
        if relaxed_ok:
            return "ok", with_timing(relaxed_detail, src, baseline)
        if relaxed_detail:
            detail += f"; {relaxed_detail}"
        return "artifact-hash", with_timing(detail, src, baseline)

    return "ok", with_timing("artifacts/text match", src, baseline)


def narrow_single_case_compare(
    src_cases: dict[str, Path],
    baseline_cases: dict[str, Path],
    src_is_single: bool,
    baseline_is_single: bool,
) -> tuple[dict[str, Path], dict[str, Path]]:
    if src_is_single and not baseline_is_single:
        case_id = next(iter(src_cases))
        if case_id in baseline_cases:
            return src_cases, {case_id: baseline_cases[case_id]}
    if baseline_is_single and not src_is_single:
        case_id = next(iter(baseline_cases))
        if case_id in src_cases:
            return {case_id: src_cases[case_id]}, baseline_cases
    return src_cases, baseline_cases


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_audiocpp_cli_path_results.py SRC_DIR BASELINE_DIR", file=sys.stderr)
        return 2
    src_root = Path(sys.argv[1]).resolve()
    baseline_root = Path(sys.argv[2]).resolve()
    src_cases, src_is_single = case_dirs(src_root)
    baseline_cases, baseline_is_single = case_dirs(baseline_root)
    src_cases, baseline_cases = narrow_single_case_compare(src_cases, baseline_cases, src_is_single, baseline_is_single)
    src_status = load_summary(src_root)
    baseline_status = load_summary(baseline_root)

    failures = 0
    print("| case | result | detail |")
    print("| --- | --- | --- |")
    for case_id in sorted(set(src_cases) | set(baseline_cases)):
        result, detail = compare_case(
            case_id,
            src_cases.get(case_id),
            baseline_cases.get(case_id),
            src_status.get(case_id, "ok"),
            baseline_status.get(case_id, "ok"),
        )
        if result != "ok":
            failures += 1
        print(f"| {case_id} | {result} | {markdown_cell(detail)} |")
    print(f"\nsummary: {len(src_cases)} source case(s), {len(baseline_cases)} baseline case(s), {failures} difference(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
