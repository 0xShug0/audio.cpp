from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_KOKORO_ROOT = REPO_ROOT / "reference" / "kokoro"
if str(REFERENCE_KOKORO_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_KOKORO_ROOT))

kFixedWarmupText = "This is a fixed warmup request for the speech session benchmark."
kCaseCatalogPath = REPO_ROOT / "tools" / "kokoro_tts" / "kokoro_tts_warm_bench_cases.txt"


def timestamp_seconds_local() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_case_catalog(path: Path) -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {}
    current_case = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_case = line[1:-1]
            cases.setdefault(current_case, [])
            continue
        if not current_case:
            raise RuntimeError("Kokoro warm bench case catalog entry is missing a [case] header")
        cases[current_case].append(line)
    return cases


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm repeated Python benchmark for the Kokoro TTS reference model.")
    parser.add_argument("--model", type=Path, default=Path("models/kokoro-82m-v1_0-ggml"))
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        default=[],
        help="Repeat to benchmark multiple requests in one report.",
    )
    parser.add_argument(
        "--case-name",
        action="append",
        dest="case_names",
        default=[],
        help="Append one named text case from the shared warm bench catalog.",
    )
    parser.add_argument("--voice-id", default="af_heart")
    parser.add_argument("--style-language", default="")
    parser.add_argument("--speaking-rate", type=float, default=1.0)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-text", default=kFixedWarmupText)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--audio-out", type=Path, default=Path("kokoro_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--timing-file", type=Path)
    parser.add_argument("--artifact-stamp", default="")
    return parser.parse_args()


def resolve_language_code(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    aliases = {
        "a": "a",
        "en": "a",
        "en-us": "a",
        "american": "a",
        "american english": "a",
        "b": "b",
        "en-gb": "b",
        "gb": "b",
        "uk": "b",
        "british": "b",
        "british english": "b",
        "e": "e",
        "es": "e",
        "spanish": "e",
        "f": "f",
        "fr": "f",
        "fr-fr": "f",
        "french": "f",
        "h": "h",
        "hi": "h",
        "hindi": "h",
        "i": "i",
        "it": "i",
        "italian": "i",
        "j": "j",
        "ja": "j",
        "japanese": "j",
        "p": "p",
        "pt": "p",
        "pt-br": "p",
        "portuguese": "p",
        "brazilian portuguese": "p",
        "z": "z",
        "zh": "z",
        "zh-cn": "z",
        "mandarin": "z",
        "chinese": "z",
        "mandarin chinese": "z",
    }
    if normalized not in aliases:
        raise RuntimeError(f"unsupported Kokoro language: {value}")
    return aliases[normalized]


def voice_language_code(voice_id: str) -> str:
    if len(voice_id) < 2 or voice_id[1] not in ("f", "m"):
        raise RuntimeError(f"invalid Kokoro voice id: {voice_id}")
    return voice_id[0]


def resolve_requested_language_code(style_language: str, voice_id: str) -> str:
    requested = resolve_language_code(style_language) if style_language else voice_language_code(voice_id)
    expected = voice_language_code(voice_id)
    if requested != expected:
        raise RuntimeError(
            f"Kokoro voice/language mismatch: voice {voice_id} requires lang_code={expected} but got {requested}"
        )
    return requested


def load_reference_model(model_root: Path, device: str):
    import torch
    from safetensors.torch import load_file
    from kokoro.model import KModel

    checkpoint_path = model_root / "kokoro-v1_0.safetensors"
    config_path = model_root / "config.json"
    original_torch_load = torch.load

    def patched_torch_load(path, *args, **kwargs):
        if Path(path) == checkpoint_path:
            flat_state = load_file(str(checkpoint_path), device="cpu")
            grouped: dict[str, dict[str, object]] = {}
            for name, tensor in flat_state.items():
                prefix, rest = name.split(".", 1)
                grouped.setdefault(prefix, {})[rest] = tensor
            return grouped
        return original_torch_load(path, *args, **kwargs)

    torch.load = patched_torch_load
    try:
        model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(config_path),
            model=str(checkpoint_path),
        ).to(device).eval()
    finally:
        torch.load = original_torch_load
    return model


def load_voice_pack(model_root: Path, voice_id: str) -> np.ndarray:
    voices_json = json.loads((model_root / "voices.json").read_text(encoding="utf-8"))
    if voice_id not in voices_json:
        raise RuntimeError(f"unknown Kokoro voice id: {voice_id}")
    info = voices_json[voice_id]
    rows = int(info["rows"])
    cols = int(info["cols"])
    values = np.fromfile(model_root / "voices" / info["path"], dtype=np.float32)
    expected = rows * cols
    if values.size != expected:
        raise RuntimeError(f"voice pack size mismatch for {voice_id}: expected {expected}, got {values.size}")
    return values.reshape(rows, cols)


def summarize(audio: np.ndarray) -> dict[str, object]:
    audio = audio.astype(np.float32, copy=False)
    first_samples = audio[:32].tolist()
    return {
        "sample_rate": 24000,
        "channels": 1,
        "samples": int(audio.shape[0]),
        "sum": float(audio.sum(dtype=np.float64)),
        "mean_abs": float(np.abs(audio).mean(dtype=np.float64)) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
        "min": float(audio.min()) if audio.size else 0.0,
        "max": float(audio.max()) if audio.size else 0.0,
        "first_samples": first_samples,
    }


def timing_line(timestamp: str, key: str, value: object) -> str:
    if isinstance(value, str):
        return f"[TIMING ts={timestamp}] {key} {value}"
    if isinstance(value, bool):
        return f"[TIMING ts={timestamp}] {key} {1 if value else 0}"
    if isinstance(value, int):
        return f"[TIMING ts={timestamp}] {key} {value}"
    return f"[TIMING ts={timestamp}] {key} {float(value):.6f}"


def write_sectioned_timing_log(path: Path, sections: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, (name, lines) in enumerate(sections):
            output.write(f"[{name}]\n")
            for line in lines:
                output.write(f"{line}\n")
            if index + 1 < len(sections):
                output.write("\n")


def main() -> int:
    args = parse_args()
    texts = list(args.texts)
    if args.case_names:
        case_catalog = load_case_catalog(kCaseCatalogPath)
        for case_name in args.case_names:
            if case_name not in case_catalog:
                raise RuntimeError(f"unknown Kokoro warm bench case: {case_name}")
            texts.extend(case_catalog[case_name])
    if not texts:
        texts = ["Hello world. This is a warm benchmark for the Kokoro framework session."]
    stamp = args.artifact_stamp or timestamp_seconds_local()
    timing_path = args.timing_file or (
        REPO_ROOT / "build" / "logs" / "parity" / "kokoro_tts" / f"kokoro_tts_python_{args.backend}-{stamp}.log"
    )

    import torch

    torch.set_num_threads(max(1, args.threads))
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")

    from kokoro.pipeline import KPipeline
    from kokoro import profiler as kokoro_profiler

    model = load_reference_model(args.model, device)
    lang_code = resolve_requested_language_code(args.style_language, args.voice_id)
    pipeline = KPipeline(lang_code=lang_code, model=model)
    voice_pack = torch.from_numpy(load_voice_pack(args.model, args.voice_id)).unsqueeze(1)

    def run_once(text: str) -> tuple[dict[str, float], dict[str, object], np.ndarray]:
        if args.backend == "cuda":
            torch.cuda.synchronize()
        started_wall = time.perf_counter()
        torch.manual_seed(int(args.rng_seed))
        if args.backend == "cuda":
            torch.cuda.manual_seed_all(int(args.rng_seed))
            torch.cuda.synchronize()
        kokoro_profiler.reset()
        outputs = [item for item in pipeline(text, voice=voice_pack, speed=float(args.speaking_rate), split_pattern=r"\n+") if item.audio is not None]
        if args.backend == "cuda":
            torch.cuda.synchronize()
        ended_wall = time.perf_counter()
        component_metrics = kokoro_profiler.snapshot()
        kokoro_profiler.clear()
        if not outputs:
            raise RuntimeError("Kokoro reference pipeline produced no audio")
        if len(outputs) != 1:
            raise RuntimeError("Kokoro framework warm bench currently requires text that stays in one pipeline segment")
        result = outputs[0]
        audio_np = result.audio.detach().cpu().numpy().astype(np.float32, copy=False)
        wall_ms = (ended_wall - started_wall) * 1000.0
        component_metrics["kokoro.predictor.plbert_ms"] = (
            component_metrics.get("kokoro.python.plbert_model_ms", 0.0)
            + component_metrics.get("kokoro.python.plbert_project_ms", 0.0)
        )
        component_metrics["kokoro.predictor.duration_compute_ms"] = (
            component_metrics.get("kokoro.python.duration_encoder_ms", 0.0)
            + component_metrics.get("kokoro.python.duration_lstm_ms", 0.0)
            + component_metrics.get("kokoro.python.duration_proj_ms", 0.0)
            + component_metrics.get("kokoro.python.duration_post_ms", 0.0)
        )
        component_metrics["kokoro.predictor.pretail_ms"] = (
            component_metrics.get("kokoro.predictor.duration_compute_ms", 0.0)
            + component_metrics.get("kokoro.predictor.text_compute_ms", 0.0)
            + component_metrics.get("kokoro.predictor.expand_ms", 0.0)
            + component_metrics.get("kokoro.python.text_expand_ms", 0.0)
        )
        component_metrics["kokoro.predictor.tail_ms"] = (
            component_metrics.get("kokoro.python.predictor_tail_f0n_ms", 0.0)
            + component_metrics.get("kokoro.python.decoder_condition_ms", 0.0)
            + component_metrics.get("kokoro.python.decoder_encode_ms", 0.0)
            + component_metrics.get("kokoro.python.decoder_residual_condition_ms", 0.0)
            + component_metrics.get("kokoro.python.decoder_decode_ms", 0.0)
        )
        component_metrics["kokoro.decoder.generator_ms"] = (
            component_metrics.get("kokoro.python.generator_source_ms", 0.0)
            + component_metrics.get("kokoro.python.generator_upsample_blocks_ms", 0.0)
            + component_metrics.get("kokoro.python.generator_post_ms", 0.0)
        )
        component_metrics["kokoro.predictor_ms"] = (
            component_metrics.get("kokoro.predictor.plbert_ms", 0.0)
            + component_metrics.get("kokoro.predictor.pretail_ms", 0.0)
            + component_metrics.get("kokoro.predictor.tail_ms", 0.0)
        )
        component_metrics["kokoro.decoder_ms"] = (
            component_metrics.get("kokoro.decoder.generator_ms", 0.0)
            + component_metrics.get("kokoro.decoder.istft_ms", 0.0)
        )
        metrics = {"kokoro.synthesize_wall_ms": wall_ms}
        metrics.update(component_metrics)
        return metrics, summarize(audio_np), audio_np

    sections: list[tuple[str, list[str]]] = []
    warmup_passes = max(0, args.warmup)
    sums: list[dict[str, float]] = [dict() for _ in texts]
    last_summaries: list[dict[str, object] | None] = [None for _ in texts]
    last_audios: list[np.ndarray | None] = [None for _ in texts]
    last_audio: np.ndarray | None = None
    for warmup_index in range(warmup_passes):
        metrics, _summary, _audio = run_once(args.warmup_text)
        ts = timestamp_seconds_local()
        lines = []
        for key in sorted(metrics):
            lines.append(timing_line(ts, key, metrics[key]))
        sections.append((f"warmup{warmup_index + 1}", lines))

    for request_index, text in enumerate(texts):
        for iteration_index in range(max(1, args.iterations)):
            metrics, summary, audio_np = run_once(text)
            ts = timestamp_seconds_local()
            lines = []
            for key in sorted(metrics):
                lines.append(timing_line(ts, key, metrics[key]))
            sections.append((f"iteration{iteration_index + 1}.request{request_index + 1}", lines))
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    sums[request_index][key] = sums[request_index].get(key, 0.0) + float(value)
            last_summaries[request_index] = summary
            last_audios[request_index] = audio_np
            last_audio = audio_np

    write_sectioned_timing_log(timing_path, sections)

    for request_index, (text, summary) in enumerate(zip(texts, last_summaries)):
        if summary is None:
            continue
        print(f"text[{request_index}]={text}")
        print(f"summary_json[{request_index}]={json.dumps(summary, ensure_ascii=False)}")
        if len(texts) == 1 and request_index == 0:
            print(f"text={text}")
            print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")

    print(f"timing_out={timing_path}")
    if last_audio is None:
        raise RuntimeError("no audio was generated")
    if args.audio_out_dir is not None:
        args.audio_out_dir.mkdir(parents=True, exist_ok=True)
        for request_index, audio_np in enumerate(last_audios):
            if audio_np is None:
                continue
            request_audio_out = args.audio_out_dir / f"request_{request_index:02d}.wav"
            write_pcm16_wav(request_audio_out, audio_np, 24000)
            print(f"audio_out[{request_index}]={request_audio_out}")
    write_pcm16_wav(args.audio_out, last_audio, 24000)
    print(f"audio_out={args.audio_out}")
    for request_index, sums_for_request in enumerate(sums):
        print(f"average[{request_index}]")
        for key, value in sums_for_request.items():
            print(f"{key}={value / max(1, args.iterations):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
