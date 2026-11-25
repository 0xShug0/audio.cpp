from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_POCKET_TTS_ROOT = REPO_ROOT / "reference" / "pocket-tts"
if str(REFERENCE_POCKET_TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_POCKET_TTS_ROOT))
kDefaultWarmupText = "At sunrise the studio monitors clicked on, and the first calibration phrase rolled across the room with steady timing."
kDefaultSharedNoiseSteps = 16384
kDefaultSharedNoisePath = REPO_ROOT / "build" / "logs" / "parity" / "pocket_tts" / "pocket_tts_noise.bin"


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
            raise RuntimeError("Pocket TTS warm bench case catalog entry is missing a [case] header")
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


def resolve_bundle_dir(model_dir: Path) -> Path:
    english_dir = model_dir / "languages" / "english"
    if english_dir.exists():
        return english_dir
    return model_dir


def write_local_config(model_dir: Path, output_dir: Path, language: str) -> Path:
    config_src = REFERENCE_POCKET_TTS_ROOT / "pocket_tts" / "config" / f"{language}.yaml"
    if not config_src.exists():
        raise FileNotFoundError(f"missing Pocket TTS local config: {config_src}")
    bundle_dir = resolve_bundle_dir(model_dir)
    config = yaml.safe_load(config_src.read_text(encoding="utf-8"))
    config["weights_path"] = str(bundle_dir / "model.safetensors")
    config["weights_path_without_voice_cloning"] = str(bundle_dir / "model.safetensors")
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(bundle_dir / "tokenizer.model")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dst = output_dir / f"pocket_tts_local_{language}.yaml"
    config_dst.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_dst


def resolve_voice_state_path(model_dir: Path, voice_id: str) -> Path:
    bundle_dir = resolve_bundle_dir(model_dir)
    for relative in (
        Path("embeddings_v3") / f"{voice_id}.safetensors",
        Path("embeddings") / f"{voice_id}.safetensors",
        Path("embeddings_v2") / f"{voice_id}.safetensors",
    ):
        candidate = bundle_dir / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing Pocket TTS voice state for {voice_id} under {bundle_dir}")


def resolve_voice_state(model, model_dir: Path, voice_id: str, clone_audio: Path | None):
    if clone_audio is not None:
        return model.get_state_for_audio_prompt(clone_audio.resolve())
    if not voice_id:
        raise RuntimeError("Pocket TTS Python warm bench requires --voice-id or --clone-audio")
    return model.get_state_for_audio_prompt(resolve_voice_state_path(model_dir, voice_id))


def fnv1a64_hex(values: np.ndarray) -> str:
    data = values.astype(np.float32, copy=False).tobytes()
    hash_value = 0xCBF29CE484222325
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & ((1 << 64) - 1)
    return f"{hash_value:016x}"


def resolve_noise_path(path: Path | None) -> Path:
    if path is None:
        return kDefaultSharedNoisePath
    return path if path.is_absolute() else (REPO_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm repeated Python benchmark for the Pocket TTS reference package.")
    parser.add_argument("--mode", choices=("parity", "performance"), default="parity")
    parser.add_argument("--cases", type=Path, help="Explicit [case] text catalog for --case-name.")
    parser.add_argument("--model", type=Path, default=Path("models/pocket-tts"))
    parser.add_argument("--text", action="append", dest="texts", default=[])
    parser.add_argument("--case-name", action="append", dest="case_names", default=[],
                        help="Append one named text case from --cases PATH.")
    parser.add_argument("--language", default="english")
    parser.add_argument("--voice-id", default="")
    parser.add_argument("--clone-audio", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--frames-after-eos", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--noise-clamp", type=float, default=-1.0)
    parser.add_argument("--eos-threshold", type=float, default=-4.0)
    parser.add_argument("--noise-file", type=Path)
    parser.add_argument("--warmup-text", default=kDefaultWarmupText)
    parser.add_argument("--audio-out", type=Path)
    parser.add_argument("--audio-out-dir", type=Path)
    parser.add_argument("--timing-file", type=Path)
    parser.add_argument("--artifact-stamp", default="")
    return parser.parse_args()


def synchronize_if_needed(torch_module, backend: str, device: int) -> None:
    if backend == "cuda":
        torch_module.cuda.synchronize(device)


def reseed_torch(torch_module, backend: str, seed: int) -> None:
    torch_module.manual_seed(seed)
    if backend == "cuda":
        torch_module.cuda.manual_seed_all(seed)


def timing_line(timestamp: str, key: str, value: object) -> str:
    if isinstance(value, str):
        return f'[TIMING ts={timestamp}] {key} "{value}"'
    if isinstance(value, bool):
        return f"[TIMING ts={timestamp}] {key} {1 if value else 0}"
    if isinstance(value, int):
        return f"[TIMING ts={timestamp}] {key} {value}"
    return f"[TIMING ts={timestamp}] {key} {float(value):.6f}"


def timing_lines_for_run(
    timestamp: str,
    text: str,
    metrics: dict[str, float | str],
    mode: str,
    noise_seed: int,
    noise_steps: int,
    noise_latent_dim: int,
    noise_hash: str,
    noise_path: Path | None,
) -> list[str]:
    lines = [
        timing_line(timestamp, "pocket_tts.request_text", text),
        timing_line(timestamp, "pocket_tts.request_char_count", metrics["pocket_tts.request_char_count"]),
        timing_line(
            timestamp,
            "pocket_tts.test_noise_mode",
            "shared_schedule_file_default" if mode == "parity" else "api_default_random",
        ),
        timing_line(timestamp, "pocket_tts.generated_steps", metrics["pocket_tts.generated_steps"]),
        timing_line(timestamp, "pocket_tts.inference_ms", metrics["pocket_tts.inference_ms"]),
        timing_line(timestamp, "pocket_tts.synthesize_wall_ms", metrics["pocket_tts.synthesize_wall_ms"]),
    ]
    if mode == "parity":
        lines[2:2] = [
            timing_line(timestamp, "pocket_tts.seed", metrics["pocket_tts.seed"]),
            timing_line(timestamp, "pocket_tts.test_noise_seed", noise_seed),
            timing_line(timestamp, "pocket_tts.test_noise_steps", noise_steps),
            timing_line(timestamp, "pocket_tts.test_noise_latent_dim", noise_latent_dim),
            timing_line(timestamp, "pocket_tts.test_noise_hash", noise_hash),
            timing_line(timestamp, "pocket_tts.test_noise_file", str(noise_path)),
        ]
    return lines


def write_sectioned_timing_log(path: Path, sections: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, (name, lines) in enumerate(sections):
            output.write(f"[{name}]\n")
            for line in lines:
                output.write(f"{line}\n")
            if index + 1 < len(sections):
                output.write("\n")


def summarize(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    mode: str,
    noise_seed: int | None = None,
    noise_steps: int | None = None,
    noise_latent_dim: int | None = None,
    noise_hash: str | None = None,
    noise_file: str | None = None,
    generated_steps: int | None = None,
) -> dict[str, object]:
    audio = audio.astype(np.float32, copy=False)
    summary: dict[str, object] = {
        "sample_rate": sample_rate,
        "channels": 1,
        "samples": int(audio.shape[0]),
        "sum": float(audio.sum(dtype=np.float64)),
        "mean_abs": float(np.abs(audio).mean(dtype=np.float64)) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
        "min": float(audio.min()) if audio.size else 0.0,
        "max": float(audio.max()) if audio.size else 0.0,
        "first_samples": audio[:32].tolist(),
        "request_char_count": len(text),
    }
    if mode == "parity":
        summary["test_noise_mode"] = "shared_schedule_file_default"
        summary["test_noise_seed"] = noise_seed
        summary["test_noise_steps"] = noise_steps
        summary["test_noise_latent_dim"] = noise_latent_dim
        summary["test_noise_hash"] = noise_hash
        summary["test_noise_file"] = noise_file
    else:
        summary["test_noise_mode"] = "api_default_random"
    if generated_steps is not None:
        summary["generated_steps"] = generated_steps
    return summary


def main() -> int:
    args = parse_args()
    texts = list(args.texts)
    if args.case_names:
        if args.cases is None:
            raise RuntimeError("--case-name requires --cases PATH")
        case_catalog = load_case_catalog(args.cases)
        for case_name in args.case_names:
            if case_name not in case_catalog:
                raise RuntimeError(f"unknown Pocket TTS warm bench case: {case_name}")
            texts.extend(case_catalog[case_name])
    if not texts:
        texts = ["We changed the benchmark request to a longer sentence so the Pocket TTS session exercises a larger prompt path during parity runs."]

    stamp = args.artifact_stamp or timestamp_seconds_local()
    timing_path = args.timing_file or (
        REPO_ROOT / "build" / "logs" / "parity" / "pocket_tts" / f"pocket_tts_python_{args.backend}-{stamp}.log"
    )
    audio_out = args.audio_out or Path(f"pocket_tts_python_{args.backend}_audio.wav")

    import torch
    import pocket_tts
    from pocket_tts import TTSModel

    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")

    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model_dir = args.model.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"missing Pocket TTS model directory: {model_dir}")
    local_config = write_local_config(
        model_dir,
        REPO_ROOT / "build" / "pocket_tts_python_warm_bench",
        args.language,
    )
    noise_clamp = None if args.noise_clamp < 0.0 else float(args.noise_clamp)
    if args.mode == "performance":
        model = TTSModel.load_model(config=local_config).to(device).eval()
    else:
        model = TTSModel.load_model(
            config=local_config,
            temp=args.temperature,
            noise_clamp=noise_clamp,
            eos_threshold=args.eos_threshold,
        ).to(device).eval()
    voice_state = resolve_voice_state(model, model_dir, args.voice_id, args.clone_audio)
    sample_rate = int(model.sample_rate)
    noise_path: Path | None = None
    master_noise_schedule: torch.Tensor | None = None
    noise_steps = 0
    noise_hash = ""
    if args.mode == "parity":
        noise_path = resolve_noise_path(args.noise_file)
        reseed_torch(torch, args.backend, args.seed)
        model.export_test_noise_file(noise_path, kDefaultSharedNoiseSteps)
        noise_schedule_np = np.fromfile(noise_path, dtype=np.float32).reshape((-1, model.flow_lm.ldim))
        master_noise_schedule = torch.from_numpy(noise_schedule_np)
        noise_steps = int(noise_schedule_np.shape[0])
        noise_hash = fnv1a64_hex(noise_schedule_np)

    def run_once(text: str) -> tuple[dict[str, float], dict[str, object], np.ndarray]:
        if args.mode == "parity":
            reseed_torch(torch, args.backend, args.seed)
        synchronize_if_needed(torch, args.backend, args.device)
        started_wall = time.perf_counter()
        generate_kwargs = {
            "frames_after_eos": None if args.frames_after_eos < 0 else args.frames_after_eos,
        }
        if args.mode == "parity":
            generate_kwargs["full_noise_schedule"] = master_noise_schedule.clone()
        audio = model.generate_audio(voice_state, text, **generate_kwargs)
        synchronize_if_needed(torch, args.backend, args.device)
        ended_wall = time.perf_counter()
        inference_ms = (ended_wall - started_wall) * 1000.0
        audio_np = audio.detach().cpu().numpy().astype(np.float32, copy=False)
        generated_steps = int(getattr(model, "last_generated_steps", 0))
        metrics = {
            "pocket_tts.request_char_count": len(text),
            "pocket_tts.inference_ms": inference_ms,
            "pocket_tts.synthesize_wall_ms": inference_ms,
            "pocket_tts.generated_steps": generated_steps,
        }
        if args.mode == "parity":
            metrics["pocket_tts.seed"] = args.seed
        return metrics, summarize(
            audio_np,
            sample_rate,
            text,
            args.mode,
            args.seed if args.mode == "parity" else None,
            noise_steps if args.mode == "parity" else None,
            model.flow_lm.ldim if args.mode == "parity" else None,
            noise_hash if args.mode == "parity" else None,
            str(noise_path) if args.mode == "parity" and noise_path is not None else None,
            generated_steps,
        ), audio_np

    sections: list[tuple[str, list[str]]] = []
    sums: list[dict[str, float]] = [dict() for _ in texts]
    warmup_summaries: list[dict[str, object]] = []
    last_summaries: list[dict[str, object] | None] = [None for _ in texts]
    last_audios: list[np.ndarray | None] = [None for _ in texts]
    last_audio: np.ndarray | None = None

    for warmup_index in range(max(0, args.warmup)):
        metrics, summary, _audio = run_once(args.warmup_text)
        warmup_summaries.append(summary)
        ts = timestamp_seconds_local()
        sections.append(
            (
                f"warmup{warmup_index + 1}",
                timing_lines_for_run(
                    ts,
                    args.warmup_text,
                    metrics,
                    args.mode,
                    args.seed,
                    noise_steps,
                    model.flow_lm.ldim,
                    noise_hash,
                    noise_path,
                ),
            )
        )

    for request_index, text in enumerate(texts):
        for iteration_index in range(max(1, args.iterations)):
            metrics, summary, audio_np = run_once(text)
            ts = timestamp_seconds_local()
            sections.append(
                (
                    f"iteration{iteration_index + 1}.request{request_index + 1}",
                    timing_lines_for_run(
                        ts,
                        text,
                        metrics,
                        args.mode,
                        args.seed,
                        noise_steps,
                        model.flow_lm.ldim,
                        noise_hash,
                        noise_path,
                    ),
                )
            )
            for key, value in metrics.items():
                if isinstance(value, str):
                    continue
                sums[request_index][key] = sums[request_index].get(key, 0.0) + float(value)
            last_summaries[request_index] = summary
            last_audios[request_index] = audio_np
            last_audio = audio_np

    write_sectioned_timing_log(timing_path, sections)

    for warmup_index, summary in enumerate(warmup_summaries):
        print(f"warmup_text[{warmup_index}]={args.warmup_text}")
        print(f"warmup_summary_json[{warmup_index}]={json.dumps(summary, ensure_ascii=False)}")

    for request_index, (text, summary) in enumerate(zip(texts, last_summaries)):
        if summary is None:
            continue
        print(f"text[{request_index}]={text}")
        print(f"summary_json[{request_index}]={json.dumps(summary, ensure_ascii=False)}")
        if len(texts) == 1 and request_index == 0:
            print(f"text={text}")
            print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")

    print(f"timing_out={timing_path}")
    print(f"noise_file={str(noise_path) if noise_path is not None else ''}")
    if last_audio is None:
        raise RuntimeError("no audio was generated")
    if args.audio_out_dir is not None:
        args.audio_out_dir.mkdir(parents=True, exist_ok=True)
        for request_index, audio_np in enumerate(last_audios):
            if audio_np is None:
                continue
            request_audio_out = args.audio_out_dir / f"request_{request_index:02d}.wav"
            write_pcm16_wav(request_audio_out, audio_np, sample_rate)
            print(f"audio_out[{request_index}]={request_audio_out}")
    write_pcm16_wav(audio_out, last_audio, sample_rate)
    print(f"audio_out={audio_out}")
    for request_index, sums_for_request in enumerate(sums):
        print(f"average[{request_index}]")
        for key, value in sums_for_request.items():
            print(f"{key}={value / max(1, args.iterations):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
