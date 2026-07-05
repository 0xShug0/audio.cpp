#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "index-tts"
DEFAULT_MODEL = REPO_ROOT / "models" / "IndexTTS-2"
OFFICIAL_EXAMPLES = REPO_ROOT / "resources" / "index_tts2" / "official_examples"
DEFAULT_VOICE = OFFICIAL_EXAMPLES / "voice_02.wav"
DEFAULT_TEXT = "The palace is strict, no false rumors, Lady Qi!"
DEFAULT_EMOTION_VOICE = OFFICIAL_EXAMPLES / "emo_sad.wav"
DEFAULT_VOICE_EMOTION_REFERENCE = OFFICIAL_EXAMPLES / "voice_07.wav"
DEFAULT_VOICE_EMOTION_VECTOR = OFFICIAL_EXAMPLES / "voice_09.wav"
DEFAULT_VOICE_EMOTION_TEXT = OFFICIAL_EXAMPLES / "voice_12.wav"

TEST_CASES: dict[str, dict[str, Any]] = {
    "voice_clone": {
        "text": DEFAULT_TEXT,
        "voice_ref": str(DEFAULT_VOICE.relative_to(REPO_ROOT)),
        "seed": 1234,
    },
    "emotion_reference": {
        "text": "The old theater was empty, but every chair still seemed to remember the audience.",
        "voice_ref": str(DEFAULT_VOICE_EMOTION_REFERENCE.relative_to(REPO_ROOT)),
        "audio": str(DEFAULT_EMOTION_VOICE.relative_to(REPO_ROOT)),
        "seed": 1235,
    },
    "emotion_reference_alpha": {
        "text": "I tried to sound calm, but the storm outside made every word feel heavier.",
        "voice_ref": str(DEFAULT_VOICE_EMOTION_REFERENCE.relative_to(REPO_ROOT)),
        "audio": str(DEFAULT_EMOTION_VOICE.relative_to(REPO_ROOT)),
        "emotion_alpha": 0.9,
        "seed": 1236,
    },
    "emotion_vector": {
        "text": "I'm sorry, I really did forget, but I promise I will remember the important things.",
        "voice_ref": str(DEFAULT_VOICE_EMOTION_VECTOR.relative_to(REPO_ROOT)),
        "emotion_vector": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        "use_random_emotion": False,
        "seed": 1237,
    },
    "emotion_text": {
        "text": "Hide quickly. Someone is coming, and I do not think they are here to help us.",
        "voice_ref": str(DEFAULT_VOICE_EMOTION_TEXT.relative_to(REPO_ROOT)),
        "use_emotion_text": True,
        "emotion_alpha": 0.6,
        "use_random_emotion": False,
        "seed": 1238,
    },
    "emotion_text_description": {
        "text": "Hide quickly. Someone is coming, and I do not think they are here to help us.",
        "voice_ref": str(DEFAULT_VOICE_EMOTION_TEXT.relative_to(REPO_ROOT)),
        "use_emotion_text": True,
        "emotion_text": "You scared me. Are you a ghost?",
        "emotion_alpha": 0.6,
        "use_random_emotion": False,
        "seed": 1239,
    },
}


def timestamp_seconds_local() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference IndexTTS2 warmbench.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--text", action="append", dest="texts", default=[])
    parser.add_argument("--case", choices=tuple(TEST_CASES), default="voice_clone")
    parser.add_argument("--case-catalog", type=Path, default=None)
    parser.add_argument("--case-name", default="")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--voice-ref", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--emotion-alpha", type=float, default=1.0)
    parser.add_argument("--emotion-vector-json", default="")
    parser.add_argument("--use-emotion-text", action="store_true")
    parser.add_argument("--emotion-text", default="")
    parser.add_argument("--use-random-emotion", action="store_true")
    parser.add_argument("--text-chunk-size", type=int, default=120)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-cuda-kernel", action="store_true")
    parser.add_argument("--use-deepspeed", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--audio-out", type=Path, default=Path("index_tts2_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--timing-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def install_transformers452_overlay() -> Path:
    spec = importlib.util.find_spec("transformers452")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("transformers452 is not installed in the active Python environment")
    source = Path(next(iter(spec.submodule_search_locations))).resolve()
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"invalid transformers452 package path: {source}")

    overlay_root = REPO_ROOT / "build" / "index_tts2_transformers452_overlay"
    overlay_root.mkdir(parents=True, exist_ok=True)
    alias = overlay_root / "transformers"
    if alias.exists() or alias.is_symlink():
        if alias.is_dir() and not alias.is_symlink():
            shutil.rmtree(alias)
        else:
            alias.unlink()
    shutil.copytree(source, alias, ignore=shutil.ignore_patterns("__pycache__"))
    sys.path.insert(0, str(overlay_root))
    return source


def add_reference_path(reference_root: Path) -> Path:
    root = resolve_repo_path(reference_root).resolve()
    package = root / "indextts" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing IndexTTS reference package: {package}")
    sys.path.insert(0, str(root))
    return root


def local_aux_paths(model: Path) -> dict[str, str]:
    model_root = resolve_repo_path(model).resolve()
    cache_root = model_root / "hf_cache"
    paths = {
        "w2v_bert": cache_root / "w2v-bert-2.0",
        "semantic_codec": cache_root / "semantic_codec_model.safetensors",
        "campplus": cache_root / "campplus_cn_common.bin",
        "bigvgan": cache_root / "bigvgan",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError("missing local IndexTTS2 auxiliary model paths: " + ", ".join(missing))
    return {key: str(path) for key, path in paths.items()}


def seed_all(seed: int, backend: str) -> None:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    if backend == "cuda":
        torch.cuda.manual_seed_all(seed)


def sync_device(backend: str, device: int) -> None:
    if backend == "cuda":
        import torch

        torch.cuda.synchronize(device)


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_sequence_json:
        payload = json.loads(args.request_sequence_json)
        if not isinstance(payload, list):
            raise RuntimeError("--request-sequence-json must decode to a list")
        return payload
    if args.request_json:
        payload = json.loads(args.request_json)
        if not isinstance(payload, dict):
            raise RuntimeError("--request-json must decode to an object")
        return [payload]
    if args.case_catalog is not None:
        catalog_path = resolve_repo_path(args.case_catalog)
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("--case-catalog must contain a JSON object")
        case_name = args.case_name or args.case
        case = payload.get(case_name)
        if not isinstance(case, dict):
            raise RuntimeError(f"case-catalog does not contain object case: {case_name}")
        requests = case.get("requests")
        if not isinstance(requests, list) or not requests:
            raise RuntimeError(f"case-catalog case has no requests: {case_name}")
        return requests
    texts = args.texts if args.texts else [DEFAULT_TEXT]
    if args.texts:
        return [{"text": text} for text in texts]
    return [dict(TEST_CASES[args.case])]


def summarize_audio(path: Path, text: str) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).reshape(-1)
    if audio.size == 0:
        raise RuntimeError(f"empty IndexTTS2 output audio: {path}")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "frames": int(frames),
        "samples": int(audio.size),
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(audio, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(audio), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
        "request_char_count": len(text),
    }


def timing_line(timestamp: str, key: str, value: object) -> str:
    if isinstance(value, str):
        return f"[TIMING ts={timestamp}] {key} {value}"
    if isinstance(value, bool):
        return f"[TIMING ts={timestamp}] {key} {1 if value else 0}"
    if isinstance(value, int):
        return f"[TIMING ts={timestamp}] {key} {value}"
    return f"[TIMING ts={timestamp}] {key} {float(value):.6f}"


def write_timing(path: Path, sections: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, (name, lines) in enumerate(sections):
            output.write(f"[{name}]\n")
            for line in lines:
                output.write(line)
                output.write("\n")
            if index + 1 < len(sections):
                output.write("\n")


def sequence_summary(summaries: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    return {
        "family": "index_tts2",
        "backend": backend,
        "sequence_steps": [
            {
                "request_index": index,
                "stems": [
                    {
                        "name": "audio",
                        "audio": item["audio_out"],
                        "summary": {
                            key: item[key]
                            for key in (
                                "sample_rate",
                                "channels",
                                "frames",
                                "samples",
                                "duration_sec",
                                "sum",
                                "mean_abs",
                                "rms",
                                "min",
                                "max",
                            )
                        },
                    }
                ],
                "metrics": {"wall_ms": item["wall_ms"]},
            }
            for index, item in enumerate(summaries)
        ],
    }


def request_path(value: object, default: Path | None) -> Path | None:
    if value is None or value == "":
        return default
    return resolve_repo_path(Path(str(value))).resolve()


def main() -> int:
    args = parse_args()
    transformer_source = install_transformers452_overlay()
    reference_root = add_reference_path(args.reference_root)

    import torch
    import transformers
    from indextts.infer_v2 import IndexTTS2

    if transformers.__version__ != "4.52.1":
        raise RuntimeError(f"IndexTTS2 warmbench expected transformers 4.52.1, got {transformers.__version__}")
    if "transformers452" not in str(Path(transformers.__file__).resolve()):
        raise RuntimeError(f"IndexTTS2 warmbench imported transformers from unexpected path: {transformers.__file__}")

    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("IndexTTS2 warmbench requested CUDA, but torch.cuda.is_available() is false")
        torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"

    model_root = resolve_repo_path(args.model).resolve()
    timing_path = args.timing_file
    if timing_path is None:
        timing_path = (
            REPO_ROOT
            / "build"
            / "logs"
            / "parity"
            / "index_tts2"
            / f"index_tts2_python_{args.backend}-{timestamp_seconds_local()}.log"
        )
    else:
        timing_path = resolve_repo_path(timing_path).resolve()

    aux_paths = local_aux_paths(model_root)
    sections: list[tuple[str, list[str]]] = []
    timestamp = timestamp_seconds_local()
    sections.append((
        "config",
        [
            timing_line(timestamp, "index_tts2.reference_root", str(reference_root)),
            timing_line(timestamp, "index_tts2.model_root", str(model_root)),
            timing_line(timestamp, "index_tts2.transformers_path", str(transformer_source)),
            timing_line(timestamp, "index_tts2.backend", args.backend),
            timing_line(timestamp, "index_tts2.device", device),
            timing_line(timestamp, "index_tts2.use_fp16", args.use_fp16),
            timing_line(timestamp, "index_tts2.use_cuda_kernel", args.use_cuda_kernel),
        ],
    ))

    sync_device(args.backend, args.device)
    load_started = time.perf_counter()
    tts = IndexTTS2(
        cfg_path=str(model_root / "config.yaml"),
        model_dir=str(model_root),
        use_fp16=args.use_fp16,
        device=device,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        use_torch_compile=args.use_torch_compile,
        aux_paths=aux_paths,
    )
    sync_device(args.backend, args.device)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    sections.append(("load", [timing_line(timestamp_seconds_local(), "index_tts2.model_load_ms", load_ms)]))

    requests = load_requests(args)
    output_dir = resolve_repo_path(args.audio_out_dir).resolve() if args.audio_out_dir else None
    default_audio_out = resolve_repo_path(args.audio_out).resolve()
    summaries: list[dict[str, Any]] = []

    def run_once(section_name: str, request: dict[str, Any], output_path: Path) -> dict[str, Any]:
        text = str(request.get("text", "")).strip()
        if not text:
            raise RuntimeError("IndexTTS2 warmbench request missing text")
        seed = int(request.get("seed", args.seed))
        seed_all(seed, args.backend)
        spk_audio = request_path(request.get("voice_ref"), resolve_repo_path(args.voice_ref).resolve())
        emo_audio = request_path(request.get("audio"), None if args.audio is None else resolve_repo_path(args.audio).resolve())
        emotion_vector = request.get("emotion_vector", None)
        if emotion_vector is None and args.emotion_vector_json:
            emotion_vector = json.loads(args.emotion_vector_json)
        use_emotion_text = bool(request.get("use_emotion_text", args.use_emotion_text))
        emotion_text = str(request.get("emotion_text", args.emotion_text)).strip() or None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sync_device(args.backend, args.device)
        started = time.perf_counter()
        result = tts.infer(
            spk_audio_prompt=str(spk_audio),
            text=text,
            output_path=str(output_path),
            emo_audio_prompt=None if emo_audio is None else str(emo_audio),
            emo_alpha=float(request.get("emotion_alpha", args.emotion_alpha)),
            emo_vector=emotion_vector,
            use_emo_text=use_emotion_text,
            emo_text=emotion_text,
            use_random=bool(request.get("use_random_emotion", args.use_random_emotion)),
            verbose=bool(request.get("verbose", False)),
            max_text_tokens_per_segment=int(request.get("text_chunk_size", args.text_chunk_size)),
        )
        sync_device(args.backend, args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        summary = summarize_audio(output_path, text)
        summary.update({
            "text": text,
            "audio_out": str(output_path),
            "wall_ms": wall_ms,
            "return": None if result is None else str(result),
        })
        sections.append((
            section_name,
            [
                timing_line(timestamp_seconds_local(), "index_tts2.request_wall_ms", wall_ms),
                timing_line(timestamp_seconds_local(), "index_tts2.request_char_count", len(text)),
                timing_line(timestamp_seconds_local(), "index_tts2.output_samples", int(summary["samples"])),
                timing_line(timestamp_seconds_local(), "index_tts2.output_sample_rate", int(summary["sample_rate"])),
            ],
        ))
        return summary

    if args.warmup > 0:
        warmup_request = dict(requests[0])
        for index in range(args.warmup):
            path = default_audio_out.parent / f"{default_audio_out.stem}_warmup_{index}{default_audio_out.suffix}"
            run_once(f"warmup{index + 1}", warmup_request, path)

    for request_index, request in enumerate(requests):
        for iteration in range(max(1, args.iterations)):
            if output_dir is not None:
                output_path = output_dir / f"request_{request_index}_iter_{iteration}.wav"
            elif len(requests) == 1 and args.iterations == 1:
                output_path = default_audio_out
            else:
                output_path = default_audio_out.parent / f"{default_audio_out.stem}_request_{request_index}_iter_{iteration}{default_audio_out.suffix}"
            summaries.append(run_once(f"iteration{iteration + 1}.request{request_index + 1}", request, output_path))

    write_timing(timing_path, sections)
    sequence = sequence_summary(summaries, args.backend)
    if args.summary_file is not None:
        summary_path = resolve_repo_path(args.summary_file).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(sequence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for index, summary in enumerate(summaries):
        print(f"text[{index}]={summary['text']}")
        print(f"audio_out[{index}]={summary['audio_out']}")
        print(f"summary_json[{index}]={json.dumps(summary, ensure_ascii=False)}")
    if len(summaries) == 1:
        print(f"text={summaries[0]['text']}")
        print(f"audio_out={summaries[0]['audio_out']}")
    print(f"summary_json={json.dumps(sequence, ensure_ascii=False)}")
    print(f"timing_out={timing_path}")
    print(f"index_tts2.model_load_ms={load_ms:.6f}")
    if summaries:
        mean_ms = sum(float(item["wall_ms"]) for item in summaries) / float(len(summaries))
        print(f"index_tts2.request_wall_ms={mean_ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
