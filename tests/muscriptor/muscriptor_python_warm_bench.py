#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "muscriptor"
sys.path.insert(0, str(REFERENCE_ROOT))

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent  # noqa: E402
import muscriptor.accelerator  # noqa: E402
from muscriptor.tokenizer.mt3 import resolve_instrument_names  # noqa: E402
from muscriptor.transcription_model import TranscriptionModel  # noqa: E402


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def parse_csv_paths(value: str, fallback: Path) -> list[Path]:
    return [Path(item) for item in value.split(",") if item] if value else [fallback]


def parse_csv_keep_empty(value: str) -> list[str]:
    if value.strip().startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("--instruments-sequence must be a JSON string array")
        return parsed
    return value.split(",") if value else []


def parse_sequence(value: str) -> list[Any]:
    if not value:
        return []
    if value.strip().startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("sequence arguments must be JSON arrays")
        return parsed
    return value.split(",")


def sequence_value(values: list[Any], index: int, fallback: Any) -> Any:
    return values[index] if index < len(values) else fallback


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid bool value: {value}")


def optional_positive_int(value: Any) -> int | None:
    parsed = int(value)
    return parsed if parsed > 0 else None


def event_to_dict(event: NoteStartEvent | NoteEndEvent) -> dict[str, Any]:
    if isinstance(event, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(event)}
    return {
        "type": "end",
        "end_time": event.end_time,
        "start_event_index": event.start_event_index,
    }


def normalized_instruments(value: str) -> list[str] | None:
    if not value:
        return None
    return resolve_instrument_names([item for item in value.split(",") if item.strip()])


def transcribe_json(
    model: TranscriptionModel,
    audio: Path,
    instruments: str,
    *,
    use_sampling: bool,
    temperature: float,
    cfg_coef: float,
    batch_size: int | None,
    beam_size: int,
    prelude_forcing: bool,
) -> str:
    events = [
        event_to_dict(event)
        for event in model.transcribe(
            audio=audio,
            instruments=normalized_instruments(instruments),
            use_sampling=use_sampling,
            temperature=temperature,
            cfg_coef=cfg_coef,
            batch_size=batch_size,
            beam_size=beam_size,
            prelude_forcing=prelude_forcing,
        )
        if not isinstance(event, ProgressEvent)
    ]
    return json.dumps(events, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Python reference MuScriptor warmbench.")
    parser.add_argument("--model", default="models/muscriptor-small/model.safetensors")
    parser.add_argument("--audio", default="build/logs/muscriptor/input/headache_by_lost_deposit_1min_16k.wav")
    parser.add_argument("--warmup-audio", default="")
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--instruments", default="")
    parser.add_argument("--instruments-sequence", default="")
    parser.add_argument("--use-sampling", default="false")
    parser.add_argument("--use-sampling-sequence", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-sequence", default="")
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--cfg-coef-sequence", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-size-sequence", default="")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--beam-size-sequence", default="")
    parser.add_argument("--prelude-forcing", default="true")
    parser.add_argument("--prelude-forcing-sequence", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seed-sequence", default="")
    parser.add_argument("--timing-file", default="")
    parser.add_argument("--disable-cuda-autocast", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    model_path = resolve_path(args.model)
    model = TranscriptionModel.load_model(weights_path=model_path, device=device)
    if args.disable_cuda_autocast:
        model._model.autocast.enabled = False

    warmup_audio = resolve_path(args.warmup_audio) if args.warmup_audio else resolve_path(args.audio)
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        transcribe_json(
            model,
            warmup_audio,
            args.instruments,
            use_sampling=parse_bool(args.use_sampling),
            temperature=args.temperature,
            cfg_coef=args.cfg_coef,
            batch_size=optional_positive_int(args.batch_size),
            beam_size=args.beam_size,
            prelude_forcing=parse_bool(args.prelude_forcing),
        )

    request_paths = parse_csv_paths(args.audio_sequence, Path(args.audio))
    instrument_values = parse_csv_keep_empty(args.instruments_sequence)
    use_sampling_values = parse_sequence(args.use_sampling_sequence)
    temperature_values = parse_sequence(args.temperature_sequence)
    cfg_coef_values = parse_sequence(args.cfg_coef_sequence)
    batch_size_values = parse_sequence(args.batch_size_sequence)
    beam_size_values = parse_sequence(args.beam_size_sequence)
    prelude_forcing_values = parse_sequence(args.prelude_forcing_sequence)
    seed_values = parse_sequence(args.seed_sequence)
    timing_lines = [
        f"muscriptor.model_root {model_path}",
        f"muscriptor.backend {args.backend}",
    ]
    steps = []
    for request_index, audio in enumerate(request_paths):
        audio_path = resolve_path(audio)
        instruments = sequence_value(instrument_values, request_index, args.instruments)
        use_sampling = parse_bool(sequence_value(use_sampling_values, request_index, args.use_sampling))
        temperature = float(sequence_value(temperature_values, request_index, args.temperature))
        cfg_coef = float(sequence_value(cfg_coef_values, request_index, args.cfg_coef))
        batch_size = optional_positive_int(sequence_value(batch_size_values, request_index, args.batch_size))
        beam_size = int(sequence_value(beam_size_values, request_index, args.beam_size))
        prelude_forcing = parse_bool(sequence_value(prelude_forcing_values, request_index, args.prelude_forcing))
        seed = int(sequence_value(seed_values, request_index, args.seed))
        text_output = ""
        total_ms = 0.0
        for _ in range(args.iterations):
            torch.manual_seed(seed)
            muscriptor.accelerator.synchronize()
            started = time.perf_counter()
            text_output = transcribe_json(
                model,
                audio_path,
                instruments,
                use_sampling=use_sampling,
                temperature=temperature,
                cfg_coef=cfg_coef,
                batch_size=batch_size,
                beam_size=beam_size,
                prelude_forcing=prelude_forcing,
            )
            muscriptor.accelerator.synchronize()
            total_ms += (time.perf_counter() - started) * 1000.0
        wall_ms = total_ms / args.iterations
        print(f"average[{request_index}]")
        print(f"muscriptor.wall_ms={wall_ms}")
        timing_lines.append(f"muscriptor.request{request_index}.wall_ms {wall_ms:.6f}")
        timing_lines.append(f"muscriptor.request{request_index}.instruments {instruments}")
        timing_lines.append(f"muscriptor.request{request_index}.use_sampling {int(use_sampling)}")
        timing_lines.append(f"muscriptor.request{request_index}.temperature {temperature}")
        timing_lines.append(f"muscriptor.request{request_index}.cfg_coef {cfg_coef}")
        timing_lines.append(f"muscriptor.request{request_index}.batch_size {batch_size or 0}")
        timing_lines.append(f"muscriptor.request{request_index}.beam_size {beam_size}")
        timing_lines.append(f"muscriptor.request{request_index}.prelude_forcing {int(prelude_forcing)}")
        timing_lines.append(f"muscriptor.request{request_index}.seed {seed}")
        steps.append({
            "request_index": request_index,
            "audio": str(audio),
            "instruments": instruments,
            "use_sampling": use_sampling,
            "temperature": temperature,
            "cfg_coef": cfg_coef,
            "batch_size": batch_size,
            "beam_size": beam_size,
            "prelude_forcing": prelude_forcing,
            "seed": seed,
            "text_output": text_output,
            "word_timestamps": [],
            "metrics": {"wall_ms": wall_ms},
        })

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    print("summary_json=" + json.dumps(
        {"family": "muscriptor", "backend": args.backend, "sequence_steps": steps},
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
