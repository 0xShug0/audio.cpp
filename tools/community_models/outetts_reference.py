#!/usr/bin/env python3
"""Run the official OuteTTS 1.0 HF reference for audio.cpp parity.

The script intentionally keeps the official sampling configuration. It resets
the PyTorch CPU and CUDA generators immediately before each model.generate()
call, after prompt/profile construction, so no unrelated torch operation can
consume the request's sampling stream.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUESTS = REPO_ROOT / "tests" / "outetts" / "warm_bench_requests.json"


def framework_chunks(text: str, budget: int | None) -> list[str]:
    text = text.strip()
    if budget is None or len(text) <= budget:
        return [text]
    words = list(re.finditer(r"\S+", text))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        hard_end = start
        while hard_end < len(words):
            width = words[hard_end].end() - words[start].start()
            if width > budget:
                break
            hard_end += 1
        if hard_end == start:
            hard_end += 1
        end = hard_end
        if hard_end < len(words) and hard_end > start + 1:
            for index in range(hard_end, start + 1, -1):
                if words[index - 1].group()[-1:] in ".!?":
                    end = index
                    break
            if end == hard_end:
                for index in range(hard_end, start + 1, -1):
                    if words[index - 1].group()[-1:] in ",;:":
                        end = index
                        break
        chunks.append(text[words[start].start() : words[end - 1].end()].strip())
        start = end
    return chunks


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def reset_seed(torch_mod: Any, seed: int, device: str) -> None:
    torch_mod.manual_seed(seed)
    if device == "cuda":
        torch_mod.cuda.manual_seed_all(seed)


def memory_snapshot(torch_mod: Any, device: str) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        import psutil

        result["rss_mb"] = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:  # psutil is optional for reference generation.
        pass
    if device == "cuda":
        result["cuda_alloc_mb"] = torch_mod.cuda.memory_allocated() / (1024.0 * 1024.0)
        result["cuda_peak_mb"] = torch_mod.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        result["cuda_reserved_mb"] = torch_mod.cuda.memory_reserved() / (1024.0 * 1024.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dac", type=Path, required=True)
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "build" / "logs" / "outetts_python_fp32"
    )
    args = parser.parse_args()

    import torch
    import torchaudio
    import outetts

    model_path = args.model.resolve()
    dac_path = args.dac.resolve()
    request_path = args.request_file.resolve()
    request_base = request_path.parent.parent.parent if request_path.parent.name == "outetts" else REPO_ROOT
    output_dir = args.out_dir.resolve()
    wav_dir = output_dir / "outputs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    config = outetts.ModelConfig(
        model_path=str(model_path),
        tokenizer_path=str(model_path),
        interface_version=outetts.InterfaceVersion.V3,
        backend=outetts.Backend.HF,
        device=args.device,
        dtype=dtype,
        audio_codec_path=str(dac_path),
        max_seq_length=8192,
    )
    interface = outetts.Interface(config)
    alignment = json.loads(args.alignment_json.read_text(encoding="utf-8"))
    requests = json.loads(request_path.read_text(encoding="utf-8"))["requests"]

    speakers: dict[tuple[str, str], dict[str, Any]] = {}
    boundaries: dict[str, list[dict[str, Any]]] = {}
    memory: list[dict[str, Any]] = []
    stdout_lines = [
        "family=outetts",
        "reference=official_hf",
        f"dtype={args.dtype}",
        f"device={args.device}",
    ]

    for request in requests:
        name = str(request["name"])
        seed = int(request["seed"])
        max_new_tokens = int(request["max_tokens"])
        profile_ms = 0.0
        speaker = None

        started = time.perf_counter()
        voice_ref = request.get("voice_ref")
        if voice_ref:
            reference_text = str(request["reference_text"])
            reference_path = resolve_path(str(voice_ref), request_base)
            key = (str(reference_path), reference_text)
            if key not in speakers:
                profile_started = time.perf_counter()
                speakers[key] = interface.audio_processor.create_speaker_from_dict(
                    {
                        "audio": {"bytes": reference_path.read_bytes()},
                        "text": reference_text,
                        "words": [
                            {
                                "word": item["word"],
                                "start": item["start_sample"] / 16000.0,
                                "end": item["end_sample"] / 16000.0,
                            }
                            for item in alignment
                        ],
                    }
                )
                speakers[key]["interface_version"] = outetts.InterfaceVersion.V3.value
                profile_ms = (time.perf_counter() - profile_started) * 1000.0
            speaker = speakers[key]

        chunks = framework_chunks(request["text"], request.get("text_chunk_size"))
        audio_chunks = []
        request_boundaries = []
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for chunk in chunks:
            current_speaker = copy.deepcopy(speaker)
            prompt_text = interface.prompt_processor.get_completion_prompt(chunk, current_speaker)
            prompt = interface._prepare_prompt(prompt_text)
            prompt_tokens = int(prompt.shape[-1])
            generation = outetts.GenerationConfig(
                text=chunk,
                speaker=current_speaker,
                generation_type=outetts.GenerationType.REGULAR,
                max_length=prompt_tokens + max_new_tokens,
                sampler_config=outetts.SamplerConfig(
                    temperature=0.4,
                    repetition_penalty=1.1,
                    repetition_range=64,
                    top_k=40,
                    top_p=0.9,
                    min_p=0.05,
                ),
            )

            # This reset is deliberately adjacent to generate(). Prompt/profile
            # work above cannot consume the model-sampling RNG stream.
            reset_seed(torch, seed, args.device)
            full_tokens = interface.model.generate(prompt, generation)
            generated = full_tokens[prompt_tokens:]
            codebooks = interface.prompt_processor.extract_audio_from_tokens(generated)
            audio = interface.get_audio(generated).detach().cpu().flatten()
            audio_chunks.append(audio)
            request_boundaries.append(
                {
                    "prompt_text": prompt_text,
                    "prompt_ids": prompt.detach().cpu().flatten().tolist(),
                    "generated_ids": generated,
                    "c1": codebooks[0],
                    "c2": codebooks[1],
                    "audio_frames_24k": int(audio.numel()),
                }
            )

        if args.device == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        audio = torch.cat(audio_chunks)
        wav_path = wav_dir / f"{name}_1.wav"
        torchaudio.save(str(wav_path), audio.unsqueeze(0), 24000, encoding="PCM_S", bits_per_sample=16)
        audio_seconds = audio.numel() / 24000.0
        rtf = wall_ms / 1000.0 / audio_seconds

        snapshot = memory_snapshot(torch, args.device)
        snapshot.update(
            {
                "request_id": name,
                "wall_ms": wall_ms,
                "profile_ms": profile_ms,
                "audio_seconds": audio_seconds,
                "rtf": rtf,
            }
        )
        memory.append(snapshot)
        boundaries[name] = request_boundaries
        stdout_lines.extend(
            [
                f"request_id={name}",
                f"[TIMING] request.{name}.wall_ms {wall_ms}",
                f"[TIMING] request.{name}.profile_ms {profile_ms}",
                f"audio_seconds={audio_seconds}",
                f"rtf={rtf}",
                f"audio_out={wav_path}",
            ]
        )
        print(f"[REF] {name}: {wall_ms:.2f} ms, {audio_seconds:.5f}s, RTF={rtf:.3f}")

    (output_dir / "command.json").write_text(
        json.dumps(
            {
                "reference": "official OuteTTS HF",
                "model": str(model_path),
                "dac": str(dac_path),
                "requests": str(request_path),
                "dtype": args.dtype,
                "device": args.device,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (output_dir / "memory.json").write_text(
        json.dumps({"requests": memory}, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "boundaries.json").write_text(
        json.dumps(boundaries, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
